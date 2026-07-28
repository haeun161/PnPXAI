from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from backend.core import uploaded_models
from backend.core.model_paths import local_file
from backend.renderers.timeseries_renderer import render_timeseries_attribution
from backend.tasks.base import TaskHandler

_loaded_models: dict[str, Any] = {}


# One entry per architecture, not per checkpoint: the picker shows "iTransformer" once,
# and `checkpoints` maps each sample dataset to the weights trained on it, so choosing
# the data chooses the model. Backend-only — the frontend receives an already-filtered
# dataset list and never learns which file backs which. A model with no `checkpoints`
# is dataset-agnostic and every sample is offered.
_TS_MODELS = {
    "itransformer": {
        "display_name": "iTransformer",
        "architecture": "",
        "description": "Long-term forecaster. Embeds each variate over the time axis, so "
                       "it runs on any channel count; the input/forecast window is read "
                       "from the checkpoint trained on the selected dataset.",
        # Candidates in preference order — models/timeseries is a volume shared with
        # another server, where files get replaced and removed mid-session, so naming a
        # single file means a deleted one takes the sample down.
        "checkpoints": {
            # The shared ETTh1 weights (24 in -> 5 out), by request; our own 96->96 run
            # was removed from the volume. Window lengths come from the file, so a swap
            # there changes the sample's horizon without any change here.
            "ETTh1.csv": ["iTransformer_etth1.pth"],
            "illness.csv": ["iTransformer_illness.pth"],
        },
    },
}


# Where a bundled sample's demonstrated forecast should start. Without an entry the
# forecast holds out the tail of the file, which is all that can be assumed about an
# arbitrary upload; with one, the sample opens on a chosen date instead.
_SAMPLE_FORECAST_START = {
    "illness.csv": "2020-03-03",
}


def sample_forecast_origin(data_name, timestamps):
    """Row the forecast horizon should start at for a bundled sample, or None for the
    default (hold out the tail of whatever was uploaded).

    Resolved against the file's own timestamps rather than stored as a row number, so the
    date stays the date if rows are ever added or removed. `searchsorted` rather than an
    equality test: it lands on the first row at or after the requested date, so a date
    that falls between samples still resolves instead of silently reverting to the tail.
    """
    start = _SAMPLE_FORECAST_START.get(data_name)
    if start is None or timestamps is None or len(timestamps) == 0:
        return None
    index = int(timestamps.searchsorted(pd.Timestamp(start)))
    return index if index < len(timestamps) else None


def timestamp_format(timestamps) -> str:
    """Shortest strftime format that still tells these timestamps apart.

    Daily and weekly series sit at midnight, where a "00:00" on every label is pure
    noise — and on a short context window it is the difference between axis labels that
    fit side by side and labels that collide into an unreadable smear.
    """
    if len(timestamps) and (timestamps.hour == 0).all() and (timestamps.minute == 0).all():
        return "%Y-%m-%d"
    return "%Y-%m-%d %H:%M"


def format_timestamps(timestamps) -> list[str]:
    fmt = timestamp_format(timestamps)
    return [t.strftime(fmt) for t in timestamps]


def _parse_time_column(time_series: pd.Series) -> list[str]:
    """Parse TIME column (e.g. '10hh45mm') into 'Day1 10:45' format labels."""
    labels = []
    day = 1
    prev_minutes = -1
    for val in time_series:
        s = str(val).strip()
        # Parse formats like "10hh45mm", "10:45", etc.
        import re
        m = re.match(r'(\d+)hh(\d+)mm', s)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
        else:
            m2 = re.match(r'(\d+):(\d+)', s)
            if m2:
                h, mi = int(m2.group(1)), int(m2.group(2))
            else:
                labels.append(s)
                continue
        total_minutes = h * 60 + mi
        if prev_minutes >= 0 and total_minutes < prev_minutes:
            day += 1
        prev_minutes = total_minutes
        labels.append(f"D{day} {h:02d}:{mi:02d}")
    return labels


# Columns that are known non-sensor (auto-detected and separated)
_NON_SENSOR_PATTERNS = {"boiler_no", "time", "timestamp", "date", "datetime", "index", "id"}


def _parse_ts_csv(raw_bytes: bytes, with_timestamps: bool = False):
    """Parse CSV into (tensor, col_names, time_labels).

    Automatically detects and separates:
    - Time columns → parsed into Day/HH:MM labels
    - ID/index columns → dropped
    Every remaining column is kept as a series to forecast — including one named "label"
    or "target", which for a forecaster is just another variable.
    Returns: (tensor, col_names, time_labels_or_None), plus a trailing pandas
    DatetimeIndex (or None) when `with_timestamps` is set — forecasters need real
    timestamps to build calendar covariates, not just display labels.
    """
    import io
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    lines = text.split("\n")

    # Detect if first line is header (non-numeric)
    first_line = lines[0].strip()
    has_header = False
    try:
        [float(v) for v in first_line.split(",")]
    except ValueError:
        has_header = True

    df = pd.read_csv(io.BytesIO(raw_bytes), header=0 if has_header else None)

    # Auto-detect special columns
    time_labels = None
    timestamps = None
    drop_cols = []

    if has_header:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            # Check for time columns
            if col_lower in _NON_SENSOR_PATTERNS or col_lower == "time":
                if df[col].dtype == object or np.issubdtype(df[col].dtype, np.datetime64):
                    # Real calendar timestamps ("2016-07-01 00:00:00") give both display
                    # labels and the covariates a forecaster needs; sensor-clock formats
                    # ("10hh45mm") only parse into labels.
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().all():
                        timestamps = pd.DatetimeIndex(parsed)
                        time_labels = format_timestamps(timestamps)
                    else:
                        time_labels = _parse_time_column(df[col])
                drop_cols.append(col)
            # Check for ID/index columns
            elif col_lower in _NON_SENSOR_PATTERNS:
                drop_cols.append(col)

        df = df.drop(columns=drop_cols, errors="ignore")

    # Drop sequential integer index columns
    if df.shape[1] > 1 and df.iloc[:, 0].dtype in (np.int64, np.float64):
        first_col = df.iloc[:, 0].values
        if np.allclose(first_col, np.arange(len(first_col))):
            df = df.iloc[:, 1:]

    col_names = list(df.columns)
    if not has_header:
        col_names = [f"var_{i+1}" for i in range(df.shape[1])]

    values = df.values.astype(np.float32)  # (seq_len, num_channels)
    # Tensor: (1, num_channels, seq_len)
    tensor = torch.tensor(values.T).unsqueeze(0)
    if with_timestamps:
        return tensor, col_names, time_labels, timestamps
    return tensor, col_names, time_labels


class TimeSeriesTaskHandler(TaskHandler):
    task_name = "timeseries"

    def get_model_datasets(self, model_name: str) -> Optional[list[str]]:
        """Sample files this model has checkpoints for, or None if it isn't tied to any."""
        checkpoints = _TS_MODELS.get(model_name, {}).get("checkpoints")
        return list(checkpoints) if checkpoints else None

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info.get("architecture", ""), "description": info["description"],
             "task": "timeseries"}
            for name, info in _TS_MODELS.items()
        ]

    # pnpxai recommends these for TS models but the pipeline can't run them here (CAM
    # needs 2D spatial maps; RAP & perturbation methods fail at runtime).
    _UNSUPPORTED = {"GradCam", "GuidedGradCam", "RAP", "Lime", "KernelShap"}

    def get_explainers(self, model_name: str) -> list[dict]:
        # Detection-driven, minus known-incompatible methods, so the list == what actually runs.
        from backend.core.explainer_catalog import detect_explainers
        model = self.load_model(model_name)
        return detect_explainers(model, self.get_modality(),
                                 cache_key=f"timeseries:{model_name}",
                                 exclude=self._UNSUPPORTED)

    def load_model(self, model_name: str, num_input_channels: int = 1,
                   dataset: Optional[str] = None) -> torch.nn.Module:
        """Resolve a model name to a loaded forecaster.

        Three kinds of name arrive here: a bundled preset, `upload:<id>` for a checkpoint
        the user supplied (by file or by URL — both land in the same store), and anything
        else, which is taken as a HuggingFace repo id. Only the presets are iTransformer;
        the other two are whatever the user brought, so the architecture is decided by
        backend.models.ts_loading rather than assumed here.
        """
        from backend.models.ts_loading import load_forecaster

        if model_name in _TS_MODELS:
            checkpoints = _TS_MODELS[model_name]["checkpoints"]
            # The chosen dataset selects the weights trained on it. Explainer detection
            # runs before any data is picked and only probes the architecture, which every
            # checkpoint shares, so an unknown dataset can answer with any of them.
            candidates = checkpoints.get(dataset) or next(iter(checkpoints.values()))
            paths = [local_file("timeseries", name) for name in candidates]
            # First candidate that is actually on disk; falling back to the first keeps
            # the "checkpoint not found" error pointing at the preferred name rather than
            # the last one tried.
            source = next((p for p in paths if p.exists()), paths[0])
        elif uploaded_models.is_upload(model_name):
            source = uploaded_models.path_for(model_name)
        else:
            source = model_name

        # Key on the source, not the model name — one preset name spans several files.
        cache_key = f"{source}_{num_input_channels}"
        from backend.core.device import to_device
        if cache_key not in _loaded_models:
            _loaded_models[cache_key] = load_forecaster(source, num_channels=num_input_channels)
        return to_device(_loaded_models[cache_key])

    def preprocess_input(self, raw_data: Any) -> Any:
        if isinstance(raw_data, bytes):
            tensor, col_names, time_labels, timestamps = _parse_ts_csv(
                raw_data, with_timestamps=True)
            result = {"tensor": tensor, "col_names": col_names}
            if time_labels is not None:
                result["time_labels"] = time_labels
            if timestamps is not None:
                result["timestamps"] = timestamps
            return result
        elif isinstance(raw_data, str):
            values = [float(v.strip()) for v in raw_data.split(",") if v.strip()]
            tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
            return {"tensor": tensor, "col_names": ["value"]}
        return raw_data

    def get_modality(self):
        from pnpxai.core.modality.modality import TimeSeriesModality
        return TimeSeriesModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        time_labels = None
        try:
            if isinstance(input_data, dict):
                tensor = input_data["tensor"]
                col_names = input_data["col_names"]
                signals = tensor.squeeze(0).detach().cpu().numpy()
                time_labels = input_data.get("time_labels")

            elif isinstance(input_data, torch.Tensor):
                signals = input_data.squeeze(0).detach().cpu().numpy()

                if signals.ndim == 1:
                    signals = signals.reshape(1, -1)

                col_names = [f"var_{i+1}" for i in range(signals.shape[0])]

            elif isinstance(input_data, bytes):
                tensor, col_names, time_labels = _parse_ts_csv(input_data)
                signals = tensor.squeeze(0).detach().cpu().numpy()

            else:
                attr_len = (
                    len(attribution.flatten())
                    if hasattr(attribution, "flatten")
                    else 10
                )
                signals = np.zeros((1, max(attr_len, 10)))
                col_names = ["value"]

        except Exception:
            attr_len = (
                len(attribution.flatten())
                if hasattr(attribution, "flatten")
                else 10
            )
            signals = np.zeros((1, max(attr_len, 10)))
            col_names = ["value"]

        return render_timeseries_attribution(signals, attribution, output_path, col_names, time_labels=time_labels)
