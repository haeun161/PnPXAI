from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from backend.core import uploaded_models
from backend.core.model_paths import local_file
from backend.renderers.timeseries_renderer import render_timeseries_attribution
from backend.tasks.base import TaskHandler

_loaded_models: dict[str, Any] = {}


class ITransformerWrapper(torch.nn.Module):
    """iTransformer long-term forecaster (ETTh1 checkpoint: 96 in -> 96 out).

    Two entry points with deliberately different jobs:
      * `forward(x)` returns a single scalar per item — the next predicted step of the
        target channel — so the classification-shaped explainer pipeline can attribute it
        ("which input timesteps drove the next value").
      * `forecast_horizon(context)` returns the whole (pred_len, channels) horizon for
        the Input & Prediction chart.

    The architecture embeds each *variate* over the full time axis, so `seq_len` and
    `pred_len` are pinned by the weights while the channel count is free — the same
    checkpoint runs on any number of columns. Values are passed in raw units: the
    model's internal instance normalization divides out any dataset-level z-scoring
    and multiplies it back on the way out, so training-set statistics aren't needed.

    Two checkpoint layouts are accepted:
      * {"state_dict", "config"} — self-describing, so layer sizes and the horizon are
        read from the file; takes no timestamp covariates.
      * a bare state_dict — the reference iTransformer, which additionally consumes
        calendar covariates alongside the variate tokens.
    """
    CKPT_NAME = "iTransformer_etth1.pth"

    def __init__(self, ckpt_path=None):
        super().__init__()
        from backend.models.itransformer import Model, ITransformerNet
        # ckpt_path lets a user-uploaded weights file stand in for the bundled one.
        ckpt_path = Path(ckpt_path) if ckpt_path else local_file("timeseries", self.CKPT_NAME)
        if not ckpt_path.exists():
            raise RuntimeError(
                f"iTransformer checkpoint not found at {ckpt_path}. Place the trained "
                f"{self.CKPT_NAME} there."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt and "config" in ckpt:
            cfg = ckpt["config"]
            self._model = ITransformerNet(**cfg)
            self._model.load_state_dict(ckpt["state_dict"])
            self.SEQ_LEN, self.PRED_LEN = cfg["seq_len"], cfg["pred_len"]
            self._uses_covariates = False
        else:
            cfg = self._infer_reference_config(ckpt)
            self._model = Model(**cfg)
            self._model.load_state_dict(ckpt)
            self.SEQ_LEN, self.PRED_LEN = cfg["seq_len"], cfg["pred_len"]
            self._uses_covariates = True
        self._model.eval()
        # Timestamp covariates for the current request's context window, shaped
        # (1, SEQ_LEN, 4). Set by set_time_context(); None falls back to no covariates.
        self._x_mark = None

    @staticmethod
    def _infer_reference_config(state_dict: dict) -> dict:
        """Recover the reference iTransformer's layer sizes from the weights themselves.

        A bare state_dict carries no config, and the horizon differs per checkpoint
        (ETTh1 was trained 96->96, another dataset may be 12->3), so hardcoding a size
        would silently reject every checkpoint but one. Every dimension except the head
        count is pinned by a weight shape; heads only re-split an unchanged d_model
        matrix, so it stays at the reference default of 8.
        """
        try:
            embed = state_dict["enc_embedding.value_embedding.weight"]   # (d_model, seq_len)
            projector = state_dict["projector.weight"]                   # (pred_len, d_model)
            d_ff = state_dict["encoder.attn_layers.0.conv1.weight"].shape[0]
        except KeyError as e:
            raise RuntimeError(
                "Unrecognised time-series checkpoint: expected either a reference "
                f"iTransformer state_dict or a {{'state_dict', 'config'}} file (missing {e})."
            )
        e_layers = 1 + max(
            int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.attn_layers.")
        )
        return {
            "seq_len": embed.shape[1],
            "pred_len": projector.shape[0],
            "d_model": embed.shape[0],
            "d_ff": d_ff,
            "e_layers": e_layers,
            "n_heads": 8,
        }

    @staticmethod
    def _time_features(timestamps) -> np.ndarray:
        """Hourly calendar features for one context window. `timestamps` is a pandas
        DatetimeIndex of length SEQ_LEN. Returns (SEQ_LEN, 4)."""
        return np.stack([
            timestamps.hour / 23.0 - 0.5,
            timestamps.dayofweek / 6.0 - 0.5,
            (timestamps.day - 1) / 30.0 - 0.5,
            (timestamps.dayofyear - 1) / 365.0 - 0.5,
        ], axis=1).astype(np.float32)

    def set_time_context(self, timestamps) -> None:
        """Attach hourly time features for the context window. `timestamps` is a
        pandas DatetimeIndex of length SEQ_LEN, or None to drop the covariates."""
        if timestamps is None or not self._uses_covariates:
            self._x_mark = None
            return
        self._x_mark = torch.from_numpy(self._time_features(timestamps)).unsqueeze(0)

    def _fit_window(self, x: torch.Tensor) -> torch.Tensor:
        """Trim/pad the time axis to exactly SEQ_LEN (the weights fix this length)."""
        seq_len = x.shape[-1]
        if seq_len > self.SEQ_LEN:
            return x[..., -self.SEQ_LEN:]
        if seq_len < self.SEQ_LEN:
            pad = x[..., :1].expand(*x.shape[:-1], self.SEQ_LEN - seq_len)
            return torch.cat([pad, x], dim=-1)
        return x

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) -> (batch, pred_len, channels)."""
        x_enc = self._fit_window(x).transpose(1, 2)
        if not self._uses_covariates:
            return self._model(x_enc)
        x_mark = None
        if self._x_mark is not None:
            x_mark = self._x_mark.to(x.device, x.dtype).expand(x.shape[0], -1, -1)
        return self._model(x_enc, x_mark)

    @staticmethod
    def channel_for(num_channels: int) -> int:
        """The forecast target is always the last column of the upload, by convention —
        which is where ETT-style CSVs put it (OT) and where the checkpoint's own
        feature_names put the label. Deriving it from the data rather than from the
        checkpoint keeps any dataset working without per-file configuration."""
        return max(num_channels - 1, 0)

    def forward(self, x):
        # Single always-selected pseudo-class, so softmax/argmax downstream stay valid.
        out = self._run(x)
        ch = self.channel_for(out.shape[-1])
        return out[:, 0, ch:ch + 1]

    @torch.no_grad()
    def forecast_horizon(self, context: torch.Tensor) -> torch.Tensor:
        """context: (channels, seq_len) or (1, channels, seq_len) -> (pred_len, channels)."""
        if context.dim() == 2:
            context = context.unsqueeze(0)
        return self._run(context)[0]

    @torch.no_grad()
    def forecast_windows(self, contexts: torch.Tensor, timestamps_list=None) -> torch.Tensor:
        """Batched sibling of forecast_horizon, for a chained back-test's worth of
        windows at once: contexts (n_windows, channels, seq_len) -> (n_windows, pred_len,
        channels). One forward pass instead of one per window -- the difference between
        milliseconds and a minute once the chain covers a whole dataset's tail.

        `timestamps_list`, if the checkpoint uses covariates, is a same-length list of
        each window's own pandas DatetimeIndex -- a single shared `set_time_context`
        would otherwise apply one window's calendar features to every window in the batch.
        """
        x_enc = self._fit_window(contexts).transpose(1, 2)
        if not self._uses_covariates or timestamps_list is None:
            return self._model(x_enc)
        feats = np.stack([self._time_features(ts) for ts in timestamps_list], axis=0)
        x_mark = torch.from_numpy(feats).to(contexts.device, contexts.dtype)
        return self._model(x_enc, x_mark)


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
        "checkpoints": {
            "ETTh1.csv": "iTransformer_etth1.pth",
            "illness.csv": "iTransformer_illness.pth",
        },
    },
}


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
                        time_labels = [t.strftime("%Y-%m-%d %H:%M") for t in timestamps]
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
        if model_name in _TS_MODELS:
            checkpoints = _TS_MODELS[model_name]["checkpoints"]
            # The chosen dataset selects the weights trained on it. Explainer detection
            # runs before any data is picked and only probes the architecture, which every
            # checkpoint shares, so an unknown dataset can answer with any of them.
            filename = checkpoints.get(dataset) or next(iter(checkpoints.values()))
            ckpt_path = local_file("timeseries", filename)
        elif uploaded_models.is_upload(model_name):
            # User-supplied weights. Only the self-describing {state_dict, config}
            # layout is loadable — a bare state_dict carries no architecture.
            ckpt_path = uploaded_models.path_for(model_name)
        else:
            raise ValueError(
                f"Unknown timeseries model: {model_name}. Pick a preset or upload a "
                f"checkpoint."
            )
        # Key on the checkpoint, not the model name — one name now spans several files.
        cache_key = f"{ckpt_path}_{num_input_channels}"
        from backend.core.device import to_device
        if cache_key not in _loaded_models:
            model = ITransformerWrapper(ckpt_path=ckpt_path)
            model.eval()
            _loaded_models[cache_key] = model
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

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str,
                      display_name: str | None = None) -> str:
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

        return render_timeseries_attribution(signals, attribution, output_path, col_names,
                                             time_labels=time_labels, display_name=display_name)
