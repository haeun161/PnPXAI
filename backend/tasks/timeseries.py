import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Any

from backend.tasks.base import TaskHandler
from backend.core.model_paths import local_dir
from backend.renderers.timeseries_renderer import render_timeseries_attribution

_loaded_models: dict[str, Any] = {}

# Fixed seed for the randomly-initialized time-series models, so weights (and therefore
# predictions/attributions) are reproducible across restarts.
_INIT_SEED = 42


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels: int, nb_filters: int = 32, kernel_sizes=(10, 20, 40), bottleneck_size: int = 32):
        super().__init__()
        self.bottleneck = torch.nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(bottleneck_size, nb_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.mp_conv = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.bn_act = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1)),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        conv_outs = [c(self.bottleneck(x)) for c in self.convs]
        mp_out = self.mp_conv(x)
        # Trim to match lengths (MaxPool can produce +1 on even-length inputs)
        min_len = min(o.shape[-1] for o in conv_outs + [mp_out])
        out = torch.cat([o[..., :min_len] for o in conv_outs] + [mp_out[..., :min_len]], dim=1)
        return self.bn_act(out)


class InceptionTimeModel(torch.nn.Module):
    """InceptionTime: time-series classification (Fawaz et al., 2020). Supports multi-variate."""
    def __init__(self, num_input_channels: int = 1, num_classes: int = 2, nb_filters: int = 32, depth: int = 3):
        super().__init__()
        nb_out = nb_filters * 4
        self.blocks = torch.nn.Sequential(*[
            InceptionBlock(num_input_channels if i == 0 else nb_out, nb_filters) for i in range(depth)
        ])
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv1d(num_input_channels, nb_out, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(nb_out),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(nb_out, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = F.relu(self.blocks(x) + self.shortcut(x))
        return self.fc(self.pool(out).squeeze(-1))


class MOMENTWrapper(torch.nn.Module):
    """Wrapper around MOMENTPipeline to make it behave like a standard nn.Module.
    MOMENT uses model(x_enc=x) and returns output.logits, which PnPXAI explainers
    can't handle directly. This wrapper accepts (batch, channels, seq_len) and returns logits tensor.
    """
    REQUIRED_SEQ_LEN = 512

    def __init__(self, num_input_channels: int = 1, num_classes: int = 5,
                 model_name: str = "AutonLab/MOMENT-1-large", local_key: str = None, **kwargs):
        super().__init__()
        # Prefer a locally-saved snapshot (models/timeseries/<local_key>); else HF Hub.
        source, local_only = model_name, False
        if local_key:
            d = local_dir("timeseries", local_key)
            if d.exists():
                source, local_only = str(d), True
        try:
            from momentfm import MOMENTPipeline
            self._pipeline = MOMENTPipeline.from_pretrained(
                source,
                local_files_only=local_only,
                model_kwargs={
                    'task_name': 'classification',
                    'n_channels': num_input_channels,
                    'num_class': num_classes,
                }
            )
            self._pipeline.init()
        except Exception as e:
            raise RuntimeError(f"Failed to load MOMENT model ({model_name}): {e}. Install with: pip install momentfm --no-deps")

    def forward(self, x):
        # x: (batch, channels, seq_len)
        seq_len = x.shape[-1]
        if seq_len < self.REQUIRED_SEQ_LEN:
            pad = torch.zeros(*x.shape[:-1], self.REQUIRED_SEQ_LEN - seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif seq_len > self.REQUIRED_SEQ_LEN:
            x = x[..., :self.REQUIRED_SEQ_LEN]

        # Build input_mask: 1 for real data, 0 for padding
        input_mask = torch.ones(x.shape[0], self.REQUIRED_SEQ_LEN, device=x.device)
        if seq_len < self.REQUIRED_SEQ_LEN:
            input_mask[:, seq_len:] = 0

        output = self._pipeline(x_enc=x, input_mask=input_mask)
        return output.logits


def _moment_large_loader(**kwargs):
    return MOMENTWrapper(model_name="AutonLab/MOMENT-1-large", local_key="moment-large", **kwargs)

def _moment_small_loader(**kwargs):
    return MOMENTWrapper(model_name="AutonLab/MOMENT-1-small", local_key="moment-small", **kwargs)


_TS_MODELS = {
    "moment-large": {
        "display_name": "MOMENT-1-Large",
        "architecture": "Transformer (T5)",
        "description": "MOMENT large: pre-trained time-series foundation model. Input auto-padded to 512 timesteps.",
        "loader": _moment_large_loader,
    },
    "moment-small": {
        "display_name": "MOMENT-1-Small",
        "architecture": "Transformer (T5)",
        "description": "MOMENT small: lightweight version, faster inference. Input auto-padded to 512 timesteps.",
        "loader": _moment_small_loader,
    },
    "inception-time": {
        "display_name": "InceptionTime",
        "architecture": "InceptionTime",
        "description": "InceptionTime classifier (Fawaz et al., 2020). Auto-adapts to multi-variate input.",
        "loader": InceptionTimeModel,
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
_LABEL_PATTERNS = {"abnormal", "label", "class", "target", "fault", "anomaly"}


def _parse_ts_csv(raw_bytes: bytes):
    """Parse CSV into (tensor, col_names, time_labels, label_info).

    Automatically detects and separates:
    - Time columns → parsed into Day/HH:MM labels
    - Label/target columns → extracted for classification display
    - ID/index columns → dropped
    Returns: (tensor, sensor_col_names, time_labels_or_None, label_info_or_None)
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
    label_info = None
    drop_cols = []

    if has_header:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            # Check for label/target columns
            if any(p in col_lower for p in _LABEL_PATTERNS):
                unique_vals = df[col].dropna().unique()
                label_info = {
                    "column": col,
                    "values": df[col].values,
                    "classes": sorted(unique_vals.tolist()),
                }
                drop_cols.append(col)
            # Check for time columns
            elif col_lower in _NON_SENSOR_PATTERNS or col_lower == "time":
                # Try to parse as time labels
                if df[col].dtype == object:
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
    return tensor, col_names, time_labels, label_info


class TimeSeriesTaskHandler(TaskHandler):
    task_name = "timeseries"

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info.get("architecture", ""), "description": info["description"],
             "task": "timeseries"}
            for name, info in _TS_MODELS.items()
        ]

    # pnpxai recommends these for TS models but the pipeline can't run them on 1D-conv /
    # MOMENT inputs (CAM needs 2D spatial maps; RAP & perturbation methods fail at runtime).
    _UNSUPPORTED = {"GradCam", "GuidedGradCam", "RAP", "Lime", "KernelShap"}

    # Per-model additions: InceptionTime concatenates branch outputs (torch.cat), which
    # breaks zennit's LRP rules the same way DenseNet's dense blocks do.
    _MODEL_UNSUPPORTED = {
        "inception-time": {"LRPUniformEpsilon", "LRPEpsilonPlus", "LRPEpsilonGammaBox",
                           "LRPEpsilonAlpha2Beta1"},
    }

    def get_explainers(self, model_name: str) -> list[dict]:
        # Detection-driven, minus known-incompatible methods, so the list == what actually runs.
        from backend.core.explainer_catalog import detect_explainers
        model = self.load_model(model_name)
        exclude = self._UNSUPPORTED | self._MODEL_UNSUPPORTED.get(model_name, set())
        return detect_explainers(model, self.get_modality(),
                                 cache_key=f"timeseries:{model_name}",
                                 exclude=exclude)

    def load_model(self, model_name: str, num_input_channels: int = 1) -> torch.nn.Module:
        if model_name not in _TS_MODELS:
            raise ValueError(f"Unknown timeseries model: {model_name}")
        # Use default_channels from model config if available
        default_ch = _TS_MODELS[model_name].get("default_channels")
        ch = default_ch if default_ch is not None else num_input_channels
        cache_key = f"{model_name}_{ch}"
        from backend.core.device import to_device
        if cache_key not in _loaded_models:
            # These models have no pretrained weights (Simple-CNN / InceptionTime are
            # randomly initialized, MOMENT's classification head likewise). Seed the
            # construction so weights are identical across restarts — otherwise results
            # aren't reproducible and precomputed caches wouldn't match the live model.
            # RNG state is saved/restored so we don't perturb anything else.
            rng_state = torch.get_rng_state()
            try:
                torch.manual_seed(_INIT_SEED)
                model = _TS_MODELS[model_name]["loader"](num_input_channels=ch)
            finally:
                torch.set_rng_state(rng_state)
            model.eval()
            _loaded_models[cache_key] = model
        return to_device(_loaded_models[cache_key])

    def preprocess_input(self, raw_data: Any) -> Any:
        if isinstance(raw_data, bytes):
            tensor, col_names, time_labels, label_info = _parse_ts_csv(raw_data)
            result = {"tensor": tensor, "col_names": col_names}
            if time_labels is not None:
                result["time_labels"] = time_labels
            if label_info is not None:
                result["label_info"] = label_info
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
                tensor, col_names, time_labels, _ = _parse_ts_csv(input_data)
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
