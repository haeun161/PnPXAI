import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Any

from backend.tasks.base import TaskHandler
from backend.renderers.timeseries_renderer import render_timeseries_attribution

_TS_EXPLAINERS = [
    {"name": "IntegratedGradients", "display_name": "Integrated Gradients", "estimated_time": 5},
    {"name": "SmoothGrad", "display_name": "SmoothGrad", "estimated_time": 5},
    {"name": "GradientXInput", "display_name": "Gradient × Input", "estimated_time": 3},
    {"name": "Gradient", "display_name": "Gradient", "estimated_time": 2},
    {"name": "Lime", "display_name": "LIME", "estimated_time": 25},
    {"name": "KernelShap", "display_name": "KernelSHAP", "estimated_time": 25},
]

_loaded_models: dict[str, Any] = {}


class SimpleTimeSeriesModel(torch.nn.Module):
    """Simple 1D CNN for time-series classification demo. Supports multi-variate input."""
    def __init__(self, num_input_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(num_input_channels, 16, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(self.conv(x).squeeze(-1))


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
    def __init__(self, num_input_channels: int = 1, num_classes: int = 5, **kwargs):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
            self._pipeline = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large",
                model_kwargs={
                    'task_name': 'classification',
                    'n_channels': num_input_channels,
                    'num_class': num_classes,
                }
            )
            self._pipeline.init()
        except Exception as e:
            raise RuntimeError(f"Failed to load MOMENT model: {e}. Install with: pip install momentfm --no-deps")

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # MOMENT expects exactly 512 timesteps; pad/truncate if needed
        if x.shape[-1] < 512:
            pad = torch.zeros(*x.shape[:-1], 512 - x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > 512:
            x = x[..., :512]
        output = self._pipeline(x_enc=x)
        return output.logits


_TS_MODELS = {
    "moment-large": {
        "display_name": "MOMENT-1-Large",
        "architecture": "Transformer (T5)",
        "description": "MOMENT: pre-trained time-series foundation model (AutonLab, 2024). Classification via linear probing on T5 encoder embeddings. Input auto-padded to 512 timesteps.",
        "loader": MOMENTWrapper,
    },
    "simple-cnn-1d": {
        "display_name": "Simple 1D CNN",
        "architecture": "1D CNN",
        "description": "Simple 1D CNN for univariate time-series. Auto-adapts to multi-variate input.",
        "loader": SimpleTimeSeriesModel,
    },
    "inception-time": {
        "display_name": "InceptionTime",
        "architecture": "InceptionTime",
        "description": "InceptionTime classifier (Fawaz et al., 2020). Auto-adapts to multi-variate input.",
        "loader": InceptionTimeModel,
    },
}


def _parse_ts_csv(raw_bytes: bytes):
    """Parse CSV into (tensor, column_names). Supports single and multi-variate."""
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

    # Drop timestamp/index columns if they look like sequential integers
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
    return tensor, col_names


class TimeSeriesTaskHandler(TaskHandler):
    task_name = "timeseries"

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info.get("architecture", ""), "description": info["description"],
             "task": "timeseries"}
            for name, info in _TS_MODELS.items()
        ]

    def get_explainers(self, model_name: str) -> list[dict]:
        return [
            {"name": e["name"], "display_name": e["display_name"],
             "estimated_compute_time_seconds": e["estimated_time"],
             "compatible": True, "incompatibility_reason": None}
            for e in _TS_EXPLAINERS
        ]

    def load_model(self, model_name: str, num_input_channels: int = 1) -> torch.nn.Module:
        if model_name not in _TS_MODELS:
            raise ValueError(f"Unknown timeseries model: {model_name}")
        # Use default_channels from model config if available
        default_ch = _TS_MODELS[model_name].get("default_channels")
        ch = default_ch if default_ch is not None else num_input_channels
        cache_key = f"{model_name}_{ch}"
        if cache_key not in _loaded_models:
            model = _TS_MODELS[model_name]["loader"](num_input_channels=ch)
            model.eval()
            _loaded_models[cache_key] = model
        return _loaded_models[cache_key]

    def preprocess_input(self, raw_data: Any) -> Any:
        if isinstance(raw_data, bytes):
            tensor, col_names = _parse_ts_csv(raw_data)
            return {"tensor": tensor, "col_names": col_names}
        elif isinstance(raw_data, str):
            values = [float(v.strip()) for v in raw_data.split(",") if v.strip()]
            tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
            return {"tensor": tensor, "col_names": ["value"]}
        return raw_data

    def get_modality(self):
        from pnpxai.core.modality.modality import TimeSeriesModality
        return TimeSeriesModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        try:
            if isinstance(input_data, dict):
                tensor = input_data["tensor"]
                col_names = input_data["col_names"]
                signals = tensor.squeeze(0).detach().cpu().numpy()

            elif isinstance(input_data, torch.Tensor):
                signals = input_data.squeeze(0).detach().cpu().numpy()

                if signals.ndim == 1:
                    signals = signals.reshape(1, -1)

                col_names = [f"var_{i+1}" for i in range(signals.shape[0])]

            elif isinstance(input_data, bytes):
                tensor, col_names = _parse_ts_csv(input_data)
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

        return render_timeseries_attribution(signals, attribution, output_path, col_names)
