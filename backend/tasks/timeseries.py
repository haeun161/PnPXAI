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
    """Simple 1D CNN for time-series classification demo."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x):
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
        out = torch.cat([c(self.bottleneck(x)) for c in self.convs] + [self.mp_conv(x)], dim=1)
        return self.bn_act(out)


class InceptionTimeModel(torch.nn.Module):
    """InceptionTime: time-series classification (Fawaz et al., 2020)."""
    def __init__(self, num_classes: int = 2, nb_filters: int = 32, depth: int = 3):
        super().__init__()
        nb_out = nb_filters * 4
        self.blocks = torch.nn.Sequential(*[
            InceptionBlock(1 if i == 0 else nb_out, nb_filters) for i in range(depth)
        ])
        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv1d(1, nb_out, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(nb_out),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(nb_out, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = F.relu(self.blocks(x) + self.shortcut(x))
        return self.fc(self.pool(out).squeeze(-1))


_TS_MODELS = {
    "simple-cnn-1d": {
        "display_name": "Simple 1D CNN",
        "architecture": "1D CNN",
        "description": "Simple 1D convolutional network for time-series demo.",
        "loader": SimpleTimeSeriesModel,
    },
    "inception-time": {
        "display_name": "InceptionTime",
        "architecture": "Inception",
        "description": "InceptionTime: deep learning classifier for time-series (Fawaz et al., 2020).",
        "loader": InceptionTimeModel,
    },
}


class TimeSeriesTaskHandler(TaskHandler):
    task_name = "timeseries"

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info["architecture"], "description": info["description"],
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

    def load_model(self, model_name: str) -> torch.nn.Module:
        if model_name not in _TS_MODELS:
            raise ValueError(f"Unknown timeseries model: {model_name}")
        if model_name not in _loaded_models:
            model = _TS_MODELS[model_name]["loader"]()
            model.eval()
            _loaded_models[model_name] = model
        return _loaded_models[model_name]

    def preprocess_input(self, raw_data: Any) -> Any:
        if isinstance(raw_data, bytes):
            import io
            df = pd.read_csv(io.BytesIO(raw_data))
            values = df.iloc[:, 0].values.astype(np.float32)
            return torch.tensor(values).unsqueeze(0)  # (1, seq_len)
        elif isinstance(raw_data, str):
            values = [float(v.strip()) for v in raw_data.split(",") if v.strip()]
            return torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        return raw_data

    def get_modality(self):
        from pnpxai.core.modality.modality import TimeSeriesModality
        return TimeSeriesModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        if isinstance(input_data, torch.Tensor):
            signal = input_data.squeeze().numpy()
        elif isinstance(input_data, bytes):
            import io
            df = pd.read_csv(io.BytesIO(input_data))
            signal = df.iloc[:, 0].values.astype(np.float32)
        else:
            signal = np.zeros(len(attribution))
        return render_timeseries_attribution(signal, attribution, output_path)
