"""Anti-corruption layer between PnPXAI output and platform schema."""

import numpy as np
from typing import Optional


def normalize_attribution(attribution, task: str = "image") -> np.ndarray:
    """Normalize attribution tensor/array to numpy [0, 1] range.

    Args:
        task: "image" → (C,H,W)→(H,W), "text" → (seq,hidden)→(seq,), "timeseries" → (ch,seq) kept as-is
    """
    if hasattr(attribution, "detach"):
        attr_np = attribution.detach().cpu().numpy()
    elif isinstance(attribution, np.ndarray):
        attr_np = attribution
    else:
        attr_np = np.array(attribution)

    # Squeeze batch dimension if present (only once — GradCam returns (1,1,H,W))
    if attr_np.ndim >= 1 and attr_np.shape[0] == 1:
        attr_np = attr_np.squeeze(0)

    # Aggregate depending on task
    if attr_np.ndim == 3:
        # Image (C, H, W) -> (H, W): mean over channels
        attr_np = np.mean(np.abs(attr_np), axis=0)
    elif attr_np.ndim == 2:
        if task == "timeseries":
            # Time-series (channels, seq_len): keep as-is for per-channel attribution
            attr_np = np.abs(attr_np)
        else:
            # Text (seq_len, hidden_dim) -> (seq_len,): mean over hidden dim
            attr_np = np.mean(np.abs(attr_np), axis=-1)
    elif attr_np.ndim == 1:
        attr_np = np.abs(attr_np)
    else:
        attr_np = np.abs(attr_np)

    attr_max = attr_np.max()
    if attr_max > 0:
        attr_np = attr_np / attr_max

    return attr_np


def extract_metric_value(metric_result) -> Optional[float]:
    """Extract a scalar float from a PnPXAI metric result."""
    if metric_result is None:
        return None
    if isinstance(metric_result, (int, float)):
        return float(metric_result)
    if hasattr(metric_result, "item"):
        return float(metric_result.item())
    if isinstance(metric_result, np.ndarray):
        return float(metric_result.mean())
    try:
        return float(metric_result)
    except (TypeError, ValueError):
        return None
