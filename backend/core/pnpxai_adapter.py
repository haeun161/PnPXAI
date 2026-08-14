"""Anti-corruption layer between PnPXAI output and platform schema."""

import math

import numpy as np
from typing import Optional


def normalize_attribution(attribution, task: str = "image") -> np.ndarray:
    """Normalize attribution to a numpy array scaled by its largest magnitude.

    Time-series keeps its sign, so the range is [-1, 1]; the other tasks aggregate with
    an absolute value or norm first and so come out in [0, 1].

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
        # Image (C, H, W) -> (H, W): L2 norm across channels avoids sign cancellation
        attr_np = np.linalg.norm(attr_np, axis=0)
    elif attr_np.ndim == 2:
        if task == "timeseries":
            # Time-series (channels, seq_len): keep per-channel *and* signed, so the
            # plot can tell "pushed the prediction up" from "pushed it down". There is
            # no axis to aggregate here, unlike image/text below.
            pass
        else:
            # Text (seq_len, hidden_dim) -> (seq_len,): mean over hidden dim
            attr_np = np.mean(np.abs(attr_np), axis=-1)
    elif attr_np.ndim == 1:
        attr_np = np.abs(attr_np)
    else:
        attr_np = np.abs(attr_np)

    # Normalize to [-1, 1] preserving sign (or [0, 1] for unsigned)
    abs_max = np.abs(attr_np).max()
    if abs_max > 0:
        attr_np = attr_np / abs_max

    return attr_np


def extract_metric_value(metric_result) -> Optional[float]:
    """Extract a scalar float from a PnPXAI metric result.

    A non-finite result counts as "not available" (None), not as a number: some metrics
    come back NaN rather than raising (LRP's Sensitivity on a transformer, for one), and a
    NaN propagates into rankings and is not JSON-serializable.
    """
    if metric_result is None:
        return None
    if isinstance(metric_result, (int, float)):
        value = float(metric_result)
    elif hasattr(metric_result, "item"):
        value = float(metric_result.item())
    elif isinstance(metric_result, np.ndarray):
        value = float(metric_result.mean())
    else:
        try:
            value = float(metric_result)
        except (TypeError, ValueError):
            return None
    return value if math.isfinite(value) else None
