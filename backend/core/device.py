"""Compute device resolution + GPU residency management.

Device priority:
  1. $PNPXAI_DEVICE  (e.g. "cuda", "cuda:1", "cpu") — explicit override
  2. cuda if available, else cpu

Models are moved onto the device on every `load_model` (a no-op when already there)
and tracked here. When the last in-flight job finishes, the tracked models are moved
back to CPU and the CUDA cache is released, so an idle server holds no VRAM. The
model objects stay in the task handlers' caches, so the next job only pays a
host->device copy — no re-download or re-instantiation.

Set PNPXAI_KEEP_ON_GPU=1 to disable offloading (keeps models resident for lowest
latency at the cost of held VRAM).
"""
import os
import threading
import weakref

import torch

_device: torch.device | None = None
_lock = threading.Lock()
_tracked: list = []          # weakrefs to modules currently moved onto the device
_active_jobs = 0

_KEEP_ON_GPU = os.environ.get("PNPXAI_KEEP_ON_GPU", "").lower() in ("1", "true", "yes")


def get_device() -> torch.device:
    global _device
    if _device is None:
        env = os.environ.get("PNPXAI_DEVICE")
        if env:
            _device = torch.device(env)
        else:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def model_device(model) -> torch.device:
    """Device a model's parameters live on (falls back to get_device() if it has none)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return get_device()


def to_device(model):
    """Move a model onto the compute device and track it for later offloading.

    Safe to call on every load: `.to()` is a no-op when the model is already there.
    """
    device = get_device()
    model = model.to(device)
    if device.type == "cuda":
        with _lock:
            if not any(ref() is model for ref in _tracked):
                _tracked.append(weakref.ref(model))
    return model


def offload_models() -> None:
    """Move all tracked models back to CPU and release cached CUDA memory."""
    if get_device().type != "cuda":
        return
    with _lock:
        refs, _tracked[:] = list(_tracked), []
    for ref in refs:
        model = ref()
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def begin_job() -> None:
    global _active_jobs
    with _lock:
        _active_jobs += 1


def end_job() -> None:
    """Mark a job finished; offload models once nothing else is in flight."""
    global _active_jobs
    with _lock:
        _active_jobs = max(0, _active_jobs - 1)
        idle = _active_jobs == 0
    if idle and not _KEEP_ON_GPU:
        offload_models()
