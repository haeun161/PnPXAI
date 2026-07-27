"""Storage for model weight files the user uploads instead of picking a preset.

An upload is addressed by an opaque id and referred to as the model name
``upload:<id>``, so it travels through the same /explain path as a preset model and
nothing downstream needs to know where the weights came from.
"""
import os
import time
import uuid

UPLOAD_DIR = os.path.join("backend", "uploaded_models")
PREFIX = "upload:"
TTL_SECONDS = 6 * 60 * 60


def is_upload(model_name: str) -> bool:
    return bool(model_name) and model_name.startswith(PREFIX)


def save(content: bytes) -> str:
    """Persist an uploaded checkpoint and return its ``upload:<id>`` model name."""
    _prune()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uid = uuid.uuid4().hex
    with open(os.path.join(UPLOAD_DIR, f"{uid}.pth"), "wb") as fh:
        fh.write(content)
    return f"{PREFIX}{uid}"


def path_for(model_name: str) -> str:
    uid = model_name[len(PREFIX):]
    # The id arrives from a query parameter and is about to become a filesystem path,
    # so require the exact shape uuid4().hex produces rather than sanitising in place.
    if len(uid) != 32 or not all(c in "0123456789abcdef" for c in uid):
        raise ValueError(f"Invalid uploaded-model id: {model_name}")
    path = os.path.join(UPLOAD_DIR, f"{uid}.pth")
    if not os.path.exists(path):
        raise ValueError("Uploaded model not found — it may have expired. Upload it again.")
    return path


def _prune() -> None:
    """Drop checkpoints older than the TTL so uploads don't accumulate on disk."""
    if not os.path.isdir(UPLOAD_DIR):
        return
    cutoff = time.time() - TTL_SECONDS
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass
