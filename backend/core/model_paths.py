"""Resolve where locally-cached model weights live.

Deployment-portable: never hardcodes an absolute path. The model root is, in
priority order:
  1. $PNPXAI_MODEL_DIR                (explicit override for any deployment)
  2. <repo_root>/models              (default; == /project/models inside the
                                      docker container, where the repo is mounted
                                      at /project and /project/models is a shared
                                      volume)

Handlers call `local_dir(task, name)` / `local_file(task, filename)` to find a
pre-downloaded model, and fall back to the HuggingFace Hub / torchvision when the
path does not exist. So a fresh deployment still works before models are
downloaded; running backend/scripts/download_models.py just makes it
offline-capable and shares weights across containers via the mounted volume.
"""
import os
from pathlib import Path

# backend/core/model_paths.py -> parents[2] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def model_root() -> Path:
    env = os.environ.get("PNPXAI_MODEL_DIR")
    return Path(env) if env else _REPO_ROOT / "models"


def local_dir(task: str, name: str) -> Path:
    """Directory for a saved model (transformers/MOMENT snapshot layout)."""
    return model_root() / task / name


def local_file(task: str, filename: str) -> Path:
    """File for a saved model (e.g. torchvision state_dict .pth)."""
    return model_root() / task / filename
