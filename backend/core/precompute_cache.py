"""Serves cached XAI results for the platform's own sample data so /explain returns
instantly instead of recomputing, for the exact (task, file, model) combinations
covered by backend/scripts/precompute_samples.py.

The manifest never stores job-specific paths (job_id changes per request) — only
per-explainer metrics/status. Visualization URLs are built fresh at serve time; the
underlying image files are copied into the new job's own heatmaps/{job_id}/ directory
so every existing job-serving route keeps working unmodified.

Cache contents are generated locally (git-ignored) — run the precompute script after
cloning if you want the speedup.
"""
import hashlib
import json
import os
import shutil
import threading
import time
from typing import Optional

from backend.core.job_manager import (
    VISUALIZATION_DIR, update_job_status, update_job_predictions,
    update_job_result, rank_completed_results,
)

PRECOMPUTED_DIR = os.path.join("backend", "precomputed")
MANIFEST_PATH = os.path.join(PRECOMPUTED_DIR, "manifest.json")

# Bump when the pipeline changes in a way that makes old cached results wrong
# (attribution normalization, renderers, metric definitions, result schema, ...).
# Entries written under an older version are ignored and recomputed.
CACHE_VERSION = 1


def model_fingerprint(model) -> str:
    """Short content hash of a model's weights + shapes.

    Stored alongside each cache entry so results are invalidated automatically when a
    model with the same name changes (retrained, re-architected, or re-initialized).
    Memoized on the module instance — models are cached, so this runs once each.
    """
    cached = getattr(model, "_pnpxai_fingerprint", None)
    if cached:
        return cached

    h = hashlib.sha256()
    try:
        state = model.state_dict()
        keys = list(state.keys())
        h.update(repr([(k, tuple(state[k].shape)) for k in keys]).encode())
        # Sample values from a few tensors — enough to catch changed weights cheaply.
        for k in keys[:5] + keys[-5:]:
            t = state[k]
            if t.numel():
                h.update(t.detach().flatten()[:64].float().cpu().numpy().tobytes())
        fp = h.hexdigest()[:16]
    except Exception:
        fp = "unknown"

    try:
        model._pnpxai_fingerprint = fp
    except Exception:
        pass
    return fp

_lock = threading.Lock()
_manifest_cache: Optional[dict] = None
_manifest_mtime: Optional[float] = None


def file_sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sample_cache_dir(task: str, model_name: str, file_hash: str) -> str:
    return os.path.join(PRECOMPUTED_DIR, task, model_name, file_hash[:16])


def _load_manifest() -> dict:
    global _manifest_cache, _manifest_mtime
    with _lock:
        try:
            mtime = os.path.getmtime(MANIFEST_PATH)
        except OSError:
            return {}
        if _manifest_cache is None or mtime != _manifest_mtime:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                _manifest_cache = json.load(f)
            _manifest_mtime = mtime
        return _manifest_cache


def get_precomputed(task: str, file_hash: str, model_name: str) -> Optional[dict]:
    """Returns {"predictions": [...], "results": [...]} or None if uncached."""
    return _load_manifest().get(task, {}).get(file_hash, {}).get(model_name)


def is_fully_cached(cached: Optional[dict], explainer_names: list[str],
                    model_fp: Optional[str] = None) -> bool:
    """True only if the entry covers every requested explainer AND is still valid.

    Validity = written by the current CACHE_VERSION and (when a fingerprint is given)
    produced by the same model weights. Anything else falls through to a real run.
    """
    if not cached:
        return False
    if cached.get("cache_version") != CACHE_VERSION:
        return False
    if model_fp is not None and cached.get("model_fp") not in (None, model_fp):
        return False
    have = {r["explainer_name"] for r in cached.get("results", [])}
    return all(name in have for name in explainer_names)


# Staged reveal so a cached run still shows per-explainer progress instead of
# snapping to "done" instantly. Set to 0 to serve as fast as possible.
STEP_DELAY_SECONDS = float(os.environ.get("PNPXAI_PRECOMPUTE_STEP_DELAY", "2.0"))


def serve_precomputed(
    job_id: str, task: str, model_name: str, file_hash: str,
    explainer_names: list[str], ranking_metric: str, cached: dict,
) -> None:
    """Populate a freshly-created job from cached results — no computation."""
    from backend.core.device import begin_job, end_job
    # Serving from cache does no compute, but the fingerprint check above loads the model
    # (onto the GPU). Pair the job counter so it gets offloaded when idle, same as a real run.
    begin_job()
    try:
        _serve_precomputed(job_id, task, model_name, file_hash,
                           explainer_names, ranking_metric, cached)
    finally:
        end_job()


def _serve_precomputed(
    job_id: str, task: str, model_name: str, file_hash: str,
    explainer_names: list[str], ranking_metric: str, cached: dict,
) -> None:
    update_job_status(job_id, "running")

    if cached.get("predictions"):
        update_job_predictions(job_id, cached["predictions"])

    src_dir = sample_cache_dir(task, model_name, file_hash)
    dest_dir = os.path.join(VISUALIZATION_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    by_name = {r["explainer_name"]: r for r in cached.get("results", [])}
    entries = []
    for name in explainer_names:
        cached_r = by_name.get(name)
        if cached_r is None:
            continue

        update_job_result(job_id, {
            "explainer_name": name,
            "display_name": cached_r["display_name"],
            "status": "running",
            "current_step": "Explaining...",
        })
        if STEP_DELAY_SECONDS:
            time.sleep(STEP_DELAY_SECONDS)

        if cached_r["status"] == "completed" and os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                if fname.startswith(name):
                    shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))
        entry = {
            "explainer_name": cached_r["explainer_name"],
            "display_name": cached_r["display_name"],
            "status": cached_r["status"],
            "visualization_url": f"/api/jobs/{job_id}/visualizations/{name}.png" if cached_r["status"] == "completed" else None,
            "mu_fidelity": cached_r.get("mu_fidelity"),
            "abpc": cached_r.get("abpc"),
            "sensitivity": cached_r.get("sensitivity"),
            "complexity": cached_r.get("complexity"),
            "error_message": cached_r.get("error_message"),
            "token_attributions": cached_r.get("token_attributions"),
        }
        entries.append(entry)
        update_job_result(job_id, entry)

    rank_completed_results(entries, ranking_metric)
    for entry in entries:
        update_job_result(job_id, entry)

    update_job_status(job_id, "completed")
