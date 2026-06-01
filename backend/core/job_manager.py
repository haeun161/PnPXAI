import asyncio
import os
import shutil
import threading
import time
from typing import Any, Optional, Union
from PIL import Image

from backend.api.schemas import JobStatus, ExplainerResult, PredictionItem

_jobs: dict[str, dict] = {}
_data: dict[str, Union[Image.Image, str, bytes]] = {}
_lock = threading.Lock()
_cleanup_task: Optional[asyncio.Task] = None

JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 10
CLEANUP_INTERVAL_SECONDS = 5 * 60
VISUALIZATION_DIR = os.path.join("backend", "heatmaps")


def create_job(job_id: str, task: str, model_name: str, explainer_names: list[str], ranking_metric: str = "mu_fidelity"):
    with _lock:
        if len(_jobs) >= MAX_JOBS:
            _evict_oldest_completed()

        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "task": task,
            "model_name": model_name,
            "explainer_names": explainer_names,
            "ranking_metric": ranking_metric,
            "predictions": None,
            "original_data_url": None,
            "results": [],
            "error_message": None,
            "created_at": time.time(),
        }


def store_uploaded_data(job_id: str, data: Union[Image.Image, str, bytes], task: str):
    _data[job_id] = data
    job_dir = os.path.join(VISUALIZATION_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    if task == "image" and isinstance(data, Image.Image):
        data.save(os.path.join(job_dir, "original.png"))
    elif task == "text" and isinstance(data, str):
        with open(os.path.join(job_dir, "original.txt"), "w", encoding="utf-8") as f:
            f.write(data)
    elif task == "timeseries" and isinstance(data, bytes):
        with open(os.path.join(job_dir, "original.csv"), "wb") as f:
            f.write(data)


def get_uploaded_data(job_id: str) -> Optional[Union[Image.Image, str, bytes]]:
    return _data.get(job_id)


def get_job(job_id: str) -> Optional[JobStatus]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None

        # Determine original data URL based on task
        task = job.get("task", "image")
        ext_map = {"image": "original.png", "text": "original.txt", "timeseries": "original.csv"}
        original_file = ext_map.get(task, "original.png")

        return JobStatus(
            job_id=job["job_id"],
            status=job["status"],
            task=task,
            model_name=job["model_name"],
            explainer_names=job["explainer_names"],
            ranking_metric=job.get("ranking_metric", "mu_fidelity"),
            predictions=[PredictionItem(**p) for p in job["predictions"]] if job["predictions"] else None,
            original_data_url=f"/api/jobs/{job_id}/original/{original_file}",
            results=[ExplainerResult(**r) for r in job["results"]],
            error_message=job["error_message"],
        )


def update_job_status(job_id: str, status: str, error_message: Optional[str] = None):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = status
            if error_message:
                _jobs[job_id]["error_message"] = error_message


def update_job_predictions(job_id: str, predictions: list[dict]):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["predictions"] = predictions


def update_result_step(job_id: str, explainer_name: str, step: str):
    """Update the current_step of a running result without replacing the full entry."""
    with _lock:
        if job_id in _jobs:
            for r in _jobs[job_id]["results"]:
                if r["explainer_name"] == explainer_name:
                    r["current_step"] = step
                    break


def update_job_result(job_id: str, result: dict):
    """Append or replace a result. No sorting - ranking is done in pipeline.py."""
    with _lock:
        if job_id in _jobs:
            results = _jobs[job_id]["results"]
            existing_idx = next(
                (i for i, r in enumerate(results) if r["explainer_name"] == result["explainer_name"]),
                None,
            )
            if existing_idx is not None:
                results[existing_idx] = result
            else:
                results.append(result)

            # Update overall status based on completion
            total = len(_jobs[job_id]["explainer_names"])
            done = len([r for r in results if r["status"] in ("completed", "not_supported", "failed")])
            if done >= total:
                _jobs[job_id]["status"] = "completed"
            elif done > 0:
                _jobs[job_id]["status"] = "partial"


def _evict_oldest_completed():
    completed_jobs = [
        (jid, j) for jid, j in _jobs.items() if j["status"] in ("completed", "failed")
    ]
    if not completed_jobs:
        return
    oldest_id = min(completed_jobs, key=lambda x: x[1]["created_at"])[0]
    _remove_job(oldest_id)


def _remove_job(job_id: str):
    _jobs.pop(job_id, None)
    _data.pop(job_id, None)
    job_dir = os.path.join(VISUALIZATION_DIR, job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)


def _cleanup_expired_jobs():
    now = time.time()
    with _lock:
        expired = [
            jid for jid, j in _jobs.items()
            if now - j["created_at"] > JOB_TTL_SECONDS
        ]
        for jid in expired:
            _remove_job(jid)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        _cleanup_expired_jobs()


def start_cleanup_task():
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_periodic_cleanup())


def stop_cleanup_task():
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        _cleanup_task = None
