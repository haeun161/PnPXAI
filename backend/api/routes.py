import asyncio
import uuid
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional

from backend.api.schemas import TaskInfo, ModelInfo, ExplainerInfo, JobStatus
from backend.tasks import get_task_handler, list_tasks
from backend.core.image_utils import load_and_validate_image
from backend.core.job_manager import create_job, get_job, store_uploaded_data, VISUALIZATION_DIR
from backend.core.pipeline import run_explanation_pipeline
from backend.optimizer.optimizer_service import (
    get_explainer_params, run_optimization, run_with_custom_params,
    save_history, get_history, get_history_record, load_record_input_data,
    delete_history_record,
)

router = APIRouter(prefix="/api")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.get("/tasks", response_model=list[TaskInfo])
async def get_tasks():
    return [TaskInfo(**t) for t in list_tasks()]


@router.get("/models", response_model=list[ModelInfo])
async def get_models(task: str = Query(...)):
    handler = get_task_handler(task)
    return [ModelInfo(**m) for m in handler.get_models()]


@router.get("/explainers", response_model=list[ExplainerInfo])
async def get_explainers(task: str = Query(...), model: Optional[str] = Query(None)):
    handler = get_task_handler(task)
    model_name = model or (handler.get_models()[0]["name"] if handler.get_models() else "")
    return [ExplainerInfo(**e) for e in handler.get_explainers(model_name)]


@router.post("/explain")
async def explain(
    task: str = Query(...),
    model_name: str = Query(...),
    explainer_names: str = Query(..., description="Comma-separated explainer names"),
    ranking_metric: str = Query("average", description="Metric for ranking: average, mu_fidelity, abpc, sensitivity, complexity"),
    file: UploadFile = File(...),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    # Validate task
    try:
        handler = get_task_handler(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Process input based on task
    if task == "image":
        try:
            data = load_and_validate_image(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif task == "text":
        data = contents.decode("utf-8", errors="replace")
    elif task == "timeseries":
        data = contents
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    # Parse explainer names
    names = [n.strip() for n in explainer_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one explainer must be selected.")

    # Normalize ranking metric - default to "average" if invalid or missing
    valid_metrics = ("average", "mu_fidelity", "abpc", "sensitivity", "complexity")
    if ranking_metric not in valid_metrics:
        ranking_metric = "average"

    # Create job
    job_id = str(uuid.uuid4())
    store_uploaded_data(job_id, data, task)
    create_job(job_id, task, model_name, names, ranking_metric)

    # Run pipeline in thread pool executor
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_explanation_pipeline, job_id, task, model_name, names, ranking_metric, {})

    return {"job_id": job_id}


def _has_unregistered_attention(model_obj) -> bool:
    """Check for attention modules not covered by pnpxai's type registry (e.g. HuggingFace ViT/Swin/CLIP)."""
    return any(
        "attention" in type(m).__name__.lower() and len(list(m.children())) > 0
        for _, m in model_obj.named_modules()
    )


@router.get("/recommend")
async def recommend_explainers(task: str = Query(...), model: str = Query(...)):
    from pnpxai.core.recommender.recommender import XaiRecommender, CAM_BASED_EXPLAINERS
    try:
        handler = get_task_handler(task)
        model_obj = handler.load_model(model)
        modality = handler.get_modality()
        output = XaiRecommender().recommend(modality, model_obj)
        explainers = output.explainers

        # pnpxai's detector only knows nn.MultiheadAttention and a few HF types.
        # For models like HuggingFace ViT/Swin/DeiT/CLIP whose attention classes
        # are not registered, we detect by class name and remove CAM-based methods.
        if _has_unregistered_attention(model_obj):
            explainers = [e for e in explainers if e not in CAM_BASED_EXPLAINERS]

        recommended_names = [e.__name__ for e in explainers]
        available_names = {e["name"] for e in handler.get_explainers(model) if e.get("compatible", True)}
        return {"recommended": [n for n in recommended_names if n in available_names]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate-model")
async def validate_model(task: str = Query(...), hf_model_id: str = Query(...)):
    """Validates a HuggingFace model ID by attempting to load it for the given task."""
    try:
        handler = get_task_handler(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if task == "timeseries":
        raise HTTPException(status_code=400, detail="Custom HuggingFace models are not supported for the timeseries task.")

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handler.load_model, hf_model_id)
        display_name = hf_model_id.split("/")[-1]
        return {"valid": True, "model_id": hf_model_id, "display_name": display_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model '{hf_model_id}': {e}")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.get("/jobs/{job_id}/visualizations/{filename}")
async def get_visualization(job_id: str, filename: str):
    viz_path = os.path.join(VISUALIZATION_DIR, job_id, filename)
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Visualization not found.")
    return FileResponse(viz_path, media_type="image/png")


@router.get("/jobs/{job_id}/original/{filename}")
async def get_original_data(job_id: str, filename: str):
    file_path = os.path.join(VISUALIZATION_DIR, job_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original data not found.")

    if filename.endswith(".png"):
        return FileResponse(file_path, media_type="image/png")
    elif filename.endswith(".txt"):
        return FileResponse(file_path, media_type="text/plain")
    elif filename.endswith(".csv"):
        return FileResponse(file_path, media_type="text/csv")
    return FileResponse(file_path)


# ── Optimizer Endpoints ──

@router.get("/samples/{task}")
async def get_samples(task: str, model: Optional[str] = Query(None)):
    """List available sample data files for a task. For timeseries, includes channel count and compatibility."""
    import glob
    sample_dir = os.path.join("sample_data", task)
    if not os.path.exists(sample_dir):
        return []
    files = glob.glob(os.path.join(sample_dir, "*"))
    result = []
    for f in sorted(files):
        name = os.path.basename(f)
        entry: dict = {"name": name, "path": f"/{task}/{name}"}

        # For timeseries, detect channel count and check model compatibility
        if task == "timeseries" and name.endswith(".csv"):
            try:
                from backend.tasks.timeseries import _parse_ts_csv, _TS_MODELS
                with open(f, "rb") as fh:
                    _, cols = _parse_ts_csv(fh.read())
                entry["channels"] = len(cols)
                entry["col_names"] = cols
                if model and model in _TS_MODELS:
                    default_ch = _TS_MODELS[model].get("default_channels")
                    if default_ch is not None and default_ch != len(cols):
                        entry["compatible"] = False
                        entry["reason"] = f"Model expects {default_ch}-channel, data has {len(cols)}"
                    else:
                        entry["compatible"] = True
                else:
                    entry["compatible"] = True
            except Exception:
                entry["compatible"] = True

        result.append(entry)
    return result


@router.get("/samples/{task}/{filename}")
async def get_sample_file(task: str, filename: str):
    """Serve a sample data file."""
    file_path = os.path.join("sample_data", task, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample not found")
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        return FileResponse(file_path, media_type="image/png")
    elif filename.endswith(".txt"):
        return FileResponse(file_path, media_type="text/plain")
    elif filename.endswith(".csv"):
        return FileResponse(file_path, media_type="text/csv")
    return FileResponse(file_path)


@router.get("/optimizer/params/{explainer_name}")
async def get_params(explainer_name: str):
    return get_explainer_params(explainer_name)


@router.post("/optimizer/optimize")
async def optimize(
    task: str = Query(...),
    model_name: str = Query(...),
    explainer_name: str = Query(...),
    metric_name: str = Query("AbPC"),
    n_trials: int = Query(20),
    file: UploadFile = File(...),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")

    handler = get_task_handler(task)
    if task == "image":
        input_data = load_and_validate_image(contents)
    elif task == "text":
        input_data = contents.decode("utf-8", errors="replace")
    else:
        input_data = contents

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, run_optimization, task, model_name, explainer_name, metric_name, input_data, n_trials
    )

    save_history(result)
    return result


@router.post("/optimizer/custom")
async def run_custom(
    task: str = Query(...),
    model_name: str = Query(...),
    explainer_name: str = Query(...),
    custom_params: str = Query("{}"),
    file: UploadFile = File(...),
):
    import json as json_mod
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")

    handler = get_task_handler(task)
    if task == "image":
        input_data = load_and_validate_image(contents)
    elif task == "text":
        input_data = contents.decode("utf-8", errors="replace")
    else:
        input_data = contents

    try:
        params = json_mod.loads(custom_params)
    except json_mod.JSONDecodeError:
        params = {}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, run_with_custom_params, task, model_name, explainer_name, params, input_data
    )
    return result


@router.get("/optimizer/history")
async def optimizer_history():
    return get_history()


@router.delete("/optimizer/history/{record_id}")
async def delete_record(record_id: str):
    from backend.optimizer.optimizer_service import delete_history_record
    success = delete_history_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "deleted"}


@router.get("/optimizer/history/{record_id}")
async def get_record(record_id: str):
    record = get_history_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.post("/optimizer/history/{record_id}/custom")
async def run_custom_from_history(
    record_id: str,
    explainer_name: str = Query(...),
    custom_params: str = Query("{}"),
):
    import json as json_mod
    record = get_history_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    task = record["task"]
    model_name = record["model_name"]
    input_data = load_record_input_data(record_id, task)
    if input_data is None:
        raise HTTPException(status_code=404, detail="Saved input data not found")

    try:
        params = json_mod.loads(custom_params)
    except json_mod.JSONDecodeError:
        params = {}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, run_with_custom_params, task, model_name, explainer_name, params, input_data
    )
    return result
