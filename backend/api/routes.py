import asyncio
import math
import uuid
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional

from backend.api.schemas import TaskInfo, ModelInfo, ExplainerInfo, JobStatus
from backend.tasks import get_task_handler, list_tasks
from backend.core.image_utils import load_and_validate_image
from backend.core.job_manager import (
    create_job, get_job, store_uploaded_data, VISUALIZATION_DIR,
    update_job_status, update_job_predictions, update_job_result, request_cancel,
)
from backend.core.pipeline import run_explanation_pipeline
from backend.core import precompute_cache, uploaded_models
from backend.optimizer.optimizer_service import (
    get_explainer_params, run_optimization, run_with_custom_params,
    save_history, get_history, get_history_record, load_record_input_data,
    delete_history_record,
)

router = APIRouter(prefix="/api")

# In-memory store for detect-rank jobs
_detect_rank_jobs: dict[str, dict] = {}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (large CSV time-series support)
MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500MB (uploaded model weights)


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
    # get_explainers now loads the model + runs pnpxai detection; keep it off the event loop.
    loop = asyncio.get_running_loop()
    try:
        explainers = await loop.run_in_executor(None, handler.get_explainers, model_name)
    except ValueError as e:
        # Unknown / malformed model name — a client error, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))
    return [ExplainerInfo(**e) for e in explainers]


@router.post("/explain")
async def explain(
    task: str = Query(...),
    model_name: str = Query(...),
    explainer_names: str = Query(..., description="Comma-separated explainer names"),
    ranking_metric: str = Query("average", description="Metric for ranking: average, mu_fidelity, abpc, sensitivity, complexity"),
    data_name: Optional[str] = Query(None, description="Sample file name, when the data came from the sample list"),
    ts_window_index: Optional[int] = Query(
        None, description="Time-series only: re-anchor the backtest so this chained "
                           "window's context is what's explained (a chart window click)."
    ),
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

    loop = asyncio.get_running_loop()

    # Serve from cache if this is one of our own sample files with precomputed results.
    # The cached entry must also match the current cache version and model weights,
    # otherwise we fall through and recompute. Precomputed results only ever describe
    # the default (window 0) backtest, so a window-click re-explain must always run live.
    file_hash = precompute_cache.file_sha256(contents)
    cached = None if ts_window_index is not None else precompute_cache.get_precomputed(task, file_hash, model_name)
    model_fp = None
    if cached:
        # Loading a model can be slow (MOMENT-large is GBs) — keep it off the event loop.
        def _fp():
            try:
                return precompute_cache.model_fingerprint(handler.load_model(model_name))
            except Exception:
                return None
        model_fp = await loop.run_in_executor(None, _fp)
    if precompute_cache.is_fully_cached(cached, names, model_fp):
        loop.run_in_executor(None, precompute_cache.serve_precomputed,
                             job_id, task, model_name, file_hash, names, ranking_metric, cached)
    else:
        loop.run_in_executor(None, run_explanation_pipeline,
                             job_id, task, model_name, names, ranking_metric,
                             {"data_name": data_name, "ts_window_index": ts_window_index})

    return {"job_id": job_id}


def _has_unregistered_attention(model_obj) -> bool:
    """Check for attention modules not covered by pnpxai's type registry (e.g. HuggingFace ViT/Swin/CLIP)."""
    return any(
        "attention" in type(m).__name__.lower() and len(list(m.children())) > 0
        for _, m in model_obj.named_modules()
    )


@router.get("/recommend")
async def recommend_explainers(task: str = Query(...), model: str = Query(...)):
    from backend.core import explainer_catalog
    try:
        handler = get_task_handler(task)
        loop = asyncio.get_running_loop()
        model_obj = await loop.run_in_executor(None, handler.load_model, model)
        modality = handler.get_modality()
        cache_key = f"{task}:{model}"

        # Detection is shared (and cached) with /explainers via explainer_catalog, so a
        # /explain click firing both requests in parallel only pays for it once.
        def _detect():
            return (
                explainer_catalog.detect_recommended_names(model_obj, modality, cache_key=cache_key),
                explainer_catalog.detect_architectures(model_obj, modality, cache_key=cache_key),
            )
        recommended_names, detected_arch_names = await loop.run_in_executor(None, _detect)

        available_names = {e["name"] for e in handler.get_explainers(model) if e.get("compatible", True)}
        recommended_names = recommended_names or []
        return {
            "recommended": [n for n in recommended_names if n in available_names],
            "detected_architectures": detected_arch_names,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_detect_rank(job_id: str, task: str, model_name: str, input_data):
    """Background: run all compatible explainers, evaluate 4 metrics, rank, and store
    results + visualizations in a linked explain job so GO never re-runs."""
    import copy
    import torch
    from pnpxai.core.recommender.recommender import XaiRecommender, CAM_BASED_EXPLAINERS
    from backend.core.pipeline import (
        _get_pnpxai_explainer, _get_pnpxai_metric,
        _run_image_inference, _run_text_inference,
        _TextInputIdsWrapper, _GRADIENT_FREE_TEXT_EXPLAINERS,
        _find_cam_target_layer,
    )
    from backend.core.pnpxai_adapter import normalize_attribution, extract_metric_value

    from backend.core.device import begin_job, end_job

    job = _detect_rank_jobs[job_id]

    # Allocate linked explain job upfront so the frontend can poll it after GO
    explain_job_id = str(uuid.uuid4())
    job["linked_job_id"] = explain_job_id

    begin_job()  # paired with end_job() below (keeps the GPU offloader from pulling
    # this model mid-computation while a concurrent /explain job finishes)
    try:
        handler = get_task_handler(task)
        model = handler.load_model(model_name)
        modality = handler.get_modality()

        # Architecture detection + compatible explainers
        output = XaiRecommender().recommend(modality, model)
        explainers_list = output.explainers
        if _has_unregistered_attention(model):
            explainers_list = [e for e in explainers_list if e not in CAM_BASED_EXPLAINERS]

        all_names = [e.__name__ for e in explainers_list]
        available = {e["name"] for e in handler.get_explainers(model_name) if e.get("compatible", True)}
        explainer_names = [n for n in all_names if n in available]
        explainer_info_map = {e["name"]: e for e in handler.get_explainers(model_name)}

        job["total"] = len(explainer_names)
        job["detected_architectures"] = sorted([a.__name__ for a in output.detected_architectures])

        # Create the linked explain job now that we know explainer names
        create_job(explain_job_id, task, model_name, explainer_names, "average")
        store_uploaded_data(explain_job_id, input_data, task)
        update_job_status(explain_job_id, "running")
        job_dir = os.path.join(VISUALIZATION_DIR, explain_job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Task-specific inference (done once)
        target_tensor = None
        input_tensor = None
        explainer_model = model
        text_input_ids = None
        tokens_for_viz = None
        viz_input = input_data  # default fallback

        if task == "image":
            proc = handler.preprocess_input(input_data, model_name)
            hf_label_map = getattr(handler, "get_hf_label_map", lambda m: {})(model_name)
            target_class, predictions, _ = _run_image_inference(model, proc, hf_label_map)
            input_tensor = proc
            target_tensor = torch.tensor([target_class], dtype=torch.long)
            explainer_model = model
            update_job_predictions(explain_job_id, predictions)
            viz_input = input_data
        elif task == "text":
            text = input_data if isinstance(input_data, str) else str(input_data)
            target_class, predictions, emb, ids, tokens_for_viz, wrap = _run_text_inference(handler, model, text, model_name)
            input_tensor = emb
            text_input_ids = ids
            target_tensor = torch.tensor([target_class], dtype=torch.long)
            explainer_model = wrap
            update_job_predictions(explain_job_id, predictions)
            viz_input = tokens_for_viz if tokens_for_viz else text
        elif task == "timeseries":
            proc = handler.preprocess_input(input_data)
            if isinstance(proc, dict) and "tensor" in proc:
                ts_tensor = proc["tensor"]
                num_ch = ts_tensor.shape[1]
                model = handler.load_model(model_name, num_input_channels=num_ch)
                explainer_model = model
                model.eval()
                with torch.no_grad():
                    out = model(ts_tensor)
                target_class = int(out.argmax(dim=1).item())
                input_tensor = ts_tensor
                viz_input = proc
            else:
                input_tensor = proc if isinstance(proc, torch.Tensor) else None
                target_class = 0
            target_tensor = torch.tensor([target_class], dtype=torch.long)

        # The metrics index the model output with this tensor, so it has to live on the
        # same device as the activations. A CPU target against a CUDA model still gets
        # through attribution but makes Sensitivity raise ("index is on cpu"), which is
        # why it used to come back None for every explainer here.
        if target_tensor is not None and input_tensor is not None:
            target_tensor = target_tensor.to(input_tensor.device)

        _STATE_MUTATING = {"LRPUniformEpsilon", "LRPEpsilonPlus", "LRPEpsilonGammaBox", "LRPEpsilonAlpha2Beta1", "RAP"}
        METRIC_KEYS = [("mu_fidelity", "MuFidelity"), ("abpc", "AbPC"),
                       ("sensitivity", "Sensitivity"), ("complexity", "Complexity")]

        results = []
        for i, exp_name in enumerate(explainer_names):
            job["current"] = i + 1
            job["current_explainer"] = exp_name
            display_name = explainer_info_map.get(exp_name, {}).get("display_name", exp_name)

            try:
                ExplainerClass = _get_pnpxai_explainer(exp_name)

                if task == "text" and exp_name in _GRADIENT_FREE_TEXT_EXPLAINERS:
                    from pnpxai.explainers.utils.feature_masks import NoMask1d
                    active_model = _TextInputIdsWrapper(model)
                    active_inp = text_input_ids.clone()
                    exp_inst = ExplainerClass(active_model, feature_mask_fn=NoMask1d())
                elif exp_name in _STATE_MUTATING:
                    active_model = copy.deepcopy(explainer_model)
                    active_inp = input_tensor.clone()
                    if active_inp.is_floating_point():
                        active_inp = active_inp.requires_grad_(True)
                    exp_inst = ExplainerClass(active_model)
                else:
                    active_model = explainer_model
                    active_inp = input_tensor.clone()
                    if active_inp.is_floating_point():
                        active_inp = active_inp.requires_grad_(True)
                    exp_inst = ExplainerClass(active_model)

                if task == "image" and exp_name in {"GradCam", "GuidedGradCam"}:
                    if hasattr(exp_inst, "set_target_layer"):
                        cam_layer = _find_cam_target_layer(active_model)
                        if cam_layer:
                            exp_inst = exp_inst.set_target_layer(cam_layer)

                job["current_step"] = "attribution"
                attr_raw = exp_inst.attribute(active_inp, target_tensor)
                attribution = normalize_attribution(attr_raw, task=task)

                metrics = {}
                for key, cls_name in METRIC_KEYS:
                    if task == "text" and cls_name == "MuFidelity":
                        metrics[key] = None
                        continue
                    job["current_step"] = key
                    try:
                        m = _get_pnpxai_metric(cls_name, active_model, exp_inst)
                        val = extract_metric_value(m.evaluate(active_inp, target_tensor, attr_raw))
                        metrics[key] = val
                    except Exception:
                        metrics[key] = None

                # Rank score, same convention as the explain pipeline and the UI:
                # Sensitivity/Complexity are lower-is-better, so their sign is flipped
                # before averaging — averaging them raw ranked the *least* robust
                # explainer first. A metric that could not be computed (exception, or a
                # NaN from the metric itself) counts as 0 rather than being dropped, so
                # an explainer that produces no metrics at all can't win by default.
                flip = {"sensitivity", "complexity"}
                scores = [
                    0.0 if v is None or math.isnan(v) else (-v if key in flip else v)
                    for key, v in metrics.items()
                    if not (task == "text" and key == "mu_fidelity")
                ]
                avg = sum(scores) / len(scores) if scores else 0.0

                # Render visualization into linked job dir
                job["current_step"] = "visualization"
                viz_path = os.path.join(job_dir, f"{exp_name}.png")
                try:
                    handler.render_result(attribution, viz_input, viz_path, display_name=display_name)
                except Exception as viz_err:
                    print(f"[detect-rank] viz failed for {exp_name}: {viz_err}")

                # Build token attributions for text
                token_attributions = None
                if task == "text" and tokens_for_viz:
                    attr_flat = attribution.flatten()
                    token_attributions = [
                        {"token": tokens_for_viz[ti], "score": float(attr_flat[ti]) if ti < len(attr_flat) else 0.0}
                        for ti in range(len(tokens_for_viz))
                    ]

                info = explainer_info_map.get(exp_name, {})
                results.append({
                    "name": exp_name,
                    "display_name": info.get("display_name", exp_name),
                    "estimated_compute_time_seconds": info.get("estimated_compute_time_seconds", 0),
                    "metrics": metrics,
                    "avg_score": avg,
                })

                # Store full result in linked explain job
                update_job_result(explain_job_id, {
                    "explainer_name": exp_name,
                    "display_name": info.get("display_name", exp_name),
                    "status": "completed",
                    "rank": None,
                    "visualization_url": f"/api/jobs/{explain_job_id}/visualizations/{exp_name}.png",
                    "mu_fidelity": round(metrics["mu_fidelity"], 4) if metrics.get("mu_fidelity") is not None else None,
                    "abpc": round(metrics["abpc"], 4) if metrics.get("abpc") is not None else None,
                    "sensitivity": round(-metrics["sensitivity"], 4) if metrics.get("sensitivity") is not None else None,
                    "complexity": round(-metrics["complexity"], 4) if metrics.get("complexity") is not None else None,
                    "token_attributions": token_attributions,
                    "not_supported_reason": None,
                    "error_message": None,
                    "current_step": None,
                })

            except Exception as e:
                print(f"[detect-rank] {exp_name} failed: {e}")
                info = explainer_info_map.get(exp_name, {})
                update_job_result(explain_job_id, {
                    "explainer_name": exp_name,
                    "display_name": info.get("display_name", exp_name),
                    "status": "failed",
                    "rank": None,
                    "visualization_url": None,
                    "mu_fidelity": None, "abpc": None, "sensitivity": None, "complexity": None,
                    "token_attributions": None,
                    "not_supported_reason": None,
                    "error_message": str(e),
                    "current_step": None,
                })

        results.sort(key=lambda x: x["avg_score"], reverse=True)
        job["results"] = results
        job["status"] = "completed"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        update_job_status(explain_job_id, "failed", str(e))
    finally:
        end_job()


@router.post("/detect-rank")
async def start_detect_rank(
    task: str = Query(...),
    model_name: str = Query(...),
    file: UploadFile = File(...),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")
    try:
        handler = get_task_handler(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if task == "image":
        input_data = load_and_validate_image(contents)
    elif task == "text":
        input_data = contents.decode("utf-8", errors="replace")
    else:
        input_data = contents

    job_id = str(uuid.uuid4())
    _detect_rank_jobs[job_id] = {
        "status": "running",
        "current": 0,
        "total": 0,
        "current_explainer": "",
        "current_step": "",
        "detected_architectures": [],
        "results": [],
        "error": None,
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_detect_rank, job_id, task, model_name, input_data)
    return {"job_id": job_id}


@router.get("/detect-rank/{job_id}")
async def get_detect_rank_status(job_id: str):
    job = _detect_rank_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/upload-model")
async def upload_model(task: str = Query(...), file: UploadFile = File(...)):
    """Accept a user-supplied weights file and return the name to explain with.

    The upload is loaded once here so a bad file fails at upload time rather than
    midway through an explanation job.
    """
    try:
        handler = get_task_handler(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = await file.read()
    if len(contents) > MAX_MODEL_SIZE:
        raise HTTPException(status_code=400, detail="Model file too large. Maximum size is 500MB.")

    model_id = uploaded_models.save(contents)
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handler.load_model, model_id)
    except Exception as e:
        # The loaders already explain what they could and couldn't read; torch's own
        # failure text is long and internal, so only fall back to it if there's nothing.
        raise HTTPException(status_code=400, detail=str(e) or f"Could not load '{file.filename}'.")
    return {"valid": True, "model_id": model_id, "display_name": file.filename or "uploaded model"}


# Weights filenames a HuggingFace repo might hold, best first.
_HF_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.pt", "model.pth")


def _hf_repo_id(url: str):
    """The `org/name` a HuggingFace *repo* reference names, or None for anything else.

    Accepts both a pasted repo URL and a bare `org/name`. A link that already points at a
    file inside a repo (`/resolve/`, `/blob/`) is not a repo reference — that one wants
    downloading, not from_pretrained.
    """
    from urllib.parse import urlparse

    if not url.lower().startswith(("http://", "https://")):
        parts = [p for p in url.split("/") if p]
        return "/".join(parts) if len(parts) == 2 else None
    parsed = urlparse(url)
    if parsed.hostname not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) == 2 and not {"resolve", "blob"} & set(parts):
        return "/".join(parts)
    return None


def _hf_repo_files(repo: str) -> list:
    """Filenames a HuggingFace repo holds."""
    import json
    import urllib.request

    api = urllib.request.Request(f"https://huggingface.co/api/models/{repo}",
                                 headers={"User-Agent": "PnPXAI"})
    with urllib.request.urlopen(api, timeout=30) as resp:
        info = json.load(resp)
    return [s["rfilename"] for s in info.get("siblings", [])]


def _download_checkpoint(url: str) -> bytes:
    """Fetch a checkpoint file from a URL, capped at MAX_MODEL_SIZE."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "PnPXAI"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        chunks, total = [], 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MODEL_SIZE:
                raise ValueError("Checkpoint file too large. Maximum size is 500MB.")
            chunks.append(chunk)
    return b"".join(chunks)


async def _load_ts_model_from_url(handler, url: str):
    """Load a forecaster named by a URL: a HuggingFace repo, or a link to a weights file.

    A repo is handed to the handler by id so transformers can fetch config.json alongside
    the weights — the config is what says which architecture to rebuild. A file link is
    downloaded and registered as if it had been uploaded, which is also how repos that
    hold a bare checkpoint (no config.json, so nothing for from_pretrained to read) are
    handled.
    """
    url = url.strip()
    loop = asyncio.get_running_loop()
    repo = _hf_repo_id(url)

    if repo:
        try:
            files = await loop.run_in_executor(None, _hf_repo_files, repo)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read the repo '{repo}': {e}")
        if "config.json" in files:
            try:
                await loop.run_in_executor(None, handler.load_model, repo)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e) or f"Could not load '{repo}'.")
            return {"valid": True, "model_id": repo, "display_name": repo.split("/")[-1]}
        weights = (next((n for n in _HF_WEIGHT_FILES if n in files), None)
                   or next((n for n in files if n.endswith((".pth", ".pt", ".safetensors", ".bin"))), None))
        if weights is None:
            raise HTTPException(
                status_code=400,
                detail=f"'{repo}' holds no weights file (found: {', '.join(files) or 'nothing'}).",
            )
        url = f"https://huggingface.co/{repo}/resolve/main/{weights}"
    elif not url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Paste a HuggingFace model URL (huggingface.co/org/name) or a direct "
                   "link to a checkpoint file.",
        )
    else:
        # A file *page* on the Hub serves HTML; its raw counterpart is under /resolve/.
        url = url.replace("/blob/", "/resolve/")

    try:
        contents = await loop.run_in_executor(None, _download_checkpoint, url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download checkpoint from URL: {e}")

    if contents[:512].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise HTTPException(
            status_code=400,
            detail="That URL returned a web page, not a file. Link the checkpoint itself "
                   "(on HuggingFace, the file's Download button).",
        )

    model_id = uploaded_models.save(contents)
    try:
        await loop.run_in_executor(None, handler.load_model, model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e) or f"Could not load {url}.")
    display_name = os.path.basename(url.split("?")[0]) or "checkpoint from URL"
    return {"valid": True, "model_id": model_id, "display_name": display_name}


@router.get("/validate-model")
async def validate_model(task: str = Query(...), hf_model_id: str = Query(...)):
    """Validates a HuggingFace model ID by attempting to load it for the given task."""
    try:
        handler = get_task_handler(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if task == "timeseries":
        # Forecasters come in two shapes the image/text path never sees: a transformers
        # repo, and a plain checkpoint file. Which one this is decides how to load it.
        return await _load_ts_model_from_url(handler, hf_model_id)

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


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a running /explain job. Takes effect between explainers —
    whichever one is currently attributing finishes first, then the job stops with
    only the results computed so far."""
    if not request_cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found or already finished")
    return {"status": "cancelling"}


@router.get("/jobs/{job_id}/visualizations/{filename}")
async def get_visualization(job_id: str, filename: str):
    viz_path = os.path.join(VISUALIZATION_DIR, job_id, filename)
    if not os.path.exists(viz_path):
        raise HTTPException(status_code=404, detail="Visualization not found.")
    if filename.endswith(".zip"):
        return FileResponse(viz_path, media_type="application/zip", filename=filename)
    if filename.endswith(".xlsx"):
        return FileResponse(viz_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)
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
    # Priority ordering: listed files appear first, rest alphabetical
    _PRIORITY = {"timeseries": ["boiler.csv", "ecg5000.csv"]}
    files = glob.glob(os.path.join(sample_dir, "*"))
    priority_list = _PRIORITY.get(task, [])
    def _sort_key(fp):
        name = os.path.basename(fp)
        if name in priority_list:
            return (0, priority_list.index(name))
        return (1, name)
    # A model trained on specific datasets only offers those. The mapping stays on the
    # server — the client just receives the already-filtered list. Models that aren't
    # tied to a corpus (ImageNet classifiers, custom uploads) offer everything.
    trained_on = None
    if model:
        try:
            handler = get_task_handler(task)
            trained_on = getattr(handler, "get_model_datasets", lambda m: None)(model)
        except ValueError:
            trained_on = None

    result = []
    for f in sorted(files, key=_sort_key):
        name = os.path.basename(f)
        if trained_on is not None and name not in trained_on:
            continue
        entry: dict = {"name": name, "path": f"/{task}/{name}"}

        # For timeseries, report the channel count. Every forecaster here embeds each
        # variate over the time axis, so any channel count runs — nothing to reject.
        if task == "timeseries" and name.endswith(".csv"):
            try:
                from backend.tasks.timeseries import _parse_ts_csv
                with open(f, "rb") as fh:
                    _, cols, _ = _parse_ts_csv(fh.read())
                entry["channels"] = len(cols)
                entry["col_names"] = cols
            except Exception:
                pass
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
    try:
        result = await loop.run_in_executor(
            None, run_with_custom_params, task, model_name, explainer_name, params, input_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    try:
        result = await loop.run_in_executor(
            None, run_with_custom_params, task, model_name, explainer_name, params, input_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
