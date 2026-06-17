"""Optimizer service using PnPXAI explainers with parameter tuning."""

import os
import json
import time
import inspect
import uuid
import io
import torch
import numpy as np
from typing import Any
from PIL import Image

from backend.tasks import get_task_handler
from backend.core.pnpxai_adapter import normalize_attribution, extract_metric_value

HISTORY_DIR = os.path.join("backend", "optimizer", "history")
DATA_DIR = os.path.join("backend", "optimizer", "data")
VISUALIZATION_DIR = os.path.join("backend", "heatmaps")
MAX_HISTORY = 5


def get_explainer_params(explainer_name: str) -> list[dict]:
    from pnpxai import explainers as exp_mod
    cls = getattr(exp_mod, explainer_name, None)
    if cls is None:
        return []

    sig = inspect.signature(cls.__init__)
    params = []
    skip = {"self", "model", "forward_func", "kwargs"}
    for name, param in sig.parameters.items():
        if name in skip or name.startswith("_"):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        default = param.default
        if isinstance(default, bool):
            params.append({"name": name, "type": "bool", "default": default, "description": ""})
        elif isinstance(default, int):
            params.append({"name": name, "type": "int", "default": default, "description": ""})
        elif isinstance(default, float):
            params.append({"name": name, "type": "float", "default": default, "description": ""})
        elif isinstance(default, str):
            params.append({"name": name, "type": "str", "default": default, "description": ""})
    return params


def _save_input_data(record_id: str, task: str, input_data: Any):
    """Save input data to disk for later restoration."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if task == "image" and isinstance(input_data, Image.Image):
        path = os.path.join(DATA_DIR, f"{record_id}.png")
        input_data.save(path)
    elif task == "text" and isinstance(input_data, str):
        path = os.path.join(DATA_DIR, f"{record_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(input_data)
    elif isinstance(input_data, bytes):
        path = os.path.join(DATA_DIR, f"{record_id}.bin")
        with open(path, "wb") as f:
            f.write(input_data)


def _load_input_data(record_id: str, task: str) -> Any:
    """Load saved input data from disk."""
    if task == "image":
        path = os.path.join(DATA_DIR, f"{record_id}.png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    elif task == "text":
        path = os.path.join(DATA_DIR, f"{record_id}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    else:
        path = os.path.join(DATA_DIR, f"{record_id}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    return None


def _run_explainer_with_params(task, model_name, explainer_name, input_data, custom_params=None):
    from backend.core.pipeline import _get_pnpxai_explainer, _get_pnpxai_metric, _run_text_inference, _run_image_inference

    handler = get_task_handler(task)
    model = handler.load_model(model_name)

    if task == "image":
        input_processed = handler.preprocess_input(input_data)
        target_class, predictions, input_tensor = _run_image_inference(model, input_processed)
        explainer_model = model
        tokens = None
    elif task == "text":
        from backend.core.pipeline import _TextInputIdsWrapper
        text = input_data if isinstance(input_data, str) else str(input_data)
        target_class, predictions, input_tensor, _input_ids, tokens, wrapper = _run_text_inference(handler, model, text, model_name)
        if explainer_name in _GRADIENT_FREE_EXPLAINERS:
            explainer_model = _TextInputIdsWrapper(model)
            input_tensor = _input_ids.clone()
        else:
            explainer_model = wrapper
    elif task == "timeseries":
        proc = handler.preprocess_input(input_data)
        tokens = None
        if isinstance(proc, dict) and "tensor" in proc:
            ts_tensor = proc["tensor"]
            num_ch = ts_tensor.shape[1]
            model = handler.load_model(model_name, num_input_channels=num_ch)
            model.eval()
            with torch.no_grad():
                out = model(ts_tensor)
            target_class = int(out.argmax(dim=1).item())
            input_tensor = ts_tensor
            explainer_model = model
            # Override viz_input for render_result
            tokens = proc  # pass dict with tensor + col_names
        else:
            input_tensor = proc if isinstance(proc, torch.Tensor) else None
            target_class = 0
            explainer_model = model
        predictions = [{"class_name": f"class_{target_class}", "probability": 100.0}]
    else:
        return None

    target_tensor = torch.tensor([target_class], dtype=torch.long)

    ExplainerClass = _get_pnpxai_explainer(explainer_name)
    try:
        if task == "text" and explainer_name in _GRADIENT_FREE_EXPLAINERS:
            from pnpxai.explainers.utils.feature_masks import NoMask1d
            extra = {"feature_mask_fn": NoMask1d()}
            if custom_params:
                sig = inspect.signature(ExplainerClass.__init__)
                # Exclude feature_mask_fn so NoMask1d() is never overridden by a string value
                valid = {k: v for k, v in custom_params.items() if k in sig.parameters and k != "feature_mask_fn"}
                extra.update(valid)
            explainer = ExplainerClass(explainer_model, **extra)
        elif custom_params:
            sig = inspect.signature(ExplainerClass.__init__)
            valid = {k: v for k, v in custom_params.items() if k in sig.parameters}
            explainer = ExplainerClass(explainer_model, **valid)
        else:
            explainer = ExplainerClass(explainer_model)
    except (TypeError, ValueError) as e:
        available = get_explainer_params(explainer_name)
        param_hints = ", ".join(f"{p['name']} ({p['type']}, default: {p['default']})" for p in available)
        raise ValueError(
            f"Invalid parameter for {explainer_name}: {e}. "
            f"Please provide valid values. Available parameters: {param_hints or 'none'}."
        )

    if explainer_name in {"GradCam", "GuidedGradCam"} and task == "image" and hasattr(explainer, "set_target_layer"):
        from backend.core.pipeline import _find_cam_target_layer
        cam_target = _find_cam_target_layer(explainer_model)
        if cam_target is not None:
            explainer = explainer.set_target_layer(cam_target)

    inp = input_tensor.clone()
    if inp.is_floating_point():
        inp = inp.requires_grad_(True)
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        attr_raw = explainer.attribute(inp, target_tensor)
    except (TypeError, ValueError, NotImplementedError, RuntimeError) as e:
        available = get_explainer_params(explainer_name)
        param_hints = ", ".join(f"{p['name']} ({p['type']}, default: {p['default']})" for p in available)
        raise ValueError(
            f"Invalid parameter value for {explainer_name}: {e}. "
            f"Please enter valid values. Available parameters: {param_hints or 'none'}."
        )
    attribution = normalize_attribution(attr_raw, task=task)

    metrics = _eval_all_metrics(explainer_model, explainer, inp, target_tensor, attr_raw)

    viz_id = f"opt_{int(time.time() * 1000)}"
    viz_dir = os.path.join(VISUALIZATION_DIR, viz_id)
    os.makedirs(viz_dir, exist_ok=True)
    viz_path = os.path.join(viz_dir, f"{explainer_name}.png")
    viz_input = tokens if tokens else input_data
    handler.render_result(attribution, viz_input, viz_path)

    return {
        "predictions": predictions if isinstance(predictions, list) else [],
        "metrics": metrics,
        "visualization_url": f"/api/jobs/{viz_id}/visualizations/{explainer_name}.png",
        "attribution_summary": {
            "mean": float(attribution.mean()),
            "max": float(attribution.max()),
            "nonzero_ratio": float((attribution > 0.1).sum() / max(attribution.size, 1)),
        },
    }


_GRADIENT_FREE_EXPLAINERS = {"Lime", "KernelShap"}
_METRIC_ORDER = ["MuFidelity", "AbPC", "Sensitivity", "Complexity"]


def _eval_all_metrics(model, explainer, inputs, targets, attr_raw) -> dict:
    from backend.core.pipeline import _get_pnpxai_metric
    metrics = {}
    for key, cls_name in [("mu_fidelity", "MuFidelity"), ("abpc", "AbPC"),
                           ("sensitivity", "Sensitivity"), ("complexity", "Complexity")]:
        try:
            m = _get_pnpxai_metric(cls_name, model, explainer)
            result = m.evaluate(inputs, targets, attr_raw)
            metrics[key] = extract_metric_value(result)
        except Exception:
            metrics[key] = None
    return metrics


def _build_output(record_id, task, model_name, explainer_name, metric_name,
                  available_params, default_params, optimized_params,
                  default_metrics, optimized_metrics,
                  predictions, visualization_url, attribution, best_trial_value=None):
    return {
        "record_id": record_id,
        "task": task,
        "model_name": model_name,
        "explainer_name": explainer_name,
        "metric_name": metric_name,
        "default_params": default_params,
        "optimized_params": optimized_params,
        "default_metrics": default_metrics,
        "optimized_metrics": optimized_metrics,
        "available_params": available_params,
        "predictions": predictions,
        "visualization_url": visualization_url,
        "attribution_summary": {
            "mean": float(attribution.mean()),
            "max": float(attribution.max()),
            "nonzero_ratio": float((attribution > 0.1).sum() / max(attribution.size, 1)),
        },
        "timestamp": time.time(),
        **({"best_trial_value": float(best_trial_value)} if best_trial_value is not None else {}),
    }


def _run_image_optimization(model_name, explainer_name, metric_name, input_data, n_trials):
    import torch

    import optuna
    from pnpxai.core.experiment.experiment import Objective, load_sampler
    from pnpxai.core.modality.modality import ImageModality
    from backend.core.pipeline import _run_image_inference, _get_pnpxai_explainer, _get_pnpxai_metric

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    handler = get_task_handler("image")
    model = handler.load_model(model_name)
    input_tensor = handler.preprocess_input(input_data, model_name)
    hf_label_map = handler.get_hf_label_map(model_name)
    target_class, predictions, _ = _run_image_inference(model, input_tensor, hf_label_map)
    target_tensor = torch.tensor([target_class])

    # Use ImageModality directly to avoid torch.fx tracing issues with some models
    image_modality = ImageModality()
    ExplainerClass = _get_pnpxai_explainer(explainer_name)
    explainer_inst = ExplainerClass(model=model)

    # For GradCam/GuidedGradCam: set target layer explicitly to avoid fx tracing
    if explainer_name in {"GradCam", "GuidedGradCam"} and hasattr(explainer_inst, "set_target_layer"):
        from backend.core.pipeline import _find_cam_target_layer
        cam_target = _find_cam_target_layer(model)
        if cam_target is not None:
            explainer_inst = explainer_inst.set_target_layer(cam_target)

    postprocessors = image_modality.get_default_postprocessors()
    postprocessor = postprocessors[0]

    metric_cls_name = metric_name if metric_name in _METRIC_ORDER else "AbPC"
    metric = _get_pnpxai_metric(metric_cls_name, model, explainer_inst)

    # Default attribution + metrics
    inp_def = input_tensor.clone().requires_grad_(True)
    default_attr = explainer_inst.attribute(inp_def, target_tensor)
    default_metrics = _eval_all_metrics(model, explainer_inst, inp_def, target_tensor, default_attr)

    available_params = get_explainer_params(explainer_name)
    default_params = {p["name"]: p["default"] for p in available_params}

    # Try Optuna optimization; fall back to default params if model can't be traced
    opt_explainer = explainer_inst
    optimized_params = dict(default_params)
    best_value = None
    try:
        inp_opt = input_tensor.clone().requires_grad_(True)
        objective = Objective(
            explainer=explainer_inst,
            postprocessor=postprocessor,
            metric=metric,
            modality=image_modality,
            inputs=inp_opt,
            targets=target_tensor,
        )
        study = optuna.create_study(sampler=load_sampler("tpe", seed=42), direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        opt_explainer = study.best_trial.user_attrs["explainer"]
        trial_params = study.best_trial.params
        opt_params = {
            k[len("explainer/"):]: v
            for k, v in trial_params.items()
            if k.startswith("explainer/") and k[len("explainer/"):] in default_params
        }
        optimized_params = {**default_params, **opt_params}
        best_value = study.best_value
    except Exception:
        pass  # use default explainer/params as fallback

    # Optimized attribution + metrics
    inp_result = input_tensor.clone().requires_grad_(True)
    torch.manual_seed(42)
    np.random.seed(42)
    opt_attr = opt_explainer.attribute(inp_result, target_tensor)
    optimized_metrics = _eval_all_metrics(model, opt_explainer, inp_result, target_tensor, opt_attr)
    opt_attribution = normalize_attribution(opt_attr)

    # Render visualization
    viz_id = f"opt_{int(time.time() * 1000)}"
    viz_dir = os.path.join(VISUALIZATION_DIR, viz_id)
    os.makedirs(viz_dir, exist_ok=True)
    handler.render_result(opt_attribution, input_data, os.path.join(viz_dir, f"{explainer_name}.png"))

    record_id = str(uuid.uuid4())[:8]
    _save_input_data(record_id, "image", input_data)

    return _build_output(
        record_id, "image", model_name, explainer_name, metric_name,
        available_params, default_params, optimized_params,
        default_metrics, optimized_metrics,
        predictions, f"/api/jobs/{viz_id}/visualizations/{explainer_name}.png",
        opt_attribution, best_value,
    )


def _run_text_optimization(model_name, explainer_name, metric_name, input_data, n_trials):
    import torch
    import optuna
    from pnpxai.core.experiment.experiment import Objective, load_sampler
    from pnpxai.core.modality.modality import TextModality
    from backend.core.pipeline import _run_text_inference, _get_pnpxai_explainer, _get_pnpxai_metric

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Gradient-free methods (Lime, KernelShap) don't benefit from embedding-level optimization
    if explainer_name in _GRADIENT_FREE_EXPLAINERS:
        res = _run_explainer_with_params("text", model_name, explainer_name, input_data)
        if res is None:
            return {"error": "Failed to run explainer"}
        available_params = [p for p in get_explainer_params(explainer_name) if p["name"] != "feature_mask_fn"]
        default_params = {p["name"]: p["default"] for p in available_params}
        record_id = str(uuid.uuid4())[:8]
        _save_input_data(record_id, "text", input_data)
        return _build_output(
            record_id, "text", model_name, explainer_name, metric_name,
            available_params, default_params, default_params,
            res["metrics"], res["metrics"],
            res["predictions"], res["visualization_url"],
            np.zeros(1), None,
        )

    handler = get_task_handler("text")
    model = handler.load_model(model_name)
    text = input_data if isinstance(input_data, str) else str(input_data)
    target_class, predictions, input_embeds, _input_ids, tokens, wrapper_model = \
        _run_text_inference(handler, model, text, model_name)
    target_tensor = torch.tensor([target_class])

    text_modality = TextModality()
    ExplainerClass = _get_pnpxai_explainer(explainer_name)
    default_explainer = ExplainerClass(wrapper_model)

    postprocessors = text_modality.get_default_postprocessors()
    default_postprocessor = postprocessors[0]

    metric_cls_name = metric_name if metric_name in _METRIC_ORDER else "AbPC"
    metric = _get_pnpxai_metric(metric_cls_name, wrapper_model, default_explainer)

    # Default attribution + metrics
    inp_def = input_embeds.clone().detach().requires_grad_(True)
    default_attr = default_explainer.attribute(inp_def, target_tensor)
    default_metrics = _eval_all_metrics(wrapper_model, default_explainer, inp_def, target_tensor, default_attr)

    available_params = get_explainer_params(explainer_name)
    default_params = {p["name"]: p["default"] for p in available_params}

    # Try Optuna optimization; fall back to default params if metric fails (e.g. pixel_flipping shape mismatch)
    opt_explainer = default_explainer
    optimized_params = dict(default_params)
    best_value = None
    try:
        inp_opt = input_embeds.clone().detach().requires_grad_(True)
        objective = Objective(
            explainer=default_explainer,
            postprocessor=default_postprocessor,
            metric=metric,
            modality=text_modality,
            inputs=inp_opt,
            targets=target_tensor,
        )
        study = optuna.create_study(sampler=load_sampler("tpe", seed=42), direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        if study.best_trial is not None:
            opt_explainer = study.best_trial.user_attrs["explainer"]
            trial_params = study.best_trial.params
            opt_params = {
                k[len("explainer/"):]: v
                for k, v in trial_params.items()
                if k.startswith("explainer/") and k[len("explainer/"):] in default_params
            }
            optimized_params = {**default_params, **opt_params}
            best_value = study.best_value
    except Exception:
        pass  # use default explainer/params as fallback

    # Optimized attribution + metrics
    inp_result = input_embeds.clone().detach().requires_grad_(True)
    torch.manual_seed(42)
    np.random.seed(42)
    opt_attr = opt_explainer.attribute(inp_result, target_tensor)
    optimized_metrics = _eval_all_metrics(wrapper_model, opt_explainer, inp_result, target_tensor, opt_attr)
    opt_attribution = normalize_attribution(opt_attr)

    # Render
    viz_id = f"opt_{int(time.time() * 1000)}"
    viz_dir = os.path.join(VISUALIZATION_DIR, viz_id)
    os.makedirs(viz_dir, exist_ok=True)
    handler.render_result(opt_attribution, tokens, os.path.join(viz_dir, f"{explainer_name}.png"))

    record_id = str(uuid.uuid4())[:8]
    _save_input_data(record_id, "text", input_data)

    return _build_output(
        record_id, "text", model_name, explainer_name, metric_name,
        available_params, default_params, optimized_params,
        default_metrics, optimized_metrics,
        predictions, f"/api/jobs/{viz_id}/visualizations/{explainer_name}.png",
        opt_attribution, best_value,
    )


def _run_timeseries_optimization(model_name, explainer_name, metric_name, input_data, n_trials):
    """Run optimization for time-series task using default params (Optuna skipped for TS)."""
    import torch

    from backend.core.pipeline import _get_pnpxai_explainer

    handler = get_task_handler("timeseries")
    proc = handler.preprocess_input(input_data)

    if isinstance(proc, dict) and "tensor" in proc:
        ts_tensor = proc["tensor"]
        num_ch = ts_tensor.shape[1]
        model = handler.load_model(model_name, num_input_channels=num_ch)
    else:
        model = handler.load_model(model_name)
        ts_tensor = proc if isinstance(proc, torch.Tensor) else None

    model.eval()
    if ts_tensor is not None:
        with torch.no_grad():
            out = model(ts_tensor)
        target_class = int(out.argmax(dim=1).item())
    else:
        target_class = 0

    target_tensor = torch.tensor([target_class], dtype=torch.long)
    predictions = [{"class_name": f"class_{target_class}", "probability": 100.0}]

    ExplainerClass = _get_pnpxai_explainer(explainer_name)
    explainer_inst = ExplainerClass(model)

    inp = ts_tensor.clone().requires_grad_(True) if ts_tensor is not None else None
    if inp is not None:
        torch.manual_seed(42)
        np.random.seed(42)
        attr_raw = explainer_inst.attribute(inp, target_tensor)
        default_metrics = _eval_all_metrics(model, explainer_inst, inp, target_tensor, attr_raw)
        attribution = normalize_attribution(attr_raw, task="timeseries")
    else:
        default_metrics = {}
        attribution = np.zeros(10)

    available_params = get_explainer_params(explainer_name)
    default_params = {p["name"]: p["default"] for p in available_params}

    # Render
    viz_id = f"opt_{int(time.time() * 1000)}"
    viz_dir = os.path.join(VISUALIZATION_DIR, viz_id)
    os.makedirs(viz_dir, exist_ok=True)
    viz_input = proc if isinstance(proc, dict) else input_data
    handler.render_result(attribution, viz_input, os.path.join(viz_dir, f"{explainer_name}.png"))

    record_id = str(uuid.uuid4())[:8]
    _save_input_data(record_id, "timeseries", input_data)

    return _build_output(
        record_id, "timeseries", model_name, explainer_name, metric_name,
        available_params, default_params, default_params,
        default_metrics, default_metrics,
        predictions, f"/api/jobs/{viz_id}/visualizations/{explainer_name}.png",
        attribution, None,
    )


def run_optimization(task, model_name, explainer_name, metric_name, input_data, n_trials=20):
    if task == "image":
        return _run_image_optimization(model_name, explainer_name, metric_name, input_data, n_trials)
    elif task == "text":
        return _run_text_optimization(model_name, explainer_name, metric_name, input_data, n_trials)
    elif task == "timeseries":
        return _run_timeseries_optimization(model_name, explainer_name, metric_name, input_data, n_trials)
    else:
        return {"error": f"Unsupported task: {task}"}


def run_with_custom_params(task, model_name, explainer_name, custom_params, input_data):
    available = get_explainer_params(explainer_name)
    type_map = {p["name"]: p["type"] for p in available}
    typed_params = {}
    for k, v in custom_params.items():
        t = type_map.get(k, "str")
        try:
            if t == "int":
                typed_params[k] = int(v)
            elif t == "float":
                typed_params[k] = float(v)
            elif t == "bool":
                typed_params[k] = bool(v)
            else:
                typed_params[k] = v
        except (ValueError, TypeError):
            typed_params[k] = v

    result = _run_explainer_with_params(task, model_name, explainer_name, input_data, typed_params)
    if result is None:
        return {"error": f"Failed to run with custom params"}

    return {
        "custom_params": typed_params,
        "metrics": result["metrics"],
        "visualization_url": result["visualization_url"],
        "attribution_summary": result["attribution_summary"],
        "predictions": result["predictions"],
    }


def save_history(record: dict):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, "records.json")
    records = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    # Clean up old data files when evicting
    if len(records) >= MAX_HISTORY:
        evicted = records[MAX_HISTORY - 1:]
        for r in evicted:
            rid = r.get("record_id")
            if rid:
                for ext in [".png", ".txt", ".bin"]:
                    p = os.path.join(DATA_DIR, f"{rid}{ext}")
                    if os.path.exists(p):
                        os.remove(p)

    records.insert(0, record)
    records = records[:MAX_HISTORY]
    with open(history_file, "w") as f:
        json.dump(records, f, indent=2, default=str)


def get_history() -> list[dict]:
    history_file = os.path.join(HISTORY_DIR, "records.json")
    if not os.path.exists(history_file):
        return []
    with open(history_file, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def get_history_record(record_id: str) -> dict | None:
    records = get_history()
    return next((r for r in records if r.get("record_id") == record_id), None)


def delete_history_record(record_id: str) -> bool:
    history_file = os.path.join(HISTORY_DIR, "records.json")
    if not os.path.exists(history_file):
        return False
    with open(history_file, "r") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            return False

    new_records = [r for r in records if r.get("record_id") != record_id]
    if len(new_records) == len(records):
        return False

    # Clean up data file
    for ext in [".png", ".txt", ".bin"]:
        p = os.path.join(DATA_DIR, f"{record_id}{ext}")
        if os.path.exists(p):
            os.remove(p)

    with open(history_file, "w") as f:
        json.dump(new_records, f, indent=2, default=str)
    return True


def load_record_input_data(record_id: str, task: str) -> Any:
    return _load_input_data(record_id, task)
