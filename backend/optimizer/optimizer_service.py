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
        text = input_data if isinstance(input_data, str) else str(input_data)
        target_class, predictions, input_tensor, _input_ids, tokens, wrapper = _run_text_inference(handler, model, text, model_name)
        explainer_model = wrapper
    else:
        return None

    target_tensor = torch.tensor([target_class], dtype=torch.long)

    ExplainerClass = _get_pnpxai_explainer(explainer_name)
    if custom_params:
        sig = inspect.signature(ExplainerClass.__init__)
        valid = {k: v for k, v in custom_params.items() if k in sig.parameters}
        explainer = ExplainerClass(explainer_model, **valid)
    else:
        explainer = ExplainerClass(explainer_model)

    inp = input_tensor.clone()
    if inp.is_floating_point():
        inp = inp.requires_grad_(True)
    attr_raw = explainer.attribute(inp, target_tensor)
    attribution = normalize_attribution(attr_raw)

    metrics = {}
    try:
        m = _get_pnpxai_metric("Complexity", explainer_model, explainer)
        result = m.evaluate(inp, target_tensor, attr_raw)
        metrics["complexity"] = extract_metric_value(result)
    except Exception:
        metrics["complexity"] = None

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
    from torch.utils.data import DataLoader, TensorDataset
    import optuna
    from pnpxai import AutoExplanationForImageClassification
    from backend.core.pipeline import _run_image_inference

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    handler = get_task_handler("image")
    model = handler.load_model(model_name)
    input_tensor = handler.preprocess_input(input_data, model_name)
    hf_label_map = handler.get_hf_label_map(model_name)
    target_class, predictions, _ = _run_image_inference(model, input_tensor, hf_label_map)
    target_tensor = torch.tensor([target_class])

    loader = DataLoader(TensorDataset(input_tensor, target_tensor), batch_size=1)
    expr = AutoExplanationForImageClassification(
        model=model,
        data=loader,
        input_extractor=lambda batch: batch[0],
        label_extractor=lambda batch: batch[1],
        target_extractor=lambda outputs: outputs.argmax(-1),
        target_labels=False,
    )

    # Locate explainer (append if not in recommended list)
    explainer_class_names = [e.__class__.__name__ for e in expr.manager.explainers]
    if explainer_name not in explainer_class_names:
        from pnpxai import explainers as exp_mod
        ExplainerClass = getattr(exp_mod, explainer_name, None)
        if ExplainerClass is None:
            raise ValueError(f"Unknown explainer: {explainer_name}")
        expr.manager.explainers.append(ExplainerClass(model=model))
    explainer_id = [e.__class__.__name__ for e in expr.manager.explainers].index(explainer_name)

    metric_id = _METRIC_ORDER.index(metric_name) if metric_name in _METRIC_ORDER else 1  # AbPC default

    # Default attribution + metrics
    inp_def = input_tensor.clone().requires_grad_(True)
    default_explainer = expr.manager.get_explainer_by_id(explainer_id)
    default_attr = default_explainer.attribute(inp_def, target_tensor)
    default_metrics = _eval_all_metrics(model, default_explainer, inp_def, target_tensor, default_attr)

    # Optimize via pnpxai (optuna TPE under the hood)
    opt_output = expr.optimize(
        data_ids=[0],
        explainer_id=explainer_id,
        metric_id=metric_id,
        direction="maximize",
        sampler="tpe",
        n_trials=n_trials,
        seed=42,
    )

    # Optimized attribution + metrics
    opt_explainer = opt_output.explainer
    inp_opt = input_tensor.clone().requires_grad_(True)
    opt_attr = opt_explainer.attribute(inp_opt, target_tensor)
    optimized_metrics = _eval_all_metrics(model, opt_explainer, inp_opt, target_tensor, opt_attr)
    opt_attribution = normalize_attribution(opt_attr)

    # Extract explainer-specific params from best trial
    available_params = get_explainer_params(explainer_name)
    default_params = {p["name"]: p["default"] for p in available_params}
    trial_params = opt_output.study.best_trial.params
    opt_params = {
        k[len("explainer/"):]: v
        for k, v in trial_params.items()
        if k.startswith("explainer/") and k[len("explainer/"):] in default_params
    }
    optimized_params = {**default_params, **opt_params}

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
        opt_attribution, opt_output.study.best_value,
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
        return _run_explainer_with_params("text", model_name, explainer_name, input_data)

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

    # Optimize via Objective + optuna
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

    opt_explainer = study.best_trial.user_attrs["explainer"]

    # Optimized attribution + metrics
    inp_result = input_embeds.clone().detach().requires_grad_(True)
    opt_attr = opt_explainer.attribute(inp_result, target_tensor)
    optimized_metrics = _eval_all_metrics(wrapper_model, opt_explainer, inp_result, target_tensor, opt_attr)
    opt_attribution = normalize_attribution(opt_attr)

    # Extract params
    available_params = get_explainer_params(explainer_name)
    default_params = {p["name"]: p["default"] for p in available_params}
    trial_params = study.best_trial.params
    opt_params = {
        k[len("explainer/"):]: v
        for k, v in trial_params.items()
        if k.startswith("explainer/") and k[len("explainer/"):] in default_params
    }
    optimized_params = {**default_params, **opt_params}

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
        opt_attribution, study.best_value,
    )


def run_optimization(task, model_name, explainer_name, metric_name, input_data, n_trials=20):
    if task == "image":
        return _run_image_optimization(model_name, explainer_name, metric_name, input_data, n_trials)
    elif task == "text":
        return _run_text_optimization(model_name, explainer_name, metric_name, input_data, n_trials)
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
