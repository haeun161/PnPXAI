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
        target_class, predictions, input_tensor, tokens, wrapper = _run_text_inference(handler, model, text, model_name)
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
    attribution = normalize_attribution(attr_raw, task=task)

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


def run_optimization(task, model_name, explainer_name, metric_name, input_data, n_trials=20):
    result = _run_explainer_with_params(task, model_name, explainer_name, input_data)
    if result is None:
        return {"error": f"Unsupported task: {task}"}

    available_params = get_explainer_params(explainer_name)
    record_id = str(uuid.uuid4())[:8]

    # Save input data for history restoration
    _save_input_data(record_id, task, input_data)

    output = {
        "record_id": record_id,
        "task": task,
        "model_name": model_name,
        "explainer_name": explainer_name,
        "metric_name": metric_name,
        "default_params": {p["name"]: p["default"] for p in available_params},
        "optimized_params": {p["name"]: p["default"] for p in available_params},
        "default_metrics": result["metrics"],
        "optimized_metrics": result["metrics"],
        "available_params": available_params,
        "predictions": result["predictions"],
        "visualization_url": result["visualization_url"],
        "attribution_summary": result["attribution_summary"],
        "timestamp": time.time(),
    }
    return output


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
