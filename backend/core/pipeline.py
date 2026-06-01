import copy
import os
import numpy as np
import torch

from backend.tasks import get_task_handler
from backend.core.pnpxai_adapter import normalize_attribution, extract_metric_value
from backend.core.job_manager import (
    get_uploaded_data, update_job_status, update_job_predictions,
    update_job_result, VISUALIZATION_DIR,
)

# ImageNet class labels
_imagenet_labels: list[str] = []


def _load_imagenet_labels() -> list[str]:
    global _imagenet_labels
    if _imagenet_labels:
        return _imagenet_labels
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        _imagenet_labels = list(weights.meta["categories"])
    except Exception:
        _imagenet_labels = [str(i) for i in range(1000)]
    return _imagenet_labels


def _get_pnpxai_explainer(name: str):
    from pnpxai import explainers as exp_mod
    cls = getattr(exp_mod, name, None)
    if cls is None:
        raise ValueError(f"PnPXAI explainer not found: {name}")
    return cls


def _get_pnpxai_metric(name: str, model, explainer_instance=None):
    from pnpxai.evaluator import metrics as met_mod
    cls = getattr(met_mod, name, None)
    if cls is None:
        raise ValueError(f"PnPXAI metric not found: {name}")
    return cls(model=model, explainer=explainer_instance)


def _run_image_inference(model, input_tensor, hf_label_map=None):
    """Run image classification inference, return (target_class, predictions, input_tensor)."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(probabilities)))

    if hf_label_map:
        get_label = lambda idx: hf_label_map.get(idx, str(idx))
    else:
        imagenet_labels = _load_imagenet_labels()
        get_label = lambda idx: imagenet_labels[idx] if idx < len(imagenet_labels) else str(idx)

    target_class = top5_indices[0].item()
    predictions = [
        {"class_name": get_label(idx.item()), "probability": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs, top5_indices)
    ]
    return target_class, predictions, input_tensor


_GRADIENT_FREE_TEXT_EXPLAINERS = {"Lime", "KernelShap"}


class _TextEmbeddingWrapper(torch.nn.Module):
    """Wrapper that takes embedding tensors (float) instead of input_ids (long).

    Captum/PnPXAI interpolates inputs (requiring float), but transformer models
    expect input_ids (long) for their embedding layer. This wrapper bypasses
    the embedding lookup by accepting pre-computed embeddings.
    """
    def __init__(self, model, embedding_layer):
        super().__init__()
        self.model = model
        self.embedding_layer = embedding_layer

    def forward(self, inputs_embeds):
        # DistilBERT: pass inputs_embeds directly, skipping token embedding lookup
        output = self.model(inputs_embeds=inputs_embeds)
        logits = output.logits if hasattr(output, "logits") else output[0]
        return logits


class _TextInputIdsWrapper(torch.nn.Module):
    """Wrapper for LIME/KernelSHAP: accepts input_ids (long) and returns logits.

    These perturbation-based methods don't need gradients and work at the token
    level using NoMask1d feature masks, so they need (bsz, seq_len) integer input.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        output = self.model(input_ids=input_ids)
        logits = output.logits if hasattr(output, "logits") else output[0]
        return logits


def _run_text_inference(handler, model, raw_text, model_name):
    """Run text classification inference. Returns wrapper model + embeddings for XAI."""
    encoded, tokens = handler.tokenize(raw_text, model_name)
    input_ids = encoded["input_ids"]

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids)
        logits = output.logits if hasattr(output, "logits") else output[0]
        probabilities = torch.softmax(logits, dim=1)[0]

    label_map = handler.get_label_map(model_name)
    target_class = probabilities.argmax().item()

    predictions = []
    for idx in range(len(probabilities)):
        label = label_map.get(idx, str(idx))
        predictions.append({
            "class_name": label,
            "probability": round(probabilities[idx].item() * 100, 2),
        })
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Get embedding layer and compute embeddings (float tensor for Captum)
    embedding_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_layer = module
            break

    input_embeds = embedding_layer(input_ids).detach().requires_grad_(True)
    wrapper_model = _TextEmbeddingWrapper(model, embedding_layer)

    return target_class, predictions, input_embeds, input_ids, tokens, wrapper_model


def run_explanation_pipeline(
    job_id: str,
    task: str,
    model_name: str,
    explainer_names: list[str],
    ranking_metric: str,
    params: dict,
):
    """Synchronous pipeline - runs in thread pool executor."""
    try:
        update_job_status(job_id, "running")

        handler = get_task_handler(task)
        model = handler.load_model(model_name)

        raw_data = get_uploaded_data(job_id)
        if raw_data is None:
            update_job_status(job_id, "failed", "Uploaded data not found.")
            return

        # Task-specific inference
        tokens_for_viz = None
        text_input_ids = None
        explainer_model = model  # model used for explainer (may be wrapper for text)
        if task == "image":
            input_data = handler.preprocess_input(raw_data, model_name)
            hf_label_map = getattr(handler, "get_hf_label_map", lambda m: {})(model_name)
            target_class, predictions, input_tensor = _run_image_inference(model, input_data, hf_label_map)
            update_job_predictions(job_id, predictions)
        elif task == "text":
            text = raw_data if isinstance(raw_data, str) else str(raw_data)
            target_class, predictions, input_tensor, text_input_ids, tokens_for_viz, wrapper_model = _run_text_inference(handler, model, text, model_name)
            explainer_model = wrapper_model  # use embedding wrapper for XAI
            update_job_predictions(job_id, predictions)
        else:
            input_data = handler.preprocess_input(raw_data)
            # Time-series returns {"tensor": ..., "col_names": ...}
            if isinstance(input_data, dict) and "tensor" in input_data:
                input_tensor = input_data["tensor"]
                data_channels = input_tensor.shape[1]

                # Check model-data channel compatibility
                from backend.tasks.timeseries import _TS_MODELS
                model_info = _TS_MODELS.get(model_name, {})
                model_default_ch = model_info.get("default_channels")

                if model_default_ch is not None and model_default_ch != data_channels:
                    update_job_status(
                        job_id, "failed",
                        f"Channel mismatch: model '{model_name}' expects {model_default_ch}-channel input, "
                        f"but data has {data_channels} channel(s). "
                        f"Please use a {'multi-variate' if data_channels > 1 else 'single-variate'} model, "
                        f"or upload {'multi-variate' if model_default_ch > 1 else 'single-variate'} data."
                    )
                    return

                model = handler.load_model(model_name, num_input_channels=data_channels)
                explainer_model = model
            else:
                input_tensor = input_data if isinstance(input_data, torch.Tensor) else None
            target_class = 0

        # For each explainer: attribution + metrics + visualization
        job_dir = os.path.join(VISUALIZATION_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        all_results = []

        for exp_name in explainer_names:
            exp_info = next((e for e in handler.get_explainers(model_name) if e["name"] == exp_name), None)
            display_name = exp_info["display_name"] if exp_info else exp_name

            update_job_result(job_id, {
                "explainer_name": exp_name,
                "display_name": display_name,
                "status": "running",
            })

            try:
                ExplainerClass = _get_pnpxai_explainer(exp_name)

                # LIME/KernelSHAP for text: perturbation-based, need input_ids + NoMask1d.
                # Gradient-based methods need float embeddings via _TextEmbeddingWrapper.
                # LRP/RAP: use a deep-copy of the model — zennit's canonizer merges BatchNorm
                # weights into conv layers in-place and may leave them corrupted on failure,
                # which would poison the shared cached model for all subsequent explainers.
                _STATE_MUTATING_EXPLAINERS = {"LRPUniformEpsilon", "LRPEpsilonPlus",
                                              "LRPEpsilonGammaBox", "LRPEpsilonAlpha2Beta1", "RAP"}
                if task == "text" and exp_name in _GRADIENT_FREE_TEXT_EXPLAINERS:
                    from pnpxai.explainers.utils.feature_masks import NoMask1d
                    active_model = _TextInputIdsWrapper(model)
                    active_inp = text_input_ids.clone()
                    explainer_instance = ExplainerClass(active_model, feature_mask_fn=NoMask1d())
                elif exp_name in _STATE_MUTATING_EXPLAINERS:
                    active_model = copy.deepcopy(explainer_model)
                    active_inp = input_tensor.clone() if input_tensor is not None else None
                    if active_inp is not None and active_inp.is_floating_point():
                        active_inp = active_inp.requires_grad_(True)
                    explainer_instance = ExplainerClass(active_model)
                else:
                    active_model = explainer_model
                    active_inp = input_tensor.clone() if input_tensor is not None else None
                    if active_inp is not None and active_inp.is_floating_point():
                        active_inp = active_inp.requires_grad_(True)
                    explainer_instance = ExplainerClass(active_model)

                # Compute attribution
                target_tensor = torch.tensor([target_class], dtype=torch.long)
                if active_inp is not None:
                    try:
                        attribution_raw = explainer_instance.attribute(active_inp, target_tensor)
                    except TypeError:
                        attribution_raw = explainer_instance.attribute(inputs=active_inp, targets=target_tensor)
                    attribution = normalize_attribution(attribution_raw)
                else:
                    attribution = np.zeros(10)
                    attribution_raw = torch.zeros(1, 10)

                # Compute metrics (MuFidelity not compatible with text embeddings)
                metric_values = {}
                metric_list = ["AbPC", "Sensitivity", "Complexity"] if task == "text" else ["MuFidelity", "AbPC", "Sensitivity", "Complexity"]
                for metric_name in metric_list:
                    try:
                        metric_instance = _get_pnpxai_metric(metric_name, active_model, explainer_instance)
                        if active_inp is not None:
                            result = metric_instance.evaluate(
                                active_inp, target_tensor, attribution_raw
                            )
                            metric_values[metric_name.lower()] = extract_metric_value(result)
                        else:
                            metric_values[metric_name.lower()] = None
                    except Exception:
                        metric_values[metric_name.lower()] = None

                # Render visualization
                viz_path = os.path.join(job_dir, f"{exp_name}.png")
                viz_input = tokens_for_viz if tokens_for_viz else raw_data
                handler.render_result(attribution, viz_input, viz_path)

                # Build token attribution data for frontend highlighting (text only)
                token_attributions = None
                if task == "text" and tokens_for_viz:
                    attr_flat = attribution.flatten()
                    token_attributions = [
                        {"token": tokens_for_viz[i], "score": float(attr_flat[i]) if i < len(attr_flat) else 0.0}
                        for i in range(len(tokens_for_viz))
                    ]

                result_entry = {
                    "explainer_name": exp_name,
                    "display_name": display_name,
                    "status": "completed",
                    "visualization_url": f"/api/jobs/{job_id}/visualizations/{exp_name}.png",
                    "mu_fidelity": round(metric_values.get("mufidelity") or 0, 4) if metric_values.get("mufidelity") is not None else None,
                    "abpc": round(metric_values.get("abpc") or 0, 4) if metric_values.get("abpc") is not None else None,
                    "sensitivity": round(metric_values.get("sensitivity") or 0, 4) if metric_values.get("sensitivity") is not None else None,
                    "complexity": round(metric_values.get("complexity") or 0, 4) if metric_values.get("complexity") is not None else None,
                    "token_attributions": token_attributions,
                }
                all_results.append(result_entry)
                update_job_result(job_id, result_entry)

            except Exception as e:
                import traceback
                traceback.print_exc()
                result_entry = {
                    "explainer_name": exp_name,
                    "display_name": display_name,
                    "status": "failed",
                    "error_message": str(e),
                }
                all_results.append(result_entry)
                update_job_result(job_id, result_entry)

        # Rank results
        def _rank_score(r):
            if ranking_metric == "average":
                vals = [r.get(k) for k in ["mu_fidelity", "sensitivity", "complexity"] if r.get(k) is not None]
                return sum(vals) / len(vals) if vals else 0
            return r.get(ranking_metric, 0) or 0

        completed = [r for r in all_results if r["status"] == "completed"]
        completed.sort(key=_rank_score, reverse=True)
        for i, r in enumerate(completed):
            r["rank"] = i + 1
            update_job_result(job_id, r)

        update_job_status(job_id, "completed")

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job_status(job_id, "failed", str(e))
