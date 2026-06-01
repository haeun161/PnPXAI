import copy
import os
from typing import Optional
import numpy as np
import torch

from backend.tasks import get_task_handler
from backend.core.pnpxai_adapter import normalize_attribution, extract_metric_value
from backend.core.job_manager import (
    get_uploaded_data, update_job_status, update_job_predictions,
    update_job_result, update_result_step, VISUALIZATION_DIR,
)

def _find_cam_target_layer(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return the layer just before global avg-pooling, matching pnpxai's intent.

    Walks named_modules to find AdaptiveAvgPool2d, navigates to its parent container,
    and returns the preceding sibling (the last encoder block). This gives the same
    result as pnpxai's find_cam_target_layer for standard architectures (layer4 for
    ResNet, encoder for HF ResNet, features for VGG) without requiring FX tracing.
    Falls back to the last Conv2d if no AdaptiveAvgPool2d is found.
    """
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.AdaptiveAvgPool2d):
            continue
        parts = name.split(".")
        # Navigate to the parent of the pool layer
        if len(parts) == 1:
            parent = model  # pool is a direct child of the root
        else:
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break
        if parent is None:
            continue
        pool_attr = parts[-1]
        prev_sibling = None
        for child_name, child_mod in parent.named_children():
            if child_name == pool_attr:
                if prev_sibling is not None:
                    return prev_sibling
            else:
                prev_sibling = child_mod

    # Fallback: last Conv2d in the model
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


def _apply_gradcam_patch() -> None:
    """Monkey-patch GradCam.attribute to honour set_target_layer.

    pnpxai's GradCam.attribute() hardcodes `layer = find_cam_target_layer(self.model)`
    (line 69 of grad_cam.py), completely ignoring self._layer / self.layer. This means
    set_target_layer() has no effect on the actual attribution. The patch replaces the
    method so it uses self.layer (which DOES respect _layer) instead.
    """
    try:
        from pnpxai.explainers.grad_cam import GradCam
        from captum.attr import LayerGradCam, LayerAttribution
        from pnpxai.utils import format_into_tuple

        def _fixed_attribute(self, inputs, targets):
            forward_args, additional_forward_args = self._extract_forward_args(inputs)
            forward_args = format_into_tuple(forward_args)
            additional_forward_args = format_into_tuple(additional_forward_args)
            assert len(forward_args) == 1, "GradCam for multiple inputs is not supported yet."
            layer = self.layer  # respects _layer set by set_target_layer
            print(f"[GradCam] using target layer: {type(layer).__name__} — {layer}")
            captum_explainer = LayerGradCam(forward_func=self.model, layer=layer)
            attrs = captum_explainer.attribute(
                forward_args[0],
                target=targets,
                additional_forward_args=additional_forward_args,
                attr_dim_summation=True,
            )
            return LayerAttribution.interpolate(
                layer_attribution=attrs,
                interpolate_dims=forward_args[0].shape[2:],
                interpolate_mode=self.interpolate_mode,
            )

        GradCam.attribute = _fixed_attribute
    except Exception as e:
        print(f"[pipeline] Could not patch GradCam.attribute: {e}")


_apply_gradcam_patch()

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

        # (No HF-model flag needed — target layer is set explicitly for all image models.)

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
            target_class = 0
            input_tensor = input_data if isinstance(input_data, torch.Tensor) else None

        # For each explainer: attribution + metrics + visualization
        job_dir = os.path.join(VISUALIZATION_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        all_results = []

        for exp_name in explainer_names:
            exp_info = next((e for e in handler.get_explainers(model_name) if e["name"] == exp_name), None)
            display_name = exp_info["display_name"] if exp_info else exp_name

            _ATTRIBUTION_STEP = {
                "Lime":                  "Generating perturbations",
                "KernelShap":            "Generating perturbations",
                "GradCam":               "Extracting feature maps",
                "GuidedGradCam":         "Extracting feature maps",
                "LRPUniformEpsilon":     "Propagating relevance",
                "LRPEpsilonPlus":        "Propagating relevance",
                "LRPEpsilonGammaBox":    "Propagating relevance",
                "LRPEpsilonAlpha2Beta1": "Propagating relevance",
                "RAP":                   "Propagating relevance",
            }
            attribution_step = _ATTRIBUTION_STEP.get(exp_name, "Computing gradients")

            update_job_result(job_id, {
                "explainer_name": exp_name,
                "display_name": display_name,
                "status": "running",
                "current_step": "Initializing explainer",
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

                # GradCam / GuidedGradCam: always set the target layer explicitly for
                # all image models (both built-in and HF). pnpxai's FX-based
                # find_cam_target_layer can fail for ResNet-50 and other standard
                # architectures if symbolic_trace encounters issues. Using our robust
                # _find_cam_target_layer (AdaptiveAvgPool2d sibling walk) avoids this.
                # Note: GradCam.attribute was monkey-patched above to honour self.layer;
                # GuidedGradCam.attribute already uses self.layer correctly.
                if task == "image" and exp_name in {"GradCam", "GuidedGradCam"} and hasattr(explainer_instance, "set_target_layer"):
                    cam_target = _find_cam_target_layer(active_model)
                    if cam_target is not None:
                        print(f"[{exp_name}] Setting target layer ({model_name}): {type(cam_target).__name__}")
                        # set_target_layer returns a new instance (set_kwargs clones) — must reassign
                        explainer_instance = explainer_instance.set_target_layer(cam_target)

                # Compute attribution
                update_result_step(job_id, exp_name, attribution_step)
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
                _METRIC_LABELS = {
                    "MuFidelity": "Evaluating Fidelity",
                    "AbPC":       "Evaluating AbPC",
                    "Sensitivity": "Evaluating Sensitivity",
                    "Complexity":  "Evaluating Complexity",
                }
                metric_list = ["AbPC", "Sensitivity", "Complexity"] if task == "text" else ["MuFidelity", "AbPC", "Sensitivity", "Complexity"]
                for metric_name in metric_list:
                    update_result_step(job_id, exp_name, _METRIC_LABELS[metric_name])
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
                update_result_step(job_id, exp_name, "Generating heatmap")
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
