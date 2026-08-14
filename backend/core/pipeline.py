import copy
import io
import math
import os
from typing import Optional
import numpy as np
import pandas as pd
import torch

from backend.tasks import get_task_handler
from backend.core.pnpxai_adapter import normalize_attribution, extract_metric_value
from backend.core.job_manager import (
    get_uploaded_data, update_job_status, update_job_predictions, update_job_forecast,
    update_job_result, update_result_step, rank_completed_results, VISUALIZATION_DIR,
    is_cancel_requested,
)

# Safety cap on how many pred_len-sized windows the Input & Prediction chart chains
# together -- the chart shows the whole available back-test region (scrollable), so this
# only guards against a pathological upload (e.g. tiny pred_len, huge file) rather than
# reflecting any real UI limit. Windows are batched into one forward pass (see
# `forecast_windows`), so even the full cap is a sub-second cost, not a per-window one.
_MAX_BACKTEST_WINDOWS = 5000

# How much of the uploaded series' tail counts as "test" for the Input & Prediction
# chart -- a plain fraction rather than a fixed date, so it holds for any dataset
# without needing to know an official train/val/test split. Skipped entirely below
# _SMALL_DATASET_ROWS timesteps -- on a small upload, a 20% slice leaves too little to
# back-test with, so the whole thing is used instead.
_TEST_FRACTION = 0.2
_SMALL_DATASET_ROWS = 100
# If that test region spans more timesteps than this, it's cut down to the most recent
# _RECENT_ROWS_CAP -- rounded *up* to the next whole pred_len window, so the boundary
# never lands mid-window.
_RECENT_ROWS_CAP = 100

# Known sample files get a pinned back-test window instead of the dynamic rows-based
# rule above, so the demo always shows the same illustrative period rather than
# whatever "most recent 20%" happens to land on. User uploads (data_name absent) always
# use the dynamic rule -- this only overrides the platform's own bundled samples.
_PINNED_SAMPLE_WINDOWS = {
    "ETTh1.csv": {"context_start": "2017-12-16 06:00:00", "n_windows": 20},
    # Context start is 2 weeks earlier than the requested 2018-08-07 -- the requested
    # end (2020-06-30) is illness.csv's very last row, so there's no data to extend
    # into for a whole 30th window; the start moved back instead of truncating short.
    "illness.csv": {"context_start": "2018-07-24 00:00:00", "n_windows": 30},
}


def _sliding_window_attribution(explainer_instance, input_tensor, target_tensor, window_size=512, stride=None):
    """Compute attribution via sliding windows for large time-series.

    Splits input_tensor (1, channels, seq_len) into overlapping windows of window_size,
    computes attribution for each, and stitches them back together.
    For overlapping regions, attributions are averaged.
    """
    seq_len = input_tensor.shape[-1]
    if stride is None:
        stride = window_size  # non-overlapping by default

    if seq_len <= window_size:
        # Small enough — compute directly
        try:
            return explainer_instance.attribute(input_tensor, target_tensor)
        except TypeError:
            return explainer_instance.attribute(inputs=input_tensor, targets=target_tensor)

    # Accumulate attributions and counts for averaging overlaps
    attr_sum = torch.zeros_like(input_tensor)
    attr_count = torch.zeros(1, 1, seq_len, device=input_tensor.device)

    starts = list(range(0, seq_len - window_size + 1, stride))
    # Ensure last window covers the tail
    if starts[-1] + window_size < seq_len:
        starts.append(seq_len - window_size)

    for s in starts:
        window = input_tensor[..., s:s + window_size].clone()
        if window.is_floating_point():
            window = window.requires_grad_(True)
        try:
            w_attr = explainer_instance.attribute(window, target_tensor)
        except TypeError:
            w_attr = explainer_instance.attribute(inputs=window, targets=target_tensor)

        if isinstance(w_attr, torch.Tensor):
            attr_sum[..., s:s + window_size] += w_attr.detach()
        else:
            attr_sum[..., s:s + window_size] += torch.as_tensor(w_attr, device=attr_sum.device)
        attr_count[..., s:s + window_size] += 1

    # Average overlapping regions
    attr_count = attr_count.clamp(min=1)
    return attr_sum / attr_count


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
    from backend.core.device import model_device
    input_tensor = input_tensor.to(model_device(model))
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
    from backend.core.device import model_device
    encoded, tokens = handler.tokenize(raw_text, model_name)
    input_ids = encoded["input_ids"].to(model_device(model))

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


def _run_ts_backtest(
    job_id: str, model, input_tensor: torch.Tensor, input_data: dict,
    window_index: Optional[int] = None, data_name: Optional[str] = None,
) -> torch.Tensor:
    """Back-test a forecasting model against the tail of the uploaded series and record
    the result for the Input & Prediction chart.

    The horizon is a chain of non-overlapping `pred_len`-sized forecasts, capped to the
    most recent `_TEST_FRACTION` of the upload (falling back to the most recent
    `_RECENT_WINDOWS_CAP` windows of that if it's a long stretch) so the chart reads as
    a held-out test-like tail rather than the whole dataset: window w's context is the
    real data right before it (already part of the upload, just held out for display)
    -- not the model's own prior output -- so this reads as a stitched back-test rather
    than an autoregressive rollout. The chain itself -- its length and every window's
    position -- never depends on `window_index`, so the chart stays put across requests;
    only which window counts as "the" explained one changes.

    By default (`window_index=None`, equivalently 0) window 0 -- the earliest, leftmost
    segment -- is the one whose context is returned and becomes the explainers' input,
    so the chart's first segment and the attributions describe the same single forward
    pass; other segments are for eyeballing consistency only. Passing `window_index`
    picks a different segment from that same already-computed chain instead -- this is
    how "explain this specific predicted window" (a user clicking a segment on the
    chart) is implemented: a full re-run, same chart, different explained segment (the
    chart then also hides everything before that segment, so the display never mixes
    two different windows' worth of history).

    When the upload is too short to hold anything out, the horizon simply runs a single
    `pred_len` past the end of the data and there is no observed series to compare against;
    `window_index` is meaningless there and is ignored.
    """
    seq_len = getattr(model, "SEQ_LEN", 96)
    pred_len = getattr(model, "PRED_LEN", 96)
    total = input_tensor.shape[-1]
    labels = input_data.get("time_labels")
    timestamps = input_data.get("timestamps")

    has_actual = total >= seq_len + pred_len
    if has_actual:
        pinned = _PINNED_SAMPLE_WINDOWS.get(data_name) if data_name else None
        pinned_start = None
        if pinned and timestamps is not None:
            try:
                pinned_start = timestamps.get_loc(pd.Timestamp(pinned["context_start"]))
            except KeyError:
                pinned_start = None  # a differently-trimmed copy of the sample -- fall through

        if pinned_start is not None and pinned_start + seq_len <= total:
            # A bundled sample -- always show this specific illustrative period rather
            # than whatever the dynamic rows-based rule below would land on.
            chain_start = pinned_start
            n_windows = min(_MAX_BACKTEST_WINDOWS, pinned["n_windows"],
                            (total - chain_start - seq_len) // pred_len)
            horizon_len = n_windows * pred_len
            hz_slice = slice(chain_start + seq_len, chain_start + seq_len + horizon_len)
        else:
            full_capacity = (total - seq_len) // pred_len  # windows the whole upload could hold
            if total < _SMALL_DATASET_ROWS:
                # Small upload -- the 20% rule would leave too little to back-test with, so
                # the test-region restriction is skipped entirely and everything is used.
                n_windows = min(_MAX_BACKTEST_WINDOWS, full_capacity)
            else:
                # Cap the chain to the most recent _TEST_FRACTION of the *whole* upload, not
                # just to _MAX_BACKTEST_WINDOWS -- otherwise, on a long series, the chain
                # would eat into data that trained the model (if it did), and the chart
                # would read as "the whole dataset" rather than a held-out test-like tail.
                # A fixed fraction generalizes across datasets without needing an official
                # split's dates.
                test_start = round(total * (1 - _TEST_FRACTION))
                region_rows = total - test_start - seq_len  # rows left for windows after
                                                              # reserving seq_len for context
                if region_rows > _RECENT_ROWS_CAP:
                    # Round *up* to the next whole pred_len window rather than down, so the
                    # window straddling the cap is kept whole instead of dropped.
                    region_rows = math.ceil(_RECENT_ROWS_CAP / pred_len) * pred_len
                n_windows = min(_MAX_BACKTEST_WINDOWS, full_capacity, max(1, region_rows // pred_len))
            horizon_len = n_windows * pred_len
            chain_start = total - seq_len - horizon_len  # window 0's context start
            hz_slice = slice(total - horizon_len, total)
    else:
        n_windows = 1
        hz_slice = None

    # Which chained window's context is "the" explained one -- the earliest (first)
    # by default, matching the chart's default leftmost scroll position and hiding
    # nothing, unless a click asked for a specific one instead (which also hides
    # every earlier window on the chart, so only the explained one's own history and
    # everything after it stays visible).
    canonical_w = max(0, min(window_index, n_windows - 1)) if window_index is not None else 0

    if has_actual:
        ctx_slice = slice(chain_start + canonical_w * pred_len,
                          chain_start + canonical_w * pred_len + seq_len)
    else:
        ctx_slice = slice(max(total - seq_len, 0), total)

    context = input_tensor[..., ctx_slice]
    actual = input_tensor[..., hz_slice] if has_actual else None

    def _set_time(sl):
        if hasattr(model, "set_time_context"):
            model.set_time_context(timestamps[sl] if timestamps is not None else None)

    if has_actual and hasattr(model, "forecast_windows"):
        # Batch every chained window into one forward pass instead of one call per
        # window -- the difference between milliseconds and (once the chain covers a
        # whole dataset's tail, potentially thousands of windows) a full minute.
        window_slices = [slice(chain_start + w * pred_len, chain_start + w * pred_len + seq_len)
                         for w in range(n_windows)]
        batched_contexts = torch.stack([input_tensor[0, :, s] for s in window_slices], dim=0)
        timestamps_list = [timestamps[s] for s in window_slices] if timestamps is not None else None
        predicted = model.forecast_windows(batched_contexts, timestamps_list)
        predicted = predicted.reshape(n_windows * pred_len, predicted.shape[-1])
        _set_time(ctx_slice)  # leave the model's covariate state matching `context`
    else:
        # Fallback for models without a batched path (or the too-short-upload case,
        # where there's only ever one window): one forecast_horizon call per window,
        # each fed the real context that precedes it.
        pred_chunks = []
        pos = (chain_start + seq_len) if has_actual else None
        for w in range(n_windows):
            if w == canonical_w:
                w_context = context
                _set_time(ctx_slice)
            else:
                w_ctx_slice = slice(pos - seq_len, pos)
                w_context = input_tensor[..., w_ctx_slice]
                _set_time(w_ctx_slice)
            pred_chunks.append(model.forecast_horizon(w_context))
            if has_actual:
                pos += pred_len
        predicted = torch.cat(pred_chunks, dim=0)  # (n_windows * pred_len, channels)
        _set_time(ctx_slice)  # restore for the explainers, which run on `context`

    ctx_labels = labels[ctx_slice] if labels else None
    if not labels:
        hz_labels = None
    elif has_actual:
        hz_labels = labels[hz_slice]
    elif timestamps is not None and len(timestamps) > 1:
        step = timestamps[-1] - timestamps[-2]
        hz_labels = [(timestamps[-1] + step * (i + 1)).strftime("%Y-%m-%d %H:%M")
                     for i in range(pred_len)]
    else:
        hz_labels = [f"+{i + 1}" for i in range(pred_len)]

    col_names = input_data.get("col_names") or [f"var_{i + 1}" for i in range(context.shape[1])]
    update_job_forecast(job_id, {
        "col_names": col_names,
        "context": context[0].transpose(0, 1).detach().cpu().tolist(),
        "predicted": predicted.detach().cpu().tolist(),
        "actual": actual[0].transpose(0, 1).detach().cpu().tolist() if actual is not None else None,
        "context_labels": ctx_labels,
        "horizon_labels": hz_labels,
        "attributed_channel": (model.channel_for(len(col_names))
                               if hasattr(model, "channel_for") else 0),
        # So the chart can mark where each chained pred_len window ends, and highlight
        # which one (0-based within *this* response's own chain) `context` describes.
        "window_len": pred_len,
        "explained_window_index": canonical_w,
        # Whether a window was actually picked, as opposed to defaulting to the first
        # one. The chart can't tell these apart from `explained_window_index` alone --
        # both are 0 -- and they display differently: the default shows the whole chain
        # to choose from, a selection shows only the chosen window.
        "window_selected": window_index is not None,
    })

    # Keep rendering aligned with what was actually explained — including the labels, or
    # the renderer sees a full-series label list against a context-length x-axis, gives up
    # on the length mismatch, and falls back to bare timestep indices.
    input_data["tensor"] = context
    if ctx_labels:
        input_data["time_labels"] = ctx_labels
    # The single step the explainers attribute: the explained window's first predicted
    # timestep, every channel. The attribution plots draw it just past the input window
    # so the history and the thing that history explains sit in one picture.
    first_step = canonical_w * pred_len
    input_data["next_pred"] = predicted[first_step].detach().cpu().numpy()
    input_data["next_pred_label"] = hz_labels[first_step] if hz_labels else None
    # Which channel that scalar belongs to. Every channel's attribution is a gradient of
    # this one value, so only this channel's forecast may be drawn next to it -- a row
    # showing its own next step would read as if the colours on that row explained it.
    input_data["attributed_channel"] = (model.channel_for(len(col_names))
                                        if hasattr(model, "channel_for") else 0)
    return context


def run_explanation_pipeline(
    job_id: str,
    task: str,
    model_name: str,
    explainer_names: list[str],
    ranking_metric: str,
    params: dict,
):
    """Synchronous pipeline - runs in thread pool executor."""
    from backend.core.device import begin_job, end_job
    begin_job()  # paired with end_job() below (frees GPU when idle)
    job_ended = False
    try:
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

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
            # Time-series returns {"tensor": ..., "col_names": ...}
            if isinstance(input_data, dict) and "tensor" in input_data:
                input_tensor = input_data["tensor"]
                data_channels = input_tensor.shape[1]

                # The selected sample decides which checkpoint backs the chosen model.
                model = handler.load_model(model_name, num_input_channels=data_channels,
                                           dataset=params.get("data_name"))
                explainer_model = model
                from backend.core.device import model_device
                input_tensor = input_tensor.to(model_device(model))
                model.eval()

                # Time-series is forecast-only: the chart gets a back-test, and the
                # explainers get the very context window that produced it. There is one
                # output (the next predicted value), so there is no class to pick.
                input_tensor = _run_ts_backtest(
                    job_id, model, input_tensor, input_data,
                    window_index=params.get("ts_window_index"),
                    data_name=params.get("data_name"),
                )
                target_class = 0
            else:
                input_tensor = input_data if isinstance(input_data, torch.Tensor) else None
                target_class = 0

        # For each explainer: attribution + metrics + visualization
        job_dir = os.path.join(VISUALIZATION_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        all_results = []

        for exp_name in explainer_names:
            if is_cancel_requested(job_id):
                break

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
                from backend.core.device import model_device
                if task == "text" and exp_name in _GRADIENT_FREE_TEXT_EXPLAINERS:
                    from pnpxai.explainers.utils.feature_masks import NoMask1d
                    active_model = _TextInputIdsWrapper(model)
                    active_inp = text_input_ids.clone()
                    explainer_instance = ExplainerClass(active_model, feature_mask_fn=NoMask1d())
                elif exp_name in _STATE_MUTATING_EXPLAINERS:
                    _dev = model_device(explainer_model)
                    try:
                        buf = io.BytesIO()
                        torch.save(explainer_model, buf)
                        buf.seek(0)
                        active_model = torch.load(buf, map_location=_dev, weights_only=False)
                    except Exception:
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
                _target_dev = active_inp.device if active_inp is not None else model_device(active_model)
                target_tensor = torch.tensor([target_class], dtype=torch.long, device=_target_dev)
                if active_inp is not None:
                    # Use sliding window for large timeseries
                    ts_window_size = params.get("window_size", 512)
                    if task == "timeseries" and active_inp.dim() == 3 and active_inp.shape[-1] > ts_window_size:
                        update_result_step(job_id, exp_name, f"{attribution_step} (sliding window)")
                        attribution_raw = _sliding_window_attribution(
                            explainer_instance, active_inp, target_tensor,
                            window_size=ts_window_size,
                        )
                    else:
                        try:
                            attribution_raw = explainer_instance.attribute(active_inp, target_tensor)
                        except TypeError:
                            attribution_raw = explainer_instance.attribute(inputs=active_inp, targets=target_tensor)
                    attribution = normalize_attribution(attribution_raw, task=task)
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
                # Both classification-only: MuFidelity masks a 2D grid of pixels, and
                # AbPC scores the target *class probability*. A forecaster emits one
                # continuous value, so AbPC's softmax is constant and its score is
                # identically 0 — a misleading number rather than a measurement.
                if task == "timeseries":
                    metric_list = ["Sensitivity", "Complexity"]
                elif task == "text":
                    metric_list = ["AbPC", "Sensitivity", "Complexity"]
                else:
                    metric_list = ["MuFidelity", "AbPC", "Sensitivity", "Complexity"]
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
                if tokens_for_viz:
                    viz_input = tokens_for_viz
                elif task == "timeseries" and isinstance(input_data, dict):
                    viz_input = input_data
                else:
                    viz_input = raw_data
                handler.render_result(attribution, viz_input, viz_path, display_name=display_name)

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
                    "sensitivity": round(-(metric_values.get("sensitivity") or 0), 4) if metric_values.get("sensitivity") is not None else None,
                    "complexity": round(-(metric_values.get("complexity") or 0), 4) if metric_values.get("complexity") is not None else None,
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

        # Rank results (shared with the precomputed-cache path)
        rank_completed_results(all_results, ranking_metric)
        for r in all_results:
            if r["status"] == "completed":
                update_job_result(job_id, r)

        # Release GPU memory (if idle) BEFORE marking the job terminal. A client
        # polling the job's status treats "cancelled"/"completed" as a signal that
        # it's safe to immediately start a new job on the same (cached) model; if we
        # flipped the status first, a fast re-click could start loading that model
        # onto the device while this thread is still mid-offload, racing two threads'
        # .to() calls on the same module and corrupting CUDA state.
        end_job()
        job_ended = True
        update_job_status(job_id, "cancelled" if is_cancel_requested(job_id) else "completed")

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job_status(job_id, "failed", str(e))
    finally:
        if not job_ended:
            end_job()
