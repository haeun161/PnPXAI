"""Detection-driven explainer catalog.

The explainers shown for a model are derived from pnpxai's XaiRecommender
(modality + architecture detection) over pnpxai's full explainer universe — not a
hardcoded per-task list. Only explainers pnpxai detects as compatible with the
specific model are exposed, so what the user sees == what actually runs.

Display metadata (name/estimated time/order) is looked up per class name; any
explainer pnpxai adds that we don't know about still appears, using its class name.
"""
from typing import Optional

# class name -> (display_name, estimated_seconds, ui_order)
_META = {
    "Gradient":               ("Gradient", 2, 1),
    "GradientXInput":         ("Gradient × Input", 3, 2),
    "SmoothGrad":             ("SmoothGrad", 5, 3),
    "VarGrad":                ("VarGrad", 5, 4),
    "IntegratedGradients":    ("Integrated Gradients", 5, 5),
    "GradCam":                ("Grad-CAM", 3, 10),
    "GuidedGradCam":          ("Guided Grad-CAM", 3, 11),
    "AttentionRollout":       ("Attention Rollout", 3, 12),
    "TransformerAttribution": ("Transformer Attribution", 5, 13),
    "LRP":                    ("LRP", 3, 20),
    # "LRPEpsilonPlus":         ("LRP (Epsilon+)", 3, 21),
    # "LRPEpsilonGammaBox":     ("LRP (Epsilon-γ-box)", 3, 22),
    # "LRPEpsilonAlpha2Beta1":  ("LRP (α2β1)", 3, 23),
    "RAP":                    ("RAP", 5, 30),
    "Gfgp":                   ("GFGP", 5, 35),
    "Lime":                   ("LIME", 25, 40),
    "KernelShap":             ("KernelSHAP", 25, 41),
}
_DEFAULT_TIME = 5
_DEFAULT_ORDER = 50

# Cache raw pnpxai detection per model, keyed the same way for both the /explainers
# and /recommend endpoints, so a given (task, model) only ever runs XaiRecommender's
# detection once — it involves a model load + a full architecture walk, so re-running
# it per endpoint made every /explain click pay for it twice.
_cache: dict[str, dict] = {}


def _entry(name: str) -> dict:
    display_name, seconds, _ = _META.get(name, (name, _DEFAULT_TIME, _DEFAULT_ORDER))
    return {
        "name": name,
        "display_name": display_name,
        "estimated_compute_time_seconds": seconds,
        "compatible": True,
        "incompatibility_reason": None,
    }


def _order(entry: dict) -> tuple:
    return (_META.get(entry["name"], (None, None, _DEFAULT_ORDER))[2], entry["name"])


def _fallback() -> list[dict]:
    """Minimal, universally-applicable set if detection fails, so the UI is never empty."""
    return sorted(
        (_entry(n) for n in ("IntegratedGradients", "SmoothGrad", "GradientXInput",
                             "Gradient", "Lime", "KernelShap")),
        key=_order,
    )


def has_unregistered_attention(model) -> bool:
    """Attention modules pnpxai's type registry doesn't know (e.g. HF ViT/Swin/CLIP).
    CAM-based methods are dropped for these since their spatial assumptions don't hold."""
    return any(
        "attention" in type(m).__name__.lower() and len(list(m.children())) > 0
        for _, m in model.named_modules()
    )


def _detect_raw(model, modality, cache_key: Optional[str]) -> dict:
    """Run (or reuse a cached) pnpxai XaiRecommender detection for (model, modality).

    Returns {"explainer_names": [...] | None, "detected_architectures": [...]}.
    `explainer_names` is None if detection failed, so callers can fall back.
    """
    if cache_key and cache_key in _cache:
        return _cache[cache_key]
    try:
        from pnpxai.core.recommender.recommender import XaiRecommender, CAM_BASED_EXPLAINERS
        output = XaiRecommender().recommend(modality, model)
        explainers = list(output.explainers)
        if has_unregistered_attention(model):
            explainers = [e for e in explainers if e not in CAM_BASED_EXPLAINERS]
        result = {
            "explainer_names": [cls.__name__ for cls in explainers],
            "detected_architectures": sorted(a.__name__ for a in output.detected_architectures),
        }
    except Exception:
        result = {"explainer_names": None, "detected_architectures": []}
    if cache_key:
        _cache[cache_key] = result
    return result


def detect_explainers(model, modality, cache_key: Optional[str] = None,
                      exclude: Optional[set] = None) -> list[dict]:
    """Return the pnpxai-detected compatible explainers for (model, modality) as UI dicts.

    pnpxai's architecture detection can over-report: it recommends explainers whose
    assumptions don't hold for the concrete pipeline (e.g. CAM/RAP/perturbation on
    1D-conv & MOMENT time-series models all fail at runtime). `exclude` drops those
    known-incompatible names so the list is exactly what actually runs ("되는 애들만").
    """
    exclude = exclude or set()
    names = _detect_raw(model, modality, cache_key)["explainer_names"]
    if names is None:
        return [e for e in _fallback() if e["name"] not in exclude]
    result = sorted(
        (_entry(n) for n in names if n not in exclude),
        key=_order,
    )
    return result or [e for e in _fallback() if e["name"] not in exclude]


def detect_architectures(model, modality, cache_key: Optional[str] = None) -> list[str]:
    """Return the sorted architecture class names pnpxai detected for (model, modality)."""
    return _detect_raw(model, modality, cache_key)["detected_architectures"]


def detect_recommended_names(model, modality, cache_key: Optional[str] = None) -> Optional[list[str]]:
    """Return the raw (pre-exclude) pnpxai-recommended explainer class names, or None on failure."""
    return _detect_raw(model, modality, cache_key)["explainer_names"]
