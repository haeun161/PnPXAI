import torch
import torchvision.models as models
import numpy as np
from typing import Any, Optional
from PIL import Image

from backend.tasks.base import TaskHandler
from backend.core.image_utils import preprocess_image, get_original_image_array
from backend.renderers.image_renderer import render_heatmap

_IMAGE_MODELS = {
    "resnet50": {
        "display_name": "ResNet-50",
        "architecture": "Residual Network",
        "description": "50-layer deep residual network, widely used in XAI research.",
        "loader": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
    },
    "vgg16": {
        "display_name": "VGG-16",
        "architecture": "Sequential CNN",
        "description": "16-layer sequential CNN, ideal for layer-wise XAI methods.",
        "loader": lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    },
    "densenet121": {
        "display_name": "DenseNet-121",
        "architecture": "Dense Connections",
        "description": "121-layer densely connected network.",
        "loader": lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
    },
}

_loaded_models: dict[str, torch.nn.Module] = {}
_hf_image_cache: dict[str, dict] = {}

_IMAGE_EXPLAINERS = [
    {"name": "IntegratedGradients", "display_name": "Integrated Gradients", "estimated_time": 5},
    {"name": "GradCam", "display_name": "Grad-CAM", "estimated_time": 3},
    {"name": "GuidedGradCam", "display_name": "Guided Grad-CAM", "estimated_time": 3},
    {"name": "SmoothGrad", "display_name": "SmoothGrad", "estimated_time": 5},
    {"name": "VarGrad", "display_name": "VarGrad", "estimated_time": 5},
    {"name": "GradientXInput", "display_name": "Gradient × Input", "estimated_time": 2},
    {"name": "Gradient", "display_name": "Gradient", "estimated_time": 2},
    {"name": "LRPUniformEpsilon", "display_name": "LRP (Uniform Epsilon)", "estimated_time": 3},
    {"name": "Lime", "display_name": "LIME", "estimated_time": 25},
    {"name": "KernelShap", "display_name": "KernelSHAP", "estimated_time": 25},
    {"name": "RAP", "display_name": "RAP", "estimated_time": 5},
]


class _HFImageWrapper(torch.nn.Module):
    """Wraps a HuggingFace image classification model to return raw logits tensor.
    Used as fallback when FX tracing fails. Handles both patched (tuple) and unpatched
    (output object) model outputs since _patch_hf_conditionals may have run first.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out.logits if hasattr(out, "logits") else out

    def __deepcopy__(self, memo):
        import copy
        # Deep-copy the underlying model, then re-patch its submodule forwards.
        # _patch_hf_conditionals closures capture bound methods from the ORIGINAL
        # model instance; after deepcopy those closures still reference the original,
        # causing FX path_of_module failures. Re-patching rebinds them to the copy.
        new_inner = copy.deepcopy(self.model, memo)
        _patch_hf_conditionals(new_inner)
        new_wrapper = _HFImageWrapper(new_inner)
        memo[id(self)] = new_wrapper
        return new_wrapper


class _FXHFLogitsWrapper(torch.nn.Module):
    """FX-traceable wrapper: traced model output has a .logits attribute."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values).logits


class _FXHFTupleWrapper(torch.nn.Module):
    """FX-traceable wrapper: traced model output is a tuple, logits at index 0."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)[0]


class _FXHFTensorWrapper(torch.nn.Module):
    """FX-traceable wrapper: traced model already returns a raw tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values)


def _patch_hf_conditionals(model: torch.nn.Module) -> None:
    """Patch all HF submodule forward methods to pass concrete defaults for
    conditional parameters (return_dict, output_hidden_states, labels, etc.).

    HF models use patterns like:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict: ...
    During torch.fx symbolic tracing these become Proxy-based conditions that raise
    TraceError. By ensuring these params arrive as concrete Python values (False/None),
    FX can evaluate the branches at trace time and proceed without error.

    Modifies model in-place. Safe on cached models (runs once per model_id).
    """
    import functools
    import inspect

    HF_CONCRETE = {
        "return_dict":            False,
        "output_hidden_states":   None,
        "output_attentions":      None,
        "labels":                 None,
        "head_mask":              None,
        "interpolate_pos_encoding": None,
        "bool_masked_pos":        None,
    }

    for _, module in model.named_modules():
        try:
            sig = inspect.signature(module.forward)
        except (TypeError, ValueError):
            continue

        to_fix = {k: v for k, v in HF_CONCRETE.items() if k in sig.parameters}
        if not to_fix:
            continue

        orig = module.forward
        defaults = dict(to_fix)

        @functools.wraps(orig)
        def _patched(*args, _f=orig, _d=defaults, **kwargs):
            for k, v in _d.items():
                if k not in kwargs:
                    kwargs[k] = v
            return _f(*args, **kwargs)

        module.forward = _patched


def _try_hf_fx_trace(raw_model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Make an HF model FX-traceable by patching submodule conditionals + symbolic_trace.

    Patches raw_model's submodule forwards in-place so HF conditional params
    (return_dict, etc.) are always concrete during FX tracing. Then runs
    torch.fx.symbolic_trace to produce a control-flow-free GraphModule that
    pnpxai can re-trace for GradCam, LRP, RAP.

    Returns a _FXHF*Wrapper on success, or None on failure (falls back to _HFImageWrapper).
    """
    try:
        import torch.fx as fx

        _patch_hf_conditionals(raw_model)
        traced = fx.symbolic_trace(raw_model)

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            test_out = traced(dummy)

        if hasattr(test_out, "logits"):
            return _FXHFLogitsWrapper(traced)
        if isinstance(test_out, (tuple, list)):
            return _FXHFTupleWrapper(traced)
        return _FXHFTensorWrapper(traced)
    except Exception as e:
        import traceback
        print(f"[_try_hf_fx_trace] FX tracing failed ({type(e).__name__}): {e}")
        traceback.print_exc()
        return None


def _check_hf_model_compatibility(model_id: str):
    """Raises ValueError with a clear message if the model can't be loaded via transformers."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        library = info.library_name or "unknown"
        if library not in ("transformers", None):
            raise ValueError(
                f"This model uses the '{library}' library, which is not supported. "
                f"Only transformers-based image classification models are supported."
            )
    except ValueError:
        raise
    except Exception:
        pass  # HF Hub unavailable or model not found — let the actual load fail with its own error


def _load_hf_image_model(model_id: str) -> dict:
    if model_id not in _hf_image_cache:
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        _check_hf_model_compatibility(model_id)

        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
        except OSError:
            raise ValueError(
                f"'{model_id}' does not have a standard image processor config. "
                f"Make sure the model is a transformers image classification model with a preprocessor_config.json."
            )

        try:
            raw_model = AutoModelForImageClassification.from_pretrained(model_id)
        except ValueError as e:
            if "Unrecognized model" in str(e) or "model_type" in str(e):
                raise ValueError(
                    f"'{model_id}' uses an unrecognized architecture. "
                    f"The model may require a third-party library that is not installed."
                )
            raise

        raw_model.eval()
        wrapper = _try_hf_fx_trace(raw_model) or _HFImageWrapper(raw_model)
        label_map = {}
        if hasattr(raw_model.config, "id2label"):
            label_map = {int(k): v for k, v in raw_model.config.id2label.items()}
        _hf_image_cache[model_id] = {
            "wrapper": wrapper,
            "processor": processor,
            "label_map": label_map,
        }
    return _hf_image_cache[model_id]


class ImageTaskHandler(TaskHandler):
    task_name = "image"

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info["architecture"], "description": info["description"],
             "task": "image"}
            for name, info in _IMAGE_MODELS.items()
        ]

    def get_explainers(self, model_name: str) -> list[dict]:
        return [
            {
                "name": e["name"],
                "display_name": e["display_name"],
                "estimated_compute_time_seconds": e["estimated_time"],
                "compatible": True,
                "incompatibility_reason": None,
            }
            for e in _IMAGE_EXPLAINERS
        ]

    def load_model(self, model_name: str) -> torch.nn.Module:
        if model_name not in _IMAGE_MODELS:
            return _load_hf_image_model(model_name)["wrapper"]
        if model_name not in _loaded_models:
            model = _IMAGE_MODELS[model_name]["loader"]()
            model.eval()
            _loaded_models[model_name] = model
        return _loaded_models[model_name]

    def get_hf_label_map(self, model_name: str) -> dict:
        if model_name not in _IMAGE_MODELS:
            return _load_hf_image_model(model_name).get("label_map", {})
        return {}

    def preprocess_input(self, raw_data: Any, model_name: Optional[str] = None) -> Any:
        if model_name and model_name not in _IMAGE_MODELS:
            cache = _load_hf_image_model(model_name)
            inputs = cache["processor"](images=raw_data, return_tensors="pt")
            return inputs["pixel_values"]
        if isinstance(raw_data, Image.Image):
            return preprocess_image(raw_data)
        return raw_data

    def get_modality(self):
        from pnpxai.core.modality.modality import ImageModality
        return ImageModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        if isinstance(input_data, Image.Image):
            original_array = get_original_image_array(input_data)
        else:
            original_array = None
        return render_heatmap(attribution, output_path)
