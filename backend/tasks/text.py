import torch
import numpy as np
from typing import Any

from backend.tasks.base import TaskHandler
from backend.core.model_paths import local_dir
from backend.renderers.text_renderer import render_text_attribution


def _resolve_text_source(model_name: str, hf_id: str):
    """Return (path_or_hf_id, local_files_only) preferring a locally-saved model."""
    d = local_dir("text", model_name)
    if d.exists():
        return str(d), True
    return hf_id, False

_TEXT_MODELS = {
    "distilbert-sst2": {
        "display_name": "DistilBERT (SST-2)",
        "architecture": "Transformer",
        "description": "DistilBERT fine-tuned on SST-2 for sentiment analysis (positive/negative).",
        "hf_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "num_labels": 2,
        "label_map": {0: "NEGATIVE", 1: "POSITIVE"},
    },
}

_loaded_models: dict[str, Any] = {}
_loaded_tokenizers: dict[str, Any] = {}
_hf_text_cache: dict[str, dict] = {}


def _is_preset(model_name: str) -> bool:
    return model_name in _TEXT_MODELS


def _get_tokenizer(model_name: str):
    if model_name not in _loaded_tokenizers:
        from transformers import AutoTokenizer
        hf_id = _TEXT_MODELS[model_name]["hf_id"]
        src, local_only = _resolve_text_source(model_name, hf_id)
        _loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    return _loaded_tokenizers[model_name]


def _load_hf_text_model(model_id: str) -> dict:
    if model_id not in _hf_text_cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        label_map = {}
        if hasattr(model.config, "id2label"):
            label_map = {int(k): v for k, v in model.config.id2label.items()}
        _hf_text_cache[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "label_map": label_map,
        }
    return _hf_text_cache[model_id]


class TextTaskHandler(TaskHandler):
    task_name = "text"

    def get_models(self) -> list[dict]:
        return [
            {"name": name, "display_name": info["display_name"],
             "architecture": info["architecture"], "description": info["description"],
             "task": "text"}
            for name, info in _TEXT_MODELS.items()
        ]

    def get_explainers(self, model_name: str) -> list[dict]:
        # Detection-driven: expose exactly what pnpxai recommends for this model.
        from backend.core.explainer_catalog import detect_explainers
        model = self.load_model(model_name)
        return detect_explainers(model, self.get_modality(), cache_key=f"text:{model_name}")

    def load_model(self, model_name: str) -> torch.nn.Module:
        from backend.core.device import to_device
        if not _is_preset(model_name):
            return to_device(_load_hf_text_model(model_name)["model"])
        if model_name not in _loaded_models:
            from transformers import AutoModelForSequenceClassification
            hf_id = _TEXT_MODELS[model_name]["hf_id"]
            src, local_only = _resolve_text_source(model_name, hf_id)
            model = AutoModelForSequenceClassification.from_pretrained(src, local_files_only=local_only)
            model.eval()
            _loaded_models[model_name] = model
        return to_device(_loaded_models[model_name])

    def get_label_map(self, model_name: str) -> dict:
        if not _is_preset(model_name):
            return _load_hf_text_model(model_name).get("label_map", {})
        return _TEXT_MODELS.get(model_name, {}).get("label_map", {})

    def tokenize(self, text: str, model_name: str):
        if not _is_preset(model_name):
            tokenizer = _load_hf_text_model(model_name)["tokenizer"]
        else:
            tokenizer = _get_tokenizer(model_name)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return encoded, tokens

    def preprocess_input(self, raw_data: Any) -> Any:
        return raw_data

    def get_modality(self):
        from pnpxai.core.modality.modality import TextModality
        return TextModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        tokens = input_data if isinstance(input_data, list) else str(input_data).split()
        return render_text_attribution(tokens, attribution, output_path)
