import torch
import numpy as np
from typing import Any

from backend.tasks.base import TaskHandler
from backend.renderers.text_renderer import render_text_attribution

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

_TEXT_EXPLAINERS = [
    {"name": "IntegratedGradients", "display_name": "Integrated Gradients", "estimated_time": 5},
    {"name": "Lime", "display_name": "LIME", "estimated_time": 25},
    {"name": "KernelShap", "display_name": "KernelSHAP", "estimated_time": 25},
    {"name": "SmoothGrad", "display_name": "SmoothGrad", "estimated_time": 5},
    {"name": "GradientXInput", "display_name": "Gradient × Input", "estimated_time": 3},
    {"name": "Gradient", "display_name": "Gradient", "estimated_time": 2},
]

_loaded_models: dict[str, Any] = {}
_loaded_tokenizers: dict[str, Any] = {}


def _get_tokenizer(model_name: str):
    if model_name not in _loaded_tokenizers:
        from transformers import AutoTokenizer
        hf_id = _TEXT_MODELS[model_name]["hf_id"]
        _loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(hf_id)
    return _loaded_tokenizers[model_name]


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
        return [
            {"name": e["name"], "display_name": e["display_name"],
             "estimated_compute_time_seconds": e["estimated_time"],
             "compatible": True, "incompatibility_reason": None}
            for e in _TEXT_EXPLAINERS
        ]

    def load_model(self, model_name: str) -> torch.nn.Module:
        if model_name not in _loaded_models:
            from transformers import AutoModelForSequenceClassification
            hf_id = _TEXT_MODELS[model_name]["hf_id"]
            model = AutoModelForSequenceClassification.from_pretrained(hf_id)
            model.eval()
            _loaded_models[model_name] = model
        return _loaded_models[model_name]

    def get_label_map(self, model_name: str) -> dict:
        return _TEXT_MODELS.get(model_name, {}).get("label_map", {})

    def tokenize(self, text: str, model_name: str):
        """Tokenize text and return (input_ids_tensor, tokens_list, attention_mask)."""
        tokenizer = _get_tokenizer(model_name)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return encoded, tokens

    def preprocess_input(self, raw_data: Any) -> Any:
        return raw_data  # Text string, tokenized later in pipeline

    def get_modality(self):
        from pnpxai.core.modality.modality import TextModality
        return TextModality()

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str) -> str:
        tokens = input_data if isinstance(input_data, list) else str(input_data).split()
        return render_text_attribution(tokens, attribution, output_path)
