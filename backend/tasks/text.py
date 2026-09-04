import torch
import numpy as np
from typing import Any

from backend.tasks.base import TaskHandler
from backend.core.model_paths import local_dir
from backend.renderers.text_renderer import render_text_attribution


def _resolve_text_source(model_name: str, hf_id: str | None):
    """Return (path_or_hf_id, local_files_only) preferring a locally-saved model.

    Presence is judged by config.json rather than the directory: an empty or half-written
    snapshot directory exists but cannot be loaded, and `local_files_only=True` against it
    fails far from the actual cause.
    """
    d = local_dir("text", model_name)
    if (d / "config.json").exists():
        return str(d), True
    if hf_id is None:
        # Locally-trained preset with no Hub fallback. Without this the caller would hit
        # from_pretrained(None, ...) and get an unrelated-looking error.
        raise RuntimeError(
            f"text model '{model_name}' is built locally and was not found at {d}. "
            f"Run: python -m backend.scripts.train_toxigen"
        )
    return hf_id, False

_TEXT_MODELS = {
    "toxigen-bert": {
        "display_name": "BERT",
        "description": "BERT fine-tuned on ToxiGen (ACL 2022) to rate toxicity severity "
                       "on the dataset's own 1-5 annotator scale. Most ToxiGen statements "
                       "carry no slur, so the attribution has to point at what makes an "
                       "otherwise plain sentence harmful.",
        # Built by backend/scripts/train_toxigen.py — it does not exist on the Hub, so
        # there is no remote to fall back to.
        "hf_id": None,
        "num_labels": 5,
        # Every word here is the paper's, not ours. Two distinct pieces of ToxiGen
        # terminology are combined:
        #  - the scale's endpoints, §4.1: annotators rate "potential harm on a 1-5 scale
        #    with 1 being clearly benign and 5 indicating very offensive or abusive text";
        #  - the bands, footnote 8: the paper bins that same scale into three named
        #    classes — "scores <3: 'non-toxic', =3: 'ambiguous', >3: 'toxic'" — and
        #    Figure 4 labels its y-axis with exactly those three ("non-toxic, toxic,
        #    ambiguous"). So 2/3/4 are no longer bare numbers.
        # One honest caveat: footnote 8 bins max(HARMFULIFAI, HARMFULIFHUMAN), whereas
        # this model regresses toxicity_human alone. Same scale, same question, one of
        # the two raters — the thresholds carry over, the aggregation does not.
        # Note "toxic/benign" elsewhere in the paper is the *generation* label of the
        # 274k prompts, a different thing from these annotator-derived bands.
        # This dict is what get_label_map() serves; config.id2label mirrors it but is
        # not read.
        "label_map": {
            0: "1 — NON-TOXIC (clearly benign)",
            1: "2 — NON-TOXIC",
            2: "3 — AMBIGUOUS",
            3: "4 — TOXIC",
            4: "5 — TOXIC (very offensive or abusive)",
        },
        # One sample per severity level, all race-targeted and all free of slurs.
        "datasets": [
            "toxigen_l1_mexican.txt",
            "toxigen_l2_chinese.txt",
            "toxigen_l3_latino.txt",
            "toxigen_l4_middle_east.txt",
            "toxigen_l5_native_american.txt",
        ],
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
        # output_attentions=False: some repos (e.g. HateXplain) ship it enabled, which the
        # default sdpa attention rejects — and returning attentions is wasted work here.
        model = AutoModelForSequenceClassification.from_pretrained(model_id, output_attentions=False)
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
            # `architecture` is required by the ModelInfo schema; every text preset here is
            # a transformer, so an entry may leave it out.
            {"name": name, "display_name": info["display_name"],
             "architecture": info.get("architecture", "Transformer"),
             "description": info["description"], "task": "text"}
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
            model = AutoModelForSequenceClassification.from_pretrained(
                src, local_files_only=local_only, output_attentions=False
            )
            model.eval()
            _loaded_models[model_name] = model
        return to_device(_loaded_models[model_name])

    def get_model_datasets(self, model_name: str) -> list[str] | None:
        """Sample files this model is meant to be demoed on, or None for "anything".

        The sample list is shared per task, so a preset trained on one corpus would
        otherwise advertise files from another one. Uploaded / HF-id models are not
        tied to a corpus and keep the full list.
        """
        return _TEXT_MODELS.get(model_name, {}).get("datasets")

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

    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str,
                      display_name: str | None = None) -> str:
        tokens = input_data if isinstance(input_data, list) else str(input_data).split()
        return render_text_attribution(tokens, attribution, output_path)
