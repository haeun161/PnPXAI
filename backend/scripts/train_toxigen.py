"""Fine-tune a 5-class toxicity-severity classifier on ToxiGen for the text task.

Produces models/text/toxigen-bert/, which backend/tasks/text.py loads with
local_files_only=True. This model does not exist on the Hub — it is built here — so
download_models.py cannot supply it.

    python -m backend.scripts.train_toxigen
    python -m backend.scripts.train_toxigen --soft-labels   # R-E fallback, see below

Design constraints (see .omc/plans/toxigen-text-task.md):
  * Output stays a (batch, 5) logit tensor so backend/core/pipeline.py's existing
    softmax/argmax text path works unmodified. A regression or ordinal (CORAL) head
    would break that, which is why the label is a rounded class and not a scalar.
  * Architecture stays BERT so pnpxai's explainer detection and the embedding-wrapper
    path in pipeline.py keep working.
  * Trained on all 13 target groups (9,900 rows). "Race-related" scopes the demo
    samples, not the training data.

Label construction: `round(toxicity_human)`, where toxicity_human is the mean of 3
annotators on the paper's 1-5 harm scale (1 = clearly benign, 5 = very offensive or
abusive; the paper reads <3 as non-toxic, 3 as ambiguous, >3 as toxic). Rounding throws
away the annotator disagreement encoded in the thirds. --soft-labels keeps it by
splitting the mass across the two adjacent classes (3.67 -> L3 0.33 / L4 0.67) and
training with KL; use it if the middle classes collapse (eval_toxigen --collapse-check).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Pin to one GPU before torch initializes CUDA. The container exposes 8, and Trainer
# silently wraps the model in DataParallel when it sees more than one — which multiplies
# per_device_train_batch_size by the device count. That turned the intended batch of 16
# into an effective 128 and cut training from 2,016 steps to 252; the resulting model
# predicted L3/L4 for ~2% of rows each against a true 13-16%, i.e. a 3-class model with a
# 5-class head. Same hyperparameters on a single device reach macro-F1 0.483 instead of
# 0.379 and put every class within a few points of its true frequency.
# Override with --gpus (e.g. --gpus 0,1) or by exporting CUDA_VISIBLE_DEVICES yourself.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpus = "0"
    for _i, _a in enumerate(sys.argv):
        if _a == "--gpus" and _i + 1 < len(sys.argv):
            _gpus = sys.argv[_i + 1]
        elif _a.startswith("--gpus="):
            _gpus = _a.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from backend.core.model_paths import local_dir  # noqa: E402
from backend.scripts.toxigen_data import load_annotated, NUM_CLASSES  # noqa: E402

BASE_MODEL = "bert-base-uncased"
OUT_NAME = "toxigen-bert"

# Entirely the paper's own vocabulary: the endpoints from §4.1 ("1 being clearly benign
# and 5 indicating very offensive or abusive text") and the band names from footnote 8,
# which bins this scale as "scores <3: 'non-toxic', =3: 'ambiguous', >3: 'toxic'".
# Kept identical to _TEXT_MODELS["toxigen-bert"]["label_map"] in backend/tasks/text.py,
# which is the authority the API actually reads; see that file for the caveat about
# footnote 8 binning the max of two ratings where we use toxicity_human alone.
ID2LABEL = {
    0: "1 — NON-TOXIC (clearly benign)",
    1: "2 — NON-TOXIC",
    2: "3 — AMBIGUOUS",
    3: "4 — TOXIC",
    4: "5 — TOXIC (very offensive or abusive)",
}


class ToxiGenDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, soft=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                             padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.soft = None if soft is None else torch.tensor(np.asarray(soft), dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        if self.soft is not None:
            item["soft_targets"] = self.soft[i]
        return item


def soft_targets(toxicity_human: np.ndarray) -> np.ndarray:
    """Spread each 1-5 mean across its two adjacent classes proportionally.

    3.67 -> class index 2 gets 0.33, index 3 gets 0.67. An exact integer puts all mass
    on one class, so this is a strict generalization of the rounded label.
    """
    x = np.clip(np.asarray(toxicity_human, dtype=float), 1.0, 5.0) - 1.0  # 0..4
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, NUM_CLASSES - 1)
    frac = x - lo
    out = np.zeros((len(x), NUM_CLASSES), dtype=float)
    rows = np.arange(len(x))
    out[rows, lo] += 1.0 - frac
    out[rows, hi] += frac
    return out


def _make_trainer_cls(use_soft: bool):
    from transformers import Trainer

    if not use_soft:
        return Trainer

    class SoftLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Pop the soft targets before the forward pass: leaving `labels` in place
            # would make the model compute its own cross-entropy, which we are replacing.
            target = inputs.pop("soft_targets")
            inputs.pop("labels", None)
            outputs = model(**inputs)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            loss = F.kl_div(log_probs, target, reduction="batchmean")
            return (loss, outputs) if return_outputs else loss

    return SoftLabelTrainer


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "within_one": float(np.mean(np.abs(preds - labels) <= 1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)  # matches text.py::tokenize
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", default="0",
                    help="CUDA_VISIBLE_DEVICES to use; more than one enables DataParallel "
                         "and multiplies the effective batch size (see module docstring)")
    ap.add_argument("--soft-labels", action="store_true",
                    help="R-E fallback: KL against adjacent-class soft targets")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              TrainingArguments, set_seed)

    set_seed(args.seed)

    # The 940-row test split is held out entirely: it is what eval_toxigen.py scores, so
    # letting it influence model selection would make that gate self-congratulatory.
    train_df, _test_df = load_annotated()
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)), test_size=args.val_frac,
        random_state=args.seed, stratify=train_df["label"].values,
    )
    tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    n_dev = torch.cuda.device_count()
    print(f"[train] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"-> {n_dev} device(s); effective batch = {args.batch * max(n_dev, 1)}")
    print(f"[train] {len(tr)} train / {len(va)} val (held-out test: {len(_test_df)})")
    print(f"[train] class counts: {tr['label'].value_counts().sort_index().to_dict()}")
    if args.soft_labels:
        print("[train] SOFT LABELS enabled (KL over adjacent classes)")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        output_attentions=False,  # sdpa attention rejects it, and text.py never asks for it
    )

    ds_tr = ToxiGenDataset(tr["text"], tr["label"], tokenizer, args.max_length,
                           soft=soft_targets(tr["toxicity_human"]) if args.soft_labels else None)
    ds_va = ToxiGenDataset(va["text"], va["label"], tokenizer, args.max_length,
                           soft=soft_targets(va["toxicity_human"]) if args.soft_labels else None)

    dest = local_dir("text", args.out_name)
    work = str(dest) + ".work"
    targs = TrainingArguments(
        output_dir=work,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=args.seed,
        eval_strategy="epoch",      # renamed from evaluation_strategy in transformers v5
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = _make_trainer_cls(args.soft_labels)(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    print(f"[train] best val metrics: {json.dumps({k: round(v, 4) for k, v in final.items() if isinstance(v, float)})}")

    dest.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    with open(dest / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base,
            "dataset": "toxigen/toxigen-data :: annotated (all 13 target groups)",
            "label_rule": "soft(adjacent)" if args.soft_labels else "round(toxicity_human)",
            "n_train": len(tr), "n_val": len(va), "held_out_test": len(_test_df),
            "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
            "max_length": args.max_length, "seed": args.seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "effective_batch": args.batch * max(torch.cuda.device_count(), 1),
            "val_metrics": {k: v for k, v in final.items() if isinstance(v, float)},
        }, f, indent=2)
    print(f"[train] saved -> {dest}")

    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e6
    print(f"[train] {size:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
