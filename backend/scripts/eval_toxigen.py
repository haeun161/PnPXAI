"""Quality gates for the ToxiGen text model.

Two independent checks, both required before the swap is considered good:

  --collapse-check      the model still distinguishes the five severity levels
  --attribution-gate    its explanations are no worse than the outgoing model's

Neither gate scores predictive accuracy. The spec makes attribution plausibility the
product goal and explicitly drops accuracy targets; these gates exist to catch the two
ways that goal can be silently faked — a model that collapsed onto one class, and
attributions that are noise.

    python -m backend.scripts.eval_toxigen --collapse-check
    python -m backend.scripts.eval_toxigen --attribution-gate
    python -m backend.scripts.eval_toxigen --all

Exit code is non-zero when a requested gate fails.

--- Why the thresholds are what they are -------------------------------------------

collapse-check runs on the 940-row `annotated` test split, which training never sees.
  (1) no class may take more than 60% of predictions. The true majority class is 35.2%,
      so 60% is ~1.7x the base rate — a model that has stopped modelling the
      distribution rather than one that is merely imperfect.
  (2) every one of the 5 classes must be predicted at least once.
  (3) every class must reach a third of its true frequency. Condition (2) alone proved
      useless in practice: the first trained model satisfied it while predicting L3 for
      20 of 940 rows (2.1% against a true 13.1%) and L4 for 25 (2.7% against 15.7%) —
      a 3-class model wearing a 5-class head. The demo shows one sample card per level,
      so a level the model almost never emits is a broken demo no matter what the
      aggregate accuracy says. (3) is the condition that actually bites.

attribution-gate compares AbPC against hatexplain-bert measured on *the same five
sample texts* (see snapshot_baseline.py). Same inputs, same explainers, only the model
differs — comparing across different texts would confound "worse model" with "harder
text", since AbPC scales with token count.

  MuFidelity is not used: it comes back None for every text result in this repo
  (pnpxai's MuFidelity does not survive the embedding-wrapper path), so the 40-result
  baseline has 0 non-null values. Sensitivity and Complexity are not faithfulness
  measures — they describe robustness and sparsity and have no meaningful zero — which
  leaves AbPC as the only usable signal.

  AbPC is signed and higher-is-better: it is the area between the MoRF and LeRF
  perturbation curves, so positive means removing high-attribution tokens hurts the
  prediction more than removing low-attribution ones. On the baseline it ranges
  -0.58..+0.65 (5 of 8 explainers positive; Gradient and SmoothGrad are reliably
  negative on this architecture).

  Because it is signed, a ratio test ("within 80% of baseline") is meaningless — 80% of
  -0.36 is -0.29, which is *better*. Both conditions are therefore absolute differences:
  (1) mean over the shared explainers of (new - baseline) >= -0.10
  (2) the count of explainers with positive median AbPC may drop by at most 1
  (3) at most 1 of the 5 samples may have no usable AbPC at all
"""
import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_NAME = "toxigen-bert"
BASELINE_PATH = os.path.join("backend", "tests", "data", "hatexplain_baseline_metrics.json")

MAX_CLASS_SHARE = 0.60
MIN_CLASS_SHARE_RATIO = 0.33
MIN_MEAN_ABPC_DELTA = -0.10
MAX_POSITIVE_EXPLAINER_DROP = 1
MAX_SAMPLES_WITHOUT_ABPC = 1


# ---------------------------------------------------------------- collapse check

def collapse_check(model_name: str) -> bool:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from backend.core.model_paths import local_dir
    from backend.scripts.toxigen_data import load_annotated, NUM_CLASSES

    _train, test = load_annotated()

    # Loaded straight from disk rather than through TextTaskHandler on purpose: this gate
    # has to be able to run *before* the model is registered in _TEXT_MODELS, so that a
    # model which fails it never becomes the served preset.
    src = local_dir("text", model_name)
    if not (src / "config.json").exists():
        raise SystemExit(f"no model at {src} — run: python -m backend.scripts.train_toxigen")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(src), local_files_only=True, output_attentions=False)
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # TextTaskHandler.tokenize() takes one string at a time; batch here with the same
    # settings it uses (max_length=128, truncation) so the gate scores the same
    # preprocessing the served model gets.
    tk = AutoTokenizer.from_pretrained(str(src), local_files_only=True)

    preds = []
    texts = list(test["text"])
    for i in range(0, len(texts), 64):
        chunk = texts[i:i + 64]
        tok = tk(chunk, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            out = model(input_ids=tok["input_ids"].to(dev),
                        attention_mask=tok["attention_mask"].to(dev))
            logits = out.logits if hasattr(out, "logits") else out[0]
        preds.extend(logits.argmax(dim=1).cpu().tolist())

    preds = np.array(preds)
    gold = test["label"].values
    n = len(preds)

    print(f"\n=== collapse check ({n} held-out rows) ===")
    counts = {c: int((preds == c).sum()) for c in range(NUM_CLASSES)}
    shares = {c: counts[c] / n for c in counts}
    print("  predicted distribution:")
    for c in range(NUM_CLASSES):
        gold_share = float((gold == c).mean())
        print(f"    L{c+1}: predicted {counts[c]:4d} ({shares[c]*100:5.1f}%)   true {gold_share*100:5.1f}%")

    worst = max(shares, key=lambda c: shares[c])
    cond1 = shares[worst] <= MAX_CLASS_SHARE
    missing = [c + 1 for c in range(NUM_CLASSES) if counts[c] == 0]
    cond2 = not missing

    # "At least once" turned out to be far too lenient: a model predicting L3 for 20 of
    # 940 rows (2.1% against a true 13.1%) satisfies it while being, in effect, a 3-class
    # model. The demo puts one sample card per level on screen, so a level the model
    # almost never emits is a broken demo. Require each class to reach a third of its
    # true frequency.
    starved = [c + 1 for c in range(NUM_CLASSES)
               if shares[c] < MIN_CLASS_SHARE_RATIO * float((gold == c).mean())]
    cond3 = not starved

    print(f"\n  [{'PASS' if cond1 else 'FAIL'}] no class exceeds {MAX_CLASS_SHARE*100:.0f}% "
          f"(max is L{worst+1} at {shares[worst]*100:.1f}%)")
    print(f"  [{'PASS' if cond2 else 'FAIL'}] every class predicted at least once"
          + ("" if cond2 else f" — never predicts: {missing}"))
    print(f"  [{'PASS' if cond3 else 'FAIL'}] every class reaches "
          f"{MIN_CLASS_SHARE_RATIO:.0%} of its true frequency"
          + ("" if cond3 else f" — starved: {starved}"))

    # Reported for the record; deliberately not gated (see module docstring).
    exact = float((preds == gold).mean())
    within1 = float((np.abs(preds - gold) <= 1).mean())
    print(f"\n  (not gated) exact accuracy {exact*100:.1f}%  |  within +-1 {within1*100:.1f}%"
          f"  |  majority baseline 35.2%")
    print("  (not gated) per-class recall:")
    for c in range(NUM_CLASSES):
        m = gold == c
        rec = float((preds[m] == c).mean()) if m.any() else float("nan")
        print(f"    L{c+1}: {rec*100:5.1f}%  (n={int(m.sum())})")

    return cond1 and cond2 and cond3


# ------------------------------------------------------------- attribution gate

def _median_abpc_per_explainer(samples: dict) -> tuple[dict, int]:
    """{explainer: median AbPC across samples}, plus count of samples with no AbPC."""
    per = defaultdict(list)
    empty = 0
    for _name, s in samples.items():
        got = False
        for exp, m in s["explainers"].items():
            v = m.get("abpc")
            if v is not None:
                per[exp].append(float(v))
                got = True
        if not got:
            empty += 1
    return {e: st.median(v) for e, v in per.items()}, empty


def _samples_from_manifest(model_name: str, prefix: str) -> dict:
    from backend.core.precompute_cache import MANIFEST_PATH
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"no manifest at {MANIFEST_PATH} — run precompute_samples first")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for _fh, per_model in manifest.get("text", {}).items():
        entry = per_model.get(model_name)
        if not entry:
            continue
        name = entry.get("sample_name", "")
        if prefix and not name.startswith(prefix):
            continue
        out[name] = {"explainers": {
            r["explainer_name"]: {k: r.get(k) for k in
                                  ("mu_fidelity", "abpc", "sensitivity", "complexity")}
            for r in entry.get("results", []) if r.get("status") == "completed"
        }}
    return out


def attribution_gate(model_name: str, baseline_path: str) -> bool:
    if not os.path.exists(baseline_path):
        raise SystemExit(f"no baseline at {baseline_path} — run snapshot_baseline.py "
                         f"before removing the old preset")
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    base_samples = baseline["samples"]
    new_samples = _samples_from_manifest(model_name, baseline.get("sample_prefix", "toxigen_"))
    if not new_samples:
        raise SystemExit(f"no precomputed '{model_name}' results found — run "
                         f"precompute_samples text first")

    shared_samples = sorted(set(base_samples) & set(new_samples))
    print(f"\n=== attribution gate ({model_name} vs {baseline['model_name']}) ===")
    print(f"  samples compared: {len(shared_samples)} "
          f"(baseline {len(base_samples)}, new {len(new_samples)})")
    for name in sorted(set(base_samples) ^ set(new_samples)):
        print(f"    WARNING sample only on one side, excluded: {name}")

    base_med, _ = _median_abpc_per_explainer({k: base_samples[k] for k in shared_samples})
    new_med, empty = _median_abpc_per_explainer({k: new_samples[k] for k in shared_samples})

    shared_exp = sorted(set(base_med) & set(new_med))
    for e in sorted(set(base_med) ^ set(new_med)):
        print(f"    WARNING explainer only on one side, excluded from comparison: {e}")
    if not shared_exp:
        print("  [FAIL] no explainer in common — cannot compare")
        return False

    print(f"\n  per-explainer median AbPC (higher is better):")
    print(f"    {'explainer':<22} {'baseline':>9} {'new':>9} {'delta':>9}")
    deltas = []
    for e in shared_exp:
        d = new_med[e] - base_med[e]
        deltas.append(d)
        print(f"    {e:<22} {base_med[e]:>+9.4f} {new_med[e]:>+9.4f} {d:>+9.4f}")

    mean_delta = sum(deltas) / len(deltas)
    base_pos = sum(1 for e in shared_exp if base_med[e] > 0)
    new_pos = sum(1 for e in shared_exp if new_med[e] > 0)

    cond1 = mean_delta >= MIN_MEAN_ABPC_DELTA
    cond2 = new_pos >= base_pos - MAX_POSITIVE_EXPLAINER_DROP
    cond3 = empty <= MAX_SAMPLES_WITHOUT_ABPC

    print(f"\n  [{'PASS' if cond1 else 'FAIL'}] mean AbPC delta {mean_delta:+.4f} "
          f">= {MIN_MEAN_ABPC_DELTA:+.2f}")
    print(f"  [{'PASS' if cond2 else 'FAIL'}] explainers with positive AbPC: "
          f"{new_pos} vs baseline {base_pos} (drop of at most {MAX_POSITIVE_EXPLAINER_DROP})")
    print(f"  [{'PASS' if cond3 else 'FAIL'}] samples with no usable AbPC: {empty} "
          f"(at most {MAX_SAMPLES_WITHOUT_ABPC})")

    return cond1 and cond2 and cond3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse-check", action="store_true")
    ap.add_argument("--attribution-gate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--baseline", default=BASELINE_PATH)
    args = ap.parse_args()

    run_collapse = args.collapse_check or args.all
    run_attr = args.attribution_gate or args.all
    if not (run_collapse or run_attr):
        ap.error("pick --collapse-check, --attribution-gate, or --all")

    results = {}
    if run_collapse:
        results["collapse-check"] = collapse_check(args.model)
    if run_attr:
        results["attribution-gate"] = attribution_gate(args.model, args.baseline)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
