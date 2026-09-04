"""Freeze the outgoing text model's attribution metrics as the comparison baseline.

The AC3 attribution gate compares the new toxigen-bert against hatexplain-bert on the
*same* sample texts. Those baseline numbers live in backend/precomputed/manifest.json,
which is git-ignored and regenerated per deployment, so they have to be lifted into a
tracked file before the old preset is removed.

Run this AFTER precomputing the old model over the new sample files and BEFORE swapping
_TEXT_MODELS, otherwise there is nothing left to compare against:

    python -m backend.scripts.snapshot_baseline
    python -m backend.scripts.snapshot_baseline --model hatexplain-bert --out <path>
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.precompute_cache import MANIFEST_PATH  # noqa: E402

DEFAULT_MODEL = "hatexplain-bert"
DEFAULT_OUT = os.path.join("backend", "tests", "data", "hatexplain_baseline_metrics.json")
# Only the samples the new model will also be measured on — comparing across different
# texts is meaningless, since AbPC scales with token count and MuFidelity with how much
# probability headroom the prediction has.
DEFAULT_SAMPLE_PREFIX = "toxigen_"

_METRIC_KEYS = ("mu_fidelity", "abpc", "sensitivity", "complexity")


def snapshot(model_name: str, sample_prefix: str, manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        raise SystemExit(f"no manifest at {manifest_path} — run precompute_samples first")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    samples = {}
    for file_hash, per_model in manifest.get("text", {}).items():
        entry = per_model.get(model_name)
        if not entry:
            continue
        sample_name = entry.get("sample_name", "")
        if sample_prefix and not sample_name.startswith(sample_prefix):
            continue
        results = {}
        for r in entry.get("results", []):
            if r.get("status") != "completed":
                continue
            results[r["explainer_name"]] = {k: r.get(k) for k in _METRIC_KEYS}
        samples[sample_name] = {"file_hash": file_hash, "explainers": results}

    return {
        "model_name": model_name,
        "sample_prefix": sample_prefix,
        "note": ("Attribution metrics for the outgoing text model, measured on the same "
                 "sample texts the incoming model is measured on. Used by "
                 "eval_toxigen.py --attribution-gate."),
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sample-prefix", default=DEFAULT_SAMPLE_PREFIX)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    args = ap.parse_args()

    snap = snapshot(args.model, args.sample_prefix, args.manifest)
    samples = snap["samples"]
    if not samples:
        raise SystemExit(
            f"no completed '{args.model}' entries for samples starting with "
            f"'{args.sample_prefix}' in {args.manifest}. Precompute the old model over "
            f"the new sample files before snapshotting."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    n_exp = sum(len(s["explainers"]) for s in samples.values())
    print(f"[snapshot] {args.model}: {len(samples)} sample(s), {n_exp} explainer result(s)")
    for name, s in sorted(samples.items()):
        exps = s["explainers"]
        abpc = [v["abpc"] for v in exps.values() if v["abpc"] is not None]
        mu = [v["mu_fidelity"] for v in exps.values() if v["mu_fidelity"] is not None]
        print(f"  {name}: {len(exps)} explainer(s), "
              f"abpc {len(abpc)} non-null, mu_fidelity {len(mu)} non-null")
    print(f"[snapshot] wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
