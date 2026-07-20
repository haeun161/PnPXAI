"""Precompute XAI results for every (built-in sample file x built-in model x
compatible explainer) combination, so the live /explain endpoint can serve an
identical request from backend/precomputed/ instantly instead of recomputing.

Run from the project root:
    python -m backend.scripts.precompute_samples [task ...]

With no arguments, precomputes image, text, and timeseries. Pass one or more
task names to limit the run (e.g. `python -m backend.scripts.precompute_samples image`).

Safe to re-run: each combination is skipped if already present in the manifest
unless --force is passed.
"""
import glob
import json
import os
import shutil
import sys
import time

from backend.tasks import get_task_handler
from backend.core.job_manager import (
    create_job, store_uploaded_data, get_job, _remove_job, VISUALIZATION_DIR,
)
from backend.core.pipeline import run_explanation_pipeline
from backend.core.image_utils import load_and_validate_image
from backend.core.precompute_cache import PRECOMPUTED_DIR, MANIFEST_PATH, file_sha256, sample_cache_dir

SAMPLE_ROOT = "sample_data"
TASKS = ["image", "text", "timeseries"]


def _sample_files(task: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(SAMPLE_ROOT, task, "*")) if os.path.isfile(f))


def _load_data(task: str, raw: bytes):
    if task == "image":
        return load_and_validate_image(raw)
    if task == "text":
        return raw.decode("utf-8", errors="replace")
    return raw  # timeseries: raw CSV bytes


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)


def _result_to_cache_entry(r) -> dict:
    return {
        "explainer_name": r.explainer_name,
        "display_name": r.display_name,
        "status": r.status,
        "mu_fidelity": r.mu_fidelity,
        "abpc": r.abpc,
        "sensitivity": r.sensitivity,
        "complexity": r.complexity,
        "error_message": r.error_message,
        "token_attributions": [t.model_dump() for t in r.token_attributions] if r.token_attributions else None,
    }


def precompute(tasks: list[str], force: bool = False) -> None:
    manifest = _load_manifest()
    skipped, done, failed_combos = 0, 0, []

    for task in tasks:
        handler = get_task_handler(task)
        samples = _sample_files(task)
        print(f"\n=== {task}: {len(samples)} sample file(s) ===")

        for sample_path in samples:
            with open(sample_path, "rb") as f:
                raw = f.read()
            file_hash = file_sha256(raw)
            sample_name = os.path.basename(sample_path)

            for model_info in handler.get_models():
                model_name = model_info["name"]
                explainers = [e for e in handler.get_explainers(model_name) if e.get("compatible", True)]
                explainer_names = [e["name"] for e in explainers]
                if not explainer_names:
                    continue

                existing = manifest.get(task, {}).get(file_hash, {}).get(model_name)
                if existing and not force:
                    have = {r["explainer_name"] for r in existing.get("results", [])}
                    if have >= set(explainer_names):
                        skipped += 1
                        continue

                print(f"  [{sample_name} + {model_name}] running {len(explainer_names)} explainer(s)...")
                t0 = time.time()

                data = _load_data(task, raw)
                job_id = f"precompute-{task}-{model_name}-{file_hash[:16]}"
                try:
                    store_uploaded_data(job_id, data, task)
                    create_job(job_id, task, model_name, explainer_names, "average")
                    run_explanation_pipeline(job_id, task, model_name, explainer_names, "average", {})

                    job = get_job(job_id)
                    if job is None:
                        raise RuntimeError("job vanished after run_explanation_pipeline")

                    dest_dir = sample_cache_dir(task, model_name, file_hash)
                    os.makedirs(dest_dir, exist_ok=True)
                    src_dir = os.path.join(VISUALIZATION_DIR, job_id)
                    if os.path.isdir(src_dir):
                        for fname in os.listdir(src_dir):
                            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

                    manifest.setdefault(task, {}).setdefault(file_hash, {})[model_name] = {
                        "sample_name": sample_name,
                        "predictions": [p.model_dump() for p in job.predictions] if job.predictions else None,
                        "results": [_result_to_cache_entry(r) for r in job.results],
                    }
                    _save_manifest(manifest)
                    done += 1
                    n_failed = sum(1 for r in job.results if r.status != "completed")
                    print(f"    done in {time.time()-t0:.1f}s (status={job.status}, {n_failed} failed)")
                    if n_failed:
                        for r in job.results:
                            if r.status != "completed":
                                print(f"      FAILED {r.explainer_name}: {r.error_message}")
                                failed_combos.append((task, sample_name, model_name, r.explainer_name, r.error_message))
                except Exception as e:
                    print(f"    ERROR: {type(e).__name__}: {e}")
                    failed_combos.append((task, sample_name, model_name, "ALL", str(e)))
                finally:
                    _remove_job(job_id)

    print(f"\n=== Done: {done} combo(s) computed, {skipped} already cached ===")
    if failed_combos:
        print(f"{len(failed_combos)} individual failure(s):")
        for task, sample_name, model_name, exp_name, err in failed_combos:
            print(f"  {task}/{sample_name}/{model_name}/{exp_name}: {err}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    precompute(args or TASKS, force=force)
