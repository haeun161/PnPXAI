"""Download / export the demo models into the local, shared model directory.

Target layout (root resolved by backend.core.model_paths — $PNPXAI_MODEL_DIR or
<repo_root>/models, which is the mounted /project/models inside the container):

    models/
      text/distilbert-sst2/            (transformers save_pretrained)
      image/resnet50.pth               (torchvision state_dict)
      image/vgg16.pth
      image/densenet121.pth
      timeseries/iTransformer_etth1.pth      (forecaster checkpoint; supplied, not downloaded)
      timeseries/iTransformer_illness.pth

Once populated, the task handlers load from here with local_files_only=True and
never touch the network. Weights are NOT committed to git (see models/.gitignore);
re-run this script on a fresh deployment.

Usage (inside the container, repo mounted at /project):
    python -m backend.scripts.download_models              # all
    python -m backend.scripts.download_models --only text image
"""
import argparse
import sys
from pathlib import Path

# Allow running as a plain file (python backend/scripts/download_models.py) too.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.model_paths import model_root, local_dir, local_file  # noqa: E402


TEXT_MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
}
IMAGE_MODELS = ["resnet50", "vgg16", "densenet121"]


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def download_text():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for name, hf_id in TEXT_MODELS.items():
        dest = local_dir("text", name)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[text] {name} <- {hf_id}")
        AutoTokenizer.from_pretrained(hf_id).save_pretrained(dest)
        AutoModelForSequenceClassification.from_pretrained(hf_id).save_pretrained(dest)
        print(f"       saved -> {dest}  ({_dir_size_mb(dest):.0f} MB)")


def download_image():
    import torch
    import torchvision.models as models
    builders = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    }
    for name in IMAGE_MODELS:
        ctor, weights = builders[name]
        dest = local_file("image", f"{name}.pth")
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[image] {name} (torchvision {weights})")
        model = ctor(weights=weights)
        torch.save(model.state_dict(), dest)
        print(f"        saved -> {dest}  ({_dir_size_mb(dest):.0f} MB)")


# Time-series has nothing to download: the forecaster is a trained checkpoint that is
# placed in models/timeseries/ by hand, not pulled from a hub.
TASKS = {"text": download_text, "image": download_image}


def main():
    ap = argparse.ArgumentParser(description="Download demo models into the local model dir.")
    ap.add_argument("--only", nargs="+", choices=list(TASKS), help="subset of tasks (default: all)")
    args = ap.parse_args()

    print(f"model root: {model_root()}")
    for task in (args.only or list(TASKS)):
        TASKS[task]()
    print("done.")


if __name__ == "__main__":
    main()
