# XAI Demo Platform

An interactive **Explainable AI (XAI)** demo platform supporting **Image Classification**, **Text Classification**, and **Time-Series Classification**.

The platform enables users to upload data, select pre-trained models and XAI explainers, visualize attribution maps, and compare explanation quality using multiple evaluation metrics.

---

## Features

### Supported Tasks

### Image Classification
- ResNet-50
- VGG-16
- DenseNet-121 (ImageNet)

### Text Classification
- BERT (HateXplain) — hate speech / normal / offensive

### Time-Series Classification
- MOMENT (Large / Small)
- InceptionTime

---

## XAI Explainers

The available explainers are automatically determined for each model using **PnPXAI's architecture detection**, ensuring that only compatible explainers are presented.

| Model | Explainers |
|---|---|
| ResNet-50, VGG-16 | 14 — Gradient, Gradient×Input, SmoothGrad, VarGrad, IntegratedGradients, GradCAM, GuidedGradCAM, LRP ×4, RAP, LIME, KernelSHAP |
| DenseNet-121 | 9 — the above minus LRP ×4 and RAP |
| DistilBERT (text) | 8 — gradient family + LRP-Epsilon + LIME + KernelSHAP |
| InceptionTime | 5 — gradient family |
| MOMENT | 6 — gradient family + LRP-Epsilon |

Detection can over-report: it recommends methods whose assumptions don't hold for the
concrete model. Those are filtered out per model/modality, verified by actually running
every sample × model × explainer combination:

- **Concatenating architectures break zennit's LRP** — DenseNet's dense blocks and
  InceptionTime's branch `torch.cat` both hit
  `size of tensor a (32) must match tensor b (96)`.
- **RAP** has no rule for `adaptive_avg_pool2d` (DenseNet) and fails on time-series.
- **CAM** assumes 2D spatial feature maps, so it can't run on 1D-conv models.
- **LIME / KernelSHAP** fail on time-series inputs.

> Note: the time-series models have **no trained classifier** — InceptionTime is randomly
> initialized and MOMENT's classification head is too (only its backbone is pretrained).
> They're there to demonstrate the XAI methods; their class predictions are not meaningful.
> Initialization is seeded so results stay reproducible across restarts.

### Evaluation Metrics
- MuFidelity, AbPC, Sensitivity, Complexity
- Results ranked by selected metric or average score

### Visualization
- **Image**: Attribution heatmaps with side-by-side comparison
- **Text**: Token-level attribution highlighting + bar chart
- **Time-Series**: Cyan→magenta background attribution + signal overlay
  - Multi-variate support (top variables ranked by importance)
  - Sliding window attribution for large data (90K+ timesteps)
  - Time axis labels (Day1 HH:MM format from CSV TIME column)
  - Expanded view (5×3 grid) + ZIP bundle download (Excel + per-variable PNGs)

### Data Handling
- Sample data with dataset info & source links
- Drag-and-drop upload (images up to 10MB, CSV up to 50MB)
- Auto-detection of non-sensor columns (TIME → time axis, labels → prediction panel)
- Boiler fault detection dataset (IEEE DataPort, 20 sensors × 90K timesteps)
- ECG5000 heartbeat classification (UCR Archive)

### Optimizer Page
- Hyperparameter optimization for XAI explainers
- History tracking with result comparison

## Tech Stack

- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS (runs on the host)
- **Backend**: FastAPI + PyTorch + PnPXAI (runs inside a GPU Docker container)
- **Time-Series Models**: MOMENT (AutonLab/MOMENT-1-large, MOMENT-1-small)
- **Metrics**: PnPXAI evaluator (MuFidelity, AbPC, Sensitivity, Complexity)

## Operating Model

The **backend runs inside a GPU Docker container** — all Python dependencies
(torch, transformers, pnpxai, momentfm, …) live in the container image, not on
the host. The **frontend runs on the host** with Node and proxies API calls to
the backend. Model weights are cached in a shared, mounted `models/` volume so
they survive container recreation and are shared across containers.

```
Host                                   Docker container (e.g. haeun_pnp)
────────────────────────────────      ─────────────────────────────────────────
frontend (Next.js) :3000  ──/api──►    backend (FastAPI/uvicorn) :8000
  repo mounted at ./                   repo mounted at /project (live --reload)
  BACKEND_URL → :8001                  /project/models  ← shared model volume
                                       GPU via /dev/nvidia* device mounts
```

Ports: **frontend `:3000`**, **backend `:8001` on host → `:8000` in container**.
(The host port is configurable via `-p`; the frontend backend URL is set with
`BACKEND_URL`, default `http://localhost:8000`.)

## Quick Start

### 1. Backend (Docker container)

This host's Docker has no NVIDIA runtime (`--gpus all` fails), so GPUs are passed
in by bind-mounting the device nodes and driver libraries. Create the container
(adjust name, host port, and mount paths for your setup):

```bash
docker run -d -it --name haeun_pnp --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8001:8000 \
  --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia0 \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v $(pwd):/project \
  -v /data8/haeun/PnPXAI/models:/project/models \
  -v /data8/haeun/PnPXAI/sample_data:/project/sample_data \
  nvcr.io/nvidia/pytorch:23.07-py3
# NOTE: add every /dev/nvidiaN and libnvidia-*.so / libcuda.so* your host exposes.
```

Install Python dependencies inside the container (pinned versions; `momentfm`
must be `--no-deps`; `setuptools` must stay `<81` so `pkg_resources` survives):

```bash
docker exec haeun_pnp pip install \
  torch==2.13.0 torchvision==0.28.0 \
  transformers==5.14.0 huggingface_hub==1.23.0 tokenizers==0.22.2 safetensors==0.8.0 \
  pnpxai==0.1.4 captum==0.9.0 \
  fastapi==0.139.0 uvicorn==0.51.0 python-multipart==0.0.32 matplotlib==3.7.2 openpyxl \
  aeon==1.3.0 soxr==1.1.0 pandas==2.3.3 setuptools==80.10.2
docker exec haeun_pnp pip install momentfm==0.1.4 --no-deps
```

Download the models into the shared `models/` volume (one time; see
[Model Storage](#model-storage)):

```bash
docker exec -w /project haeun_pnp python -m backend.scripts.download_models
```

Run the backend.

```bash
docker exec -d -w /project haeun_pnp \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at http://localhost:8001

### 2. Frontend (host)

```bash
cd frontend
npm install
echo "BACKEND_URL=http://localhost:8001" > .env.local   # point at your backend port
npm run dev
```

App runs at http://localhost:3000

### Everyday use / after a reboot

The container uses `--restart unless-stopped`, so it comes back automatically
after a host reboot — but the uvicorn process (started via `docker exec`) does
not, so restart it:

```bash
docker start haeun_pnp                                   # if stopped
docker exec -d -w /project haeun_pnp \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
docker exec haeun_pnp tail -f /var/log/uvicorn_haeun.log # tail logs (if redirected there)
```

Editing code on the host is reflected live (`--reload`) — no container restart
needed. You only need the container running to **run/test** the backend, not to
edit code.

## Precomputed Sample Results (optional)

XAI results for the built-in sample data can be precomputed so `/explain` serves an
identical request instantly instead of recomputing:

```bash
docker run \
    --gpus all \
    -it \
    -p 8000:8000 \
    -v $(pwd)/models:/project/models \
    pnpxai-demo
```

- Cache lives in `backend/precomputed/` and is **git-ignored** — regenerate after cloning.
- Matched by **file SHA-256 + model name**, so it only hits when the uploaded file is
  byte-identical to a sample. Everything else computes normally.
- **Self-invalidating**: each entry records `CACHE_VERSION` and a fingerprint of the
  model's weights. Swapping in a retrained/re-architected model makes stale entries
  recompute automatically — no manual clearing. Bump `CACHE_VERSION` in
  `backend/core/precompute_cache.py` when pipeline/rendering logic changes (weights are
  unchanged then, so the fingerprint can't catch it).
- Cached hits are revealed one explainer at a time (2s each) so progress still shows;
  set `PNPXAI_PRECOMPUTE_STEP_DELAY=0` to serve instantly.

## Compute Device (GPU)

Models and all inference/attribution tensors run on GPU when one is available.

- Resolution order (`backend/core/device.py`): **`$PNPXAI_DEVICE`** (e.g. `cuda`, `cuda:1`,
  `cpu`) → else `cuda` if available → else `cpu`.
- Models are moved to the device on load; the pipeline moves inputs/targets to the
  *model's own* device, so everything stays consistent.
- Force CPU with `PNPXAI_DEVICE=cpu` (useful for debugging).

### VRAM release when idle

Models are **offloaded back to CPU as soon as the last in-flight job finishes**, so an
idle server holds no model VRAM. A job counter guards this — concurrent jobs never have
their models pulled out from under them.

The model objects stay cached in the task handlers, so the next job only pays a
host→device copy (no re-download or re-instantiation). Measured on VGG-16:

| | torch allocated | model device |
|---|---|---|
| after GPU load | 554.7 MB | `cuda:0` |
| after offload | **0 MB** | `cpu` |

A residual ~160–230 MiB stays in `nvidia-smi` while the server runs — that's PyTorch's
**CUDA context**, which cannot be freed without exiting the process. All model memory is
returned.

Set `PNPXAI_KEEP_ON_GPU=1` to disable offloading (lowest latency, holds VRAM).

Verify it's actually on GPU:
```bash
docker exec <container> nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

> GPU matters a lot here: Integrated Gradients on ResNet-50 measured **11.4s on CPU vs
> 0.067s on GPU (~170×)**. Note the container needs the GPU device/library mounts (see
> Quick Start) — without them `torch.cuda.is_available()` is False and it silently
> falls back to CPU.

## Model Storage

Model weights are cached locally instead of relying on the ephemeral
`~/.cache` inside the container.

- **Location**: `$PNPXAI_MODEL_DIR`, else `<repo>/models` — which resolves to the
  mounted `/project/models` volume in the container (shared across containers,
  survives recreation). Resolved by `backend/core/model_paths.py`.
- **Populate**: `python -m backend.scripts.download_models [--only text image timeseries]`
- **Layout**: `models/text/hatexplain-bert/` (transformers), `models/image/*.pth`
  (torchvision state_dicts), `models/timeseries/moment-{large,small}/` (HF snapshot).
  Simple-CNN and InceptionTime are randomly initialized in code — nothing to download.
- **Loading**: handlers load from this directory with `local_files_only=True` when
  present (offline-capable), and **fall back to the HuggingFace Hub / torchvision**
  when absent — so a fresh deployment works before the download script is run.
- Weights are git-ignored (`models/.gitignore`); re-run the download script on a new
  deployment.

## Architecture

```
User
 │
 ├── Upload Data
 │
 ├── Select Model
 │
 ├── Select Explainers
 │
 ▼
Frontend (Next.js)
 │
 ▼
FastAPI Backend
 │
 ├── Load Model
 ├── Prediction
 ├── Attribution
 ├── Evaluation Metrics
 └── Visualization
 │
 ▼
Frontend Visualization
```

---

# REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/tasks` | Supported tasks |
| GET | `/api/models` | Supported models |
| GET | `/api/explainers` | Supported explainers |
| POST | `/api/explain` | Run explanation |
| GET | `/api/jobs/{id}` | Job status |
| GET | `/api/jobs/{id}/visualizations/{name}` | Visualization |
| POST | `/api/optimizer/optimize` | Hyperparameter optimization |

---

# Project Structure

```
backend/
    api/
    core/
    models/
    tasks/
    scripts/

frontend/
    src/
    public/

models/

sample_data/
```

---

# Notes

- The backend runs in the container only — the host has no torch/transformers, so
  code can be edited on the host but must be **run/tested inside the container**.
- MOMENT models require the `momentfm` package (install with `--no-deps` — its pins
  otherwise conflict with the newer transformers/huggingface_hub used here).
- Keep `setuptools<81` in the container: newer setuptools removes `pkg_resources`,
  which breaks the librosa → transformers audio import chain.
- Files written into mounted volumes by the container are owned by `root`; fix with
  `docker exec <container> chown -R <uid>:<gid> /project/models` if the host user
  needs to manage them.
- LIME and KernelSHAP are slower (~20-30s) due to the perturbation-based approach.
- For large time-series (>512 timesteps), sliding window attribution is used automatically.
- LRP explainers use deep-copied models to prevent state corruption.
