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
- DistilBERT (SST-2)

### Time-Series Classification
- MOMENT (Large / Small)
- InceptionTime

---

## XAI Explainers

The available explainers are automatically determined for each model using **PnPXAI's architecture detection**, ensuring that only compatible explainers are presented.

| Model | Supported Explainers |
|-------|----------------------|
| ResNet-50, VGG-16 | Gradient, Gradient×Input, SmoothGrad, VarGrad, Integrated Gradients, GradCAM, Guided GradCAM, LRP, RAP, LIME, KernelSHAP |
| DenseNet-121 | Gradient family, CAM methods, LIME, KernelSHAP |
| DistilBERT | Gradient family, LRP, LIME, KernelSHAP |
| InceptionTime | Gradient family |
| MOMENT | Gradient family + LRP |

Some explainers are filtered automatically because certain model architectures are incompatible with them (e.g., DenseNet with Zennit LRP, CAM on 1D CNNs, perturbation methods on time-series).

---

## Evaluation Metrics

- MuFidelity
- AbPC
- Sensitivity
- Complexity

Results can be ranked using any metric or their average score.

---

## Visualization

### Image
- Attribution heatmaps
- Side-by-side comparison

### Text
- Token-level attribution
- Importance bar charts

### Time-Series
- Signal overlay
- Background attribution map
- Multi-variable visualization
- Sliding-window attribution for long sequences
- Excel & ZIP export

---

## Sample Datasets

| Dataset | Task |
|----------|------|
| ImageNet Samples | Image Classification |
| SST-2 | Text Classification |
| Boiler Fault Detection | Time-Series |
| ECG5000 | Time-Series |

---

# Installation

## Requirements

- Python 3.10+
- Node.js 20+
- Docker (recommended)
- CUDA-enabled GPU (optional but recommended)

---

## 1. Clone Repository

```bash
git clone https://github.com/<your-repository>.git
cd <repository>
```

---

## 2. Backend

Install Python dependencies.

```bash
pip install -r backend/requirements.txt
```

Download pretrained models.

```bash
python -m backend.scripts.download_models
```

Run the backend.

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Backend:

```
http://localhost:8000
```

---

## 3. Frontend

```bash
cd frontend
npm install
```

Create `.env.local`

```text
BACKEND_URL=http://localhost:8000
```

Run

```bash
npm run dev
```

Frontend:

```
http://localhost:3000
```

---

# Docker (Recommended)

Build

```bash
docker build -t pnpxai-demo .
```

Run

```bash
docker run \
    --gpus all \
    -it \
    -p 8000:8000 \
    -v $(pwd)/models:/project/models \
    pnpxai-demo
```

---

# GPU Support

The backend automatically selects the compute device.

Priority:

```
PNPXAI_DEVICE
        ↓
CUDA
        ↓
CPU
```

Examples

```bash
PNPXAI_DEVICE=cpu
```

```bash
PNPXAI_DEVICE=cuda:1
```

Models are automatically moved to GPU during inference and offloaded afterward to reduce VRAM usage.

---

# Model Storage

Downloaded models are stored in

```
models/
```

To download all models

```bash
python -m backend.scripts.download_models
```

The directory is automatically ignored by Git.

---

# Architecture

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

- GPU is recommended for faster attribution.
- LIME and KernelSHAP are perturbation-based methods and may require additional computation time.
- Sliding-window attribution is automatically applied to long time-series.
- LRP explainers operate on deep-copied models to avoid state corruption.
- Model weights are stored locally and are not included in the repository.

---

# License

MIT License

---

# Citation

If you use this project in your research, please cite

```bibtex
@misc{pnpxai,
  title={PnPXAI},
  author={OpenXAIProject},
  year={2026},
  url={https://github.com/OpenXAIProject/pnpxai}
}
```
