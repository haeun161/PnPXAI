# XAI Demo Platform

Interactive eXplainable AI demo platform supporting **Image Classification**, **Text Classification**, and **Time-Series Classification**. Upload data, select a pre-trained model and XAI explainers, and visualize attribution explanations with quality metrics.

## Features

### Multi-Task Support
- **Image Classification**: ResNet-50, VGG-16, DenseNet-121 (ImageNet)
- **Text Classification**: DistilBERT (SST-2 sentiment analysis)
- **Time-Series Classification**: MOMENT foundation model (large/small), InceptionTime, Simple 1D CNN

### XAI Explainers
- Integrated Gradients, SmoothGrad, Gradient × Input, Gradient
- LIME, KernelSHAP (perturbation-based)
- GradCAM, Guided GradCAM (image only)
- LRP variants (LRP-Epsilon, LRP-Alpha2Beta1, etc.)

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

- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Backend**: FastAPI + PyTorch + PnPXAI
- **Time-Series Models**: MOMENT (AutonLab/MOMENT-1-large, MOMENT-1-small)
- **Metrics**: PnPXAI evaluator (MuFidelity, AbPC, Sensitivity, Complexity)

## Quick Start

### Backend

```bash
pip install -r backend/requirements.txt
pip install momentfm --no-deps  # MOMENT time-series foundation model
uvicorn backend.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App runs at http://localhost:3000

## Architecture

```
User selects task (image / text / timeseries)
  → Selects model + XAI explainers
  → Uploads data or picks sample
  → Frontend sends to FastAPI backend
  → Backend loads pre-trained model
  → Runs inference (predictions)
  → For each selected XAI explainer:
     → Computes attribution (sliding window for large TS)
     → Computes evaluation metrics
     → Renders visualization
  → Results returned progressively via polling
  → Frontend displays ranked results + metrics
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/models/{task} | List models for task |
| GET | /api/explainers/{task}/{model} | List explainers |
| POST | /api/explain | Submit analysis job |
| GET | /api/jobs/{id} | Get job status + results |
| GET | /api/jobs/{id}/visualizations/{name}.png | Get visualization |
| GET | /api/samples/{task} | List sample data files |
| GET | /api/samples/{task}/{filename} | Serve sample file |
| POST | /api/optimizer/run | Run optimizer |
| GET | /api/optimizer/history | Get optimization history |

## Sample Data

| Dataset | Task | Channels | Timesteps | Source |
|---------|------|----------|-----------|--------|
| Boiler Fault Detection | timeseries | 20 | 90,120 | [IEEE DataPort](https://ieee-dataport.org/open-access/simulated-boiler-data-fault-detection-and-classification) |
| ECG5000 | timeseries | 1 | 140 | [UCR Archive](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) |

## Notes

- MOMENT models require `momentfm` package (install with `--no-deps` to avoid numpy conflicts)
- LIME and KernelSHAP are slower (~20-30s) due to perturbation-based approach
- For large time-series (>512 timesteps), sliding window attribution is used automatically
- LRP explainers use deep-copied models to prevent state corruption
