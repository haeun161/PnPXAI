from pydantic import BaseModel
from typing import Optional


class TaskInfo(BaseModel):
    name: str
    display_name: str
    description: str


class ModelInfo(BaseModel):
    name: str
    display_name: str
    architecture: str
    description: str
    task: str


class ExplainerInfo(BaseModel):
    name: str
    display_name: str
    estimated_compute_time_seconds: float = 5.0
    compatible: bool = True
    incompatibility_reason: Optional[str] = None


class TokenAttribution(BaseModel):
    token: str
    score: float


class ExplainerResult(BaseModel):
    explainer_name: str
    display_name: str
    status: str  # "completed", "running", "pending", "not_supported", "failed"
    rank: Optional[int] = None
    visualization_url: Optional[str] = None
    mu_fidelity: Optional[float] = None
    abpc: Optional[float] = None
    sensitivity: Optional[float] = None
    complexity: Optional[float] = None
    not_supported_reason: Optional[str] = None
    error_message: Optional[str] = None
    token_attributions: Optional[list[TokenAttribution]] = None
    current_step: Optional[str] = None


class PredictionItem(BaseModel):
    class_name: str
    probability: float


class ForecastInfo(BaseModel):
    """Populated instead of `predictions` for forecasting time-series models, whose
    output is a future trajectory rather than class probabilities.

    Holds a backtest: the last `pred_len` points of the uploaded series are held out,
    the `context` window immediately before them is fed to the model, and `predicted`
    is what it produced. `actual` is the held-out truth, so the UI can draw predicted
    against observed — it is None only when the upload is too short to hold anything
    out, in which case the horizon runs past the end of the data.

    Series are (timestep, channel) so every channel can be plotted, not just the first.
    """
    col_names: list[str]
    context: list[list[float]]
    predicted: list[list[float]]
    actual: Optional[list[list[float]]] = None
    context_labels: Optional[list[str]] = None
    horizon_labels: Optional[list[str]] = None
    # Channel the explainers attributed, so the chart can flag when the viewer is
    # looking at a different one than the attributions describe.
    attributed_channel: int = 0


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "partial", "completed", "failed"
    task: str
    model_name: str
    explainer_names: list[str]
    ranking_metric: str = "mu_fidelity"
    predictions: Optional[list[PredictionItem]] = None
    forecast: Optional[ForecastInfo] = None
    original_data_url: Optional[str] = None
    results: list[ExplainerResult] = []
    error_message: Optional[str] = None
