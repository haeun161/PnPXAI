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


class PredictionItem(BaseModel):
    class_name: str
    probability: float


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "partial", "completed", "failed"
    task: str
    model_name: str
    explainer_names: list[str]
    ranking_metric: str = "mu_fidelity"
    predictions: Optional[list[PredictionItem]] = None
    original_data_url: Optional[str] = None
    results: list[ExplainerResult] = []
    error_message: Optional[str] = None
