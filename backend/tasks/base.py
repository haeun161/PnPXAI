from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import numpy as np


class TaskHandler(ABC):
    """Abstract base class for task-specific handlers."""

    task_name: str

    @abstractmethod
    def get_models(self) -> list[dict]:
        """Return list of available models for this task."""
        ...

    @abstractmethod
    def get_explainers(self, model_name: str) -> list[dict]:
        """Return list of available explainers for this task and model."""
        ...

    @abstractmethod
    def load_model(self, model_name: str) -> torch.nn.Module:
        """Load and return a pre-trained model."""
        ...

    @abstractmethod
    def preprocess_input(self, raw_data: Any) -> Any:
        """Preprocess raw input data into model-ready format."""
        ...

    @abstractmethod
    def get_modality(self):
        """Return the PnPXAI Modality instance for this task."""
        ...

    @abstractmethod
    def render_result(self, attribution: np.ndarray, input_data: Any, output_path: str,
                      display_name: str | None = None) -> str:
        """Render visualization and save to output_path. Returns file path.

        `display_name` is the explainer's display name (e.g. "SmoothGrad") -- only
        time-series rendering uses it today, as the in-image chart title.
        """
        ...
