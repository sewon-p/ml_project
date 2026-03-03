"""Abstract base class for all estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseEstimator(ABC):
    """Common interface for all models (tree-based and deep learning)."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the model. Returns a dict of training info/metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Save model to disk. Returns resolved path."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseEstimator:
        """Load model from disk."""

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters. Override in subclasses."""
        return {}
