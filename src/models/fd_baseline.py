"""Fundamental Diagram baseline: Greenshields model inversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseEstimator
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class FDBaselineEstimator(BaseEstimator):
    """Estimate density from speed using Greenshields fundamental diagram.

    Greenshields: v = v_f * (1 - k/k_j)
    => k = k_j * (1 - v/v_f)
    => q = k * v

    Parameters:
        v_free: free-flow speed (m/s)
        k_jam: jam density (veh/km)
    """

    def __init__(self, v_free: float = 33.33, k_jam: float = 150.0, **kwargs: Any):
        self.v_free = v_free
        self.k_jam = k_jam

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.get("calibrate", False) and len(X_train) > 0:
            speeds = X_train[:, 0]
            densities = y_train
            self.v_free = float(np.percentile(speeds, 95))
            valid = speeds > 0
            if valid.sum() > 2:
                coeffs = np.polyfit(speeds[valid], densities[valid], 1)
                intercept = coeffs[1]
                if intercept > 0:
                    self.k_jam = float(intercept)
        return {"v_free": self.v_free, "k_jam": self.k_jam}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict density from speed (first column of X)."""
        speeds = X[:, 0] if X.ndim > 1 else X
        return self.k_jam * (1 - np.clip(speeds / self.v_free, 0, 1))

    def predict_flow(self, X: np.ndarray) -> np.ndarray:
        """Predict flow: q = k * v."""
        speeds = X[:, 0] if X.ndim > 1 else X
        density = self.predict(X)
        return density * speeds

    def save(self, path: str | Path) -> Path:
        params = {"v_free": self.v_free, "k_jam": self.k_jam}
        return save_checkpoint(params, Path(path).with_suffix(".pkl"))

    @classmethod
    def load(cls, path: str | Path) -> FDBaselineEstimator:
        params = load_checkpoint(path)
        return cls(**params)

    def get_params(self) -> dict[str, Any]:
        return {"v_free": self.v_free, "k_jam": self.k_jam}
