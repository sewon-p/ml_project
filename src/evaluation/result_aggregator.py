"""Collect and aggregate experiment results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics


class ResultAggregator:
    """Collect and aggregate experiment results into a DataFrame."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def add_result(
        self,
        experiment: str,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str = "density",
        **extra_fields: object,
    ) -> None:
        """Add a single experiment result."""
        metrics = compute_all_metrics(y_true, y_pred)
        record = {
            "experiment": experiment,
            "model": model_name,
            "target": target,
            **metrics,
            **extra_fields,
        }
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all results as a DataFrame."""
        return pd.DataFrame(self._records)

    def summary(self) -> pd.DataFrame:
        """Return mean metrics grouped by experiment and model."""
        df = self.to_dataframe()
        if df.empty:
            return df
        metric_cols = ["rmse", "mae", "mape", "r2"]
        return df.groupby(["experiment", "model"])[metric_cols].mean().reset_index()
