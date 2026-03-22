"""Trainer for tabular (tree-based) models with GroupKFold CV."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.models.base import BaseEstimator
from src.models.factory import create_model
from src.training.cross_validation import grouped_kfold_split

logger = logging.getLogger(__name__)


class TabularTrainer:
    """Train and evaluate tabular models with GroupKFold cross-validation."""

    def __init__(
        self,
        model_type: str,
        model_params: dict[str, Any] | None = None,
        n_splits: int = 5,
        group_column: str = "scenario_id",
        early_stopping_rounds: int = 50,
    ):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.n_splits = n_splits
        self.group_column = group_column
        self.early_stopping_rounds = early_stopping_rounds
        self.fold_models: list[BaseEstimator] = []
        self.fold_metrics: list[dict[str, float]] = []

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "density",
        sample_weight_column: str | None = None,
    ) -> dict[str, Any]:
        """Run GroupKFold CV training.

        Returns dict with per-fold and mean metrics.
        """
        splits = grouped_kfold_split(df, self.n_splits, self.group_column)
        self.fold_models = []
        self.fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info("Fold %d/%d", fold_idx + 1, self.n_splits)
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[feature_columns].values
            y_train = train_df[target_column].values
            X_val = val_df[feature_columns].values
            y_val = val_df[target_column].values

            sw_train = train_df[sample_weight_column].values if sample_weight_column else None

            model = create_model(self.model_type, **self.model_params)
            model.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                early_stopping_rounds=self.early_stopping_rounds,
                sample_weight=sw_train,
            )

            preds = model.predict(X_val)
            metrics = compute_all_metrics(y_val, preds)
            logger.info("Fold %d metrics: %s", fold_idx + 1, metrics)

            self.fold_models.append(model)
            self.fold_metrics.append(metrics)

        mean_metrics = {
            key: float(np.mean([m[key] for m in self.fold_metrics])) for key in self.fold_metrics[0]
        }
        logger.info("Mean CV metrics: %s", mean_metrics)
        return {
            "fold_metrics": self.fold_metrics,
            "mean_metrics": mean_metrics,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction: average across fold models."""
        if not self.fold_models:
            raise RuntimeError("No models trained yet. Call fit() first.")
        preds = np.stack([m.predict(X) for m in self.fold_models])
        return preds.mean(axis=0)

    @property
    def best_model(self) -> BaseEstimator:
        """Return the fold model with lowest RMSE."""
        if not self.fold_models:
            raise RuntimeError("No models trained yet.")
        best_idx = int(np.argmin([m["rmse"] for m in self.fold_metrics]))
        return self.fold_models[best_idx]
