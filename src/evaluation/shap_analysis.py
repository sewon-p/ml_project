"""SHAP analysis for tree-based model interpretability."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: A tree-based model (XGBoost, LightGBM sklearn API).
        X: Feature matrix.
        feature_names: Optional feature names for the explanation.
        max_samples: Maximum number of samples to explain.

    Returns:
        shap.Explanation object.
    """
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    if feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values


def get_feature_importance_from_shap(
    shap_values: shap.Explanation,
) -> dict[str, float]:
    """Extract mean absolute SHAP values per feature."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    names = shap_values.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
    importance = dict(zip(names, mean_abs.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
