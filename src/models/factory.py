"""Model factory: create model instances from config."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseEstimator

_MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {}


def _register_models() -> None:
    """Lazy-register all built-in models."""
    if _MODEL_REGISTRY:
        return
    from src.models.cnn1d import CNN1DEstimator
    from src.models.fd_baseline import FDBaselineEstimator
    from src.models.lstm import LSTMEstimator
    from src.models.tabular import LightGBMEstimator, XGBoostEstimator

    _MODEL_REGISTRY.update(
        {
            "xgboost": XGBoostEstimator,
            "lightgbm": LightGBMEstimator,
            "cnn1d": CNN1DEstimator,
            "lstm": LSTMEstimator,
            "fd_baseline": FDBaselineEstimator,
        }
    )


def create_model(model_type: str, **params: Any) -> BaseEstimator:
    """Create a model instance by type name.

    Args:
        model_type: One of "xgboost", "lightgbm", "cnn1d", "lstm", "fd_baseline".
        **params: Model-specific hyperparameters.
    """
    _register_models()
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    return _MODEL_REGISTRY[model_type](**params)
