"""Tree-based estimators: XGBoost and LightGBM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseEstimator
from src.utils.checkpoint import load_checkpoint, save_checkpoint

_DL_ONLY_KEYS = {
    "in_channels",
    "seq_len",
    "n_filters",
    "kernel_size",
    "dropout",
    "n_conditions",
    "hidden_size",
    "num_layers",
}


class XGBoostEstimator(BaseEstimator):
    """XGBoost regressor wrapper."""

    def __init__(self, **params: Any):
        import xgboost as xgb

        # Filter out DL-specific params
        for k in _DL_ONLY_KEYS:
            params.pop(k, None)

        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        defaults.update(params)
        self.params = defaults
        self.model = xgb.XGBRegressor(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            early_stopping = kwargs.get("early_stopping_rounds", 50)
            self.model.set_params(early_stopping_rounds=early_stopping)
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, verbose=kwargs.get("verbose", False), **fit_params)
        best_iter = getattr(self.model.get_booster(), "best_iteration", None)
        n_used = best_iter if best_iter is not None else self.params["n_estimators"]
        return {"n_estimators_used": n_used}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(self.model, Path(path).with_suffix(".pkl"))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostEstimator:
        instance = cls.__new__(cls)
        instance.model = load_checkpoint(path)
        instance.params = {}
        return instance

    def get_params(self) -> dict[str, Any]:
        return self.params


class LightGBMEstimator(BaseEstimator):
    """LightGBM regressor wrapper."""

    def __init__(self, **params: Any):
        import lightgbm as lgb

        # Filter out DL-specific and unsupported params
        for k in _DL_ONLY_KEYS | {"device"}:
            params.pop(k, None)

        defaults = {
            "n_estimators": 500,
            "max_depth": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "random_state": 42,
            "verbosity": -1,
        }
        defaults.update(params)
        self.params = defaults
        self.model = lgb.LGBMRegressor(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import lightgbm as lgb

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            early_stopping = kwargs.get("early_stopping_rounds", 50)
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping),
                lgb.log_evaluation(period=0),
            ]
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_params)
        return {"n_estimators_used": self.model.best_iteration_ or self.params["n_estimators"]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(self.model, Path(path).with_suffix(".pkl"))

    @classmethod
    def load(cls, path: str | Path) -> LightGBMEstimator:
        instance = cls.__new__(cls)
        instance.model = load_checkpoint(path)
        instance.params = {}
        return instance

    def get_params(self) -> dict[str, Any]:
        return self.params
