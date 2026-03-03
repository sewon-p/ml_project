"""Hyperparameter optimization with Optuna."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

from src.evaluation.metrics import rmse
from src.models.factory import create_model
from src.training.cross_validation import grouped_kfold_split

logger = logging.getLogger(__name__)


def _default_xgboost_search_space(
    trial: optuna.Trial,
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _default_lightgbm_search_space(
    trial: optuna.Trial,
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }


_DEFAULT_SPACES: dict[str, Callable] = {
    "xgboost": _default_xgboost_search_space,
    "lightgbm": _default_lightgbm_search_space,
}


def optimize_hyperparams(
    model_type: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "density",
    group_column: str = "scenario_id",
    n_splits: int = 3,
    n_trials: int = 50,
    search_space_fn: (Callable[[optuna.Trial], dict[str, Any]] | None) = None,
    direction: str = "minimize",
    metric_fn: Callable = rmse,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search with GroupKFold CV.

    Returns dict with best_params and best_value.
    """
    if search_space_fn is None:
        search_space_fn = _DEFAULT_SPACES.get(model_type)
        if search_space_fn is None:
            raise ValueError(
                f"No default search space for '{model_type}'. Provide search_space_fn."
            )

    splits = grouped_kfold_split(df, n_splits, group_column)

    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        scores = []
        for train_idx, val_idx in splits:
            X_train = df.iloc[train_idx][feature_columns].values
            y_train = df.iloc[train_idx][target_column].values
            X_val = df.iloc[val_idx][feature_columns].values
            y_val = df.iloc[val_idx][target_column].values

            model = create_model(model_type, **params)
            model.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                early_stopping_rounds=50,
            )
            preds = model.predict(X_val)
            scores.append(metric_fn(y_val, preds))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Best trial: value=%.4f params=%s",
        study.best_value,
        study.best_params,
    )
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
