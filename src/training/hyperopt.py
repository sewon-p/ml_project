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


def optimize_with_window(
    datasets: dict[int, pd.DataFrame],
    window_sizes: list[int],
    feature_columns_map: dict[int, list[str]],
    target_column: str = "density",
    group_column: str = "scenario_id",
    n_splits: int = 3,
    n_trials: int = 100,
    direction: str = "minimize",
    metric_fn: Callable = rmse,
    storage: str | None = None,
    search_space: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Search window_size + XGBoost hyperparameters jointly with Optuna.

    Parameters
    ----------
    datasets:
        Mapping from window_size to its feature DataFrame.
    window_sizes:
        Candidate window sizes to search over.
    feature_columns_map:
        Mapping from window_size to its feature column names.
    target_column:
        Target column name in each DataFrame.
    group_column:
        Group column for GroupKFold.
    n_splits:
        Number of CV folds.
    n_trials:
        Number of Optuna trials.
    direction:
        "minimize" (RMSE) or "maximize".
    metric_fn:
        Scoring function(y_true, y_pred) -> float.
    storage:
        Optional SQLite path for Optuna persistence.
    search_space:
        Optional dict of param ranges, e.g. {"n_estimators": [100, 1000]}.

    Returns
    -------
    dict with best_params (including window_size), best_value,
    study object, and per-window best scores.
    """
    sp = search_space or {}

    # Pre-compute GroupKFold splits per window size
    splits_map: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ws in window_sizes:
        splits_map[ws] = grouped_kfold_split(datasets[ws], n_splits, group_column)

    def objective(trial: optuna.Trial) -> float:
        ws = trial.suggest_categorical("window_size", window_sizes)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *sp.get("n_estimators", [100, 1000])),
            "max_depth": trial.suggest_int("max_depth", *sp.get("max_depth", [3, 10])),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                *sp.get("learning_rate", [0.01, 0.3]),
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", *sp.get("subsample", [0.5, 1.0])),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *sp.get("colsample_bytree", [0.5, 1.0])
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *sp.get("min_child_weight", [1, 10])
            ),
        }

        df = datasets[ws]
        feat_cols = feature_columns_map[ws]
        splits = splits_map[ws]
        scores = []
        for train_idx, val_idx in splits:
            X_train = df.iloc[train_idx][feat_cols].values
            y_train = df.iloc[train_idx][target_column].values
            X_val = df.iloc[val_idx][feat_cols].values
            y_val = df.iloc[val_idx][target_column].values

            model = create_model("xgboost", **params)
            model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
            preds = model.predict(X_val)
            scores.append(metric_fn(y_val, preds))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_url = f"sqlite:///{storage}" if storage else None
    study = optuna.create_study(
        direction=direction,
        storage=storage_url,
        study_name="hyperopt_window",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    # Collect per-window best scores
    window_best: dict[int, float] = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        ws = trial.params["window_size"]
        val = trial.value
        if ws not in window_best or val < window_best[ws]:
            window_best[ws] = val

    logger.info(
        "Best trial: value=%.4f params=%s",
        study.best_value,
        study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "window_best": window_best,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
