"""Evaluate trained model and generate plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.io import load_data_filter_mask, read_parquet
from src.data.preprocessing import grouped_train_test_split
from src.evaluation.metrics import compute_all_metrics
from src.models.cnn1d import CNN1DEstimator
from src.models.fd_baseline import FDBaselineEstimator
from src.models.lstm import LSTMEstimator
from src.models.tabular import (
    LightGBMEstimator,
    XGBoostEstimator,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.visualization.prediction_plots import (
    plot_predicted_vs_actual,
    plot_residuals,
)

logger = logging.getLogger(__name__)

DL_MODELS = {"cnn1d", "lstm"}
WINDOW_DL_MODELS = {"window_cnn1d", "window_lstm"}
WINDOW_TABULAR_MODELS = {"window_xgboost"}
WINDOW_MODELS = WINDOW_DL_MODELS | WINDOW_TABULAR_MODELS
SHAP_MODELS = {"xgboost", "lightgbm", "window_xgboost"}

_LOADERS = {
    "xgboost": XGBoostEstimator,
    "lightgbm": LightGBMEstimator,
    "fd_baseline": FDBaselineEstimator,
    "cnn1d": CNN1DEstimator,
    "lstm": LSTMEstimator,
    "window_cnn1d": CNN1DEstimator,
    "window_lstm": LSTMEstimator,
    "window_xgboost": XGBoostEstimator,
}


def _evaluate_dl(
    cfg: dict,
    model_type: str,
    model_path: str,
    target: str,
) -> None:
    """Evaluate a DL model on test split of .npz data."""
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    ts_path = data_cfg.get("timeseries_path", "data/features/timeseries.npz")
    data = np.load(ts_path)
    sequences = data["sequences"]
    scenario_ids = data["scenario_ids"]

    # Load conditions (needed early for per-lane fallback)
    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)

    # Per-lane target: density → density_per_lane, flow → flow_per_lane
    per_lane_target = f"{target}_per_lane"
    if per_lane_target in data.files:
        actual_targets = data[per_lane_target]
    else:
        logger.info("Fallback: computing %s from %s / num_lanes", per_lane_target, target)
        actual_targets = data[target].astype(np.float32) / meta_df["num_lanes"].values

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    if residual_enabled:
        delta_key = "delta_density" if target == "density" else "delta_flow"
        fd_key = "k_fd" if target == "density" else "q_fd"
        num_lanes = meta_df["num_lanes"].values
        if per_lane_target in data.files:
            train_targets = data[delta_key]
            fd_estimates = data[fd_key]
        else:
            # Old data: k_fd/delta are absolute → convert to per-lane
            train_targets = data[delta_key].astype(np.float32) / num_lanes
            fd_estimates = data[fd_key].astype(np.float32) / num_lanes
        logger.info("Residual mode: delta=%s, fd=%s", delta_key, fd_key)
    else:
        train_targets = actual_targets

    # Reproduce test split
    rng = np.random.RandomState(cfg.get("seed", 42))
    unique_ids = np.unique(scenario_ids)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_test = max(1, int(n * train_cfg.get("test_ratio", 0.2)))
    test_ids = set(unique_ids[:n_test])
    test_idx = np.where(np.isin(scenario_ids, list(test_ids)))[0]

    # Load model
    model_params = {}
    model_config_path = cfg.get("model", {}).get("config")
    if model_config_path:
        model_params = load_config(model_config_path)

    # Apply config-driven data filters
    mask = load_data_filter_mask(cfg, meta_df)
    if not mask.all():
        sequences = sequences[mask]
        actual_targets = actual_targets[mask]
        scenario_ids = scenario_ids[mask]
        if residual_enabled:
            train_targets = train_targets[mask]
            fd_estimates = fd_estimates[mask]
        meta_df = meta_df[mask].reset_index(drop=True)

    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)
    n_conditions = conditions.shape[1]

    model_params["n_conditions"] = n_conditions
    loader_cls = _LOADERS[model_type]
    model = loader_cls.load(model_path, **model_params)

    device = train_cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    test_ds = TimeSeriesDataset(sequences[test_idx], train_targets[test_idx], conditions[test_idx])
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.get("batch_size", 128),
        shuffle=False, num_workers=0,
    )

    model.model.to(torch.device(device))
    model.model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            cond = batch[2].to(device) if len(batch) == 3 else None
            preds = model.model(X_batch, cond)
            all_preds.append(preds.cpu().numpy())

    y_pred = np.concatenate(all_preds)

    # Restore predictions if residual mode
    if residual_enabled:
        y_pred = fd_estimates[test_idx] + y_pred
    y_test = actual_targets[test_idx]

    metrics = compute_all_metrics(y_test, y_pred)
    logger.info("Test metrics: %s", metrics)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    plots_dir = output_dir / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_residuals.png",
    )
    logger.info("Plots saved to %s", plots_dir)

    # Save metrics + predictions JSON for dashboard
    _save_eval_results(output_dir, model_type, target, metrics, y_test, y_pred)


def _evaluate_window(
    cfg: dict,
    model_type: str,
    model_path: str,
    target: str,
) -> None:
    """Evaluate a window-feature DL model on test split."""
    from src.features.window_features import extract_window_features

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    ts_path = data_cfg.get("timeseries_path", "data/features/timeseries.npz")
    data = np.load(ts_path)
    sequences = data["sequences"]
    scenario_ids = data["scenario_ids"]

    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)
    speed_limits = meta_df["speed_limit"].values

    per_lane_target = f"{target}_per_lane"
    if per_lane_target in data.files:
        actual_targets = data[per_lane_target]
    else:
        actual_targets = data[target].astype(np.float32) / meta_df["num_lanes"].values

    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    if residual_enabled:
        delta_key = "delta_density" if target == "density" else "delta_flow"
        fd_key = "k_fd" if target == "density" else "q_fd"
        num_lanes = meta_df["num_lanes"].values
        if per_lane_target in data.files:
            train_targets = data[delta_key]
            fd_estimates = data[fd_key]
        else:
            train_targets = data[delta_key].astype(np.float32) / num_lanes
            fd_estimates = data[fd_key].astype(np.float32) / num_lanes
    else:
        train_targets = actual_targets
        fd_estimates = None

    # Apply data filters
    mask = load_data_filter_mask(cfg, meta_df)
    if not mask.all():
        sequences = sequences[mask]
        actual_targets = actual_targets[mask]
        scenario_ids = scenario_ids[mask]
        speed_limits = speed_limits[mask]
        train_targets = train_targets[mask]
        if fd_estimates is not None:
            fd_estimates = fd_estimates[mask]
        meta_df = meta_df[mask].reset_index(drop=True)

    # Reproduce test split
    rng = np.random.RandomState(cfg.get("seed", 42))
    unique_ids = np.unique(scenario_ids)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_test = max(1, int(n * train_cfg.get("test_ratio", 0.2)))
    n_val = max(1, int(n * train_cfg.get("val_ratio", 0.1)))
    test_ids = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test + n_val:])
    test_idx = np.where(np.isin(scenario_ids, list(test_ids)))[0]
    train_idx = np.where(np.isin(scenario_ids, list(train_ids)))[0]

    # Extract window features
    window_cfg = cfg.get("window_features", {})
    window_size = window_cfg.get("window_size", 30)
    exclude_wf = window_cfg.get("exclude", [])
    win_features, used_names = extract_window_features(
        sequences, speed_limits, window_size, exclude=exclude_wf,
    )
    n_win_features = win_features.shape[1]

    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)

    # Normalize using train statistics
    N_tr, C, W = win_features[train_idx].shape
    flat_tr = win_features[train_idx].transpose(1, 0, 2).reshape(C, -1)
    mean_f = flat_tr.mean(axis=1, keepdims=True)
    std_f = flat_tr.std(axis=1, keepdims=True) + 1e-8

    def norm_seq(arr: np.ndarray) -> np.ndarray:
        return ((arr.transpose(1, 0, 2).reshape(C, -1) - mean_f) / std_f
                ).reshape(C, arr.shape[0], W).transpose(1, 0, 2).astype(np.float32)

    wf_test = norm_seq(win_features[test_idx])

    cond_mean = conditions[train_idx].mean(axis=0, keepdims=True)
    cond_std = conditions[train_idx].std(axis=0, keepdims=True) + 1e-8
    cond_test = ((conditions[test_idx] - cond_mean) / cond_std).astype(np.float32)

    # Load model
    model_params = {}
    model_config_path = cfg.get("model", {}).get("config")
    if model_config_path:
        model_params = load_config(model_config_path)
    model_params["n_conditions"] = conditions.shape[1]

    base_type = model_type.replace("window_", "")

    # Override input dimensions for window features
    if base_type == "cnn1d":
        model_params["in_channels"] = n_win_features
        model_params["seq_len"] = win_features.shape[2]
    elif base_type == "lstm":
        model_params["input_size"] = n_win_features

    loader_cls = _LOADERS[model_type]
    model = loader_cls.load(model_path, **model_params)

    device = train_cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    test_ds = TimeSeriesDataset(wf_test, train_targets[test_idx], cond_test)
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.get("batch_size", 128),
        shuffle=False, num_workers=0,
    )

    model.model.to(torch.device(device))
    model.model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            cond = batch[2].to(device) if len(batch) == 3 else None
            preds = model.model(X_batch, cond)
            all_preds.append(preds.cpu().numpy())

    y_pred = np.concatenate(all_preds)

    if residual_enabled and fd_estimates is not None:
        y_pred = fd_estimates[test_idx] + y_pred
    y_test = actual_targets[test_idx]

    metrics = compute_all_metrics(y_test, y_pred)
    logger.info("Test metrics (%s): %s", model_type, metrics)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    plots_dir = output_dir / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_residuals.png",
    )
    _save_eval_results(output_dir, model_type, target, metrics, y_test, y_pred)


def _evaluate_window_tabular(
    cfg: dict,
    model_type: str,
    model_path: str,
    target: str,
) -> None:
    """Evaluate a window-feature tabular model (e.g. window_xgboost)."""
    from src.features.window_features import extract_window_features

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    ts_path = data_cfg.get("timeseries_path", "data/features/timeseries.npz")
    data = np.load(ts_path)
    sequences = data["sequences"]
    scenario_ids = data["scenario_ids"]

    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)
    speed_limits = meta_df["speed_limit"].values

    per_lane_target = f"{target}_per_lane"
    if per_lane_target in data.files:
        actual_targets = data[per_lane_target]
    else:
        actual_targets = data[target].astype(np.float32) / meta_df["num_lanes"].values

    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    fd_estimates = None
    if residual_enabled:
        delta_key = "delta_density" if target == "density" else "delta_flow"
        fd_key = "k_fd" if target == "density" else "q_fd"
        num_lanes = meta_df["num_lanes"].values
        if per_lane_target in data.files:
            train_targets = data[delta_key]
            fd_estimates = data[fd_key]
        else:
            train_targets = data[delta_key].astype(np.float32) / num_lanes
            fd_estimates = data[fd_key].astype(np.float32) / num_lanes
    else:
        train_targets = actual_targets

    mask = load_data_filter_mask(cfg, meta_df)
    if not mask.all():
        sequences = sequences[mask]
        actual_targets = actual_targets[mask]
        scenario_ids = scenario_ids[mask]
        speed_limits = speed_limits[mask]
        train_targets = train_targets[mask]
        if fd_estimates is not None:
            fd_estimates = fd_estimates[mask]
        meta_df = meta_df[mask].reset_index(drop=True)

    # Reproduce test split
    rng = np.random.RandomState(cfg.get("seed", 42))
    unique_ids = np.unique(scenario_ids)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_test = max(1, int(n * train_cfg.get("test_ratio", 0.2)))
    test_ids = set(unique_ids[:n_test])
    test_idx = np.where(np.isin(scenario_ids, list(test_ids)))[0]

    # Extract and flatten window features
    window_cfg = cfg.get("window_features", {})
    window_size = window_cfg.get("window_size", 30)
    exclude_wf = window_cfg.get("exclude", [])
    win_features, used_names = extract_window_features(
        sequences, speed_limits, window_size, exclude=exclude_wf,
    )
    _N, C, W = win_features.shape
    columns = [f"w{w}_{name}" for w in range(W) for name in used_names]
    flat = win_features.transpose(0, 2, 1).reshape(_N, -1)

    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)
    feature_columns = columns + ["num_lanes", "speed_limit"]

    # Build test feature matrix
    X_test = np.column_stack([flat[test_idx], conditions[test_idx]])

    # Load model
    loader_cls = _LOADERS[model_type]
    model = loader_cls.load(model_path)

    y_pred = model.predict(X_test)

    if residual_enabled and fd_estimates is not None:
        y_pred = fd_estimates[test_idx] + y_pred
    y_test = actual_targets[test_idx]

    metrics = compute_all_metrics(y_test, y_pred)
    logger.info("Test metrics (%s): %s", model_type, metrics)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    plots_dir = output_dir / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_residuals.png",
    )

    # SHAP
    feature_importance = _compute_shap_importance(
        cfg, model, X_test, feature_columns, model_type, plots_dir,
    )

    _save_eval_results(
        output_dir, model_type, target, metrics, y_test, y_pred,
        feature_importance=feature_importance,
    )


def _evaluate_tabular(
    cfg: dict,
    model_type: str,
    model_path: str,
    target: str,
    data_path: str,
) -> None:
    """Evaluate a tabular model on test split."""
    train_cfg = cfg.get("training", {})

    df = read_parquet(data_path)

    # Apply config-driven data filters
    mask = load_data_filter_mask(cfg, df)
    if not mask.all():
        df = df[mask].reset_index(drop=True)

    exclude = {
        "scenario_id", "probe_idx",
        "density", "flow", "demand_vehph",
        "density_per_lane", "flow_per_lane",
        "k_fd", "q_fd", "delta_density", "delta_flow",
    }
    # Apply user-selected feature exclusions from dashboard
    user_exclude = train_cfg.get("exclude_features") or []
    if user_exclude:
        exclude.update(user_exclude)
        logger.info("Excluding %d user-selected features: %s", len(user_exclude), user_exclude)
    feature_columns = [c for c in df.columns if c not in exclude]

    # Per-lane target: density → density_per_lane, flow → flow_per_lane
    per_lane_target = f"{target}_per_lane"
    old_data = per_lane_target not in df.columns
    if old_data:
        logger.info("Fallback: computing %s from %s / num_lanes", per_lane_target, target)
        df[per_lane_target] = df[target] / df["num_lanes"]

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)

    # Ensure residual columns are per-lane scale for old data
    if residual_enabled and old_data:
        delta_col = "delta_density" if target == "density" else "delta_flow"
        fd_col_name = "k_fd" if target == "density" else "q_fd"
        df[delta_col] = df[delta_col] / df["num_lanes"]
        df[fd_col_name] = df[fd_col_name] / df["num_lanes"]
        logger.info("Converted %s, %s to per-lane scale", delta_col, fd_col_name)

    _, test_df = grouped_train_test_split(
        df,
        group_column=train_cfg.get("group_column", "scenario_id"),
        test_ratio=train_cfg.get("test_ratio", 0.2),
    )

    loader_cls = _LOADERS.get(model_type)
    if loader_cls is None:
        logger.error("Unsupported model type: %s", model_type)
        return
    model = loader_cls.load(model_path)

    X_test = test_df[feature_columns]
    y_pred = model.predict(X_test)

    if residual_enabled:
        fd_col = "k_fd" if target == "density" else "q_fd"
        y_pred = test_df[fd_col].values + y_pred
    y_test = test_df[per_lane_target].values

    metrics = compute_all_metrics(y_test, y_pred)
    logger.info("Test metrics: %s", metrics)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    plots_dir = output_dir / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target}, per-lane)",
        save_path=plots_dir / f"{model_type}_residuals.png",
    )
    logger.info("Plots saved to %s", plots_dir)

    # SHAP feature importance (tree-based models only)
    feature_importance = _compute_shap_importance(
        cfg, model, X_test, feature_columns, model_type, plots_dir,
    )

    # Save metrics + predictions JSON for dashboard
    _save_eval_results(
        output_dir, model_type, target, metrics, y_test, y_pred,
        feature_importance=feature_importance,
    )


def _compute_shap_importance(
    cfg: dict,
    model: object,
    X_test: np.ndarray | object,
    feature_columns: list[str],
    model_type: str,
    plots_dir: Path,
) -> dict[str, float] | None:
    """Compute SHAP feature importance for tree-based models."""
    eval_cfg = cfg.get("evaluation", {}).get("shap", {})
    if not eval_cfg.get("enabled", False):
        return None
    if model_type not in SHAP_MODELS:
        return None

    try:
        import shap
    except ImportError:
        logger.warning("shap not installed, skipping feature importance")
        return None

    max_samples = eval_cfg.get("max_samples", 500)
    X = np.asarray(X_test)
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X = X[idx]

    try:
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X)

        # mean |SHAP| per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_columns, mean_abs.tolist()))
        # Sort descending, top 20
        top = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        # Plot
        plots_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        names = list(top.keys())
        values = list(top.values())
        ax.barh(names[::-1], values[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_type} — Feature Importance (Top 20)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{model_type}_feature_importance.png", dpi=150)
        plt.close(fig)
        logger.info("SHAP feature importance saved (%d features)", len(top))
        return top
    except Exception:
        logger.warning("SHAP computation failed", exc_info=True)
        return None


def _save_eval_results(
    output_dir: Path, model_type: str, target: str,
    metrics: dict, y_test: np.ndarray, y_pred: np.ndarray,
    feature_importance: dict[str, float] | None = None,
) -> None:
    """Save evaluation metrics and scatter data as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for scatter plot (max 2000 points)
    n = len(y_test)
    if n > 2000:
        idx = np.random.default_rng(42).choice(n, 2000, replace=False)
        scatter_actual = y_test[idx]
        scatter_pred = y_pred.ravel()[idx]
    else:
        scatter_actual = y_test
        scatter_pred = y_pred.ravel()

    result: dict = {
        "model_type": model_type,
        "target": target,
        "n_test": int(n),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "scatter": {
            "actual": [round(float(x), 3) for x in scatter_actual],
            "predicted": [round(float(x), 3) for x in scatter_pred],
        },
    }
    if feature_importance is not None:
        result["feature_importance"] = {
            k: round(float(v), 6) for k, v in feature_importance.items()
        }

    path = output_dir / f"eval_{model_type}.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    logger.info("Eval results saved to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    target = train_cfg.get("target", "density")
    model_type = cfg.get("model", {}).get("type", "cnn1d")

    output_dir = cfg.get("output_dir", "outputs")

    if model_type in WINDOW_TABULAR_MODELS:
        model_path = (
            args.model_path
            or f"{output_dir}/{model_type}_best.pkl"
        )
        _evaluate_window_tabular(cfg, model_type, model_path, target)
    elif model_type in WINDOW_DL_MODELS:
        model_path = (
            args.model_path
            or f"{output_dir}/{model_type}_best.pt"
        )
        _evaluate_window(cfg, model_type, model_path, target)
    elif model_type in DL_MODELS:
        model_path = (
            args.model_path
            or f"{output_dir}/{model_type}_best.pt"
        )
        _evaluate_dl(cfg, model_type, model_path, target)
    else:
        data_path = (
            args.data
            or cfg.get("data", {}).get(
                "tabular_path", "data/features/dataset.parquet"
            )
        )
        model_path = (
            args.model_path
            or f"{output_dir}/{model_type}_best.pkl"
        )
        _evaluate_tabular(
            cfg, model_type, model_path, target, data_path,
        )


if __name__ == "__main__":
    main()
