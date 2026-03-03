"""Evaluate trained model and generate plots."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.io import read_parquet
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

_LOADERS = {
    "xgboost": XGBoostEstimator,
    "lightgbm": LightGBMEstimator,
    "fd_baseline": FDBaselineEstimator,
    "cnn1d": CNN1DEstimator,
    "lstm": LSTMEstimator,
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
    actual_targets = data[target]
    scenario_ids = data["scenario_ids"]

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    if residual_enabled:
        delta_key = "delta_density" if target == "density" else "delta_flow"
        fd_key = "k_fd" if target == "density" else "q_fd"
        train_targets = data[delta_key]
        fd_estimates = data[fd_key]
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

    # Load conditions
    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)
    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)
    n_conditions = conditions.shape[1]

    model_params["n_conditions"] = n_conditions
    loader_cls = _LOADERS[model_type]
    model = loader_cls.load(model_path, **model_params)

    device = train_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
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

    plots_dir = Path(cfg.get("output_dir", "outputs")) / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target})",
        save_path=plots_dir / "predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target})",
        save_path=plots_dir / "residuals.png",
    )
    logger.info("Plots saved to %s", plots_dir)


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
    exclude = {
        "scenario_id", "probe_idx",
        "density", "flow", "demand_vehph",
        "k_fd", "q_fd", "delta_density", "delta_flow",
    }
    feature_columns = [c for c in df.columns if c not in exclude]

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

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)

    X_test = test_df[feature_columns].values
    y_pred = model.predict(X_test)

    if residual_enabled:
        fd_col = "k_fd" if target == "density" else "q_fd"
        y_pred = test_df[fd_col].values + y_pred
    y_test = test_df[target].values

    metrics = compute_all_metrics(y_test, y_pred)
    logger.info("Test metrics: %s", metrics)

    plots_dir = Path(cfg.get("output_dir", "outputs")) / "plots"
    plot_predicted_vs_actual(
        y_test, y_pred,
        title=f"{model_type} — Predicted vs Actual ({target})",
        save_path=plots_dir / "predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, y_pred,
        title=f"{model_type} — Residuals ({target})",
        save_path=plots_dir / "residuals.png",
    )
    logger.info("Plots saved to %s", plots_dir)


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

    if model_type in DL_MODELS:
        model_path = (
            args.model_path
            or f"outputs/{model_type}_best.pt"
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
            or f"outputs/{model_type}_best.pkl"
        )
        _evaluate_tabular(
            cfg, model_type, model_path, target, data_path,
        )


if __name__ == "__main__":
    main()
