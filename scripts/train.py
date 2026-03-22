"""Model training entry point — DL-primary with tabular fallback.

DL models (cnn1d, lstm): Load .npz time series → DataLoader → DLTrainer (CUDA)
Tabular models (xgboost, lightgbm, fd_baseline): Load .parquet → TabularTrainer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.io import load_data_filter_mask, read_parquet
from src.data.preprocessing import grouped_train_test_split
from src.evaluation.metrics import compute_all_metrics
from src.models.factory import create_model
from src.training.trainer_dl import DLTrainer
from src.training.trainer_tabular import TabularTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

DL_MODELS = {"cnn1d", "lstm"}
TABULAR_MODELS = {"xgboost", "lightgbm", "fd_baseline"}


def _split_indices_by_group(
    scenario_ids: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test by scenario_id groups."""
    rng = np.random.RandomState(seed)
    unique_ids = np.unique(scenario_ids)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_ids = set(unique_ids[:n_test])
    val_ids = set(unique_ids[n_test : n_test + n_val])
    train_ids = set(unique_ids[n_test + n_val :])

    train_idx = np.where(np.isin(scenario_ids, list(train_ids)))[0]
    val_idx = np.where(np.isin(scenario_ids, list(val_ids)))[0]
    test_idx = np.where(np.isin(scenario_ids, list(test_ids)))[0]

    return train_idx, val_idx, test_idx


def train_dl(cfg: dict, args: argparse.Namespace) -> None:
    """Train a DL model (CNN1D or LSTM) on raw time series."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_type = model_cfg.get("type", "cnn1d")

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
    logger.info("Using device: %s", device)

    # Load time series data
    ts_path = args.data or data_cfg.get(
        "timeseries_path", "data/features/timeseries.npz"
    )
    logger.info("Loading time series from %s", ts_path)
    data = np.load(ts_path)
    sequences = data["sequences"]  # (N, 6, seq_len)
    target_name = train_cfg.get("target", "density")
    scenario_ids = data["scenario_ids"]  # (N,)

    # Load conditions (num_lanes, speed_limit) from metadata (needed early for fallback)
    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)

    # Per-lane target: density → density_per_lane, flow → flow_per_lane
    per_lane_target = f"{target_name}_per_lane"
    if per_lane_target in data.files:
        actual_targets = data[per_lane_target]
    else:
        logger.info("Fallback: computing %s from %s / num_lanes", per_lane_target, target_name)
        actual_targets = data[target_name].astype(np.float32) / meta_df["num_lanes"].values

    # Residual correction: switch target to delta if enabled
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    fd_estimates = None

    if residual_enabled:
        delta_key = "delta_density" if target_name == "density" else "delta_flow"
        fd_key = "k_fd" if target_name == "density" else "q_fd"
        num_lanes = meta_df["num_lanes"].values
        # Ensure residual targets and FD estimates are per-lane scale
        if per_lane_target in data.files:
            # prepare_residuals already produced per-lane k_fd/delta
            targets = data[delta_key]
            fd_estimates = data[fd_key]
        else:
            # Old data: k_fd/delta are absolute → convert to per-lane
            targets = data[delta_key].astype(np.float32) / num_lanes
            fd_estimates = data[fd_key].astype(np.float32) / num_lanes
        logger.info("Residual mode: target=%s, fd_key=%s", delta_key, fd_key)
    else:
        targets = actual_targets

    # Apply config-driven data filters
    mask = load_data_filter_mask(cfg, meta_df)
    if not mask.all():
        sequences = sequences[mask]
        scenario_ids = scenario_ids[mask]
        actual_targets = actual_targets[mask]
        targets = targets[mask]
        if fd_estimates is not None:
            fd_estimates = fd_estimates[mask]
        meta_df = meta_df[mask].reset_index(drop=True)

    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)
    n_conditions = conditions.shape[1]

    logger.info(
        "Loaded %d samples, shape %s, target=%s (residual=%s), conditions=%d",
        len(sequences), sequences.shape, target_name, residual_enabled, n_conditions,
    )

    # Split by scenario_id
    train_idx, val_idx, test_idx = _split_indices_by_group(
        scenario_ids,
        test_ratio=train_cfg.get("test_ratio", 0.2),
        val_ratio=train_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # Create datasets and loaders
    train_ds = TimeSeriesDataset(sequences[train_idx], targets[train_idx], conditions[train_idx])
    val_ds = TimeSeriesDataset(sequences[val_idx], targets[val_idx], conditions[val_idx])
    test_ds = TimeSeriesDataset(sequences[test_idx], targets[test_idx], conditions[test_idx])

    batch_size = train_cfg.get("batch_size", 128)
    # MPS: multiprocessing DataLoader is slower due to IPC overhead
    num_workers = 0 if device == "mps" else train_cfg.get("num_workers", 4)
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # Create model
    model_params = {}
    model_config_path = model_cfg.get("config")
    if model_config_path:
        model_params = load_config(model_config_path)
    model_params.pop("device", None)  # device managed by trainer

    estimator = create_model(model_type, device=device, n_conditions=n_conditions, **model_params)
    model = estimator.model

    # Create trainer
    output_dir = Path(cfg.get("output_dir", "outputs"))
    trainer = DLTrainer(
        model=model,
        device=device,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        optimizer_name=train_cfg.get("optimizer", "adam"),
        scheduler_config=train_cfg.get("scheduler"),
        max_epochs=train_cfg.get("max_epochs", 200),
        patience=train_cfg.get("early_stopping_patience", 20),
        checkpoint_dir=output_dir / model_type,
        tensorboard_dir=cfg.get("logging", {}).get(
            "tensorboard_dir", "runs/"
        ),
    )

    # Train
    logger.info("Training %s on device=%s", model_type, device)
    results = trainer.fit(train_loader, val_loader)
    logger.info(
        "Training done. Best val_loss=%.4f",
        results["best_val_loss"],
    )

    # Evaluate on test set
    test_preds = trainer.predict(test_loader)
    if residual_enabled:
        # Restore: k_pred = k_fd + Δk_pred
        restored_preds = fd_estimates[test_idx] + test_preds
        test_metrics = compute_all_metrics(
            actual_targets[test_idx], restored_preds,
        )
        logger.info("Test metrics (restored from residual): %s", test_metrics)
    else:
        test_metrics = compute_all_metrics(
            targets[test_idx], test_preds,
        )
        logger.info("Test metrics: %s", test_metrics)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator.save(output_dir / f"{model_type}_best")
    logger.info("Model saved to %s", output_dir)


def train_tabular(cfg: dict, args: argparse.Namespace) -> None:
    """Train a tabular model (XGBoost, LightGBM, FD baseline)."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_type = model_cfg.get("type", "xgboost")

    data_path = args.data or data_cfg.get(
        "tabular_path", "data/features/dataset.parquet"
    )
    df = read_parquet(data_path)

    # Apply config-driven data filters
    mask = load_data_filter_mask(cfg, df)
    if not mask.all():
        df = df[mask].reset_index(drop=True)

    logger.info(
        "Loaded dataset: %d rows, %d columns", len(df), len(df.columns),
    )

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
    target_name = train_cfg.get("target", "density")

    # Per-lane target: density → density_per_lane, flow → flow_per_lane
    per_lane_target = f"{target_name}_per_lane"
    old_data = per_lane_target not in df.columns
    if old_data:
        logger.info("Fallback: computing %s from %s / num_lanes", per_lane_target, target_name)
        df[per_lane_target] = df[target_name] / df["num_lanes"]

    # Residual correction: switch target
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    if residual_enabled:
        delta_col = "delta_density" if target_name == "density" else "delta_flow"
        fd_col = "k_fd" if target_name == "density" else "q_fd"
        if old_data:
            # Old data: k_fd/delta are absolute → convert to per-lane
            df[delta_col] = df[delta_col] / df["num_lanes"]
            df[fd_col] = df[fd_col] / df["num_lanes"]
            logger.info("Converted %s, %s to per-lane scale", delta_col, fd_col)
        target = delta_col
        logger.info("Residual mode: target=%s, fd_col=%s", target, fd_col)
    else:
        target = per_lane_target

    train_df, test_df = grouped_train_test_split(
        df,
        group_column=train_cfg.get("group_column", "scenario_id"),
        test_ratio=train_cfg.get("test_ratio", 0.2),
    )
    logger.info("Train: %d, Test: %d", len(train_df), len(test_df))

    # Density-weighted sample weights: upweight high-density samples
    sw_col: str | None = None
    if train_cfg.get("density_weighted", False):
        density_vals = train_df[per_lane_target].values
        train_df = train_df.copy()
        train_df["_sample_weight"] = 1.0 + (density_vals / density_vals.max()) * 2.0
        sw_col = "_sample_weight"
        logger.info("Density weighting enabled: weight range [1.0, 3.0]")

    model_params = {}
    model_config_path = model_cfg.get("config")
    if model_config_path:
        model_params = load_config(model_config_path)

    trainer = TabularTrainer(
        model_type=model_type,
        model_params=model_params,
        n_splits=train_cfg.get("n_splits", 5),
        group_column=train_cfg.get("group_column", "scenario_id"),
    )
    results = trainer.fit(train_df, feature_columns, target, sample_weight_column=sw_col)
    logger.info("CV Results: %s", results["mean_metrics"])

    test_preds = trainer.predict(test_df[feature_columns])
    if residual_enabled:
        restored_preds = test_df[fd_col].values + test_preds
        test_metrics = compute_all_metrics(
            test_df[per_lane_target].values, restored_preds,
        )
        logger.info("Test metrics (restored from residual): %s", test_metrics)
    else:
        test_metrics = compute_all_metrics(
            test_df[target].values, test_preds,
        )
        logger.info("Test metrics: %s", test_metrics)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.best_model.save(output_dir / f"{model_type}_best")
    logger.info("Model saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None, help="Override data path")
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model_type = cfg.get("model", {}).get("type", "cnn1d")

    if model_type in DL_MODELS:
        train_dl(cfg, args)
    elif model_type in TABULAR_MODELS:
        train_tabular(cfg, args)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {DL_MODELS | TABULAR_MODELS}"
        )


if __name__ == "__main__":
    main()
