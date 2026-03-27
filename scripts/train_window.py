"""Train window-feature models (LSTM, CNN1D, XGBoost).

Loads timeseries.npz + metadata → extracts (N, 8, 10) window features →
DL models: train with DLTrainer pipeline.
Tabular models: flatten to (N, 80) + conditions → TabularTrainer.

Usage:
    python scripts/train_window.py --config configs/default.yaml
    python scripts/train_window.py --config path/to/run_config.yaml
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
from src.evaluation.metrics import compute_all_metrics
from src.features.window_features import extract_window_features
from src.models.factory import create_model
from src.training.trainer_dl import DLTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

WINDOW_MODELS = {"window_cnn1d", "window_lstm", "window_xgboost"}
WINDOW_DL_MODELS = {"window_cnn1d", "window_lstm"}
WINDOW_TABULAR_MODELS = {"window_xgboost"}


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


def _normalize_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    conditions_train: np.ndarray,
    conditions_val: np.ndarray,
    conditions_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """StandardScaler on window features (per channel) and conditions."""
    # Window features: (N, 8, 10) — normalize per channel (axis 0, 2)
    N_tr, C, W = train.shape
    flat_tr = train.transpose(1, 0, 2).reshape(C, -1)  # (C, N*W)
    mean_f = flat_tr.mean(axis=1, keepdims=True)  # (C, 1)
    std_f = flat_tr.std(axis=1, keepdims=True) + 1e-8

    def norm_seq(arr: np.ndarray) -> np.ndarray:
        return (
            ((arr.transpose(1, 0, 2).reshape(C, -1) - mean_f) / std_f)
            .reshape(C, arr.shape[0], W)
            .transpose(1, 0, 2)
            .astype(np.float32)
        )

    train_n = norm_seq(train)
    val_n = norm_seq(val)
    test_n = norm_seq(test)

    # Conditions: (N, 2) — normalize per feature
    cond_mean = conditions_train.mean(axis=0, keepdims=True)
    cond_std = conditions_train.std(axis=0, keepdims=True) + 1e-8

    cond_train = ((conditions_train - cond_mean) / cond_std).astype(np.float32)
    cond_val = ((conditions_val - cond_mean) / cond_std).astype(np.float32)
    cond_test = ((conditions_test - cond_mean) / cond_std).astype(np.float32)

    scaler_params = {
        "feature_mean": mean_f.squeeze().tolist(),
        "feature_std": std_f.squeeze().tolist(),
        "cond_mean": cond_mean.squeeze().tolist(),
        "cond_std": cond_std.squeeze().tolist(),
    }

    return train_n, val_n, test_n, cond_train, cond_val, cond_test, scaler_params


def train_window(cfg: dict, args: argparse.Namespace) -> None:
    """Train a window-feature DL model."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_type = model_cfg.get("type", "window_cnn1d")

    # Map window_* to base model type for create_model
    base_model_type = model_type.replace("window_", "")

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

    # Load raw timeseries
    ts_path = args.data or data_cfg.get("timeseries_path", "data/features/timeseries.npz")
    logger.info("Loading time series from %s", ts_path)
    data = np.load(ts_path)
    sequences = data["sequences"]  # (N, 6, 300)
    target_name = train_cfg.get("target", "density")
    scenario_ids = data["scenario_ids"]

    # Load metadata
    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    meta_df = read_parquet(meta_path)
    speed_limits = meta_df["speed_limit"].values

    # Per-lane target
    per_lane_target = f"{target_name}_per_lane"
    if per_lane_target in data.files:
        actual_targets = data[per_lane_target]
    else:
        logger.info("Fallback: computing %s from %s / num_lanes", per_lane_target, target_name)
        actual_targets = data[target_name].astype(np.float32) / meta_df["num_lanes"].values

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    fd_estimates = None

    if residual_enabled:
        delta_key = "delta_density" if target_name == "density" else "delta_flow"
        fd_key = "k_fd" if target_name == "density" else "q_fd"
        num_lanes = meta_df["num_lanes"].values
        if per_lane_target in data.files:
            targets = data[delta_key]
            fd_estimates = data[fd_key]
        else:
            targets = data[delta_key].astype(np.float32) / num_lanes
            fd_estimates = data[fd_key].astype(np.float32) / num_lanes
        logger.info("Residual mode: target=%s, fd_key=%s", delta_key, fd_key)
    else:
        targets = actual_targets

    # Apply data filters
    mask = load_data_filter_mask(cfg, meta_df)
    if not mask.all():
        sequences = sequences[mask]
        scenario_ids = scenario_ids[mask]
        actual_targets = actual_targets[mask]
        targets = targets[mask]
        speed_limits = speed_limits[mask]
        if fd_estimates is not None:
            fd_estimates = fd_estimates[mask]
        meta_df = meta_df[mask].reset_index(drop=True)

    # Extract window features: (N, 6, 300) → (N, C, 10)
    window_cfg = cfg.get("window_features", {})
    window_size = window_cfg.get("window_size", 30)
    exclude_wf = window_cfg.get("exclude", [])
    win_features, used_names = extract_window_features(
        sequences,
        speed_limits,
        window_size,
        exclude=exclude_wf,
    )
    n_win_features = win_features.shape[1]

    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)

    logger.info(
        "Loaded %d samples, window shape %s, target=%s (residual=%s)",
        len(win_features),
        win_features.shape,
        target_name,
        residual_enabled,
    )

    # Split by scenario_id
    train_idx, val_idx, test_idx = _split_indices_by_group(
        scenario_ids,
        test_ratio=train_cfg.get("test_ratio", 0.2),
        val_ratio=train_cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )
    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    output_dir = Path(cfg.get("output_dir", "outputs"))

    # ---- Branch: tabular (window_xgboost) vs DL (window_cnn1d, window_lstm) ----
    if model_type in WINDOW_TABULAR_MODELS:
        _train_window_tabular(
            cfg,
            model_type,
            base_model_type,
            win_features,
            used_names,
            conditions,
            targets,
            actual_targets,
            fd_estimates,
            scenario_ids,
            train_idx,
            val_idx,
            test_idx,
            residual_enabled,
            per_lane_target,
            output_dir,
        )
    else:
        _train_window_dl(
            cfg,
            model_type,
            base_model_type,
            device,
            win_features,
            n_win_features,
            conditions,
            targets,
            actual_targets,
            fd_estimates,
            train_idx,
            val_idx,
            test_idx,
            residual_enabled,
            output_dir,
        )


def _train_window_tabular(
    cfg: dict,
    model_type: str,
    base_model_type: str,
    win_features: np.ndarray,
    used_names: list[str],
    conditions: np.ndarray,
    targets: np.ndarray,
    actual_targets: np.ndarray,
    fd_estimates: np.ndarray | None,
    scenario_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    residual_enabled: bool,
    per_lane_target: str,
    output_dir: Path,
) -> None:
    """Train window-feature tabular model (XGBoost) with flattened features."""
    import pandas as pd

    from src.training.trainer_tabular import TabularTrainer

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # Flatten (N, C, W) → (N, C*W) with column names w0_feat, w1_feat, ...
    N, C, W = win_features.shape
    columns = []
    for w in range(W):
        for name in used_names:
            columns.append(f"w{w}_{name}")

    flat = win_features.transpose(0, 2, 1).reshape(N, -1)  # (N, W*C)
    logger.info("Flattened window features: %s → %d columns", win_features.shape, len(columns))

    # Build DataFrame with features + conditions + targets + scenario_id
    df = pd.DataFrame(flat, columns=columns)
    df["num_lanes"] = conditions[:, 0]
    df["speed_limit"] = conditions[:, 1]
    df["scenario_id"] = scenario_ids
    df["_target"] = targets
    df["_actual_target"] = actual_targets
    if fd_estimates is not None:
        df["_fd_estimates"] = fd_estimates

    feature_columns = columns + ["num_lanes", "speed_limit"]
    target_col = "_target"

    # Density-weighted sample weights
    sw_col: str | None = None
    if train_cfg.get("density_weighted", False):
        density_vals = df["_actual_target"].values
        df["_sample_weight"] = 1.0 + (density_vals / density_vals.max()) * 2.0
        sw_col = "_sample_weight"
        logger.info("Density weighting enabled: weight range [1.0, 3.0]")

    # Split — reuse same scenario_id groups
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    model_params = {}
    model_config_path = model_cfg.get("config")
    if model_config_path:
        model_params = load_config(model_config_path)

    trainer = TabularTrainer(
        model_type=base_model_type,
        model_params=model_params,
        n_splits=train_cfg.get("n_splits", 5),
        group_column="scenario_id",
    )
    results = trainer.fit(train_df, feature_columns, target_col, sample_weight_column=sw_col)
    logger.info("CV Results: %s", results["mean_metrics"])

    test_preds = trainer.predict(test_df[feature_columns].values)
    if residual_enabled and "_fd_estimates" in test_df.columns:
        restored_preds = test_df["_fd_estimates"].values + test_preds
        test_metrics = compute_all_metrics(test_df["_actual_target"].values, restored_preds)
        logger.info("Test metrics (restored from residual): %s", test_metrics)
    else:
        test_metrics = compute_all_metrics(test_df[target_col].values, test_preds)
        logger.info("Test metrics: %s", test_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.best_model.save(output_dir / f"{model_type}_best")
    logger.info("Model saved to %s", output_dir)


def _train_window_dl(
    cfg: dict,
    model_type: str,
    base_model_type: str,
    device: str,
    win_features: np.ndarray,
    n_win_features: int,
    conditions: np.ndarray,
    targets: np.ndarray,
    actual_targets: np.ndarray,
    fd_estimates: np.ndarray | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    residual_enabled: bool,
    output_dir: Path,
) -> None:
    """Train window-feature DL model (CNN1D, LSTM)."""
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    n_conditions = conditions.shape[1]

    # Normalize window features and conditions
    wf_train, wf_val, wf_test, c_train, c_val, c_test, scaler = _normalize_features(
        win_features[train_idx],
        win_features[val_idx],
        win_features[test_idx],
        conditions[train_idx],
        conditions[val_idx],
        conditions[test_idx],
    )
    logger.info("Scaler: feature_mean=%s", scaler["feature_mean"])

    # Create datasets
    train_ds = TimeSeriesDataset(wf_train, targets[train_idx], c_train)
    val_ds = TimeSeriesDataset(wf_val, targets[val_idx], c_val)
    test_ds = TimeSeriesDataset(wf_test, targets[test_idx], c_test)

    batch_size = train_cfg.get("batch_size", 128)
    num_workers = 0 if device == "mps" else train_cfg.get("num_workers", 4)
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    model_params = {}
    model_config_path = model_cfg.get("config")
    if model_config_path:
        model_params = load_config(model_config_path)
    model_params.pop("device", None)

    if base_model_type == "cnn1d":
        model_params["in_channels"] = n_win_features
        model_params["seq_len"] = win_features.shape[2]
    elif base_model_type == "lstm":
        model_params["input_size"] = n_win_features

    estimator = create_model(
        base_model_type,
        device=device,
        n_conditions=n_conditions,
        **model_params,
    )
    model = estimator.model

    trainer = DLTrainer(
        model=model,
        device=device,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        optimizer_name=train_cfg.get("optimizer", "adam"),
        scheduler_config=train_cfg.get("scheduler"),
        max_epochs=train_cfg.get("max_epochs", 200),
        patience=train_cfg.get("early_stopping_patience", 20),
        checkpoint_dir=output_dir / model_type,
        tensorboard_dir=cfg.get("logging", {}).get("tensorboard_dir", "runs/"),
    )

    logger.info("Training %s on device=%s", model_type, device)
    results = trainer.fit(train_loader, val_loader)
    logger.info("Training done. Best val_loss=%.4f", results["best_val_loss"])

    test_preds = trainer.predict(test_loader)
    if residual_enabled:
        restored_preds = fd_estimates[test_idx] + test_preds
        test_metrics = compute_all_metrics(actual_targets[test_idx], restored_preds)
        logger.info("Test metrics (restored from residual): %s", test_metrics)
    else:
        test_metrics = compute_all_metrics(targets[test_idx], test_preds)
        logger.info("Test metrics: %s", test_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    estimator.save(output_dir / f"{model_type}_best")

    import json

    scaler_path = output_dir / f"{model_type}_scaler.json"
    scaler_path.write_text(json.dumps(scaler, indent=2))
    logger.info("Model saved to %s, scaler to %s", output_dir, scaler_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train window-feature DL model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None, help="Override timeseries path")
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    model_type = cfg.get("model", {}).get("type", "window_cnn1d")
    if model_type not in WINDOW_MODELS:
        logger.error("Expected window model type, got: %s", model_type)
        return

    train_window(cfg, args)


if __name__ == "__main__":
    main()
