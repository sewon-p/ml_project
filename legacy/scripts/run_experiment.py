"""Run a specific experiment configuration (DL + tabular)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.data.io import read_parquet
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


def _run_dl_experiment(cfg: dict, exp_name: str) -> dict:
    """Run a DL experiment (CNN1D / LSTM)."""
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
        logger.info("Auto-detected device: %s", device)
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU.")
        device = "cpu"

    # Load time series
    ts_path = data_cfg.get(
        "timeseries_path", "data/features/timeseries.npz"
    )
    data = np.load(ts_path)
    sequences = data["sequences"]
    target_name = train_cfg.get("target", "density")
    scenario_ids = data["scenario_ids"]

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    actual_targets = data[target_name]
    fd_estimates = None

    if residual_enabled:
        delta_key = "delta_density" if target_name == "density" else "delta_flow"
        fd_key = "k_fd" if target_name == "density" else "q_fd"
        targets = data[delta_key]
        fd_estimates = data[fd_key]
        logger.info("Residual mode: target=%s, fd_key=%s", delta_key, fd_key)
    else:
        targets = actual_targets

    # Load conditions
    meta_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    from src.data.io import read_parquet
    meta_df = read_parquet(meta_path)
    conditions = meta_df[["num_lanes", "speed_limit"]].values.astype(np.float32)
    n_conditions = conditions.shape[1]

    logger.info(
        "Loaded %d samples, shape %s", len(sequences), sequences.shape,
    )

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

    batch_size = train_cfg.get("batch_size", 128)
    num_workers = train_cfg.get("num_workers", 4)
    if len(sequences) < 1000:
        num_workers = 0
        logger.info("Small dataset (%d samples), using num_workers=0", len(sequences))

    train_loader = DataLoader(
        TimeSeriesDataset(sequences[train_idx], targets[train_idx], conditions[train_idx]),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        TimeSeriesDataset(sequences[val_idx], targets[val_idx], conditions[val_idx]),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        TimeSeriesDataset(sequences[test_idx], targets[test_idx], conditions[test_idx]),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # Create model
    model_params = {}
    model_config_path = model_cfg.get("config")
    if model_config_path:
        model_params = load_config(model_config_path)
    model_params.pop("device", None)

    estimator = create_model(model_type, device=device, n_conditions=n_conditions, **model_params)

    output_dir = Path(cfg.get("output_dir", "outputs")) / exp_name
    trainer = DLTrainer(
        model=estimator.model,
        device=device,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        optimizer_name=train_cfg.get("optimizer", "adam"),
        scheduler_config=train_cfg.get("scheduler"),
        max_epochs=train_cfg.get("max_epochs", 200),
        patience=train_cfg.get("early_stopping_patience", 20),
        checkpoint_dir=output_dir,
        tensorboard_dir=cfg.get("logging", {}).get(
            "tensorboard_dir", "runs/"
        ),
    )

    logger.info("Training %s on device=%s", model_type, device)
    results = trainer.fit(train_loader, val_loader)
    logger.info("Best val_loss=%.4f", results["best_val_loss"])

    test_preds = trainer.predict(test_loader)
    if residual_enabled:
        restored_preds = fd_estimates[test_idx] + test_preds
        test_metrics = compute_all_metrics(actual_targets[test_idx], restored_preds)
    else:
        test_metrics = compute_all_metrics(targets[test_idx], test_preds)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator.save(output_dir / f"{model_type}_best")

    return test_metrics


def _run_tabular_experiment(cfg: dict, exp_name: str) -> dict:
    """Run a tabular experiment (XGBoost / LightGBM / FD baseline)."""
    model_type = cfg.get("model", {}).get("type", "xgboost")
    target = cfg.get("training", {}).get("target", "density")
    data_path = cfg.get(
        "data_path",
        cfg.get("data", {}).get(
            "tabular_path", "data/features/dataset.parquet"
        ),
    )

    df = read_parquet(data_path)

    exclude = {
        "scenario_id", "probe_idx",
        "density", "flow", "demand_vehph",
        "k_fd", "q_fd", "delta_density", "delta_flow",
    }
    # Apply user-selected feature exclusions from dashboard
    user_exclude = cfg.get("training", {}).get("exclude_features") or []
    if user_exclude:
        exclude.update(user_exclude)
        logger.info("Excluding %d user-selected features: %s", len(user_exclude), user_exclude)
    feature_columns = [c for c in df.columns if c not in exclude]

    feature_names = cfg.get("features", {}).get("selected")
    if feature_names:
        feature_columns = [
            c for c in feature_columns if c in feature_names
        ]

    # Residual correction
    rc_cfg = cfg.get("residual_correction", {})
    residual_enabled = rc_cfg.get("enabled", False)
    target_name = target
    if residual_enabled:
        train_target = "delta_density" if target == "density" else "delta_flow"
        fd_col = "k_fd" if target == "density" else "q_fd"
        logger.info("Residual mode: target=%s, fd_col=%s", train_target, fd_col)
    else:
        train_target = target

    train_cfg = cfg.get("training", {})
    train_df, test_df = grouped_train_test_split(
        df,
        group_column=train_cfg.get("group_column", "scenario_id"),
        test_ratio=train_cfg.get("test_ratio", 0.2),
    )

    # Density-weighted sample weights
    sw_col: str | None = None
    per_lane_target = f"{target}_per_lane"
    if train_cfg.get("density_weighted", False) and per_lane_target in train_df.columns:
        density_vals = train_df[per_lane_target].values
        train_df = train_df.copy()
        train_df["_sample_weight"] = 1.0 + (density_vals / density_vals.max()) * 2.0
        sw_col = "_sample_weight"
        logger.info("Density weighting enabled: weight range [1.0, 3.0]")

    model_params = {}
    model_config_path = cfg.get("model", {}).get("config")
    if model_config_path:
        model_params = load_config(model_config_path)

    trainer = TabularTrainer(
        model_type=model_type,
        model_params=model_params,
        n_splits=train_cfg.get("n_splits", 5),
        group_column=train_cfg.get("group_column", "scenario_id"),
    )
    trainer.fit(train_df, feature_columns, train_target, sample_weight_column=sw_col)

    test_preds = trainer.predict(test_df[feature_columns].values)
    if residual_enabled:
        restored_preds = test_df[fd_col].values + test_preds
        test_metrics = compute_all_metrics(
            test_df[target_name].values, restored_preds,
        )
    else:
        test_metrics = compute_all_metrics(
            test_df[target].values, test_preds,
        )

    output_dir = Path(cfg.get("output_dir", "outputs")) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.best_model.save(output_dir / f"{model_type}_best")

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument(
        "--config", required=True, help="Experiment config YAML.",
    )
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    exp_name = cfg.get("experiment_name", "unnamed")
    model_type = cfg.get("model", {}).get("type", "xgboost")
    logger.info("Starting experiment: %s (model=%s)", exp_name, model_type)

    if model_type in DL_MODELS:
        test_metrics = _run_dl_experiment(cfg, exp_name)
    else:
        test_metrics = _run_tabular_experiment(cfg, exp_name)

    logger.info("Experiment %s — Test: %s", exp_name, test_metrics)

    output_dir = Path(cfg.get("output_dir", "outputs")) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
