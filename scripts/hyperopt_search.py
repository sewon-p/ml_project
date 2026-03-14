"""Optuna hyperparameter search with window size optimization.

Usage:
  python scripts/hyperopt_search.py --config configs/hyperopt.yaml
  python scripts/hyperopt_search.py --config configs/hyperopt.yaml --n-trials 10
  python scripts/hyperopt_search.py --n-trials 200 --target density

3-phase pipeline:
  Phase 1: Pre-compute feature datasets for each window size
  Phase 2: Optuna search (window_size + XGBoost hyperparameters)
  Phase 3: Train final model with best parameters + save
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import CHANNELS
from src.evaluation.metrics import compute_all_metrics
from src.features.pipeline import extract_features
from src.models.factory import create_model
from src.training.cross_validation import grouped_kfold_split
from src.training.hyperopt import optimize_with_window
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = logging.getLogger(__name__)

# Columns that are NOT features (targets, metadata, FD columns)
NON_FEATURE_COLUMNS = {
    "scenario_id",
    "probe_idx",
    "num_lanes",
    "speed_limit",
    "density",
    "flow",
    "demand_vehph",
    "k_fd",
    "q_fd",
    "delta_density",
    "delta_flow",
}


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify feature columns by excluding known non-feature columns."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def _extract_one(args: tuple) -> dict:
    """Extract features from a single sliced sequence (for multiprocessing)."""
    seq, meta_dict, window_size, feature_names = args
    sliced = seq[:, -window_size:]  # (6, window_size)
    trajectory = pd.DataFrame(sliced.T, columns=CHANNELS)
    feats = extract_features(trajectory, feature_names=feature_names)
    feats.update(meta_dict)
    return feats


def _build_window_dataset(
    sequences: np.ndarray,
    metadata: pd.DataFrame,
    window_size: int,
    feature_names: list[str] | None = None,
    n_workers: int = 0,
) -> pd.DataFrame:
    """Slice time series to window_size and extract tabular features.

    Uses multiprocessing for speed.
    """
    n_samples = len(sequences)

    # Pre-build metadata dicts
    meta_cols = ["scenario_id", "density", "flow", "num_lanes", "speed_limit", "demand_vehph"]
    meta_cols = [c for c in meta_cols if c in metadata.columns]
    meta_dicts = []
    for i in range(n_samples):
        row = metadata.iloc[i]
        d = {}
        for c in meta_cols:
            v = row[c]
            d[c] = int(v) if c == "scenario_id" else float(v)
        meta_dicts.append(d)

    work_items = [
        (sequences[i], meta_dicts[i], window_size, feature_names)
        for i in range(n_samples)
    ]

    if n_workers <= 1:
        records = [_extract_one(item) for item in work_items]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            records = pool.map(_extract_one, work_items, chunksize=500)

    return pd.DataFrame(records)


def phase1_precompute(
    cfg: dict,
    window_sizes: list[int],
    force: bool = False,
) -> dict[int, Path]:
    """Phase 1: Pre-compute feature datasets for each window size.

    Returns mapping from window_size to parquet path.
    """
    data_cfg = cfg.get("data", {})
    timeseries_path = data_cfg.get("timeseries_path", "data/features/timeseries.npz")
    metadata_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    tabular_path = Path(data_cfg.get("tabular_path", "data/features/dataset.parquet"))
    features_dir = tabular_path.parent
    seq_len = data_cfg.get("seq_len", 300)

    # Load feature names from config
    feature_cfg_path = cfg.get("features", {}).get("config")
    feature_names = None
    if feature_cfg_path and Path(feature_cfg_path).exists():
        feature_names = load_config(feature_cfg_path).get("features")

    # Workers for parallel feature extraction
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    logger.info("Using %d workers for feature extraction", n_workers)

    # Only load timeseries.npz if we actually need to extract
    sequences = None
    metadata = None
    dataset_paths: dict[int, Path] = {}

    for ws in window_sizes:
        out_path = features_dir / f"dataset_w{ws}.parquet"
        dataset_paths[ws] = out_path

        if out_path.exists() and not force:
            logger.info("Window %ds: %s already exists, skipping", ws, out_path)
            continue

        # If ws == seq_len (300) and dataset.parquet exists, just copy it
        if ws == seq_len and tabular_path.exists() and not force:
            shutil.copy2(tabular_path, out_path)
            logger.info("Window %ds: copied from %s", ws, tabular_path)
            continue

        # Lazy-load timeseries data
        if sequences is None:
            logger.info("Loading time series from %s", timeseries_path)
            npz = np.load(timeseries_path)
            sequences = npz["sequences"]  # (N, 6, 300)
            logger.info("Loaded %d sequences of shape %s", len(sequences), sequences.shape)
            metadata = pd.read_parquet(metadata_path)
            logger.info("Loaded metadata: %d rows", len(metadata))

        logger.info(
            "Window %ds: extracting features from %d samples (%d workers)...",
            ws, len(sequences), n_workers,
        )
        df = _build_window_dataset(sequences, metadata, ws, feature_names, n_workers=n_workers)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Window %ds: saved %d rows to %s", ws, len(df), out_path)

    return dataset_paths


def phase2_search(
    cfg: dict,
    dataset_paths: dict[int, Path],
    n_trials: int,
    target: str,
) -> dict:
    """Phase 2: Run Optuna search over window sizes + XGBoost params."""
    hyperopt_cfg = cfg.get("hyperopt", {})
    window_sizes = sorted(dataset_paths.keys())
    n_splits = hyperopt_cfg.get("n_splits", 3)
    direction = hyperopt_cfg.get("direction", "minimize")
    storage = hyperopt_cfg.get("storage")
    search_space = hyperopt_cfg.get("search_space")

    # Load all datasets into memory
    datasets: dict[int, pd.DataFrame] = {}
    feature_columns_map: dict[int, list[str]] = {}

    for ws in window_sizes:
        df = pd.read_parquet(dataset_paths[ws])
        datasets[ws] = df
        feature_columns_map[ws] = _get_feature_columns(df)
        logger.info(
            "Window %ds: %d samples, %d features",
            ws,
            len(df),
            len(feature_columns_map[ws]),
        )

    # Ensure storage directory exists
    if storage:
        Path(storage).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting Optuna search: %d trials, windows=%s, target=%s",
        n_trials,
        window_sizes,
        target,
    )

    result = optimize_with_window(
        datasets=datasets,
        window_sizes=window_sizes,
        feature_columns_map=feature_columns_map,
        target_column=target,
        n_splits=n_splits,
        n_trials=n_trials,
        direction=direction,
        storage=storage,
        search_space=search_space,
    )

    return result


def phase3_final_train(
    cfg: dict,
    dataset_paths: dict[int, Path],
    best_params: dict,
    target: str,
) -> dict:
    """Phase 3: Train final model with best params and evaluate with 5-fold CV."""
    window_size = best_params["window_size"]
    df = pd.read_parquet(dataset_paths[window_size])
    feature_columns = _get_feature_columns(df)

    # Extract XGBoost params (exclude window_size)
    model_params = {k: v for k, v in best_params.items() if k != "window_size"}

    # 5-fold CV for robust evaluation
    n_splits = cfg.get("training", {}).get("n_splits", 5)
    splits = grouped_kfold_split(df, n_splits=n_splits)

    fold_metrics = []
    best_model = None
    best_fold_score = float("inf")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train = df.iloc[train_idx][feature_columns].values
        y_train = df.iloc[train_idx][target].values
        X_val = df.iloc[val_idx][feature_columns].values
        y_val = df.iloc[val_idx][target].values

        model = create_model("xgboost", **model_params)
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        preds = model.predict(X_val)
        metrics = compute_all_metrics(y_val, preds)
        fold_metrics.append(metrics)

        if metrics["rmse"] < best_fold_score:
            best_fold_score = metrics["rmse"]
            best_model = model

        logger.info("Fold %d: %s", fold_idx + 1, metrics)

    # Aggregate metrics
    mean_metrics = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        mean_metrics[key] = float(np.mean(values))

    # Save best model
    output_dir = Path(cfg.get("output_dir", "outputs_xgboost"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_best.pkl"
    best_model.save(model_path)
    logger.info("Saved best model to %s", model_path)

    return {"mean_metrics": mean_metrics, "fold_metrics": fold_metrics}


def _print_results(
    best_params: dict,
    best_value: float,
    window_best: dict,
    final_metrics: dict,
    n_trials: int,
) -> None:
    """Print formatted results to console."""
    print("\n=== Hyperopt Results ===")
    print(f"Best trial  RMSE: {best_value:.2f}")
    print()
    print("Best params:")
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print()
    print("Window comparison:")
    best_ws = best_params.get("window_size")
    for ws in sorted(window_best):
        marker = "  *" if ws == best_ws else ""
        print(f"  {ws:>3d}s -> best RMSE: {window_best[ws]:.2f}{marker}")
    print()
    mm = final_metrics["mean_metrics"]
    print("Final model (5-fold CV):")
    print(
        f"  RMSE: {mm['rmse']:.2f}  MAE: {mm['mae']:.2f}  "
        f"MAPE: {mm['mape']:.1f}%  R2: {mm['r2']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search with window size optimization."
    )
    parser.add_argument("--config", default="configs/hyperopt.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of windowed datasets",
    )
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)

    hyperopt_cfg = cfg.get("hyperopt", {})
    window_sizes = hyperopt_cfg.get("window_sizes", [60, 120, 180, 300])
    n_trials = args.n_trials or hyperopt_cfg.get("n_trials", 100)
    target = args.target or hyperopt_cfg.get("target", "density")

    # Phase 1: Pre-compute windowed datasets
    logger.info("=== Phase 1: Pre-compute windowed datasets ===")
    dataset_paths = phase1_precompute(cfg, window_sizes, force=args.force)

    # Phase 2: Optuna search
    logger.info("=== Phase 2: Optuna search ===")
    result = phase2_search(cfg, dataset_paths, n_trials, target)

    best_params = result["best_params"]
    best_value = result["best_value"]
    window_best = result["window_best"]

    # Phase 3: Final training
    logger.info("=== Phase 3: Final training with best params ===")
    final_metrics = phase3_final_train(cfg, dataset_paths, best_params, target)

    # Save results to JSON
    output_dir = Path(cfg.get("output_dir", "outputs_xgboost"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hyperopt_results.json"
    results_data = {
        "best_params": best_params,
        "best_value": best_value,
        "window_best": {str(k): v for k, v in window_best.items()},
        "final_metrics": final_metrics["mean_metrics"],
        "all_trials": result["all_trials"],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", results_path)

    _print_results(best_params, best_value, window_best, final_metrics, n_trials)


if __name__ == "__main__":
    main()
