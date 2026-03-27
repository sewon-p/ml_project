"""Multi-probe penetration rate experiment.

Compares DeepSets (LSTM, CNN1D) and XGBoost across N=1,2,3,5 probes.
Outputs R² / MAE / RMSE tables by probe count and density range.

Usage:
    python scripts/train_multi_probe.py
    python scripts/train_multi_probe.py --probes 1 2 3 5
    python scripts/train_multi_probe.py --skip-dl          # XGBoost only
    python scripts/train_multi_probe.py --skip-xgb         # DL only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import compute_all_metrics
from src.models.multi_probe import MultiProbeModel
from src.training.trainer_dl import DLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Constants ----
DATA_DIR = Path("data/features")
RESULTS_DIR = Path("results/multi_probe")
SEED = 42

TOP_FEATURES = [
    "speed_mean",
    "speed_std",
    "speed_min",
    "speed_cv",
    "ax_std",
    "ax_mean",
    "stop_count",
    "brake_time_ratio",
]

DENSITY_BINS = [
    ("low (0-8)", 0, 8),
    ("mid (8-16)", 8, 16),
    ("high (16-24)", 16, 24),
]


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


# ---- Multi-Probe Dataset ----
class MultiProbeDataset(Dataset):
    """Groups N probes per scenario for DeepSets training."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        scenario_ids: np.ndarray,
        n_probes: int,
        conditions: np.ndarray | None = None,
        scenario_subset: np.ndarray | None = None,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        valid_sids = (
            set(scenario_subset) if scenario_subset is not None else set(np.unique(scenario_ids))
        )

        self.items: list[dict] = []
        for sid in sorted(valid_sids):
            indices = np.where(scenario_ids == sid)[0]
            if len(indices) < n_probes:
                continue
            chosen = rng.choice(indices, size=n_probes, replace=False)
            chosen.sort()
            self.items.append(
                {
                    "probe_indices": chosen,
                    "target": float(targets[indices[0]]),
                    "cond_idx": int(indices[0]),
                }
            )

        self.sequences = sequences
        self.conditions = conditions

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        x = torch.from_numpy(self.sequences[item["probe_indices"]].copy()).float()
        y = torch.tensor(item["target"], dtype=torch.float32)
        if self.conditions is not None:
            cond = torch.from_numpy(self.conditions[item["cond_idx"]].copy()).float()
            return x, y, cond
        return x, y


# ---- Data Loading ----
def load_data(
    target: str = "density_per_lane",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    logger.info("Loading data from %s (target=%s)...", DATA_DIR, target)

    # Use 1km-resampled data if available, else fall back to original
    npz_path = DATA_DIR / "timeseries_1km.npz"
    if not npz_path.exists():
        npz_path = DATA_DIR / "timeseries.npz"
        logger.info("Using original 300s timeseries")
    else:
        logger.info("Using 1km-resampled timeseries (100 points)")

    npz = np.load(npz_path)
    sequences = npz["sequences"]  # (N, 6, 100) or (N, 6, 300)
    scenario_ids = npz["scenario_ids"]

    # Load target from parquet (supports density, density_per_lane, flow, etc.)
    df_path = DATA_DIR / "dataset_1km.parquet"
    if not df_path.exists():
        df_path = DATA_DIR / "dataset.parquet"
    df = pd.read_parquet(df_path)

    if target in df.columns:
        density = df[target].values.astype(np.float32)
        logger.info("Target '%s': mean=%.2f, std=%.2f", target, density.mean(), density.std())
    else:
        density = npz["density"].astype(np.float32)
        logger.warning("Target '%s' not found, falling back to npz density (absolute)", target)

    # Load conditions: num_lanes, speed_limit, traversal_time
    meta_path = DATA_DIR / "metadata_1km.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        conditions = meta[["num_lanes", "speed_limit", "traversal_time"]].values.astype(np.float32)
        logger.info("Conditions: num_lanes, speed_limit, traversal_time")
    else:
        df = pd.read_parquet(DATA_DIR / "dataset.parquet")
        conditions = df[["num_lanes", "speed_limit"]].values.astype(np.float32)
        logger.info("Conditions: num_lanes, speed_limit")

    df_path = DATA_DIR / "dataset_1km.parquet"
    if not df_path.exists():
        df_path = DATA_DIR / "dataset.parquet"
    df = pd.read_parquet(df_path)

    # Filter to scenarios with >= 5 probes
    sid_counts = pd.Series(scenario_ids).value_counts()
    valid_sids = sid_counts[sid_counts >= 5].index.values

    logger.info(
        "Loaded %d samples, %d scenarios (>= 5 probes: %d)",
        len(sequences),
        sid_counts.shape[0],
        len(valid_sids),
    )
    return sequences, density, scenario_ids, conditions, df, valid_sids


def split_scenarios(
    valid_sids: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_sids, temp_sids = train_test_split(valid_sids, test_size=0.3, random_state=seed)
    val_sids, test_sids = train_test_split(temp_sids, test_size=0.5, random_state=seed)
    logger.info("Split: train=%d, val=%d, test=%d", len(train_sids), len(val_sids), len(test_sids))
    return train_sids, val_sids, test_sids


# ---- DL Training ----
def train_deepsets(
    encoder_type: str,
    n_probes: int,
    sequences: np.ndarray,
    density: np.ndarray,
    scenario_ids: np.ndarray,
    conditions: np.ndarray,
    train_sids: np.ndarray,
    val_sids: np.ndarray,
    test_sids: np.ndarray,
    device: str,
    pooling: str = "mean",
) -> tuple[dict, np.ndarray, np.ndarray]:
    train_ds = MultiProbeDataset(
        sequences,
        density,
        scenario_ids,
        n_probes,
        conditions=conditions,
        scenario_subset=train_sids,
        seed=SEED,
    )
    val_ds = MultiProbeDataset(
        sequences,
        density,
        scenario_ids,
        n_probes,
        conditions=conditions,
        scenario_subset=val_sids,
        seed=SEED + 1,
    )
    test_ds = MultiProbeDataset(
        sequences,
        density,
        scenario_ids,
        n_probes,
        conditions=conditions,
        scenario_subset=test_sids,
        seed=SEED + 2,
    )

    nw = 0  # Windows spawn + large array = pickle error with nw>0
    pin = device == "cuda"
    bs = 512 if device == "cuda" else 64
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        train_ds if len(val_ds) == 0 else val_ds, batch_size=bs, num_workers=nw, pin_memory=pin
    )
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw, pin_memory=pin)

    if encoder_type == "lstm":
        model = MultiProbeModel(
            encoder_type="lstm",
            n_conditions=conditions.shape[1],
            pooling=pooling,
            input_size=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
    else:
        model = MultiProbeModel(
            encoder_type="cnn1d",
            n_conditions=conditions.shape[1],
            pooling=pooling,
            in_channels=6,
            n_filters=[32, 64, 128],
            kernel_size=3,
            dropout=0.2,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        trainer = DLTrainer(
            model=model,
            device=device,
            learning_rate=1e-3,
            max_epochs=100,
            patience=15,
            checkpoint_dir=str(ckpt_dir),
        )

        logger.info(
            "Training DeepSets-%s/%s (N=%d, %d scenarios)...",
            encoder_type.upper(),
            pooling,
            n_probes,
            len(train_ds),
        )
        trainer.fit(train_loader, val_loader)

        # Load best checkpoint
        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(state)

    # Evaluate on test set
    y_pred = trainer.predict(test_loader)
    y_true = np.array([test_ds.items[i]["target"] for i in range(len(test_ds))])

    metrics = compute_all_metrics(y_true, y_pred)
    logger.info(
        "DeepSets-%s/%s N=%d: R²=%.4f MAE=%.4f RMSE=%.4f",
        encoder_type.upper(),
        pooling,
        n_probes,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics, y_true, y_pred


# ---- XGBoost Training ----
def build_aggregated_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_probes: int,
    scenario_subset: np.ndarray,
    seed: int = 42,
    target_col: str = "density_per_lane",
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    subset_df = df[df["scenario_id"].isin(set(scenario_subset))]

    rows = []
    for sid, group in subset_df.groupby("scenario_id"):
        if len(group) < n_probes:
            continue
        sampled = group.iloc[rng.choice(len(group), size=n_probes, replace=False)]
        feats = sampled[feature_cols].values

        row: dict = {}
        for i, col in enumerate(feature_cols):
            row[f"{col}_mean"] = float(feats[:, i].mean())
            row[f"{col}_std"] = float(feats[:, i].std()) if n_probes > 1 else 0.0

        row["num_lanes"] = float(group["num_lanes"].iloc[0])
        row["speed_limit"] = float(group["speed_limit"].iloc[0])
        if "traversal_time" in group.columns:
            row["traversal_time_mean"] = float(sampled["traversal_time"].mean())
            row["traversal_time_std"] = (
                float(sampled["traversal_time"].std()) if n_probes > 1 else 0.0
            )
        row["target"] = float(group[target_col].iloc[0])
        row["scenario_id"] = int(sid)
        rows.append(row)

    return pd.DataFrame(rows)


def train_xgboost_multi(
    n_probes: int,
    df: pd.DataFrame,
    feature_cols: list[str],
    train_sids: np.ndarray,
    val_sids: np.ndarray,
    test_sids: np.ndarray,
    target_col: str = "density_per_lane",
) -> tuple[dict, np.ndarray, np.ndarray]:
    import xgboost as xgb

    train_df = build_aggregated_df(
        df, feature_cols, n_probes, train_sids, seed=SEED, target_col=target_col
    )
    val_df = build_aggregated_df(
        df, feature_cols, n_probes, val_sids, seed=SEED + 1, target_col=target_col
    )
    test_df = build_aggregated_df(
        df, feature_cols, n_probes, test_sids, seed=SEED + 2, target_col=target_col
    )

    agg_cols: list[str] = []
    for col in feature_cols:
        agg_cols.append(f"{col}_mean")
        agg_cols.append(f"{col}_std")
    agg_cols.extend(["num_lanes", "speed_limit"])
    if "traversal_time_mean" in train_df.columns:
        agg_cols.extend(["traversal_time_mean", "traversal_time_std"])

    X_train, y_train = train_df[agg_cols].values, train_df["target"].values
    X_val, y_val = val_df[agg_cols].values, val_df["target"].values
    X_test, y_test = test_df[agg_cols].values, test_df["target"].values

    logger.info(
        "Training XGBoost (N=%d, %d train, %d features)...", n_probes, len(train_df), len(agg_cols)
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=SEED,
        tree_method="hist",
        device="cpu",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)
    logger.info(
        "XGBoost N=%d: R²=%.4f MAE=%.4f RMSE=%.4f",
        n_probes,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics, y_test, y_pred


# ---- Evaluation by Density Range ----
def evaluate_density_ranges(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    results = {}
    for label, lo, hi in DENSITY_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n > 10:
            m = compute_all_metrics(y_true[mask], y_pred[mask])
            m["n"] = n
            results[label] = m
    return results


# ---- Pretty Print ----
def print_results(all_results: dict, probes_list: list[int], models: list[str]) -> None:
    print("\n" + "=" * 70)
    print("MULTI-PROBE PENETRATION RATE EXPERIMENT")
    print("=" * 70)

    # Overall metrics
    for metric_name in ["r2", "mae", "rmse"]:
        print(f"\n{metric_name.upper()}")
        header = f"{'Model':<20}"
        for n in probes_list:
            header += f"  N={n:<8}"
        print(header)
        print("-" * (20 + 10 * len(probes_list)))

        for model_name in models:
            row = f"{model_name:<20}"
            for n in probes_list:
                key = f"{model_name}_n{n}"
                if key in all_results:
                    val = all_results[key]["overall"][metric_name]
                    row += f"  {val:<10.4f}"
                else:
                    row += f"  {'N/A':<10}"
            print(row)

    # Density range breakdown (R² only)
    for label, lo, hi in DENSITY_BINS:
        print(f"\nR² by density range: {label}")
        header = f"{'Model':<20}"
        for n in probes_list:
            header += f"  N={n:<8}"
        print(header)
        print("-" * (20 + 10 * len(probes_list)))

        for model_name in models:
            row = f"{model_name:<20}"
            for n in probes_list:
                key = f"{model_name}_n{n}"
                ranges = all_results.get(key, {}).get("ranges", {})
                if label in ranges:
                    row += f"  {ranges[label]['r2']:<10.4f}"
                else:
                    row += f"  {'N/A':<10}"
            print(row)


# ---- Main ----
def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-probe penetration rate experiment")
    parser.add_argument("--probes", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-dl", action="store_true", help="Skip DeepSets (DL) experiments")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost experiments")
    parser.add_argument(
        "--pooling",
        nargs="+",
        default=["mean", "attention", "cf_score"],
        help="Pooling methods for DeepSets (mean, attention, cf_score)",
    )
    parser.add_argument(
        "--target",
        default="density_per_lane",
        help="Target variable (density, density_per_lane, flow, flow_per_lane)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    sequences, density, scenario_ids, conditions, df, valid_sids = load_data(target=args.target)
    train_sids, val_sids, test_sids = split_scenarios(valid_sids)

    all_results: dict = {}
    active_models: list[str] = []

    if not args.skip_dl:
        for pool in args.pooling:
            active_models.extend([f"lstm_{pool}", f"cnn1d_{pool}"])
    if not args.skip_xgb:
        active_models.append("xgboost")

    for n_probes in args.probes:
        logger.info("=" * 50)
        logger.info("N_PROBES = %d", n_probes)
        logger.info("=" * 50)

        if not args.skip_dl:
            for enc in ["lstm", "cnn1d"]:
                for pool in args.pooling:
                    metrics, y_true, y_pred = train_deepsets(
                        enc,
                        n_probes,
                        sequences,
                        density,
                        scenario_ids,
                        conditions,
                        train_sids,
                        val_sids,
                        test_sids,
                        device,
                        pooling=pool,
                    )
                    ranges = evaluate_density_ranges(y_true, y_pred)
                    all_results[f"{enc}_{pool}_n{n_probes}"] = {
                        "overall": metrics,
                        "ranges": ranges,
                    }

        if not args.skip_xgb:
            metrics, y_true, y_pred = train_xgboost_multi(
                n_probes,
                df,
                TOP_FEATURES,
                train_sids,
                val_sids,
                test_sids,
                target_col=args.target,
            )
            ranges = evaluate_density_ranges(y_true, y_pred)
            all_results[f"xgboost_n{n_probes}"] = {
                "overall": metrics,
                "ranges": ranges,
            }

    # Print summary
    print_results(all_results, args.probes, active_models)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
