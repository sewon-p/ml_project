"""Extract features and raw time series from simulation FCD output.

Per-scenario 5-probe extraction:
  - Probe candidates: vehicles present during 200 <= time < 300
  - Trajectory: 300 <= time < 600 (exactly 300 seconds, step=1s → (6, 300))
  - Ground truth: Edie density/flow from all vehicles in 300-600

Produces:
  - data/features/dataset.parquet   (tabular features for XGBoost/LightGBM)
  - data/features/timeseries.npz    (raw 6-channel time series for CNN1D/LSTM)
  - data/features/metadata.parquet  (per-sample metadata: scenario_id, density, etc.)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import CHANNELS, build_trajectory
from src.features.pipeline import extract_features
from src.simulation.trajectory_collector import parse_fcd
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = logging.getLogger(__name__)

# Features to drop from tabular output (redundant or count-based)
FEATURES_DROP = {
    "vx_mean", "vx_std", "vx_min", "vx_max",
    "vx_autocorr_lag1", "vx_fft_dominant_freq",
    "harsh_accel_count", "harsh_decel_count", "lane_change_count",
}


def _process_scenario(
    args: tuple,
) -> tuple[list[dict], list[np.ndarray], list[dict]] | None:
    """Process a single scenario: parse FCD → select 5 probes → extract."""
    (
        row_dict, fcd_dir, num_probes, warmup, collect,
        seq_len, feature_names,
    ) = args

    sid = int(row_dict["scenario_id"])
    link_length = float(row_dict["link_length"])
    seed = int(row_dict.get("seed", 42))

    fcd_path = Path(f"{fcd_dir}/scenario_{sid}/fcd.csv")
    if not fcd_path.exists():
        fcd_path = Path(f"{fcd_dir}/scenario_{sid}/fcd.xml")
    if not fcd_path.exists():
        return None

    fcd_df = parse_fcd(fcd_path)

    # Time boundaries
    traj_start = warmup + collect - seq_len  # 200 + 400 - 300 = 300
    traj_end = warmup + collect              # 600
    probe_start = warmup                     # 200
    probe_end = traj_start                   # 300

    # --- Ground truth: Edie density/flow from ALL vehicles in 300-600 ---
    gt_df = fcd_df[(fcd_df["time"] >= traj_start) & (fcd_df["time"] < traj_end)]
    density, flow = _fast_edie(gt_df, link_length, traj_end - traj_start)

    # --- Probe selection: vehicles present during 200 <= time < 300 ---
    probe_df = fcd_df[
        (fcd_df["time"] >= probe_start) & (fcd_df["time"] < probe_end)
    ]
    if probe_df.empty:
        return None

    candidate_ids = probe_df["vehicle_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(candidate_ids)
    selected_ids = candidate_ids[:num_probes]

    if not selected_ids:
        return None

    # --- Extract trajectory for each probe ---
    tabular_records: list[dict] = []
    raw_sequences: list[np.ndarray] = []
    meta_records: list[dict] = []

    for probe_idx, probe_id in enumerate(selected_ids):
        veh_df = gt_df[gt_df["vehicle_id"] == probe_id].sort_values("time")

        if len(veh_df) < seq_len:
            continue

        # Take exactly seq_len timesteps
        veh_df = veh_df.head(seq_len)
        trajectory = build_trajectory(veh_df)

        if len(trajectory) != seq_len:
            continue

        # Tabular features
        feats = extract_features(trajectory, feature_names=feature_names)
        # Drop VX-related features
        for vx_key in FEATURES_DROP:
            feats.pop(vx_key, None)

        feats["scenario_id"] = sid
        feats["probe_idx"] = probe_idx
        feats["num_lanes"] = row_dict.get("num_lanes", 1)
        feats["speed_limit"] = row_dict.get("speed_limit", 33.33)
        feats["density"] = density
        feats["flow"] = flow
        feats["demand_vehph"] = row_dict["demand_vehph"]
        tabular_records.append(feats)

        # Raw time series (6, 300)
        ts_array = trajectory[CHANNELS].values.T  # (6, seq_len)
        raw_sequences.append(ts_array.astype(np.float32))

        meta_records.append(
            {
                "scenario_id": sid,
                "probe_idx": probe_idx,
                "probe_id": str(probe_id),
                "density": density,
                "flow": flow,
                "demand_vehph": row_dict["demand_vehph"],
                "num_lanes": row_dict.get("num_lanes", 1),
                "speed_limit": row_dict.get("speed_limit", 33.33),
                "link_length": link_length,
                "seq_length": seq_len,
            }
        )

    if not tabular_records:
        return None

    return tabular_records, raw_sequences, meta_records


def _fast_edie(
    tw_df: pd.DataFrame,
    dx_m: float,
    dt: float,
) -> tuple[float, float]:
    """Compute Edie density and flow in one pass on pre-filtered subset."""
    if tw_df.empty or dt <= 0 or dx_m <= 0:
        return 0.0, 0.0
    times = tw_df["time"].unique()
    step = float(np.min(np.diff(np.sort(times)))) if len(times) >= 2 else 1.0
    dx_km = dx_m / 1000.0
    area = dx_km * dt
    density = float(len(tw_df) * step / area)
    flow = float(np.sum(tw_df["speed"].values) * step / 1000.0 / area * 3600.0)
    return density, flow


def _backup_and_load_existing(
    tabular_path: str,
    timeseries_path: str,
    metadata_path: str,
) -> tuple[pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None, set[int]]:
    """Load existing feature files and return their data + extracted scenario IDs.

    Also creates timestamped backup copies (e.g. dataset.bak_20260307.parquet).
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_ids: set[int] = set()
    old_df = None
    old_seqs = None
    old_meta = None

    tab_p = Path(tabular_path)
    ts_p = Path(timeseries_path)
    meta_p = Path(metadata_path)

    if tab_p.exists() and meta_p.exists() and ts_p.exists():
        old_df = pd.read_parquet(tab_p)
        old_meta = pd.read_parquet(meta_p)
        old_npz = np.load(ts_p)
        old_seqs = old_npz["sequences"]
        existing_ids = set(old_meta["scenario_id"].unique().tolist())

        # Drop FD residual columns from old data to avoid NaN on merge.
        # These columns are added by prepare_residuals.py and must be
        # recomputed after the merge anyway.
        fd_cols = {"k_fd", "q_fd", "delta_density", "delta_flow"}
        drop_cols = [c for c in fd_cols if c in old_df.columns]
        if drop_cols:
            old_df = old_df.drop(columns=drop_cols)
            logger.info("Dropped stale FD columns from old data: %s", drop_cols)

        # Backup
        import shutil

        for p in (tab_p, meta_p, ts_p):
            bak = p.with_suffix(f".bak_{stamp}{p.suffix}")
            shutil.copy2(p, bak)
            logger.info("Backup: %s -> %s", p.name, bak.name)

        logger.info(
            "Loaded existing: %d samples, %d scenarios",
            len(old_df),
            len(existing_ids),
        )

    return old_df, old_seqs, old_meta, existing_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features and time series from FCD.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenarios-csv", default=None)
    parser.add_argument(
        "--workers",
        default=None,
        type=str,
        help="Number of parallel workers (default: from config, 'auto' or int)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only extract new scenarios, append to existing features. "
        "Backs up old files with timestamp suffix.",
    )
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)

    output_cfg = cfg.get("output", {})
    scenarios_csv = args.scenarios_csv or output_cfg.get(
        "scenarios_csv",
        cfg.get("simulation", {}).get("scenarios_csv", "data/scenarios.csv"),
    )
    scenarios = pd.read_csv(scenarios_csv)

    sim_cfg = cfg.get("simulation", {})
    warmup = sim_cfg.get("warmup_time", 200)
    collect = sim_cfg.get("collect_time", 400)
    num_probes = sim_cfg.get("num_probes", 5)
    seq_len = cfg.get("data", {}).get("seq_len", 300)

    feature_cfg_path = cfg.get("features", {}).get("config")
    feature_names = None
    if feature_cfg_path:
        feature_names = load_config(feature_cfg_path).get("features")

    fcd_dir = output_cfg.get("fcd_dir", "data/fcd")
    features_dir = output_cfg.get("features_dir", "data/features")
    data_cfg = cfg.get("data", {})
    tabular_path = data_cfg.get("tabular_path", f"{features_dir}/dataset.parquet")
    timeseries_path = data_cfg.get("timeseries_path", f"{features_dir}/timeseries.npz")
    metadata_path = data_cfg.get("metadata_path", f"{features_dir}/metadata.parquet")

    # Resolve workers
    workers_arg = args.workers or sim_cfg.get("max_workers", "auto")
    if workers_arg is None or workers_arg == "auto":
        max_workers = max(1, (os.cpu_count() or 1) - 1)
    else:
        max_workers = int(workers_arg)

    # Incremental mode: load existing data, skip already-extracted scenarios
    old_df = None
    old_seqs = None
    old_meta = None
    existing_ids: set[int] = set()

    if args.incremental:
        old_df, old_seqs, old_meta, existing_ids = _backup_and_load_existing(
            tabular_path, timeseries_path, metadata_path,
        )
        before = len(scenarios)
        scenarios = scenarios[~scenarios["scenario_id"].isin(existing_ids)]
        logger.info(
            "Incremental: %d total scenarios, %d already extracted, %d new to process",
            before,
            before - len(scenarios),
            len(scenarios),
        )
        if scenarios.empty:
            logger.info("No new scenarios to extract. Done.")
            return

    # Build work items (link_length comes from each scenario row)
    work_items = [
        (
            row.to_dict(), fcd_dir, num_probes, warmup, collect,
            seq_len, feature_names,
        )
        for _, row in scenarios.iterrows()
    ]

    logger.info(
        "Extracting features from %d scenarios (%d probes each) with %d workers",
        len(scenarios),
        num_probes,
        max_workers,
    )

    # Process scenarios in parallel, streaming results
    tabular_records: list[dict] = []
    raw_sequences: list[np.ndarray] = []
    meta_records: list[dict] = []

    total = len(work_items)
    done = 0
    extracted = 0

    def _collect(result: tuple | None) -> None:
        nonlocal done, extracted
        done += 1
        if result is not None:
            tab, seqs, metas = result
            tabular_records.extend(tab)
            raw_sequences.extend(seqs)
            meta_records.extend(metas)
            extracted += 1
        if done % 500 == 0 or done == total:
            logger.info(
                "Progress: %d/%d scenarios (%d extracted, %d samples)",
                done,
                total,
                extracted,
                len(raw_sequences),
            )

    if max_workers <= 1:
        for item in work_items:
            _collect(_process_scenario(item))
    else:
        with multiprocessing.Pool(processes=max_workers) as pool:
            for result in pool.imap_unordered(_process_scenario, work_items, chunksize=50):
                _collect(result)

    logger.info(
        "Done: %d/%d scenarios, %d extracted, %d new samples.",
        done, total, extracted, len(raw_sequences),
    )

    if not tabular_records and old_df is None:
        logger.warning("No records extracted.")
        return

    # Build new DataFrames
    new_df = pd.DataFrame(tabular_records) if tabular_records else None
    new_meta = pd.DataFrame(meta_records) if meta_records else None
    new_seqs = (
        np.array(raw_sequences, dtype=np.float32) if raw_sequences else None
    )

    # Merge with existing data if incremental
    if args.incremental and old_df is not None:
        df = pd.concat([old_df, new_df], ignore_index=True) if new_df is not None else old_df
        meta_df = (
            pd.concat([old_meta, new_meta], ignore_index=True)
            if new_meta is not None
            else old_meta
        )
        stacked = (
            np.concatenate([old_seqs, new_seqs], axis=0)
            if new_seqs is not None
            else old_seqs
        )
        logger.info(
            "Merged: %d old + %d new = %d total samples",
            len(old_df),
            len(new_df) if new_df is not None else 0,
            len(df),
        )
    else:
        df = new_df
        meta_df = new_meta
        stacked = new_seqs

    # Save tabular features
    Path(tabular_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tabular_path, index=False)
    logger.info("Saved %d tabular records to %s", len(df), tabular_path)

    # Save metadata
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(metadata_path, index=False)
    logger.info("Saved metadata to %s", metadata_path)

    # Save time series
    targets = meta_df["density"].values.astype(np.float32)
    flow_targets = meta_df["flow"].values.astype(np.float32)
    scenario_ids = meta_df["scenario_id"].values

    Path(timeseries_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        timeseries_path,
        sequences=stacked,  # (N, 6, seq_len)
        density=targets,  # (N,)
        flow=flow_targets,  # (N,)
        scenario_ids=scenario_ids,  # (N,)
    )
    logger.info(
        "Saved %d time series (shape %s) to %s",
        len(stacked),
        stacked.shape,
        timeseries_path,
    )


if __name__ == "__main__":
    main()
