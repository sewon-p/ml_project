"""Pre-compute Underwood FD estimates and residual targets.

Reads metadata.parquet + dataset.parquet + timeseries.npz, computes
k_fd, q_fd, delta_density, delta_flow, and writes them back into the
same data files. Idempotent — safe to re-run.

Run after extract_features.py and before train.py:
    python scripts/prepare_residuals.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from src.data.io import read_parquet
from src.models.underwood import compute_fd_estimates
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FD estimates and residual targets.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    rc_cfg = cfg.get("residual_correction", {})

    vehicle_length = rc_cfg.get("vehicle_length", 4.5)
    min_gap = rc_cfg.get("min_gap", 2.5)

    metadata_path = data_cfg.get("metadata_path", "data/features/metadata.parquet")
    tabular_path = data_cfg.get("tabular_path", "data/features/dataset.parquet")
    ts_path = data_cfg.get("timeseries_path", "data/features/timeseries.npz")

    # --- Load metadata (source of per-sample speed_limit, num_lanes) ---
    meta_df = read_parquet(metadata_path)
    logger.info("Loaded metadata: %d rows", len(meta_df))

    speed_limit = meta_df["speed_limit"].values  # m/s
    num_lanes = meta_df["num_lanes"].values

    # v_free = speed_limit × factor (제한속도에서도 k>0 보장)
    v_free_factor = rc_cfg.get("v_free_factor", 1.1)
    v_free = speed_limit * v_free_factor
    logger.info("v_free_factor=%.2f (v_free = speed_limit × %.2f)", v_free_factor, v_free_factor)

    # --- Compute FD estimates ---
    tab_df = read_parquet(tabular_path)
    logger.info("Loaded tabular dataset: %d rows", len(tab_df))

    assert len(meta_df) == len(tab_df), (
        f"Row count mismatch: metadata={len(meta_df)}, tabular={len(tab_df)}"
    )

    speed_mean = tab_df["speed_mean"].values
    fd = compute_fd_estimates(
        speed_mean, v_free, num_lanes,
        vehicle_length=vehicle_length, min_gap=min_gap,
    )
    k_fd = fd["k_fd"].astype(np.float32)
    q_fd = fd["q_fd"].astype(np.float32)

    # --- Compute residuals ---
    density_actual = tab_df["density"].values.astype(np.float32)
    flow_actual = tab_df["flow"].values.astype(np.float32)
    delta_density = (density_actual - k_fd).astype(np.float32)
    delta_flow = (flow_actual - q_fd).astype(np.float32)

    logger.info(
        "FD stats — k_fd: mean=%.2f, std=%.2f | q_fd: mean=%.2f, std=%.2f",
        k_fd.mean(), k_fd.std(), q_fd.mean(), q_fd.std(),
    )
    logger.info(
        "Residual stats — Δk: mean=%.2f, std=%.2f | Δq: mean=%.2f, std=%.2f",
        delta_density.mean(), delta_density.std(),
        delta_flow.mean(), delta_flow.std(),
    )

    # --- Update tabular dataset ---
    tab_df["k_fd"] = k_fd
    tab_df["q_fd"] = q_fd
    tab_df["delta_density"] = delta_density
    tab_df["delta_flow"] = delta_flow
    tab_df.to_parquet(tabular_path, index=False)
    logger.info("Updated tabular dataset: %s", tabular_path)

    # --- Update timeseries npz ---
    data = np.load(ts_path)
    assert len(data["sequences"]) == len(k_fd), (
        f"Row count mismatch: npz={len(data['sequences'])}, fd={len(k_fd)}"
    )

    # Re-save with additional arrays
    arrays = {key: data[key] for key in data.files}
    arrays["k_fd"] = k_fd
    arrays["q_fd"] = q_fd
    arrays["delta_density"] = delta_density
    arrays["delta_flow"] = delta_flow
    np.savez_compressed(ts_path, **arrays)
    logger.info("Updated timeseries npz: %s", ts_path)

    logger.info("Residual preparation complete.")


if __name__ == "__main__":
    main()
