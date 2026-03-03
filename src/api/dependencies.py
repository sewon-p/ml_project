"""Model registry — loads XGBoost model and feature metadata at startup."""

from __future__ import annotations

import pyarrow.parquet as pq

from src.models.tabular import XGBoostEstimator
from src.utils.config import load_config

# Columns that are metadata/targets, not features.
_EXCLUDE_COLUMNS = {
    "scenario_id",
    "probe_idx",
    "density",
    "flow",
    "demand_vehph",
    "k_fd",
    "q_fd",
    "delta_density",
    "delta_flow",
}


class ModelRegistry:
    """Singleton loaded at startup — holds model + feature column order."""

    def __init__(self, config_path: str) -> None:
        cfg = load_config(config_path)

        # --- model ---
        api_cfg = cfg.get("api", {})
        self.model_path = api_cfg.get(
            "model_path", cfg.get("output_dir", "outputs/") + "xgboost_best.pkl"
        )
        self.model = XGBoostEstimator.load(self.model_path)

        # --- feature column order (from training parquet schema) ---
        tabular_path = cfg.get("data", {}).get("tabular_path", "data/features/dataset.parquet")
        schema = pq.read_schema(tabular_path)
        all_columns = [f.name for f in schema]
        self.feature_columns: list[str] = [c for c in all_columns if c not in _EXCLUDE_COLUMNS]

        # --- residual correction config ---
        rc = cfg.get("residual_correction", {})
        self.residual_enabled: bool = rc.get("enabled", True)
        self.v_free_factor: float = rc.get("v_free_factor", 1.1)
        self.vehicle_length: float = rc.get("vehicle_length", 4.5)
        self.min_gap: float = rc.get("min_gap", 2.5)

        # --- features to drop (same as extract_features.py) ---
        self.features_drop: set[str] = {
            "vx_mean",
            "vx_std",
            "vx_min",
            "vx_max",
            "vx_autocorr_lag1",
            "vx_fft_dominant_freq",
            "harsh_accel_count",
            "harsh_decel_count",
            "lane_change_count",
        }
