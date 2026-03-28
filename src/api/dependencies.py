"""Model registry — loads XGBoost model and feature metadata at startup."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from src.models.tabular import XGBoostEstimator
from src.utils.config import load_config

# Columns that are metadata/targets, not features.
_EXCLUDE_COLUMNS = {
    "scenario_id",
    "probe_idx",
    "density",
    "flow",
    "density_per_lane",
    "flow_per_lane",
    "demand_vehph",
    "traversal_time",
    "k_fd",
    "q_fd",
    "delta_density",
    "delta_flow",
    "gap_mean",
}


class ModelRegistry:
    """Singleton loaded at startup — holds model + feature column order."""

    def __init__(self, config_path: str) -> None:
        cfg = load_config(config_path)
        config_root = Path(config_path).resolve().parent.parent

        # --- model ---
        api_cfg = cfg.get("api", {})
        self.model_path = api_cfg.get(
            "model_path", cfg.get("output_dir", "outputs/") + "xgboost_best.pkl"
        )
        self.model = XGBoostEstimator.load(self.model_path)

        # --- feature column order (from training parquet schema) ---
        feature_columns_path = api_cfg.get("feature_columns_path")
        if feature_columns_path:
            path = Path(feature_columns_path)
            if not path.is_absolute():
                path = (config_root / path).resolve()
            with open(path, encoding="utf-8") as f:
                self.feature_columns = json.load(f)
        else:
            tabular_path = cfg.get("data", {}).get("tabular_path", "data/features/dataset.parquet")
            try:
                schema = pq.read_schema(tabular_path)
                all_columns = [f.name for f in schema]
                self.feature_columns = [c for c in all_columns if c not in _EXCLUDE_COLUMNS]
            except Exception:
                # Fallback: known 32 feature columns matching the deployed model
                self.feature_columns = [
                    "speed_mean", "speed_std", "speed_cv", "speed_iqr",
                    "speed_min", "speed_max", "speed_median", "speed_p10", "speed_p90",
                    "vy_mean", "vy_std", "vy_min", "vy_max",
                    "ax_mean", "ax_std", "ay_mean", "ay_std",
                    "jerk_mean", "jerk_std",
                    "stop_count", "stop_time_ratio", "mean_stop_duration",
                    "speed_autocorr_lag1", "speed_fft_dominant_freq", "sample_entropy",
                    "brake_count", "brake_time_ratio", "mean_brake_duration",
                    "vy_variance", "vy_energy",
                    "num_lanes", "speed_limit",
                ]

        # --- residual correction config ---
        rc = cfg.get("residual_correction", {})
        self.residual_enabled: bool = rc.get("enabled", True)
        self.fd_model: str = rc.get("fd_model", "underwood")
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
