"""Inference helpers for raw-FCD and feature-only prediction paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.api.dependencies import ModelRegistry
from src.data.preprocessing import build_trajectory
from src.features.pipeline import extract_features
from src.models.fd_models import compute_fd_density


def _predict_from_feature_map(
    feats: dict[str, float],
    speed_limit: float,
    num_lanes: int,
    registry: ModelRegistry,
) -> dict[str, float]:
    """Run XGBoost + FD correction from an already prepared feature mapping."""
    for key in registry.features_drop:
        feats.pop(key, None)
    feats["num_lanes"] = float(num_lanes)
    feats["speed_limit"] = float(speed_limit)

    feature_vector = np.array(
        [[feats.get(col, 0.0) for col in registry.feature_columns]],
        dtype=np.float64,
    )
    density_total = float(registry.model.predict(feature_vector)[0])
    density = density_total / max(num_lanes, 1)  # per-lane

    speed_mean = float(feats.get("speed_mean", 0.0))
    flow = density * speed_mean * 3.6

    # FD baseline for reference (not used in prediction)
    v_free = speed_limit * registry.v_free_factor
    fd = compute_fd_density(
        registry.fd_model,
        np.array(speed_mean),
        np.array(v_free),
        num_lanes,
        vehicle_length=registry.vehicle_length,
        min_gap=registry.min_gap,
    )
    k_fd = float(fd["k_fd"])
    q_fd = float(fd["q_fd"])

    return {
        "density": density,
        "flow": flow,
        "fd_density": k_fd,
        "fd_flow": q_fd,
        "residual_density": density - k_fd,
    }


def predict_density(
    fcd_records: list[dict],
    speed_limit: float,
    num_lanes: int,
    registry: ModelRegistry,
) -> dict[str, float]:
    """Run the full inference pipeline and return predictions.

    Steps:
        1. Build 6-channel trajectory from raw FCD records.
        2. Extract scalar features via the feature registry.
        3. Drop redundant features, add num_lanes / speed_limit.
        4. Align to training column order → numpy array.
        5. XGBoost predict → density (direct).
        6. Compute Underwood FD baseline → k_fd, q_fd (reference only).
        7. Final density = XGBoost output.
    """
    # 1. raw FCD → 6-channel trajectory
    raw_df = pd.DataFrame(fcd_records)
    trajectory = build_trajectory(raw_df)

    # 2. extract features
    feats: dict[str, float] = extract_features(trajectory)

    return _predict_from_feature_map(feats, speed_limit, num_lanes, registry)


def predict_density_from_traversal(
    traversal: dict,  # type: ignore[type-arg]
    registry: ModelRegistry,
) -> dict[str, float]:
    """Run inference on a link-based traversal (variable length).

    Accepts a LinkTraversal dict with fcd_records from one or more
    consecutive links. The statistical features (mean, std, ratio, etc.)
    are naturally length-invariant, so the same model handles 30s and 120s
    trajectories. link_length_m and traversal_time are injected as
    additional features if the model was trained with them.

    Also returns the CF score for downstream ensemble weighting.
    """
    fcd_records: list[dict] = list(traversal.get("fcd_records", []))  # type: ignore[arg-type]
    speed_limit = float(traversal.get("speed_limit", 22.22))
    num_lanes = int(traversal.get("num_lanes", 2))
    link_length_m = float(traversal.get("total_distance_m", 0.0))
    traversal_time = float(traversal.get("traversal_time", 0.0))

    # Guard: too few records
    if len(fcd_records) < 10:
        return {
            "density": 0.0,
            "flow": 0.0,
            "fd_density": 0.0,
            "fd_flow": 0.0,
            "residual_density": 0.0,
            "cf_score": 0.0,
        }

    # 1. Build trajectory from FCD
    raw_df = pd.DataFrame(fcd_records)

    # If raw data already has 6-channel columns (VX,VY,AX,AY,speed,brake),
    # use directly. Otherwise go through build_trajectory.
    from src.data.preprocessing import CHANNELS

    if all(c in raw_df.columns for c in CHANNELS):
        trajectory = raw_df[CHANNELS].copy()
    else:
        trajectory = build_trajectory(raw_df)

    # 1b. Resample to 100 timesteps (match training data from 1km npz)
    _TARGET_TS = 100
    if len(trajectory) > _TARGET_TS:
        idx = np.linspace(0, len(trajectory) - 1, _TARGET_TS, dtype=int)
        trajectory = trajectory.iloc[idx].reset_index(drop=True)
    elif len(trajectory) < _TARGET_TS:
        pad = pd.concat([trajectory.iloc[[-1]]] * (_TARGET_TS - len(trajectory)), ignore_index=True)
        trajectory = pd.concat([trajectory, pad], ignore_index=True)

    # 2. Extract features (length-invariant statistics)
    feats: dict[str, float] = extract_features(trajectory, speed_limit=speed_limit)

    # 3. Compute CF score before dropping features
    cf_score = (
        feats.get("ax_std", 0.0) + feats.get("brake_time_ratio", 0.0) + feats.get("speed_cv", 0.0)
    )

    # 4. Inject traversal metadata as features (if model supports them)
    feats["link_length_m"] = link_length_m
    feats["traversal_time"] = traversal_time

    # 5. Standard prediction pipeline
    result = _predict_from_feature_map(feats, speed_limit, num_lanes, registry)
    result["cf_score"] = cf_score
    result["cf_features"] = {  # type: ignore[assignment]
        "ax_std": feats.get("ax_std", 0.0),
        "brake_time_ratio": feats.get("brake_time_ratio", 0.0),
        "speed_cv": feats.get("speed_cv", 0.0),
    }
    result["link_length_m"] = link_length_m
    result["traversal_time"] = traversal_time

    return result


def predict_density_from_features(
    features: dict[str, float],
    speed_limit: float,
    num_lanes: int,
    registry: ModelRegistry,
) -> dict[str, float]:
    """Run prediction directly from a client-computed feature vector."""
    return _predict_from_feature_map(dict(features), speed_limit, num_lanes, registry)
