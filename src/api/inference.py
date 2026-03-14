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
    delta_k = float(registry.model.predict(feature_vector)[0])

    speed_mean = float(feats.get("speed_mean", 0.0))
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

    density = k_fd + delta_k
    flow = density * speed_mean * 3.6

    return {
        "density": density,
        "flow": flow,
        "fd_density": k_fd,
        "fd_flow": q_fd,
        "residual_density": delta_k,
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
        5. XGBoost predict → Δk (residual).
        6. Compute Underwood FD baseline → k_fd, q_fd.
        7. Final density = k_fd + Δk.
    """
    # 1. raw FCD → 6-channel trajectory
    raw_df = pd.DataFrame(fcd_records)
    trajectory = build_trajectory(raw_df)

    # 2. extract features
    feats: dict[str, float] = extract_features(trajectory)

    return _predict_from_feature_map(feats, speed_limit, num_lanes, registry)


def predict_density_from_features(
    features: dict[str, float],
    speed_limit: float,
    num_lanes: int,
    registry: ModelRegistry,
) -> dict[str, float]:
    """Run prediction directly from a client-computed feature vector."""
    return _predict_from_feature_map(dict(features), speed_limit, num_lanes, registry)
