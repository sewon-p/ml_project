"""Inference logic: raw FCD → build_trajectory → extract_features → predict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.api.dependencies import ModelRegistry
from src.data.preprocessing import build_trajectory
from src.features.pipeline import extract_features
from src.models.underwood import compute_fd_estimates


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

    # 3. drop redundant features, add context columns
    for key in registry.features_drop:
        feats.pop(key, None)
    feats["num_lanes"] = float(num_lanes)
    feats["speed_limit"] = float(speed_limit)

    # 4. align to training column order
    feature_vector = np.array(
        [[feats.get(col, 0.0) for col in registry.feature_columns]],
        dtype=np.float64,
    )

    # 5. predict residual
    delta_k = float(registry.model.predict(feature_vector)[0])

    # 6. FD baseline
    speed_mean = float(np.mean(trajectory["speed"].values))
    v_free = speed_limit * registry.v_free_factor
    fd = compute_fd_estimates(
        np.array(speed_mean),
        np.array(v_free),
        num_lanes,
        vehicle_length=registry.vehicle_length,
        min_gap=registry.min_gap,
    )
    k_fd = float(fd["k_fd"])
    q_fd = float(fd["q_fd"])

    # 7. final predictions
    density = k_fd + delta_k
    flow = density * speed_mean * 3.6  # veh/km × m/s × 3.6 → veh/hr

    return {
        "density": density,
        "flow": flow,
        "fd_density": k_fd,
        "fd_flow": q_fd,
        "residual_density": delta_k,
    }
