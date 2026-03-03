"""Lateral dynamics features from VY channel for lane-change estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature


@register_feature("vy_variance")
def vy_variance(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Variance of lateral velocity (VY)."""
    vals = trajectory["VY"].values
    if len(vals) == 0:
        return 0.0
    return float(np.var(vals))


@register_feature("lane_change_count")
def lane_change_count(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Estimated lane-change count based on VY sign changes.

    A lane change is detected when VY crosses zero (sign change),
    filtered by a minimum magnitude threshold to avoid noise.
    """
    vy = trajectory["VY"].values
    if len(vy) < 2:
        return 0.0
    threshold = float(kwargs.get("lane_change_vy_threshold", 0.3))
    significant = np.abs(vy) > threshold
    sign = np.sign(vy)
    changes = 0
    for i in range(1, len(sign)):
        if significant[i] and significant[i - 1] and sign[i] != sign[i - 1] and sign[i] != 0:
            changes += 1
    return float(changes)


@register_feature("vy_energy")
def vy_energy(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Sum of squared lateral velocity (energy measure)."""
    vals = trajectory["VY"].values
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals**2))
