"""Traffic state classification based on speed thresholds."""

from __future__ import annotations

import numpy as np


def classify_traffic_state(
    speeds: np.ndarray,
    congested_threshold: float = 10.0,
    saturated_threshold: float = 20.0,
) -> np.ndarray:
    """Classify traffic state based on speed thresholds.

    Returns array of strings: "congested", "saturated", or "free".
    - speed < congested_threshold -> "congested"
    - congested_threshold <= speed < saturated_threshold -> "saturated"
    - speed >= saturated_threshold -> "free"
    """
    states = np.empty(len(speeds), dtype=object)
    states[speeds < congested_threshold] = "congested"
    states[(speeds >= congested_threshold) & (speeds < saturated_threshold)] = "saturated"
    states[speeds >= saturated_threshold] = "free"
    return states
