"""Brake-pattern features from single-vehicle trajectory brake channel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _brake_segments(brake: np.ndarray) -> list[int]:
    """Return durations (timesteps) of each contiguous braking segment."""
    if len(brake) == 0:
        return []

    is_braking = brake > 0
    segments: list[int] = []
    current_len = 0

    for b in is_braking:
        if b:
            current_len += 1
        else:
            if current_len > 0:
                segments.append(current_len)
                current_len = 0
    if current_len > 0:
        segments.append(current_len)

    return segments


@register_feature("brake_count")
def brake_count(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Number of distinct braking events."""
    segments = _brake_segments(trajectory["brake"].values)
    return float(len(segments))


@register_feature("brake_time_ratio")
def brake_time_ratio(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Fraction of timesteps where braking is active."""
    brake = trajectory["brake"].values
    if len(brake) == 0:
        return 0.0
    return float(np.sum(brake > 0) / len(brake))


@register_feature("mean_brake_duration")
def mean_brake_duration(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Average duration (timesteps) of braking events."""
    segments = _brake_segments(trajectory["brake"].values)
    if len(segments) == 0:
        return 0.0
    return float(np.mean(segments))
