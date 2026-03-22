"""Stop-pattern features derived from single-vehicle trajectory speed channel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _stop_segments(
    speeds: np.ndarray,
    stop_threshold: float = 0.5,
) -> list[int]:
    """Return a list of durations (in timesteps) for each contiguous stop segment."""
    if len(speeds) == 0:
        return []

    is_stopped = speeds < stop_threshold
    segments: list[int] = []
    current_len = 0

    for stopped in is_stopped:
        if stopped:
            current_len += 1
        else:
            if current_len > 0:
                segments.append(current_len)
                current_len = 0
    if current_len > 0:
        segments.append(current_len)

    return segments


@register_feature("stop_count")
def stop_count(trajectory: pd.DataFrame, **kwargs: object) -> float:
    stop_threshold = float(kwargs.get("stop_threshold", 0.5))
    speeds = trajectory["speed"].values
    return float(len(_stop_segments(speeds, stop_threshold)))


@register_feature("stop_time_ratio")
def stop_time_ratio(trajectory: pd.DataFrame, **kwargs: object) -> float:
    stop_threshold = float(kwargs.get("stop_threshold", 0.5))
    speeds = trajectory["speed"].values
    if len(speeds) == 0:
        return 0.0
    return float(np.sum(speeds < stop_threshold) / len(speeds))


@register_feature("mean_stop_duration")
def mean_stop_duration(trajectory: pd.DataFrame, **kwargs: object) -> float:
    stop_threshold = float(kwargs.get("stop_threshold", 0.5))
    speeds = trajectory["speed"].values
    segments = _stop_segments(speeds, stop_threshold)
    if len(segments) == 0:
        return 0.0
    return float(np.mean(segments))


@register_feature("slow_duration_ratio")
def slow_duration_ratio(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Fraction of time spent below half the speed limit."""
    speeds = trajectory["speed"].values
    speed_limit = float(kwargs.get("speed_limit", 13.89))  # m/s default ~50km/h
    if len(speeds) == 0:
        return 0.0
    return float(np.sum(speeds < speed_limit * 0.5) / len(speeds))
