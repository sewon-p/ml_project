"""Descriptive statistics for single-vehicle trajectory channels (VX, VY, speed)."""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: I001

from src.features.registry import register_feature

# --- speed channel ---


@register_feature("speed_mean")
def speed_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.mean(trajectory["speed"].values))


@register_feature("speed_std")
def speed_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["speed"].values
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


@register_feature("speed_cv")
def speed_cv(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["speed"].values
    m = float(np.mean(vals))
    if m == 0.0:
        return 0.0
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return s / m


@register_feature("speed_iqr")
def speed_iqr(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["speed"].values
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


@register_feature("speed_min")
def speed_min(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.min(trajectory["speed"].values))


@register_feature("speed_max")
def speed_max(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.max(trajectory["speed"].values))


@register_feature("speed_median")
def speed_median(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.median(trajectory["speed"].values))


@register_feature("speed_p10")
def speed_p10(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.percentile(trajectory["speed"].values, 10))


@register_feature("speed_p90")
def speed_p90(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.percentile(trajectory["speed"].values, 90))


# --- VX channel ---


@register_feature("vx_mean")
def vx_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.mean(trajectory["VX"].values))


@register_feature("vx_std")
def vx_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["VX"].values
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


@register_feature("vx_min")
def vx_min(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.min(trajectory["VX"].values))


@register_feature("vx_max")
def vx_max(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.max(trajectory["VX"].values))


# --- VY channel ---


@register_feature("vy_mean")
def vy_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.mean(trajectory["VY"].values))


@register_feature("vy_std")
def vy_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["VY"].values
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


@register_feature("vy_min")
def vy_min(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.min(trajectory["VY"].values))


@register_feature("vy_max")
def vy_max(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return float(np.max(trajectory["VY"].values))
