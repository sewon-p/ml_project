"""Acceleration features derived from AX, AY channels of single-vehicle trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: I001

from src.features.registry import register_feature

# --- AX (longitudinal acceleration) ---


@register_feature("ax_mean")
def ax_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AX"].values
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


@register_feature("ax_std")
def ax_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AX"].values
    if len(vals) == 0:
        return 0.0
    return float(np.std(vals))


@register_feature("harsh_accel_count")
def harsh_accel_count(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AX"].values
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals > 3.0))


@register_feature("harsh_decel_count")
def harsh_decel_count(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AX"].values
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals < -3.0))


# --- AY (lateral acceleration) ---


@register_feature("ay_mean")
def ay_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AY"].values
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


@register_feature("ay_std")
def ay_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    vals = trajectory["AY"].values
    if len(vals) == 0:
        return 0.0
    return float(np.std(vals))


# --- Jerk (derivative of AX) ---


@register_feature("jerk_mean")
def jerk_mean(trajectory: pd.DataFrame, **kwargs: object) -> float:
    dt = float(kwargs.get("dt", 1.0))
    ax = trajectory["AX"].values
    if len(ax) < 2:
        return 0.0
    jerk = np.diff(ax) / dt
    return float(np.mean(jerk))


@register_feature("jerk_std")
def jerk_std(trajectory: pd.DataFrame, **kwargs: object) -> float:
    dt = float(kwargs.get("dt", 1.0))
    ax = trajectory["AX"].values
    if len(ax) < 2:
        return 0.0
    jerk = np.diff(ax) / dt
    return float(np.std(jerk))
