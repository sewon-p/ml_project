"""Compute ground-truth traffic state using Edie's generalized definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_edie_density(
    fcd_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    x_start: float = 0.0,
    x_end: float = 1000.0,
) -> float:
    """Edie's generalized density: k = total_time_in_region / (dx * dt).

    Args:
        fcd_df: FCD DataFrame with columns [time, vehicle_id, x, speed].
        t_start, t_end: Time window boundaries (seconds).
        x_start, x_end: Spatial boundaries (meters). Default 1km segment.

    Returns:
        Density in vehicles per kilometer.
    """
    dt = t_end - t_start
    dx = (x_end - x_start) / 1000.0  # convert to km

    if dt <= 0 or dx <= 0:
        return 0.0

    mask = (
        (fcd_df["time"] >= t_start)
        & (fcd_df["time"] < t_end)
        & (fcd_df["x"] >= x_start)
        & (fcd_df["x"] < x_end)
    )
    subset = fcd_df.loc[mask]

    if subset.empty:
        return 0.0

    times = sorted(subset["time"].unique())
    if len(times) < 2:
        step_length = 0.1
    else:
        step_length = times[1] - times[0]

    total_time_spent = len(subset) * step_length  # veh*s
    density = total_time_spent / (dx * dt)  # veh/km
    return float(density)


def compute_edie_flow(
    fcd_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    x_start: float = 0.0,
    x_end: float = 1000.0,
) -> float:
    """Edie's generalized flow: q = total_distance_in_region / (dx * dt).

    Returns:
        Flow in vehicles per hour.
    """
    dt = t_end - t_start
    dx = (x_end - x_start) / 1000.0

    if dt <= 0 or dx <= 0:
        return 0.0

    mask = (
        (fcd_df["time"] >= t_start)
        & (fcd_df["time"] < t_end)
        & (fcd_df["x"] >= x_start)
        & (fcd_df["x"] < x_end)
    )
    subset = fcd_df.loc[mask]

    if subset.empty:
        return 0.0

    times = sorted(subset["time"].unique())
    if len(times) < 2:
        step_length = 0.1
    else:
        step_length = times[1] - times[0]

    total_distance_m = float(np.sum(subset["speed"].values * step_length))
    total_distance_km = total_distance_m / 1000.0

    flow = total_distance_km / (dx * dt) * 3600.0  # veh/hr
    return float(flow)
