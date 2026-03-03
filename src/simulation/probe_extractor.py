"""Select a single probe vehicle per segment and extract 6-channel trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_single_probe(
    fcd_df: pd.DataFrame,
    x_start: float,
    x_end: float,
    t_start: float | None = None,
    t_end: float | None = None,
    seed: int = 42,
) -> str | None:
    """Select one vehicle that traverses the segment [x_start, x_end].

    Picks the vehicle with the most data points in the segment.
    Ties are broken randomly using *seed*.
    """
    df = fcd_df.copy()
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_end is not None:
        df = df[df["time"] < t_end]

    mask = (df["x"] >= x_start) & (df["x"] < x_end)
    segment_df = df[mask]

    if segment_df.empty:
        return None

    counts = segment_df.groupby("vehicle_id").size()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()

    rng = np.random.RandomState(seed)
    return str(rng.choice(candidates))


def extract_probe_trajectory(
    fcd_df: pd.DataFrame,
    vehicle_id: str,
    x_start: float,
    x_end: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> pd.DataFrame:
    """Extract 6-channel trajectory for a single vehicle within a segment.

    Returns DataFrame with columns: VX, VY, AX, AY, speed, brake.
    VX and VY are computed from position differences.
    AX and AY are computed from velocity differences.
    brake is taken directly from TraCI getSignals() if available,
    otherwise estimated from AX < -0.5 m/s².
    """
    df = fcd_df[fcd_df["vehicle_id"] == vehicle_id].copy()
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_end is not None:
        df = df[df["time"] < t_end]

    mask = (df["x"] >= x_start) & (df["x"] < x_end)
    df = df[mask].sort_values("time").reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame(columns=["VX", "VY", "AX", "AY", "speed", "brake"])

    has_brake = "brake" in df.columns

    times = df["time"].values
    dt = np.diff(times)
    dt[dt == 0] = 1e-6  # avoid division by zero

    # Velocity components from position differences
    vx = np.diff(df["x"].values) / dt
    vy = np.diff(df["y"].values) / dt

    # Use the speed column from FCD (magnitude)
    speed = df["speed"].values[1:]  # align with diff

    # Acceleration from velocity differences
    if len(vx) >= 2:
        dt2 = dt[1:]
        dt2[dt2 == 0] = 1e-6
        ax = np.diff(vx) / dt2
        ay = np.diff(vy) / dt2
        # Trim to match acceleration length
        vx = vx[1:]
        vy = vy[1:]
        speed = speed[1:]
    else:
        return pd.DataFrame(columns=["VX", "VY", "AX", "AY", "speed", "brake"])

    n = min(len(vx), len(ax))

    # Brake signal: use TraCI data if available, otherwise estimate
    if has_brake:
        # Align brake with acceleration-trimmed indices (skip first 2 rows)
        brake = df["brake"].values[2 : 2 + n].astype(float)
    else:
        brake = (ax[:n] < -0.5).astype(float)

    trajectory = pd.DataFrame(
        {
            "VX": vx[:n],
            "VY": vy[:n],
            "AX": ax[:n],
            "AY": ay[:n],
            "speed": speed[:n],
            "brake": brake,
        }
    )
    return trajectory


def get_segment_boundaries(
    link_length: float = 5000.0,
    segment_length: float = 1000.0,
) -> list[tuple[float, float]]:
    """Return list of (x_start, x_end) for each segment."""
    segments = []
    x = 0.0
    while x + segment_length <= link_length + 1e-6:
        segments.append((x, x + segment_length))
        x += segment_length
    return segments
