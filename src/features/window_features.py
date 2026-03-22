"""Window-feature extraction: (N, 6, 300) raw timeseries → (N, 8, 10) window features.

300 seconds are split into 10 windows of 30 seconds (30 data points each).
Per window, 8 features are computed from the 6-channel data:
  1. speed_mean
  2. speed_std
  3. speed_min
  4. ax_std
  5. brake_time_ratio
  6. stop_time_ratio
  7. slow_duration_ratio  (speed < speed_limit * 0.5)
  8. vy_energy            (sum of VY^2)

Channels in timeseries.npz: [VX=0, VY=1, AX=2, AY=3, speed=4, brake=5]
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Channel indices in (N, 6, seq_len) timeseries
CH_VY = 1
CH_AX = 2
CH_SPEED = 4
CH_BRAKE = 5

N_FEATURES = 8
WINDOW_SEC = 30  # seconds per window


def extract_window_features(
    sequences: np.ndarray,
    speed_limits: np.ndarray,
    window_size: int = WINDOW_SEC,
    exclude: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert raw timeseries to window-feature representation.

    Args:
        sequences: (N, 6, seq_len) raw 6-channel time series.
        speed_limits: (N,) speed limit per sample in m/s.
        window_size: Number of timesteps per window (default 30).

    Returns:
        Tuple of (N, C, n_windows) window-feature tensor and list of used feature names.
    """
    N, _C, seq_len = sequences.shape
    n_windows = seq_len // window_size
    if n_windows == 0:
        raise ValueError(f"seq_len={seq_len} < window_size={window_size}")

    # Trim to exact multiple of window_size
    trimmed = sequences[:, :, : n_windows * window_size]
    # Reshape to (N, 6, n_windows, window_size)
    reshaped = trimmed.reshape(N, 6, n_windows, window_size)

    speed = reshaped[:, CH_SPEED]  # (N, n_windows, window_size)
    ax = reshaped[:, CH_AX]  # (N, n_windows, window_size)
    brake = reshaped[:, CH_BRAKE]  # (N, n_windows, window_size)
    vy = reshaped[:, CH_VY]  # (N, n_windows, window_size)

    ws = float(window_size)

    # 1. speed_mean
    speed_mean = speed.mean(axis=2)  # (N, n_windows)
    # 2. speed_std (ddof=1)
    speed_std = speed.std(axis=2, ddof=1)
    # 3. speed_min
    speed_min = speed.min(axis=2)
    # 4. ax_std (ddof=1)
    ax_std = ax.std(axis=2, ddof=1)
    # 5. brake_time_ratio
    brake_time_ratio = (brake > 0).sum(axis=2) / ws
    # 6. stop_time_ratio (speed < 0.5 m/s)
    stop_time_ratio = (speed < 0.5).sum(axis=2) / ws
    # 7. slow_duration_ratio (speed < speed_limit * 0.5)
    threshold = (speed_limits * 0.5)[:, np.newaxis, np.newaxis]  # (N, 1, 1)
    slow_duration_ratio = (speed < threshold).sum(axis=2) / ws
    # 8. vy_energy (sum of VY^2 per window)
    vy_energy = (vy**2).sum(axis=2)

    # Build feature list, excluding any specified
    all_features = [
        ("speed_mean", speed_mean),
        ("speed_std", speed_std),
        ("speed_min", speed_min),
        ("ax_std", ax_std),
        ("brake_time_ratio", brake_time_ratio),
        ("stop_time_ratio", stop_time_ratio),
        ("slow_duration_ratio", slow_duration_ratio),
        ("vy_energy", vy_energy),
    ]

    exclude_set = set(exclude or [])
    selected = [(n, f) for n, f in all_features if n not in exclude_set]
    if not selected:
        raise ValueError("All window features excluded — nothing to train on")

    used_names = [n for n, _ in selected]
    features = np.stack([f for _, f in selected], axis=1).astype(np.float32)

    logger.info(
        "Window features: %s → %s (%d windows of %ds, %d features: %s)",
        sequences.shape,
        features.shape,
        n_windows,
        window_size,
        len(used_names),
        used_names,
    )
    return features, used_names


FEATURE_NAMES = [
    "speed_mean",
    "speed_std",
    "speed_min",
    "ax_std",
    "brake_time_ratio",
    "stop_time_ratio",
    "slow_duration_ratio",
    "vy_energy",
]
