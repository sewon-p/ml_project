"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_trajectory(
    n: int,
    speed_base: float = 25.0,
    speed_noise: float = 1.0,
    vy_scale: float = 0.1,
    brake_ratio: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper to create a synthetic trajectory DataFrame."""
    rng = np.random.RandomState(seed)
    speed = np.clip(speed_base + rng.randn(n) * speed_noise, 0.0, None)
    vx = speed + rng.randn(n) * 0.1
    vy = rng.randn(n) * vy_scale
    ax = np.zeros(n)
    ax[1:] = np.diff(vx)
    ay = np.zeros(n)
    ay[1:] = np.diff(vy)
    brake = np.zeros(n)
    if brake_ratio > 0:
        n_brake = max(1, int(n * brake_ratio))
        brake_idx = rng.choice(n, size=n_brake, replace=False)
        brake[brake_idx] = 1.0
    return pd.DataFrame(
        {
            "VX": vx,
            "VY": vy,
            "AX": ax,
            "AY": ay,
            "speed": speed,
            "brake": brake,
        }
    )


@pytest.fixture
def free_flow_trajectory():
    """Synthetic free-flow trajectory (~25 m/s with small noise)."""
    return _make_trajectory(100, speed_base=25.0, speed_noise=1.0)


@pytest.fixture
def congested_trajectory():
    """Synthetic congested trajectory (~5 m/s with stops and braking)."""
    rng = np.random.RandomState(42)
    n = 100
    speed = np.clip(5.0 + rng.randn(n) * 2.0, 0.0, None)
    speed[30:40] = 0.0  # stopped segment
    speed[70:75] = 0.0  # another stop
    vx = speed + rng.randn(n) * 0.1
    vy = rng.randn(n) * 0.05
    ax = np.zeros(n)
    ax[1:] = np.diff(vx)
    ay = np.zeros(n)
    ay[1:] = np.diff(vy)
    brake = np.zeros(n)
    brake[28:40] = 1.0  # braking into stop
    brake[68:75] = 1.0  # braking into stop
    return pd.DataFrame(
        {
            "VX": vx,
            "VY": vy,
            "AX": ax,
            "AY": ay,
            "speed": speed,
            "brake": brake,
        }
    )


@pytest.fixture
def constant_trajectory():
    """Constant speed trajectory for edge case testing."""
    n = 50
    return pd.DataFrame(
        {
            "VX": np.full(n, 20.0),
            "VY": np.zeros(n),
            "AX": np.zeros(n),
            "AY": np.zeros(n),
            "speed": np.full(n, 20.0),
            "brake": np.zeros(n),
        }
    )
