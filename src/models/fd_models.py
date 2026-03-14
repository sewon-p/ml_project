"""Fundamental Diagram models — speed-to-density inverse functions.

Supported models:
  - greenshields : Linear   v = v_f(1 - k/k_j)
  - greenberg    : Logarithmic  v = c_0 ln(k_j/k)
  - underwood    : Exponential  v = v_f exp(-k/k_m)
  - drake        : Bell-shaped  v = v_f exp(-0.5(k/k_m)^2)
  - multi_regime : Greenshields (free) + Greenberg (congested)

All functions follow the signature:
    compute_fd_density_<name>(speed, v_free, k_jam, k_m) -> density
"""

from __future__ import annotations

import numpy as np

FD_MODEL_NAMES = ["greenshields", "greenberg", "underwood", "drake", "multi_regime"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_k_jam(
    num_lanes: np.ndarray | float,
    vehicle_length: float = 4.5,
    min_gap: float = 2.5,
) -> np.ndarray:
    """Jam density: k_j = num_lanes × 1000 / (L + s)  (veh/km)."""
    return np.asarray(num_lanes, dtype=np.float64) * 1000.0 / (vehicle_length + min_gap)


def compute_k_optimum(k_jam: np.ndarray) -> np.ndarray:
    """Optimal density (density at max flow): k_m = k_j / e."""
    return k_jam / np.e


def compute_fd_flow(speed_mean: np.ndarray, k_fd: np.ndarray) -> np.ndarray:
    """Flow from density and speed: q = k × v × 3.6  (veh/hr)."""
    return np.asarray(k_fd, dtype=np.float64) * np.asarray(speed_mean, dtype=np.float64) * 3.6


# ---------------------------------------------------------------------------
# Individual FD models  (speed → density)
# ---------------------------------------------------------------------------


def _greenshields(
    speed: np.ndarray, v_free: np.ndarray, k_jam: np.ndarray, _k_m: np.ndarray
) -> np.ndarray:
    """Greenshields (linear): k = k_j × (1 - v/v_f)."""
    ratio = np.clip(speed / v_free, 0.0, 1.0)
    return k_jam * (1.0 - ratio)


def _greenberg(
    speed: np.ndarray, v_free: np.ndarray, k_jam: np.ndarray, _k_m: np.ndarray
) -> np.ndarray:
    """Greenberg (logarithmic): k = k_j × exp(-v / c_0), c_0 = v_f / e."""
    c0 = v_free / np.e
    return k_jam * np.exp(-speed / c0)


def _underwood(
    speed: np.ndarray, v_free: np.ndarray, _k_jam: np.ndarray, k_m: np.ndarray
) -> np.ndarray:
    """Underwood (exponential): k = k_m × ln(v_f / v)."""
    ratio = np.clip(speed / v_free, 1e-6, 1.0)
    return -k_m * np.log(ratio)


def _drake(
    speed: np.ndarray, v_free: np.ndarray, _k_jam: np.ndarray, k_m: np.ndarray
) -> np.ndarray:
    """Drake (bell-shaped): k = k_m × sqrt(-2 ln(v / v_f))."""
    ratio = np.clip(speed / v_free, 1e-6, 1.0)
    return k_m * np.sqrt(-2.0 * np.log(ratio))


def _multi_regime(
    speed: np.ndarray, v_free: np.ndarray, k_jam: np.ndarray, k_m: np.ndarray
) -> np.ndarray:
    """Multi-regime: Greenshields (free flow) + Greenberg (congested).

    Split at critical speed v_c = v_f / 2.
    """
    v_c = v_free / 2.0
    k_gs = _greenshields(speed, v_free, k_jam, k_m)
    k_gb = _greenberg(speed, v_free, k_jam, k_m)
    return np.where(speed > v_c, k_gs, k_gb)


_MODEL_MAP = {
    "greenshields": _greenshields,
    "greenberg": _greenberg,
    "underwood": _underwood,
    "drake": _drake,
    "multi_regime": _multi_regime,
}


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def compute_fd_density(
    model_name: str,
    speed_mean: np.ndarray,
    v_free: np.ndarray,
    num_lanes: np.ndarray | float,
    vehicle_length: float = 4.5,
    min_gap: float = 2.5,
) -> dict[str, np.ndarray]:
    """Compute FD density and flow for any supported model.

    Returns dict with keys: k_jam, k_m, k_fd, q_fd.
    """
    speed_mean = np.asarray(speed_mean, dtype=np.float64)
    v_free = np.asarray(v_free, dtype=np.float64)

    fn = _MODEL_MAP.get(model_name)
    if fn is None:
        raise ValueError(f"Unknown FD model '{model_name}'. Choose from: {FD_MODEL_NAMES}")

    k_jam = compute_k_jam(num_lanes, vehicle_length, min_gap)
    k_m = compute_k_optimum(k_jam)

    k_fd = fn(speed_mean, v_free, k_jam, k_m)
    # Clamp density to non-negative
    k_fd = np.maximum(k_fd, 0.0)
    q_fd = compute_fd_flow(speed_mean, k_fd)

    return {"k_jam": k_jam, "k_m": k_m, "k_fd": k_fd, "q_fd": q_fd}
