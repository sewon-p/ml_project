"""Underwood (exponential) Fundamental Diagram — pure function module.

u = u_f × exp(-k / k_m)   →   k = k_m × ln(u_f / u)

Greenshields(선형)는 자유류 근처에서 k≈0으로 붕괴하지만,
Underwood(지수)는 중간대 추정이 훨씬 안정적이다.

Used by prepare_residuals.py to pre-compute k_fd, q_fd, delta_density, delta_flow.
"""

from __future__ import annotations

import numpy as np


def compute_k_optimum(
    num_lanes: np.ndarray | float,
    vehicle_length: float = 4.5,
    min_gap: float = 2.5,
) -> np.ndarray:
    """Optimal density (density at max flow).

    k_m = k_jam / e, where k_jam = num_lanes × 1000 / (vehicle_length + min_gap).
    Units: veh/km.
    """
    k_jam = np.asarray(num_lanes) * 1000.0 / (vehicle_length + min_gap)
    return k_jam / np.e


def compute_fd_density(
    speed_mean: np.ndarray,
    v_free: np.ndarray,
    k_m: np.ndarray,
) -> np.ndarray:
    """Underwood density: k_fd = k_m × ln(v_f / v̄).

    Speed is clipped to [ε, v_f] to avoid ln(0) and negative density.

    Args:
        speed_mean: mean probe speed (m/s).
        v_free: free-flow speed per sample (m/s).
        k_m: optimal density per sample (veh/km).
    """
    speed_mean = np.asarray(speed_mean, dtype=np.float64)
    v_free = np.asarray(v_free, dtype=np.float64)
    k_m = np.asarray(k_m, dtype=np.float64)
    ratio = np.clip(speed_mean / v_free, 1e-6, 1.0)
    return -k_m * np.log(ratio)


def compute_fd_flow(
    speed_mean: np.ndarray,
    k_fd: np.ndarray,
) -> np.ndarray:
    """FD flow: q_fd = k_fd × v̄ × 3.6.

    Converts m/s speed to km/h so result is veh/hr.
    """
    speed_mean = np.asarray(speed_mean, dtype=np.float64)
    k_fd = np.asarray(k_fd, dtype=np.float64)
    return k_fd * speed_mean * 3.6


def compute_fd_estimates(
    speed_mean: np.ndarray,
    v_free: np.ndarray,
    num_lanes: np.ndarray | float,
    vehicle_length: float = 4.5,
    min_gap: float = 2.5,
) -> dict[str, np.ndarray]:
    """Compute all Underwood FD estimates in one call.

    Returns:
        dict with keys: k_m, k_fd, q_fd
    """
    k_m = compute_k_optimum(num_lanes, vehicle_length, min_gap)
    k_fd = compute_fd_density(speed_mean, v_free, k_m)
    q_fd = compute_fd_flow(speed_mean, k_fd)
    return {"k_m": k_m, "k_fd": k_fd, "q_fd": q_fd}
