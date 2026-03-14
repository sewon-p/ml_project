"""Regression metrics for traffic density estimation."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
    min_denominator: float = 1.0,
) -> float:
    """Mean Absolute Percentage Error (%).

    Filters out samples where y_true <= min_denominator to avoid
    inflated percentages from near-zero denominators (e.g. density < 1 veh/km).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > min_denominator
    if not np.any(mask):
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    return float(np.mean(np.abs((yt - yp) / (yt + epsilon))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_min_denominator: float = 1.0,
) -> dict[str, float]:
    """Compute all metrics at once."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred, min_denominator=mape_min_denominator),
        "r2": r2_score(y_true, y_pred),
    }
