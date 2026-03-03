"""Predicted vs actual and residual plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    target_label: str = "Density (veh/km)",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of predicted vs actual values with 1:1 line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Residual plot (actual - predicted) vs actual."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    residuals = y_true - y_pred
    ax.scatter(y_true, residuals, alpha=0.3, s=10)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
