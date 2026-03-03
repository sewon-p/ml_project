"""Fundamental diagram plots: k-q and k-v scatter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_kv_diagram(
    density: np.ndarray,
    speed: np.ndarray,
    title: str = "Speed-Density (k-v)",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of density vs speed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(density, speed, alpha=0.3, s=10)
    ax.set_xlabel("Density (veh/km)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_kq_diagram(
    density: np.ndarray,
    flow: np.ndarray,
    title: str = "Flow-Density (k-q)",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of density vs flow."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(density, flow, alpha=0.3, s=10)
    ax.set_xlabel("Density (veh/km)")
    ax.set_ylabel("Flow (veh/hr)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
