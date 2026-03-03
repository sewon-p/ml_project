"""Probe count vs prediction error curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_probe_ratio_vs_rmse(
    probe_ratios: list[float],
    rmse_values: list[float],
    title: str = "Probe Ratio vs RMSE",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line plot showing how RMSE changes with probe vehicle ratio."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(probe_ratios, rmse_values, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Probe Ratio")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
