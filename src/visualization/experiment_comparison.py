"""Model comparison bar charts across experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str | None = None,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bar chart comparing models across experiments.

    Args:
        results_df: DataFrame with columns
            [experiment, model, <metric>].
        metric: Metric column name to plot.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    pivot = results_df.pivot(index="experiment", columns="model", values=metric)
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model Comparison: {metric.upper()}")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
