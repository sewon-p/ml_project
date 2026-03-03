"""Feature importance and SHAP visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_feature_importance_bar(
    importance: dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance (mean |SHAP|)",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar chart of feature importances."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in reversed(sorted_items)]
    values = [x[1] for x in reversed(sorted_items)]
    ax.barh(names, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_shap_summary(
    shap_values: Any,
    save_path: str | Path | None = None,
) -> None:
    """SHAP beeswarm summary plot.

    Args:
        shap_values: shap.Explanation object.
        save_path: Optional path to save the figure.
    """
    import shap

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
