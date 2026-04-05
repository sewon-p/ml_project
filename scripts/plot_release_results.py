"""Render the latest README results figures.

Builds two standalone figures:
1. aligned same-slice full-feature rerun by probe count
2. deployable link-level fusion comparison by probe count

Usage:
    python scripts/plot_release_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
ALIGNED_PATH = ROOT / "results/multi_probe/results_all_config.json"
DEPLOY_PATH = ROOT / "results/multi_probe/cf_comparison_runtime32_full.json"
ALIGNED_OUT_PATH = ROOT / "docs/images/aligned_research_results.png"
DEPLOY_OUT_PATH = ROOT / "docs/images/deployable_fusion_results.png"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_aligned(aligned: dict, probes: list[int]) -> None:
    aligned_r2 = [aligned[f"xgboost_n{n}"]["overall"]["r2"] for n in probes]
    aligned_mae = [aligned[f"xgboost_n{n}"]["overall"]["mae"] for n in probes]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.plot(probes, aligned_r2, color="#1f5aa6", marker="o", linewidth=2.8, markersize=8)
    ax.fill_between(probes, aligned_r2, [min(aligned_r2)] * len(aligned_r2), color="#1f5aa6", alpha=0.08)
    ax.set_title("Aligned Research Setting", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Probes")
    ax.set_ylabel("R²")
    ax.set_xticks(probes)
    ax.set_ylim(0.45, 0.78)

    for x, y, mae in zip(probes, aligned_r2, aligned_mae):
        ax.annotate(
            f"R² {y:.3f}\nMAE {mae:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.text(
        0.03,
        0.05,
        "Latest full-feature rerun\nsame 1 km slice, XGBoost",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#355070",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef4ff", "edgecolor": "#c9daf8"},
    )

    ALIGNED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ALIGNED_OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {ALIGNED_OUT_PATH}")


def plot_deployable(deploy: dict, probes: list[int]) -> None:
    deploy_r2 = {
        "Simple mean": [deploy[f"N={n}"]["simple_mean"]["r2"] for n in probes],
        "CF-softmax": [deploy[f"N={n}"]["cf_add_softmax"]["r2"] for n in probes],
        "Bayesian+CF": [deploy[f"N={n}"]["bayesian_cf_add"]["r2"] for n in probes],
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("white")

    colors = {
        "Simple mean": "#adb5bd",
        "CF-softmax": "#f4a261",
        "Bayesian+CF": "#2a9d8f",
    }
    for label, values in deploy_r2.items():
        ax.plot(
            probes,
            values,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=colors[label],
            label=label,
        )
    ax.set_title("Deployable Link-Level Fusion", fontsize=14, weight="bold")
    ax.set_xlabel("Number of Probes")
    ax.set_ylabel("R²")
    ax.set_xticks(probes)
    ax.set_ylim(0.50, 0.68)

    for label, values in deploy_r2.items():
        x = probes[-1]
        y = values[-1]
        offsets = {
            "Simple mean": (10, -2),
            "CF-softmax": (12, 10),
            "Bayesian+CF": (12, 18),
        }
        ax.annotate(
            f"{label}\nR² {y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=offsets[label],
            ha="left",
            fontsize=9,
            color=colors[label],
        )

    ax.legend(frameon=False, loc="upper left")

    ax.text(
        0.03,
        0.05,
        "Unequal traversal boundaries\n32-input single-probe model",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#245c54",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#edf8f6", "edgecolor": "#b7e4dc"},
    )

    DEPLOY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEPLOY_OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {DEPLOY_OUT_PATH}")


def main() -> None:
    aligned = load_json(ALIGNED_PATH)
    deploy = load_json(DEPLOY_PATH)
    probes = [1, 2, 3, 5]
    plot_aligned(aligned, probes)
    plot_deployable(deploy, probes)


if __name__ == "__main__":
    main()
