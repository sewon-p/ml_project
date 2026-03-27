"""Run the full pipeline: simulate -> extract -> [prepare_residuals] -> train -> evaluate."""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.utils.config import load_config


def run_step(description: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"FAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip SUMO simulation (use existing FCD data).",
    )
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_simulation:
        run_step(
            "Generate scenarios",
            [
                py,
                "scripts/generate_scenarios.py",
                "--config",
                args.config,
            ],
        )
        run_step(
            "Run SUMO simulations",
            [
                py,
                "scripts/run_simulation.py",
                "--config",
                args.config,
            ],
        )

    run_step(
        "Extract features",
        [
            py,
            "scripts/extract_features.py",
            "--config",
            args.config,
        ],
    )

    cfg = load_config(args.config)
    if cfg.get("residual_correction", {}).get("enabled", False):
        run_step(
            "Prepare residual targets",
            [
                py,
                "scripts/prepare_residuals.py",
                "--config",
                args.config,
            ],
        )

    run_step(
        "Train model",
        [
            py,
            "scripts/train.py",
            "--config",
            args.config,
        ],
    )
    run_step(
        "Evaluate model",
        [
            py,
            "scripts/evaluate.py",
            "--config",
            args.config,
        ],
    )

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
