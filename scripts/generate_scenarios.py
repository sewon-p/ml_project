"""Generate simulation scenario matrix CSV via stochastic sampling."""

from __future__ import annotations

import argparse

import pandas as pd

from src.simulation.scenario_generator import (
    generate_scenario_matrix,
    save_scenario_matrix,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate scenario matrix."
    )
    parser.add_argument(
        "--config", default="configs/simulation/scenarios.yaml"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--num", type=int, default=None,
        help="Number of scenarios to generate (overrides config)",
    )
    parser.add_argument(
        "--start-id", type=int, default=0,
        help="Starting scenario_id (use to append new scenarios)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base seed for RNG (default: 42)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing scenarios CSV instead of overwriting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg.get("simulation", {})
    scenario_cfg = cfg.get("scenarios", {})

    num_simulations = args.num or scenario_cfg.get("num_simulations", 20000)

    df = generate_scenario_matrix(
        sim_cfg=sim_cfg,
        num_simulations=num_simulations,
        base_seed=args.base_seed,
        start_id=args.start_id,
    )

    output = args.output or cfg.get("output", {}).get(
        "scenarios_csv", "data/scenarios.csv"
    )

    if args.append:
        existing = pd.read_csv(output)
        df = pd.concat([existing, df], ignore_index=True)
        print(f"Appended {num_simulations} scenarios (total: {len(df)})")

    path = save_scenario_matrix(df, output)
    print(f"Saved {len(df)} scenarios to {path}")


if __name__ == "__main__":
    main()
