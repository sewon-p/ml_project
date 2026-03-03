"""Generate simulation scenario matrix CSV via stochastic sampling."""

from __future__ import annotations

import argparse

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
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg.get("simulation", {})
    scenario_cfg = cfg.get("scenarios", {})

    num_simulations = scenario_cfg.get("num_simulations", 20000)

    df = generate_scenario_matrix(
        sim_cfg=sim_cfg,
        num_simulations=num_simulations,
    )

    output = args.output or cfg.get("output", {}).get(
        "scenarios_csv", "data/scenarios.csv"
    )
    path = save_scenario_matrix(df, output)
    print(f"Saved {len(df)} scenarios to {path}")


if __name__ == "__main__":
    main()
