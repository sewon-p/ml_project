"""Generate scenario parameter matrix via stochastic sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sample_clipped_normal(
    rng: np.random.RandomState,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    """Draw from a normal distribution, clipped to [low, high]."""
    return float(np.clip(rng.normal(mean, std), low, high))


def _sample_vehicle_params(
    rng: np.random.RandomState,
    vtype_cfg: dict[str, Any],
) -> dict[str, float]:
    """Sample vehicle type parameters from config distributions."""
    params: dict[str, float] = {}
    for key, spec in vtype_cfg.items():
        if key == "length":
            params[key] = float(spec)
        elif isinstance(spec, dict) and "mean" in spec:
            params[key] = _sample_clipped_normal(
                rng,
                spec["mean"],
                spec["std"],
                spec["min"],
                spec["max"],
            )
    return params


def generate_scenario_matrix(
    sim_cfg: dict[str, Any],
    num_simulations: int = 10000,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Generate stochastic scenario matrix by sampling from config distributions.

    Each row is a unique scenario with sampled parameters:
    - num_lanes, speed_limit (m/s), demand_vehph, truck_ratio
    - passenger_* and truck_* vehicle type params
    """
    rng = np.random.RandomState(base_seed)

    net_cfg = sim_cfg.get("network", {})
    demand_cfg = sim_cfg.get("demand", {})
    vtype_cfg = sim_cfg.get("vehicle_types", {})

    # Network ranges
    lanes_min = net_cfg.get("num_lanes", {}).get("min", 1)
    lanes_max = net_cfg.get("num_lanes", {}).get("max", 3)
    speed_limits_kmh = net_cfg.get("speed_limit_kmh", [50, 60, 80, 100])

    # Per-lane demand ranges
    per_lane_min = demand_cfg.get("per_lane_min_vehph", 800)
    per_lane_max = demand_cfg.get("per_lane_max_vehph", 2200)

    # Truck ratio distribution
    truck_ratio_cfg = vtype_cfg.get("truck_ratio", {})
    passenger_cfg = vtype_cfg.get("passenger", {})
    truck_cfg = vtype_cfg.get("truck", {})

    records = []
    for i in range(num_simulations):
        seed = base_seed + i

        # Sample network params
        num_lanes = int(rng.randint(lanes_min, lanes_max + 1))
        speed_limit_kmh = float(rng.choice(speed_limits_kmh))
        speed_limit = speed_limit_kmh / 3.6  # km/h -> m/s

        # Compute dynamic link_length: speed_limit * 600s * 1.2 margin
        link_length = round(speed_limit * 600.0 * 1.2, 1)

        # Sample per-lane demand, then multiply by num_lanes
        per_lane_demand = int(rng.randint(per_lane_min, per_lane_max + 1))
        demand_vehph = per_lane_demand * num_lanes

        # Sample truck ratio
        truck_ratio = _sample_clipped_normal(
            rng,
            truck_ratio_cfg.get("mean", 0.16),
            truck_ratio_cfg.get("std", 0.05),
            truck_ratio_cfg.get("min", 0.0),
            truck_ratio_cfg.get("max", 0.40),
        )

        # Sample vehicle type params
        pass_params = _sample_vehicle_params(rng, passenger_cfg)
        truck_params = _sample_vehicle_params(rng, truck_cfg)

        row: dict[str, Any] = {
            "scenario_id": i,
            "seed": seed,
            "num_lanes": num_lanes,
            "speed_limit_kmh": speed_limit_kmh,
            "speed_limit": round(speed_limit, 4),
            "link_length": link_length,
            "demand_vehph": demand_vehph,
            "truck_ratio": round(truck_ratio, 4),
        }
        for k, v in pass_params.items():
            row[f"passenger_{k}"] = round(v, 4)
        for k, v in truck_params.items():
            row[f"truck_{k}"] = round(v, 4)

        records.append(row)

    return pd.DataFrame(records)


def save_scenario_matrix(df: pd.DataFrame, path: str | Path) -> Path:
    """Save scenario matrix to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path.resolve()
