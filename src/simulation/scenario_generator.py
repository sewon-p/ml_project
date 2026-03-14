"""Generate scenario parameter matrix via stochastic sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sample_param(rng: np.random.RandomState, cfg: dict, n: int) -> np.ndarray:
    """Sample *n* values from a parameter config.

    cfg format:
      {"dist": "uniform", "min": 0.8, "max": 2.0}
      {"dist": "normal", "mean": 1.4, "std": 0.3, "min": 0.8, "max": 2.0}

    If ``dist`` is missing, defaults to ``"uniform"``.
    """
    if cfg.get("dist", "uniform") == "normal":
        vals = rng.normal(cfg["mean"], cfg["std"], size=n)
        return np.clip(vals, cfg["min"], cfg["max"])
    return rng.uniform(cfg["min"], cfg["max"], size=n)


def _sample_vehicle_params(
    rng: np.random.RandomState,
    vtype_cfg: dict[str, Any],
) -> dict[str, float]:
    """Sample vehicle type parameters from configured distributions."""
    params: dict[str, float] = {}
    for key, spec in vtype_cfg.items():
        if key == "length":
            params[key] = float(spec)
        elif isinstance(spec, dict) and "min" in spec and "max" in spec:
            params[key] = float(_sample_param(rng, spec, 1)[0])
    return params


def generate_scenario_matrix(
    sim_cfg: dict[str, Any],
    num_simulations: int = 10000,
    base_seed: int = 42,
    start_id: int = 0,
) -> pd.DataFrame:
    """Generate stochastic scenario matrix by sampling from config distributions.

    Each row is a unique scenario with sampled parameters:
    - num_lanes, speed_limit (m/s), demand_vehph, truck_ratio
    - passenger_* and truck_* vehicle type params

    Parameters
    ----------
    start_id:
        Starting scenario_id. Use to append new scenarios without
        overwriting existing ones (e.g. start_id=20000).
    """
    rng = np.random.RandomState(base_seed)

    net_cfg = sim_cfg.get("network", {})
    demand_cfg = sim_cfg.get("demand", {})
    vtype_cfg = sim_cfg.get("vehicle_types", {})

    # Network ranges
    num_lanes_cfg = net_cfg.get("num_lanes", {})
    lanes_min = num_lanes_cfg.get("min", 1)
    lanes_max = num_lanes_cfg.get("max", 3)
    speed_limits_kmh = net_cfg.get("speed_limit_kmh", [50, 60, 80, 100])

    # Per-lane demand — support both old flat keys and new dict format
    if "per_lane_demand_vehph" in demand_cfg and isinstance(
        demand_cfg["per_lane_demand_vehph"], dict
    ):
        per_lane_cfg = demand_cfg["per_lane_demand_vehph"]
    else:
        per_lane_cfg = {
            "min": demand_cfg.get("per_lane_min_vehph", 800),
            "max": demand_cfg.get("per_lane_max_vehph", 2200),
        }

    # Truck ratio distribution
    truck_ratio_cfg = vtype_cfg.get("truck_ratio", {})
    passenger_cfg = vtype_cfg.get("passenger", {})
    truck_cfg = vtype_cfg.get("truck", {})

    records = []
    for i in range(num_simulations):
        seed = base_seed + i

        # Sample num_lanes — always use discrete randint for integer params
        num_lanes = int(rng.randint(lanes_min, lanes_max + 1))
        speed_limit_kmh = float(rng.choice(speed_limits_kmh))
        speed_limit = speed_limit_kmh / 3.6  # km/h -> m/s

        # Compute dynamic link_length: speed_limit * 600s * 1.2 margin
        link_length = round(speed_limit * 600.0 * 1.2, 1)

        # Sample per-lane demand, then multiply by num_lanes
        per_lane_demand = int(round(_sample_param(rng, per_lane_cfg, 1)[0]))

        # Sample vehicle type params (needed for capacity clipping)
        pass_params = _sample_vehicle_params(rng, passenger_cfg)
        truck_params = _sample_vehicle_params(rng, truck_cfg)

        # Clip per-lane demand to 150% of theoretical Krauss capacity
        # Allowing demand > capacity triggers congestion at the entry,
        # producing higher-density scenarios needed for ML training.
        tau = pass_params.get("tau", 1.45)
        veh_length = pass_params.get("length", 4.5)
        min_gap = pass_params.get("minGap", 2.5)
        theoretical_cap = 3600.0 / (tau + (veh_length + min_gap) / speed_limit)
        per_lane_demand = min(per_lane_demand, int(theoretical_cap * 1.5))

        demand_vehph = per_lane_demand * num_lanes

        # Sample truck ratio
        truck_ratio = float(
            _sample_param(
                rng,
                {
                    "dist": truck_ratio_cfg.get("dist", "uniform"),
                    "min": truck_ratio_cfg.get("min", 0.0),
                    "max": truck_ratio_cfg.get("max", 0.40),
                    **({k: truck_ratio_cfg[k] for k in ("mean", "std") if k in truck_ratio_cfg}),
                },
                1,
            )[0]
        )

        row: dict[str, Any] = {
            "scenario_id": start_id + i,
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
