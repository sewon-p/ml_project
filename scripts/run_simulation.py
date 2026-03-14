"""Run SUMO simulations for all scenarios with optional parallelism."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulation.network_builder import (
    build_route_file,
    build_single_link_network,
)
from src.simulation.runner import run_sumo_simulation
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = logging.getLogger(__name__)


def _run_single_scenario(
    row_dict: dict[str, Any],
    sim_cfg: dict[str, Any],
    fcd_dir: Path,
) -> dict[str, Any] | None:
    """Run one SUMO scenario. Designed to be called from a worker."""
    sid = int(row_dict["scenario_id"])
    net_dir = fcd_dir / f"scenario_{sid}"
    net_dir.mkdir(parents=True, exist_ok=True)

    try:
        net_file = build_single_link_network(
            net_dir / "network",
            link_length=row_dict["link_length"],
            num_lanes=int(row_dict["num_lanes"]),
            speed_limit=row_dict["speed_limit"],
        )

        # Extract vehicle type params from scenario row
        pass_params = {}
        truck_params = {}
        for key in ("length", "tau", "decel", "minGap", "speedFactor", "sigma", "accel"):
            pcol = f"passenger_{key}"
            tcol = f"truck_{key}"
            if pcol in row_dict:
                pass_params[key] = float(row_dict[pcol])
            if tcol in row_dict:
                truck_params[key] = float(row_dict[tcol])

        truck_ratio = float(row_dict.get("truck_ratio", 0.0))

        route_file = build_route_file(
            net_dir / "routes.rou.xml",
            demand_vehph=int(row_dict["demand_vehph"]),
            begin=0.0,
            end=float(sim_cfg.get("warmup_time", 300) + sim_cfg.get("collect_time", 600)),
            truck_ratio=truck_ratio,
            passenger_params=pass_params,
            truck_params=truck_params,
            speed_limit=row_dict["speed_limit"],
        )

        fcd_out = net_dir / "fcd.csv"
        result = run_sumo_simulation(
            net_file=net_file,
            route_file=route_file,
            output_fcd=fcd_out,
            sumo_binary=sim_cfg.get("sumo_binary", "sumo"),
            step_length=sim_cfg.get("step_length", 0.1),
            warmup_time=sim_cfg.get("warmup_time", 300),
            collect_time=sim_cfg.get("collect_time", 600),
            seed=int(row_dict.get("seed", 42)),
        )
        return result
    except Exception as e:
        logger.error("Scenario %d failed: %s", sid, e, exc_info=True)
        return None


def _is_scenario_done(fcd_dir: Path, sid: int) -> bool:
    """Check if a scenario already has a non-empty FCD output."""
    fcd_csv = fcd_dir / f"scenario_{sid}" / "fcd.csv"
    return fcd_csv.exists() and fcd_csv.stat().st_size > 0


def _worker(args: tuple[dict, dict, str, bool]) -> tuple[int, bool]:
    """Multiprocessing worker wrapper. Returns (scenario_id, success)."""
    row_dict, sim_cfg, fcd_dir_str, resume = args
    sid = int(row_dict["scenario_id"])
    fcd_dir = Path(fcd_dir_str)
    if resume and _is_scenario_done(fcd_dir, sid):
        return sid, True
    result = _run_single_scenario(row_dict, sim_cfg, fcd_dir)
    return sid, result is not None


def _resolve_max_workers(value: str | int | None) -> int:
    """Resolve max_workers setting."""
    if value is None or value == "auto":
        return max(1, (os.cpu_count() or 1) - 1)
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUMO simulations.")
    parser.add_argument("--config", default="configs/simulation/scenarios.yaml")
    parser.add_argument("--scenarios-csv", default=None)
    parser.add_argument(
        "--workers",
        default=None,
        type=str,
        help="Number of parallel workers (default: from config, 'auto' or int)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that already have FCD output.",
    )
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)
    sim_cfg = cfg.get("simulation", {})
    output_cfg = cfg.get("output", {})

    csv_path = args.scenarios_csv or output_cfg.get("scenarios_csv", "data/scenarios.csv")
    scenarios = pd.read_csv(csv_path)
    fcd_dir = Path(output_cfg.get("fcd_dir", "data/fcd"))

    max_workers_cfg = args.workers or sim_cfg.get("max_workers", "auto")
    max_workers = _resolve_max_workers(max_workers_cfg)

    print(f"[Simulation] {len(scenarios)} scenarios, {max_workers} workers", flush=True)

    # Resume: count already-done scenarios
    if args.resume:
        done = sum(
            1
            for _, row in scenarios.iterrows()
            if _is_scenario_done(fcd_dir, int(row["scenario_id"]))
        )
        total = len(scenarios)
        remaining = total - done
        print(f"[Simulation] Resume: {done}/{total} done, {remaining} remaining", flush=True)
    else:
        done = 0

    # Prepare args for workers
    work_items = [
        (row.to_dict(), sim_cfg, str(fcd_dir), args.resume) for _, row in scenarios.iterrows()
    ]

    total = len(work_items)
    completed = 0
    failed = 0

    if max_workers <= 1:
        for item in work_items:
            sid, ok = _worker(item)
            completed += 1
            if not ok:
                failed += 1
            if completed % 20 == 0 or completed == total:
                print(f"[Simulation] {completed}/{total} done ({failed} failed)", flush=True)
    else:
        # imap_unordered with chunksize=1 for optimal load balancing
        with multiprocessing.Pool(processes=max_workers) as pool:
            for sid, ok in pool.imap_unordered(_worker, work_items, chunksize=1):
                completed += 1
                if not ok:
                    failed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"[Simulation] {completed}/{total} done ({failed} failed)", flush=True)

    print(f"[Simulation] All done: {total} scenarios ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
