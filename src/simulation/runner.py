"""Run SUMO simulation via TraCI and collect FCD + brake signals."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Brake light is bit 3 (value 8) in SUMO signal bitmask
_BRAKE_LIGHT_BIT = 0b1000


def run_sumo_simulation(
    net_file: str | Path,
    route_file: str | Path,
    output_fcd: str | Path,
    sumo_binary: str = "sumo",
    step_length: float = 0.1,
    warmup_time: float = 300.0,
    collect_time: float = 600.0,
    seed: int = 42,
    additional_args: list[str] | None = None,
) -> dict[str, Any]:
    """Launch SUMO with TraCI, collect FCD with brake signals.

    Collects vehicle data via TraCI (position, speed, signals) instead of
    relying solely on SUMO's FCD XML output. Brake state is determined from
    traci.vehicle.getSignals() bit 3 (brake light).

    Output is saved as CSV with columns:
        time, vehicle_id, x, y, speed, edge_id, brake

    Requires the 'simulation' extra:
        pip install -e ".[simulation]"
    """
    try:
        import traci
    except ImportError:
        raise ImportError("traci not found. Install with: pip install -e '.[simulation]'")

    output_fcd = Path(output_fcd).with_suffix(".csv")
    output_fcd.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sumo_binary,
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "--step-length",
        str(step_length),
        "--seed",
        str(seed),
        "--no-warnings",
        "true",
    ]
    if additional_args:
        cmd.extend(additional_args)

    logger.info("Starting SUMO: %s", " ".join(cmd))
    traci.start(cmd)

    total_time = warmup_time + collect_time
    step = 0
    n_vehicles_seen = 0

    try:
        with open(output_fcd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "vehicle_id", "x", "y", "speed", "edge_id", "brake"])

            while traci.simulation.getTime() < total_time:
                traci.simulationStep()
                step += 1
                current_time = traci.simulation.getTime()

                # Only collect data after warmup
                if current_time >= warmup_time:
                    n_vehicles_seen += traci.simulation.getDepartedNumber()

                    for veh_id in traci.vehicle.getIDList():
                        pos = traci.vehicle.getPosition(veh_id)
                        speed = traci.vehicle.getSpeed(veh_id)
                        signals = traci.vehicle.getSignals(veh_id)
                        lane_id = traci.vehicle.getLaneID(veh_id)
                        edge_id = lane_id.rsplit("_", 1)[0] if lane_id else ""
                        brake = 1 if signals & _BRAKE_LIGHT_BIT else 0

                        writer.writerow(
                            [
                                round(current_time, 2),
                                veh_id,
                                round(pos[0], 4),
                                round(pos[1], 4),
                                round(speed, 4),
                                edge_id,
                                brake,
                            ]
                        )
    finally:
        traci.close()

    logger.info(
        "Simulation done. Steps=%d, vehicles_seen=%d, output=%s",
        step,
        n_vehicles_seen,
        output_fcd,
    )
    return {
        "fcd_path": str(output_fcd),
        "steps": step,
        "vehicles_seen": n_vehicles_seen,
    }
