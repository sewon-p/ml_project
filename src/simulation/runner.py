"""Run SUMO simulation via TraCI and collect FCD + brake signals."""

from __future__ import annotations

import csv
import logging
import os
import socket
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Brake light is bit 3 (value 8) in SUMO signal bitmask
_BRAKE_LIGHT_BIT = 0b1000


def _resolve_sumo_binary(name: str) -> str:
    """Resolve SUMO binary using SUMO_HOME if bare name fails PATH lookup."""
    import shutil

    if shutil.which(name):
        return name
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        full = os.path.join(sumo_home, "bin", name)
        if os.path.isfile(full):
            return full
        # Try with .exe on Windows
        full_exe = full + ".exe"
        if os.path.isfile(full_exe):
            return full_exe
    raise FileNotFoundError(
        f"SUMO binary '{name}' not found in PATH or SUMO_HOME={sumo_home!r}. "
        "Install SUMO and set SUMO_HOME."
    )


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    port: int | None = None,
) -> dict[str, Any]:
    """Launch SUMO with TraCI, collect FCD with brake signals.

    Collects vehicle data via TraCI (position, speed, signals) instead of
    relying solely on SUMO's FCD XML output. Brake state is determined from
    traci.vehicle.getSignals() bit 3 (brake light).

    Output is saved as CSV with columns:
        time, vehicle_id, x, y, speed, edge_id, brake

    Parameters
    ----------
    port:
        Explicit TraCI port. If None, a free port is auto-assigned.
        Prevents port conflicts in parallel execution.

    Requires the 'simulation' extra:
        pip install -e ".[simulation]"
    """
    try:
        import traci
    except ImportError:
        raise ImportError("traci not found. Install with: pip install -e '.[simulation]'")

    resolved_binary = _resolve_sumo_binary(sumo_binary)

    output_fcd = Path(output_fcd).with_suffix(".csv")
    output_fcd.parent.mkdir(parents=True, exist_ok=True)

    # Assign explicit port to prevent conflicts in parallel execution
    if port is None:
        port = _find_free_port()

    cmd = [
        resolved_binary,
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
        "--no-step-log",
        "true",
    ]
    if additional_args:
        cmd.extend(additional_args)

    logger.info("Starting SUMO on port %d: %s", port, " ".join(cmd))

    # Defensively close any stale connection (prevents "already active" cascade)
    try:
        traci.close()
    except Exception:
        pass

    traci.start(cmd, port=port, numRetries=10)

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
        try:
            traci.close()
        except Exception:
            pass

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
