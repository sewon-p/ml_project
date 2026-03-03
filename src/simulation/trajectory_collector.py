"""Parse SUMO FCD output (CSV or XML) into DataFrames."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def parse_fcd_csv(fcd_path: str | Path) -> pd.DataFrame:
    """Parse TraCI-collected CSV with brake signals.

    Expected columns: time, vehicle_id, x, y, speed, edge_id, brake
    """
    fcd_path = Path(fcd_path)
    if not fcd_path.exists():
        raise FileNotFoundError(f"FCD file not found: {fcd_path}")

    dtype = {
        "time": "float64",
        "x": "float64",
        "y": "float64",
        "speed": "float64",
        "brake": "int32",
        "vehicle_id": "str",
    }
    try:
        df = pd.read_csv(fcd_path, dtype=dtype, engine="pyarrow")
    except ImportError:
        df = pd.read_csv(fcd_path, dtype=dtype)
    return df


def parse_fcd_xml(fcd_path: str | Path) -> pd.DataFrame:
    """Parse SUMO FCD XML file into a DataFrame.

    Returns DataFrame with columns:
        time, vehicle_id, x, y, speed, edge_id.

    Note: XML format does not contain brake signals.
    Use parse_fcd_csv for TraCI-collected data with brake info.
    """
    fcd_path = Path(fcd_path)
    if not fcd_path.exists():
        raise FileNotFoundError(f"FCD file not found: {fcd_path}")

    records = []
    tree = ET.iterparse(str(fcd_path), events=("end",))
    current_time = 0.0

    for event, elem in tree:
        if elem.tag == "timestep":
            current_time = float(elem.get("time", 0))
        elif elem.tag == "vehicle":
            records.append(
                {
                    "time": current_time,
                    "vehicle_id": elem.get("id", ""),
                    "x": float(elem.get("x", 0)),
                    "y": float(elem.get("y", 0)),
                    "speed": float(elem.get("speed", 0)),
                    "edge_id": elem.get("lane", "").rsplit("_", 1)[0],
                }
            )
            elem.clear()

    return pd.DataFrame(records)


def parse_fcd(fcd_path: str | Path) -> pd.DataFrame:
    """Auto-detect format (CSV or XML) and parse FCD data.

    CSV files include brake column; XML files do not.
    """
    fcd_path = Path(fcd_path)
    if fcd_path.suffix == ".csv":
        return parse_fcd_csv(fcd_path)
    return parse_fcd_xml(fcd_path)


def filter_time_range(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """Filter FCD DataFrame to a time window."""
    return df[(df["time"] >= t_start) & (df["time"] < t_end)].reset_index(drop=True)
