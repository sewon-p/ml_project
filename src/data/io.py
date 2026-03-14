"""Parquet I/O utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def load_data_filter_mask(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Create a boolean mask from ``cfg["data"]["filters"]``.

    Supported filter keys:
    - ``lanes``: list of int — keep rows where ``num_lanes`` is in the list.
    - ``speed_limits_kmh``: list of numeric — keep rows where ``speed_limit``
      (m/s) matches after km/h→m/s conversion (tolerance 0.1 m/s).

    Returns an all-True mask when no filters are configured.
    """
    filters = (cfg.get("data") or {}).get("filters")
    mask = np.ones(len(df), dtype=bool)
    if not filters:
        return mask

    total = len(df)

    lanes = filters.get("lanes")
    if lanes and "num_lanes" in df.columns:
        mask &= df["num_lanes"].isin(lanes).values

    speed_limits_kmh = filters.get("speed_limits_kmh")
    if speed_limits_kmh and "speed_limit" in df.columns:
        sl_ms = [v / 3.6 for v in speed_limits_kmh]
        sl_mask = np.zeros(len(df), dtype=bool)
        for target in sl_ms:
            sl_mask |= (df["speed_limit"] - target).abs().values < 0.1
        mask &= sl_mask

    kept = int(mask.sum())
    logger.info("Data filter: %d/%d samples kept", kept, total)
    return mask


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet.

    Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.resolve()
