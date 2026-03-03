"""Feature extraction pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

import src.features.acceleration as _acceleration  # noqa: F401, I001
import src.features.basic_stats as _basic_stats  # noqa: F401
import src.features.brake_patterns as _brake_patterns  # noqa: F401
import src.features.lateral as _lateral  # noqa: F401
import src.features.stop_patterns as _stop_patterns  # noqa: F401
import src.features.time_series as _time_series  # noqa: F401
from src.features.registry import _registry


def extract_features(
    trajectory: pd.DataFrame,
    feature_names: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Extract features from a single-vehicle trajectory DataFrame.

    Parameters
    ----------
    trajectory:
        DataFrame with columns: VX, VY, AX, AY, speed, brake.
    feature_names:
        Optional list of feature names to extract.  If ``None``, all
        registered features are extracted.
    **kwargs:
        Forwarded to every feature function (e.g. ``dt``, ``stop_threshold``).

    Returns
    -------
    dict[str, float]
        Flat mapping of feature name to its scalar value.
    """
    if feature_names is None:
        funcs = _registry.get_subset(_registry.list_all())
    else:
        funcs = _registry.get_subset(feature_names)

    results: dict[str, float] = {}
    for name, func in funcs.items():
        value = func(trajectory, **kwargs)
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                results[f"{name}_{sub_key}"] = float(sub_val)
        else:
            results[name] = float(value)

    return results
