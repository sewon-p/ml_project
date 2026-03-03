"""Cross-validation utilities with group-aware splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def grouped_kfold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = "scenario_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate GroupKFold train/val index pairs.

    Ensures that samples from the same group (scenario) never appear
    in both train and validation within the same fold.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_column].values
    splits = []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        splits.append((train_idx, val_idx))
    return splits
