"""Data preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CHANNELS = ["VX", "VY", "AX", "AY", "speed", "brake"]


def build_trajectory(veh_df: pd.DataFrame) -> pd.DataFrame:
    """Build 6-channel trajectory using np.gradient (no row loss).

    Input DataFrame must have columns: time, x, y, speed.
    Optional column: brake (inferred from AX < -0.5 if absent).

    np.gradient uses central differences in the interior and one-sided
    differences at the boundaries, preserving the exact input length.
    """
    n = len(veh_df)
    if n < 2:
        return pd.DataFrame(columns=CHANNELS)

    times = veh_df["time"].values.astype(float)
    x = veh_df["x"].values.astype(float)
    y = veh_df["y"].values.astype(float)
    speed = veh_df["speed"].values.astype(float)
    has_brake = "brake" in veh_df.columns

    vx = np.gradient(x, times)
    vy = np.gradient(y, times)
    ax = np.gradient(vx, times)
    ay = np.gradient(vy, times)

    if has_brake:
        brake = veh_df["brake"].values.astype(float)
    else:
        brake = (ax < -0.5).astype(float)

    return pd.DataFrame({"VX": vx, "VY": vy, "AX": ax, "AY": ay, "speed": speed, "brake": brake})


def grouped_train_test_split(
    df: pd.DataFrame,
    group_column: str = "scenario_id",
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame ensuring no group leakage between
    train and test.
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_state,
    )
    groups = df[group_column]
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def pad_sequences(
    sequences: list[np.ndarray],
    max_len: int | None = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad variable-length sequences to uniform length.

    Supports both 1D sequences (N arrays of shape (L,)) and
    2D multi-channel sequences (N arrays of shape (C, L)).

    Args:
        sequences: List of arrays with varying lengths.
        max_len: Target length. If None, use the max length.
        pad_value: Value to use for padding.

    Returns:
        For 1D: array of shape (N, max_len).
        For 2D: array of shape (N, C, max_len).
    """
    if not sequences:
        return np.array([])

    if sequences[0].ndim == 1:
        # 1D case
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        padded = np.full(
            (len(sequences), max_len),
            pad_value,
            dtype=np.float32,
        )
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        return padded
    else:
        # Multi-channel case: (C, L)
        n_channels = sequences[0].shape[0]
        if max_len is None:
            max_len = max(s.shape[1] for s in sequences)
        padded = np.full(
            (len(sequences), n_channels, max_len),
            pad_value,
            dtype=np.float32,
        )
        for i, seq in enumerate(sequences):
            length = min(seq.shape[1], max_len)
            padded[i, :, :length] = seq[:, :length]
        return padded
