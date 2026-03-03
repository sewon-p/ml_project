"""Dataset classes for traffic estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrafficDataset:
    """Tabular dataset for tree-based models (XGBoost, LightGBM).

    Wraps a DataFrame with feature columns, target column,
    and optional group column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "density",
        group_column: str | None = "scenario_id",
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.group_column = group_column

    @property
    def X(self) -> np.ndarray:
        return self.df[self.feature_columns].values

    @property
    def y(self) -> np.ndarray:
        return self.df[self.target_column].values

    @property
    def groups(self) -> np.ndarray | None:
        if self.group_column and self.group_column in self.df.columns:
            return self.df[self.group_column].values
        return None

    def __len__(self) -> int:
        return len(self.df)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for 6-channel time-series models (CNN1D, LSTM).

    Each sample is a multi-channel sequence (VX, VY, AX, AY, speed, brake),
    a scalar target, and optional condition variables (num_lanes, speed_limit).
    """

    def __init__(
        self,
        sequences: np.ndarray,  # shape: (N, channels, seq_len) or (N, seq_len)
        targets: np.ndarray,  # shape: (N,)
        conditions: np.ndarray | None = None,  # shape: (N, n_cond)
    ):
        self.sequences = torch.FloatTensor(sequences)
        if self.sequences.ndim == 2:
            # Single-channel fallback: (N, seq_len) -> (N, 1, seq_len)
            self.sequences = self.sequences.unsqueeze(1)
        self.targets = torch.FloatTensor(targets)
        self.conditions = torch.FloatTensor(conditions) if conditions is not None else None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conditions is not None:
            return self.sequences[idx], self.targets[idx], self.conditions[idx]
        return self.sequences[idx], self.targets[idx]
