"""Tests for data pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import TimeSeriesDataset, TrafficDataset
from src.data.preprocessing import (
    grouped_train_test_split,
    pad_sequences,
)


class TestTrafficDataset:
    def test_basic_properties(self):
        df = pd.DataFrame(
            {
                "speed_mean": [10.0, 20.0, 30.0],
                "speed_std": [1.0, 2.0, 3.0],
                "density": [50.0, 30.0, 10.0],
                "scenario_id": [0, 0, 1],
            }
        )
        ds = TrafficDataset(df, feature_columns=["speed_mean", "speed_std"])
        assert ds.X.shape == (3, 2)
        assert ds.y.shape == (3,)
        assert ds.groups is not None
        assert len(ds) == 3

    def test_no_group_column(self):
        df = pd.DataFrame(
            {
                "speed_mean": [10.0, 20.0],
                "density": [50.0, 30.0],
            }
        )
        ds = TrafficDataset(
            df,
            feature_columns=["speed_mean"],
            group_column=None,
        )
        assert ds.groups is None


class TestTimeSeriesDataset:
    def test_shapes_2d(self):
        sequences = np.random.randn(10, 60)
        targets = np.random.randn(10)
        ds = TimeSeriesDataset(sequences, targets)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (1, 60)  # single channel fallback
        assert y.shape == ()

    def test_shapes_6channel(self):
        sequences = np.random.randn(10, 6, 60)
        targets = np.random.randn(10)
        ds = TimeSeriesDataset(sequences, targets)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (6, 60)  # 6-channel
        assert y.shape == ()


class TestPreprocessing:
    def test_grouped_split_no_leakage(self):
        df = pd.DataFrame(
            {
                "x": range(100),
                "scenario_id": [i // 10 for i in range(100)],
            }
        )
        train_df, test_df = grouped_train_test_split(df, test_ratio=0.2)
        train_groups = set(train_df["scenario_id"])
        test_groups = set(test_df["scenario_id"])
        assert train_groups.isdisjoint(test_groups)

    def test_pad_sequences_1d(self):
        seqs = [
            np.array([1, 2, 3]),
            np.array([4, 5]),
        ]
        padded = pad_sequences(seqs, max_len=4, pad_value=0.0)
        assert padded.shape == (2, 4)
        assert padded[1, 2] == 0.0  # padded
        assert padded[0, 2] == 3.0  # original value

    def test_pad_sequences_auto_length(self):
        seqs = [
            np.array([1, 2, 3]),
            np.array([4, 5]),
        ]
        padded = pad_sequences(seqs)
        assert padded.shape == (2, 3)

    def test_pad_sequences_multichannel(self):
        seqs = [
            np.random.randn(6, 50),
            np.random.randn(6, 30),
        ]
        padded = pad_sequences(seqs, max_len=60)
        assert padded.shape == (2, 6, 60)
        # Check padding region is zero
        assert padded[1, 0, 30] == 0.0
