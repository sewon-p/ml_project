"""Tests for hyperparameter optimization with window size search."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import extract_features
from src.training.hyperopt import optimize_with_window


def _make_dummy_dataset(n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a dummy feature dataset with scenario_id and density."""
    rng = np.random.RandomState(seed)
    records = []
    for i in range(n_samples):
        scenario_id = i // 5  # 5 samples per scenario
        speed_mean = 15.0 + rng.randn() * 5.0
        speed_std = 2.0 + rng.rand()
        accel_mean = rng.randn() * 0.5
        density = max(0.0, 50.0 - speed_mean + rng.randn() * 2.0)
        records.append(
            {
                "speed_mean": speed_mean,
                "speed_std": speed_std,
                "accel_mean": accel_mean,
                "scenario_id": scenario_id,
                "density": density,
            }
        )
    return pd.DataFrame(records)


class TestOptimizeWithWindow:
    """Tests for optimize_with_window function."""

    def test_basic_search(self):
        """optimize_with_window returns best_params with window_size."""
        df60 = _make_dummy_dataset(50, seed=1)
        df120 = _make_dummy_dataset(50, seed=2)

        datasets = {60: df60, 120: df120}
        feature_cols = ["speed_mean", "speed_std", "accel_mean"]
        feature_columns_map = {60: feature_cols, 120: feature_cols}

        result = optimize_with_window(
            datasets=datasets,
            window_sizes=[60, 120],
            feature_columns_map=feature_columns_map,
            target_column="density",
            n_splits=2,
            n_trials=5,
        )

        assert "best_params" in result
        assert "best_value" in result
        assert "window_size" in result["best_params"]
        assert result["best_params"]["window_size"] in [60, 120]
        assert result["best_value"] > 0

    def test_returns_xgboost_params(self):
        """best_params contains all XGBoost hyperparameters."""
        df = _make_dummy_dataset(30, seed=42)
        datasets = {60: df}
        feature_cols = ["speed_mean", "speed_std", "accel_mean"]

        result = optimize_with_window(
            datasets=datasets,
            window_sizes=[60],
            feature_columns_map={60: feature_cols},
            n_splits=2,
            n_trials=3,
        )

        expected_keys = {
            "window_size",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
        }
        assert expected_keys == set(result["best_params"].keys())

    def test_window_best_scores(self):
        """Result includes per-window best scores."""
        df60 = _make_dummy_dataset(50, seed=1)
        df120 = _make_dummy_dataset(50, seed=2)

        result = optimize_with_window(
            datasets={60: df60, 120: df120},
            window_sizes=[60, 120],
            feature_columns_map={
                60: ["speed_mean", "speed_std", "accel_mean"],
                120: ["speed_mean", "speed_std", "accel_mean"],
            },
            n_splits=2,
            n_trials=6,
        )

        assert "window_best" in result
        # At least one window should have a score
        assert len(result["window_best"]) >= 1

    def test_all_trials_recorded(self):
        """All completed trials are returned."""
        df = _make_dummy_dataset(30)
        n_trials = 4

        result = optimize_with_window(
            datasets={60: df},
            window_sizes=[60],
            feature_columns_map={60: ["speed_mean", "speed_std", "accel_mean"]},
            n_splits=2,
            n_trials=n_trials,
        )

        assert "all_trials" in result
        assert len(result["all_trials"]) == n_trials
        for trial in result["all_trials"]:
            assert "number" in trial
            assert "value" in trial
            assert "params" in trial

    def test_custom_search_space(self):
        """Custom search space ranges are respected."""
        df = _make_dummy_dataset(30)

        result = optimize_with_window(
            datasets={60: df},
            window_sizes=[60],
            feature_columns_map={60: ["speed_mean", "speed_std", "accel_mean"]},
            n_splits=2,
            n_trials=3,
            search_space={
                "n_estimators": [50, 100],
                "max_depth": [2, 4],
                "learning_rate": [0.05, 0.2],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.6, 0.8],
                "min_child_weight": [1, 3],
            },
        )

        params = result["best_params"]
        assert 50 <= params["n_estimators"] <= 100
        assert 2 <= params["max_depth"] <= 4
        assert 1 <= params["min_child_weight"] <= 3


class TestWindowFeatureExtraction:
    """Test that slicing window from 300s time series produces valid features."""

    def test_slice_and_extract(self):
        """Slicing a 300-step sequence to 60 steps produces valid features."""
        rng = np.random.RandomState(42)
        # Simulate a (6, 300) time series
        seq = rng.randn(6, 300).astype(np.float32)
        seq[4, :] = np.abs(seq[4, :]) * 10  # speed channel positive
        seq[5, :] = (rng.rand(300) > 0.8).astype(np.float32)  # brake binary

        # Slice last 60 timesteps
        sliced = seq[:, -60:]
        assert sliced.shape == (6, 60)

        from src.data.preprocessing import CHANNELS

        trajectory = pd.DataFrame(sliced.T, columns=CHANNELS)
        feats = extract_features(trajectory)

        assert isinstance(feats, dict)
        assert len(feats) > 0
        assert "speed_mean" in feats
        assert np.isfinite(feats["speed_mean"])

    def test_different_windows_different_features(self):
        """Different window sizes from same sequence produce different feature values."""
        rng = np.random.RandomState(42)
        seq = rng.randn(6, 300).astype(np.float32)
        seq[4, :] = np.cumsum(np.abs(rng.randn(300))) * 0.1  # trending speed

        from src.data.preprocessing import CHANNELS

        feats_60 = extract_features(
            pd.DataFrame(seq[:, -60:].T, columns=CHANNELS),
            feature_names=["speed_mean"],
        )
        feats_300 = extract_features(
            pd.DataFrame(seq.T, columns=CHANNELS),
            feature_names=["speed_mean"],
        )

        # Different windows should give different mean speed
        assert feats_60["speed_mean"] != pytest.approx(feats_300["speed_mean"], abs=0.01)


class TestResultsSerialization:
    """Test that hyperopt results can be serialized to JSON."""

    def test_results_json_roundtrip(self, tmp_path):
        """Hyperopt results dict is JSON-serializable and round-trips."""
        results = {
            "best_params": {
                "window_size": 120,
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.72,
                "min_child_weight": 3,
            },
            "best_value": 2.31,
            "window_best": {"60": 3.15, "120": 2.31, "180": 2.45, "300": 2.52},
            "final_metrics": {"rmse": 2.28, "mae": 1.71, "mape": 8.3, "r2": 0.94},
        }

        path = tmp_path / "results.json"
        with open(path, "w") as f:
            json.dump(results, f)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["best_params"]["window_size"] == 120
        assert loaded["best_value"] == pytest.approx(2.31)
        assert "120" in loaded["window_best"]
