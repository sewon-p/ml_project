"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mape,
    r2_score,
    rmse,
)


class TestMetrics:
    def test_rmse_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1 / 3)
        assert rmse(y_true, y_pred) == pytest.approx(expected)

    def test_mae_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_mape_perfect(self) -> None:
        y = np.array([2.0, 3.0, 4.0])
        assert mape(y, y) == pytest.approx(0.0, abs=0.01)

    def test_mape_known(self) -> None:
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.1 + 0.1 → mean = 0.1 → 10%
        assert mape(y_true, y_pred) == pytest.approx(10.0, abs=0.1)

    def test_mape_filters_low_density(self) -> None:
        # Samples with y_true <= 1.0 should be excluded
        y_true = np.array([0.5, 100.0, 200.0])
        y_pred = np.array([50.0, 110.0, 180.0])
        # Only [100, 200] kept → |10/100| + |20/200| = 0.1 + 0.1 → 10%
        assert mape(y_true, y_pred) == pytest.approx(10.0, abs=0.1)

    def test_mape_no_min_denominator(self) -> None:
        # With min_denominator=0, all samples included
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([2.2, 3.6])
        # |0.2/2| + |0.4/4| = 0.1 + 0.1 → mean = 0.1 → 10%
        assert mape(y_true, y_pred, min_denominator=0.0) == pytest.approx(10.0, abs=0.1)

    def test_r2_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_constant_true(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        # ss_tot = 0, should return 0.0
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_compute_all_metrics(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.1, 20.1, 29.9])
        result = compute_all_metrics(y_true, y_pred)
        assert set(result.keys()) == {"rmse", "mae", "mape", "r2"}
        assert result["rmse"] >= 0
        assert result["r2"] <= 1.0
