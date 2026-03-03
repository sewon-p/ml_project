"""Extended tests for Fundamental Diagram baseline."""

from __future__ import annotations

import numpy as np

from src.models.fd_baseline import FDBaselineEstimator


class TestFDBaselineCalibration:
    def test_calibrate_from_data(self):
        est = FDBaselineEstimator()
        speeds = np.linspace(1, 30, 50)
        densities = 150.0 * (1 - speeds / 33.33)
        X = speeds.reshape(-1, 1)
        result = est.fit(X, densities, calibrate=True)
        assert "v_free" in result
        assert "k_jam" in result
        preds = est.predict(X)
        error = np.mean(np.abs(preds - densities))
        assert error < 20.0

    def test_no_calibrate(self):
        est = FDBaselineEstimator(v_free=30.0, k_jam=100.0)
        X = np.random.randn(10, 1)
        y = np.random.randn(10)
        result = est.fit(X, y)
        assert result["v_free"] == 30.0
        assert result["k_jam"] == 100.0
