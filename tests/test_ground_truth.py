"""Tests for Edie ground truth computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.ground_truth import (
    compute_edie_density,
    compute_edie_flow,
)


@pytest.fixture
def sample_fcd():
    """Synthetic FCD data: 5 vehicles over 10 seconds on a 1km segment."""
    records = []
    for t in np.arange(0, 10, 0.5):
        for vid in range(5):
            records.append(
                {
                    "time": t,
                    "vehicle_id": f"veh_{vid}",
                    "x": 50.0 + vid * 150.0 + t * 10.0,
                    "y": 0.0,
                    "speed": 10.0,
                    "edge_id": "link0",
                }
            )
    return pd.DataFrame(records)


class TestEdieDensity:
    def test_positive_density(self, sample_fcd):
        k = compute_edie_density(
            sample_fcd,
            t_start=0,
            t_end=10,
            x_start=0,
            x_end=1000,
        )
        assert k > 0

    def test_zero_for_empty_region(self, sample_fcd):
        k = compute_edie_density(
            sample_fcd,
            t_start=0,
            t_end=10,
            x_start=9000,
            x_end=10000,
        )
        assert k == 0.0

    def test_zero_for_invalid_window(self, sample_fcd):
        k = compute_edie_density(
            sample_fcd,
            t_start=5,
            t_end=5,
            x_start=0,
            x_end=1000,
        )
        assert k == 0.0


class TestEdieFlow:
    def test_positive_flow(self, sample_fcd):
        q = compute_edie_flow(
            sample_fcd,
            t_start=0,
            t_end=10,
            x_start=0,
            x_end=1000,
        )
        assert q > 0

    def test_zero_for_empty_region(self, sample_fcd):
        q = compute_edie_flow(
            sample_fcd,
            t_start=0,
            t_end=10,
            x_start=9000,
            x_end=10000,
        )
        assert q == 0.0
