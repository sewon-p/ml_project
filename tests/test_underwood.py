"""Unit tests for Underwood (exponential) FD functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.underwood import (
    compute_fd_density,
    compute_fd_estimates,
    compute_fd_flow,
    compute_k_optimum,
)


class TestKOptimum:
    def test_single_lane(self):
        """1 lane: k_m = (1000 / 7.0) / e ≈ 52.55."""
        result = compute_k_optimum(1, vehicle_length=4.5, min_gap=2.5)
        expected = 1000.0 / 7.0 / np.e
        assert result == pytest.approx(expected, rel=1e-6)

    def test_multi_lane(self):
        """3 lanes: k_m = 3 × (1000 / 7.0) / e."""
        result = compute_k_optimum(3, vehicle_length=4.5, min_gap=2.5)
        expected = 3 * 1000.0 / 7.0 / np.e
        assert result == pytest.approx(expected, rel=1e-6)

    def test_vectorized(self):
        lanes = np.array([1, 2, 3])
        result = compute_k_optimum(lanes)
        expected = lanes * 1000.0 / 7.0 / np.e
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestFDDensity:
    def test_free_flow_speed(self):
        """v = v_f → ln(1) = 0 → k_fd = 0."""
        k_m = compute_k_optimum(1)
        k_fd = compute_fd_density(
            speed_mean=np.array([33.33]),
            v_free=np.array([33.33]),
            k_m=np.array([k_m]),
        )
        assert k_fd[0] == pytest.approx(0.0, abs=1e-6)

    def test_speed_at_vf_over_e(self):
        """v = v_f/e → ln(e) = 1 → k_fd = k_m (optimal density)."""
        k_m = compute_k_optimum(2)
        v_free = 27.78
        k_fd = compute_fd_density(
            speed_mean=np.array([v_free / np.e]),
            v_free=np.array([v_free]),
            k_m=np.array([k_m]),
        )
        assert k_fd[0] == pytest.approx(k_m, rel=1e-6)

    def test_low_speed_high_density(self):
        """v ≪ v_f → k_fd ≫ k_m (exponential, not capped at k_jam)."""
        k_m = compute_k_optimum(1)
        k_fd = compute_fd_density(
            speed_mean=np.array([1.0]),
            v_free=np.array([27.78]),
            k_m=np.array([k_m]),
        )
        # ln(27.78) ≈ 3.3 → k ≈ 3.3 × k_m
        assert k_fd[0] > k_m * 2

    def test_never_negative(self):
        """Density >= 0 for any speed in (0, v_f]."""
        k_m = compute_k_optimum(2)
        speeds = np.array([0.01, 5.0, 15.0, 27.78, 50.0])
        v_free = np.full(5, 27.78)
        k_fd = compute_fd_density(speeds, v_free, np.full(5, k_m))
        assert np.all(k_fd >= 0)

    def test_speed_exceeds_vfree_clipped(self):
        """Speed > v_free clips to ratio=1 → k_fd = 0."""
        k_m = compute_k_optimum(1)
        k_fd = compute_fd_density(
            speed_mean=np.array([50.0]),
            v_free=np.array([33.33]),
            k_m=np.array([k_m]),
        )
        assert k_fd[0] == pytest.approx(0.0, abs=1e-6)


class TestFDFlow:
    def test_units(self):
        """q_fd = k_fd × v_mean × 3.6 → veh/hr."""
        k_fd = np.array([50.0])
        speed_mean = np.array([10.0])
        q_fd = compute_fd_flow(speed_mean, k_fd)
        assert q_fd[0] == pytest.approx(50.0 * 10.0 * 3.6, rel=1e-6)

    def test_zero_density(self):
        """k_fd=0 → q_fd=0."""
        q_fd = compute_fd_flow(np.array([20.0]), np.array([0.0]))
        assert q_fd[0] == pytest.approx(0.0, abs=1e-10)


class TestComputeFDEstimates:
    def test_integration(self):
        speed_mean = np.array([15.0, 0.01, 27.78])
        v_free = np.array([27.78, 27.78, 27.78])
        num_lanes = np.array([2, 2, 2])

        result = compute_fd_estimates(speed_mean, v_free, num_lanes)

        assert "k_m" in result
        assert "k_fd" in result
        assert "q_fd" in result

        k_m_expected = 2 * 1000.0 / 7.0 / np.e
        np.testing.assert_allclose(result["k_m"], k_m_expected, rtol=1e-6)

        # v ≈ 0 → very high density
        assert result["k_fd"][1] > k_m_expected * 3
        # v = v_free → k_fd = 0
        assert result["k_fd"][2] == pytest.approx(0.0, abs=1e-6)


class TestResidualRoundtrip:
    def test_roundtrip(self):
        """k_actual = k_fd + delta_k, exact restoration."""
        rng = np.random.RandomState(42)
        n = 100
        speed_mean = rng.uniform(1, 30, n)
        v_free = np.full(n, 33.33)
        num_lanes = rng.choice([1, 2, 3], n)
        k_actual = rng.uniform(0, 200, n)

        fd = compute_fd_estimates(speed_mean, v_free, num_lanes)
        k_fd = fd["k_fd"]
        delta_k = k_actual - k_fd
        restored = k_fd + delta_k

        np.testing.assert_allclose(restored, k_actual, rtol=1e-10)
