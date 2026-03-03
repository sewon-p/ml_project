"""Tests for feature extraction from single-vehicle trajectory."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.pipeline import extract_features


class TestBasicStats:
    def test_speed_stats(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "speed_mean",
                "speed_std",
                "speed_cv",
                "speed_iqr",
                "speed_min",
                "speed_max",
                "speed_median",
                "speed_p10",
                "speed_p90",
            ],
        )
        assert result["speed_mean"] == pytest.approx(25.0, abs=1.0)
        assert result["speed_std"] > 0
        assert result["speed_cv"] > 0
        assert result["speed_iqr"] > 0
        assert result["speed_min"] < result["speed_max"]
        assert result["speed_p10"] < result["speed_p90"]

    def test_vx_stats(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "vx_mean",
                "vx_std",
                "vx_min",
                "vx_max",
            ],
        )
        assert result["vx_mean"] == pytest.approx(25.0, abs=1.5)
        assert result["vx_std"] > 0
        assert result["vx_min"] < result["vx_max"]

    def test_vy_stats(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "vy_mean",
                "vy_std",
                "vy_min",
                "vy_max",
            ],
        )
        assert abs(result["vy_mean"]) < 1.0
        assert result["vy_std"] > 0


class TestAcceleration:
    def test_ax_features(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "ax_mean",
                "ax_std",
                "harsh_accel_count",
                "harsh_decel_count",
                "jerk_mean",
                "jerk_std",
            ],
        )
        assert abs(result["ax_mean"]) < 5.0
        assert result["ax_std"] >= 0
        assert result["harsh_accel_count"] >= 0
        assert result["harsh_decel_count"] >= 0

    def test_ay_features(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "ay_mean",
                "ay_std",
            ],
        )
        assert abs(result["ay_mean"]) < 1.0
        assert result["ay_std"] >= 0


class TestStopPatterns:
    def test_stop_patterns(self, congested_trajectory):
        result = extract_features(
            congested_trajectory,
            feature_names=[
                "stop_count",
                "stop_time_ratio",
                "mean_stop_duration",
            ],
        )
        assert result["stop_count"] >= 2  # two stop segments
        assert result["stop_time_ratio"] > 0
        assert result["mean_stop_duration"] > 0


class TestTimeSeries:
    def test_time_series(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "speed_autocorr_lag1",
                "speed_fft_dominant_freq",
                "vx_autocorr_lag1",
                "vx_fft_dominant_freq",
                "sample_entropy",
            ],
        )
        assert -1.0 <= result["speed_autocorr_lag1"] <= 1.0
        assert result["speed_fft_dominant_freq"] >= 0
        assert -1.0 <= result["vx_autocorr_lag1"] <= 1.0
        assert result["vx_fft_dominant_freq"] >= 0
        assert result["sample_entropy"] >= 0


class TestBrakePatterns:
    def test_brake_features(self, congested_trajectory):
        result = extract_features(
            congested_trajectory,
            feature_names=[
                "brake_count",
                "brake_time_ratio",
                "mean_brake_duration",
            ],
        )
        assert result["brake_count"] >= 1
        assert result["brake_time_ratio"] > 0
        assert result["mean_brake_duration"] > 0

    def test_no_brake(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "brake_count",
                "brake_time_ratio",
                "mean_brake_duration",
            ],
        )
        assert result["brake_count"] == 0
        assert result["brake_time_ratio"] == 0.0
        assert result["mean_brake_duration"] == 0.0


class TestLateral:
    def test_lateral_features(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=[
                "vy_variance",
                "lane_change_count",
                "vy_energy",
            ],
        )
        assert result["vy_variance"] >= 0
        assert result["lane_change_count"] >= 0
        assert result["vy_energy"] >= 0


class TestPipeline:
    def test_extract_all(self, free_flow_trajectory):
        result = extract_features(free_flow_trajectory)
        assert isinstance(result, dict)
        assert len(result) >= 30  # at least 30 features registered

    def test_extract_subset(self, free_flow_trajectory):
        result = extract_features(
            free_flow_trajectory,
            feature_names=["speed_mean", "speed_std"],
        )
        assert set(result.keys()) == {"speed_mean", "speed_std"}


class TestEdgeCases:
    def test_single_element(self):
        trajectory = pd.DataFrame(
            {
                "VX": [15.0],
                "VY": [0.0],
                "AX": [0.0],
                "AY": [0.0],
                "speed": [15.0],
                "brake": [0.0],
            }
        )
        result = extract_features(trajectory)
        assert result["speed_mean"] == 15.0
        assert result["speed_std"] == 0.0

    def test_constant_trajectory(self, constant_trajectory):
        result = extract_features(constant_trajectory)
        assert result["speed_std"] == 0.0
        assert result["speed_cv"] == 0.0
        assert result["speed_autocorr_lag1"] == 0.0
