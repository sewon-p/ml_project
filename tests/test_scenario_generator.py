"""Tests for stochastic scenario generator."""

from __future__ import annotations

import pytest

from src.simulation.scenario_generator import generate_scenario_matrix


@pytest.fixture
def sim_cfg():
    """Minimal simulation config for testing."""
    return {
        "network": {
            "num_lanes": {"min": 1, "max": 3},
            "speed_limit_kmh": [50, 60, 80, 100],
        },
        "demand": {
            "per_lane_min_vehph": 800,
            "per_lane_max_vehph": 2200,
        },
        "vehicle_types": {
            "truck_ratio": {"mean": 0.16, "std": 0.05, "min": 0.0, "max": 0.40},
            "passenger": {
                "tau": {"mean": 1.45, "std": 1.07, "min": 0.2, "max": 4.0},
                "decel": {"mean": 4.5, "std": 1.0, "min": 1.0, "max": 7.0},
                "minGap": {"mean": 2.5, "std": 0.8, "min": 0.5, "max": 6.0},
                "speedFactor": {"mean": 1.0, "std": 0.1, "min": 0.2, "max": 2.0},
                "sigma": {"mean": 0.5, "std": 0.15, "min": 0.0, "max": 1.0},
                "accel": {"mean": 2.6, "std": 0.6, "min": 1.0, "max": 4.0},
                "length": 4.5,
            },
            "truck": {
                "tau": {"mean": 1.45, "std": 1.07, "min": 0.2, "max": 4.0},
                "decel": {"mean": 3.0, "std": 0.8, "min": 1.0, "max": 5.0},
                "minGap": {"mean": 3.5, "std": 1.0, "min": 1.0, "max": 8.0},
                "speedFactor": {"mean": 0.85, "std": 0.08, "min": 0.2, "max": 1.5},
                "sigma": {"mean": 0.4, "std": 0.1, "min": 0.0, "max": 1.0},
                "accel": {"mean": 1.3, "std": 0.4, "min": 0.5, "max": 2.5},
                "length": 12.0,
            },
        },
    }


class TestScenarioGenerator:
    def test_generates_correct_count(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=100)
        assert len(df) == 100

    def test_scenario_ids_unique(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=50)
        assert df["scenario_id"].nunique() == 50

    def test_num_lanes_in_range(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=200)
        assert df["num_lanes"].min() >= 1
        assert df["num_lanes"].max() <= 3

    def test_speed_limit_from_choices(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=200)
        valid_kmh = {50.0, 60.0, 80.0, 100.0}
        assert set(df["speed_limit_kmh"].unique()).issubset(valid_kmh)

    def test_speed_limit_converted(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=10)
        for _, row in df.iterrows():
            expected_ms = row["speed_limit_kmh"] / 3.6
            assert abs(row["speed_limit"] - expected_ms) < 0.01

    def test_demand_in_range(self, sim_cfg):
        """demand is clipped to 95% of theoretical Krauss capacity per lane."""
        df = generate_scenario_matrix(sim_cfg, num_simulations=500)
        for _, row in df.iterrows():
            n = int(row["num_lanes"])
            assert row["demand_vehph"] >= 1 * n  # at least 1 veh/hr/lane
            assert row["demand_vehph"] <= 2200 * n  # upper bound from config

    def test_dynamic_link_length(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=200)
        for _, row in df.iterrows():
            expected = round(row["speed_limit"] * 600.0 * 1.2, 1)
            assert row["link_length"] == expected

    def test_truck_ratio_clipped(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=500)
        assert df["truck_ratio"].min() >= 0.0
        assert df["truck_ratio"].max() <= 0.40

    def test_vehicle_params_present(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=10)
        for prefix in ("passenger", "truck"):
            for key in ("tau", "decel", "minGap", "speedFactor", "sigma", "accel", "length"):
                col = f"{prefix}_{key}"
                assert col in df.columns, f"Missing column: {col}"

    def test_vehicle_params_clipped(self, sim_cfg):
        df = generate_scenario_matrix(sim_cfg, num_simulations=500)
        # Check passenger tau is within bounds
        assert df["passenger_tau"].min() >= 0.2
        assert df["passenger_tau"].max() <= 4.0
        # Check truck decel is within bounds
        assert df["truck_decel"].min() >= 1.0
        assert df["truck_decel"].max() <= 5.0

    def test_reproducible_with_same_seed(self, sim_cfg):
        df1 = generate_scenario_matrix(sim_cfg, num_simulations=10, base_seed=99)
        df2 = generate_scenario_matrix(sim_cfg, num_simulations=10, base_seed=99)
        assert df1.equals(df2)
