"""Tests for the runtime Bayesian ensemble aggregator."""

from __future__ import annotations

import pytest

from src.api.ensemble import (
    EnsembleAggregator,
    bayesian_posterior_mean,
    compute_cf_score,
    compute_observation_std,
    precision_weighted_mean,
)


def test_compute_cf_score_sums_expected_terms() -> None:
    score = compute_cf_score(
        {
            "ax_std": 0.8,
            "brake_time_ratio": 0.15,
            "speed_cv": 0.25,
            "ignored": 999.0,
        }
    )
    assert score == pytest.approx(1.2)


def test_single_probe_passthrough_is_unchanged() -> None:
    agg = EnsembleAggregator(
        window_seconds=900.0,
        base_obs_std=2.0,
        cf_sensitivity=0.5,
        min_obs_std=0.1,
        prior_density=5.0,
        prior_density_std=1.0,
    )

    state = agg.add_prediction(
        link_id="L1",
        density=11.5,
        flow=420.0,
        features={"ax_std": 0.6, "brake_time_ratio": 0.1, "speed_cv": 0.2},
        timestamp=100.0,
    )

    assert state.probe_count == 1
    assert state.ensemble_density == pytest.approx(11.5)
    assert state.ensemble_flow == pytest.approx(420.0)


def test_bayesian_fusion_matches_expected_posterior() -> None:
    agg = EnsembleAggregator(
        window_seconds=900.0,
        base_obs_std=2.0,
        cf_sensitivity=0.5,
        min_obs_std=0.1,
        prior_density=12.0,
        prior_density_std=10.0,
    )

    agg.add_prediction(
        link_id="L1",
        density=10.0,
        flow=300.0,
        features={"ax_std": 0.5, "brake_time_ratio": 0.0, "speed_cv": 0.0},
        timestamp=100.0,
    )
    state = agg.add_prediction(
        link_id="L1",
        density=20.0,
        flow=500.0,
        features={"ax_std": 1.5, "brake_time_ratio": 0.0, "speed_cv": 0.0},
        timestamp=120.0,
    )

    std_1 = compute_observation_std(
        cf_score=0.5,
        base_obs_std=2.0,
        cf_sensitivity=0.5,
        min_obs_std=0.1,
    )
    std_2 = compute_observation_std(
        cf_score=1.5,
        base_obs_std=2.0,
        cf_sensitivity=0.5,
        min_obs_std=0.1,
    )
    precisions = [1.0 / (std_1**2), 1.0 / (std_2**2)]

    expected_density = bayesian_posterior_mean(
        values=[10.0, 20.0],
        precisions=precisions,
        prior_mean=12.0,
        prior_precision=1.0 / (10.0**2),
    )
    expected_flow = precision_weighted_mean(
        values=[300.0, 500.0],
        precisions=precisions,
    )

    assert state.probe_count == 2
    assert state.ensemble_density == pytest.approx(expected_density)
    assert state.ensemble_flow == pytest.approx(expected_flow)
    assert state.ensemble_density > 15.0
    assert state.ensemble_flow > 400.0
