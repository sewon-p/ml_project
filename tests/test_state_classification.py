"""Tests for traffic state classification."""

from __future__ import annotations

import numpy as np

from src.evaluation.state_classification import classify_traffic_state


class TestStateClassification:
    def test_all_free(self) -> None:
        speeds = np.array([25.0, 30.0, 22.0])
        states = classify_traffic_state(speeds)
        assert all(s == "free" for s in states)

    def test_all_congested(self) -> None:
        speeds = np.array([2.0, 5.0, 8.0])
        states = classify_traffic_state(speeds)
        assert all(s == "congested" for s in states)

    def test_mixed(self) -> None:
        speeds = np.array([5.0, 15.0, 25.0])
        states = classify_traffic_state(speeds)
        assert states[0] == "congested"
        assert states[1] == "saturated"
        assert states[2] == "free"

    def test_custom_thresholds(self) -> None:
        speeds = np.array([5.0, 15.0, 25.0])
        states = classify_traffic_state(
            speeds,
            congested_threshold=6.0,
            saturated_threshold=16.0,
        )
        assert states[0] == "congested"
        assert states[1] == "saturated"
        assert states[2] == "free"
