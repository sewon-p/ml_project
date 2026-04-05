"""In-memory ensemble aggregator with Bayesian CF-aware rolling fusion.

Manages per-link prediction aggregation: when multiple probes traverse
the same link within a configurable time window (default 15 min, per HCM),
their predictions are combined with a Bayesian update whose observation
noise shrinks as car-following intensity rises.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbePrediction:
    """Single probe's prediction for a link."""

    density: float
    flow: float
    cf_raw_score: float  # ax_std + brake_time_ratio + speed_cv
    timestamp: float  # unix timestamp
    prediction_id: int | None = None
    session_id: str | None = None


@dataclass
class EnsembleState:
    """Current ensemble state for a link."""

    link_id: str
    predictions: list[ProbePrediction] = field(default_factory=list)
    ensemble_density: float = 0.0
    ensemble_flow: float = 0.0
    probe_count: int = 0
    window_start: float = 0.0
    window_end: float = 0.0
    is_frozen: bool = False


class EnsembleAggregator:
    """Per-link rolling ensemble with Bayesian CF-aware aggregation.

    Parameters
    ----------
    window_seconds:
        Maximum gap between consecutive probes before freezing.
        Default 900s = 15 minutes (HCM LOS analysis period).
    base_obs_std:
        Base observation standard deviation before CF adjustment.
    cf_sensitivity:
        Controls how strongly higher CF scores reduce observation noise.
    min_obs_std:
        Lower bound on the observation standard deviation.
    prior_density:
        Weak prior mean used for the Bayesian density update.
    prior_density_std:
        Weak prior standard deviation used for the Bayesian density update.
    """

    def __init__(
        self,
        window_seconds: float = 900.0,
        base_obs_std: float = 1.0,
        cf_sensitivity: float = 1.0,
        min_obs_std: float = 0.1,
        prior_density: float = 0.0,
        prior_density_std: float = 100.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.base_obs_std = base_obs_std
        self.cf_sensitivity = cf_sensitivity
        self.min_obs_std = min_obs_std
        self.prior_density = prior_density
        self.prior_density_precision = (
            0.0 if prior_density_std <= 0 else 1.0 / (prior_density_std**2)
        )
        self._active: dict[str, EnsembleState] = {}

    def add_prediction(
        self,
        link_id: str,
        density: float,
        flow: float,
        features: dict[str, float],
        timestamp: float | None = None,
        prediction_id: int | None = None,
        session_id: str | None = None,
    ) -> EnsembleState:
        """Add a probe prediction and return updated ensemble state."""
        ts = timestamp or time.time()
        cf_score = compute_cf_score(features)

        probe = ProbePrediction(
            density=density,
            flow=flow,
            cf_raw_score=cf_score,
            timestamp=ts,
            prediction_id=prediction_id,
            session_id=session_id,
        )

        state = self._active.get(link_id)

        # Start new ensemble if none exists or window expired
        if state is None or state.is_frozen or (ts - state.window_end) > self.window_seconds:
            state = EnsembleState(
                link_id=link_id,
                predictions=[probe],
                window_start=ts,
                window_end=ts,
            )
            self._active[link_id] = state
        else:
            state.predictions.append(probe)
            state.window_end = ts
            # Prune predictions older than window_seconds from the newest
            cutoff = ts - self.window_seconds
            state.predictions = [p for p in state.predictions if p.timestamp >= cutoff]
            if state.predictions:
                state.window_start = state.predictions[0].timestamp

        # Recompute ensemble
        self._recompute(state)
        return state

    def get_state(self, link_id: str) -> EnsembleState | None:
        """Get current ensemble state for a link."""
        return self._active.get(link_id)

    def freeze_stale(self, now: float | None = None) -> int:
        """Freeze ensembles with no new probes within the window.

        Returns number of ensembles frozen.
        """
        now = now or time.time()
        frozen = 0
        for state in self._active.values():
            if not state.is_frozen and (now - state.window_end) > self.window_seconds:
                state.is_frozen = True
                frozen += 1
        return frozen

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove frozen ensembles older than max_age_seconds."""
        now = time.time()
        stale = [
            lid
            for lid, s in self._active.items()
            if s.is_frozen and (now - s.window_end) > max_age_seconds
        ]
        for lid in stale:
            del self._active[lid]
        return len(stale)

    def _recompute(self, state: EnsembleState) -> None:
        """Recompute Bayesian CF-aware ensemble density and flow."""
        preds = state.predictions
        if not preds:
            return

        if len(preds) == 1:
            state.ensemble_density = preds[0].density
            state.ensemble_flow = preds[0].flow
            state.probe_count = 1
            return

        obs_stds = [
            compute_observation_std(
                cf_score=p.cf_raw_score,
                base_obs_std=self.base_obs_std,
                cf_sensitivity=self.cf_sensitivity,
                min_obs_std=self.min_obs_std,
            )
            for p in preds
        ]
        precisions = [1.0 / (std**2) for std in obs_stds]

        state.ensemble_density = bayesian_posterior_mean(
            values=[p.density for p in preds],
            precisions=precisions,
            prior_mean=self.prior_density,
            prior_precision=self.prior_density_precision,
        )
        state.ensemble_flow = precision_weighted_mean(
            values=[p.flow for p in preds],
            precisions=precisions,
        )
        state.probe_count = len(preds)

    @property
    def active_links(self) -> int:
        """Number of links with active (non-frozen) ensembles."""
        return sum(1 for s in self._active.values() if not s.is_frozen)

    def snapshot(self) -> dict[str, Any]:
        """Return summary for monitoring."""
        return {
            "total_links": len(self._active),
            "active_links": self.active_links,
            "frozen_links": len(self._active) - self.active_links,
            "window_seconds": self.window_seconds,
        }


def compute_cf_score(features: dict[str, float]) -> float:
    """Compute car-following intensity score from feature dict.

    Higher score = more car-following behavior = more informative probe.
    Uses ax_std + brake_time_ratio + speed_cv as CF indicators.
    """
    ax_std = features.get("ax_std", 0.0)
    brake_ratio = features.get("brake_time_ratio", 0.0)
    speed_cv = features.get("speed_cv", 0.0)
    return ax_std + brake_ratio + speed_cv


def compute_observation_std(
    *,
    cf_score: float,
    base_obs_std: float,
    cf_sensitivity: float,
    min_obs_std: float,
) -> float:
    """Convert CF intensity to observation noise for Bayesian fusion."""
    std = base_obs_std * math.exp(-cf_sensitivity * cf_score)
    return max(min_obs_std, std)


def precision_weighted_mean(values: list[float], precisions: list[float]) -> float:
    """Compute a precision-weighted average."""
    total_precision = sum(precisions)
    if total_precision <= 0:
        return sum(values) / len(values)
    return sum(value * precision for value, precision in zip(values, precisions)) / total_precision


def bayesian_posterior_mean(
    *,
    values: list[float],
    precisions: list[float],
    prior_mean: float,
    prior_precision: float,
) -> float:
    """Compute posterior mean from a Gaussian prior and Gaussian observations."""
    obs_precision = sum(precisions)
    if obs_precision <= 0:
        return prior_mean
    numerator = prior_mean * prior_precision + sum(
        value * precision for value, precision in zip(values, precisions)
    )
    denominator = prior_precision + obs_precision
    return numerator / denominator
