"""In-memory ensemble aggregator with CF-weighted rolling window.

Manages per-link prediction aggregation: when multiple probes traverse
the same link within a configurable time window (default 15 min, per HCM),
their predictions are combined using car-following intensity weights.
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
    """Per-link rolling ensemble with CF-weighted aggregation.

    Parameters
    ----------
    window_seconds:
        Maximum gap between consecutive probes before freezing.
        Default 900s = 15 minutes (HCM LOS analysis period).
    temperature:
        Softmax temperature for CF-score weighting.
        Higher = more uniform weights, lower = sharper weighting.
    """

    def __init__(
        self,
        window_seconds: float = 900.0,
        temperature: float = 1.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.temperature = temperature
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
        """Recompute CF-weighted ensemble density and flow."""
        preds = state.predictions
        if not preds:
            return

        if len(preds) == 1:
            state.ensemble_density = preds[0].density
            state.ensemble_flow = preds[0].flow
            state.probe_count = 1
            return

        # Softmax over CF scores
        scores = [p.cf_raw_score / self.temperature for p in preds]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        weights = [e / total for e in exps]

        state.ensemble_density = sum(w * p.density for w, p in zip(weights, preds))
        state.ensemble_flow = sum(w * p.flow for w, p in zip(weights, preds))
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
