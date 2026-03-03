"""Time-series and frequency-domain features from speed and VX channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _autocorr_lag1(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x, y = values[:-1], values[1:]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _fft_dominant_freq(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(values))
    if len(spectrum) <= 1:
        return 0.0
    dominant_idx = int(np.argmax(spectrum[1:])) + 1
    return float(dominant_idx / len(values))


@register_feature("speed_autocorr_lag1")
def speed_autocorr_lag1(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return _autocorr_lag1(trajectory["speed"].values)


@register_feature("speed_fft_dominant_freq")
def speed_fft_dominant_freq(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return _fft_dominant_freq(trajectory["speed"].values)


@register_feature("vx_autocorr_lag1")
def vx_autocorr_lag1(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return _autocorr_lag1(trajectory["VX"].values)


@register_feature("vx_fft_dominant_freq")
def vx_fft_dominant_freq(trajectory: pd.DataFrame, **kwargs: object) -> float:
    return _fft_dominant_freq(trajectory["VX"].values)


@register_feature("sample_entropy")
def sample_entropy(trajectory: pd.DataFrame, **kwargs: object) -> float:
    """Approximate sample entropy with template length m=2 and tolerance r=0.2*std."""
    speeds = trajectory["speed"].values
    m = 2
    r_ratio = 0.2

    n = len(speeds)
    if n < m + 2:
        return 0.0

    std = float(np.std(speeds))
    if std == 0.0:
        return 0.0
    r = r_ratio * std

    def _count_templates(template_len: int) -> int:
        templates = np.array([speeds[i : i + template_len] for i in range(n - template_len + 1)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    b = _count_templates(m)
    a = _count_templates(m + 1)

    if b == 0 or a == 0:
        return 0.0

    return float(-np.log(a / b))
