#!/usr/bin/env python3
"""Lightweight benchmark for the deployed UrbanFlow service."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from random import Random


def _build_predict_payload() -> bytes:
    rng = Random(42)
    records: list[dict[str, float]] = []
    x = 0.0
    for i in range(300):
        speed = max(0.1, 20.0 + rng.gauss(0, 1.0))
        x += speed
        records.append(
            {
                "time": float(i),
                "x": x,
                "y": rng.gauss(0, 0.1),
                "speed": speed,
                "brake": 0.0,
            }
        )
    return json.dumps(
        {
            "fcd_records": records,
            "speed_limit": 22.22,
            "num_lanes": 2,
        }
    ).encode()


def _build_feature_payload() -> bytes:
    return json.dumps(
        {
            "session_id": "benchmark-session",
            "timestamp": 1710000000.0,
            "speed_limit": 22.22,
            "num_lanes": 2,
            "buffer_count": 300,
            "features": {
                "speed_mean": 18.0,
                "speed_std": 1.2,
                "speed_cv": 0.066,
                "speed_iqr": 1.5,
                "speed_min": 15.0,
                "speed_max": 21.0,
                "speed_median": 18.1,
                "speed_p10": 16.0,
                "speed_p90": 20.0,
                "vy_mean": 0.0,
                "vy_std": 0.1,
                "vy_min": -0.2,
                "vy_max": 0.2,
                "ax_mean": 0.0,
                "ax_std": 0.4,
                "ay_mean": 0.0,
                "ay_std": 0.1,
                "jerk_mean": 0.0,
                "jerk_std": 0.2,
                "stop_count": 0.0,
                "stop_time_ratio": 0.0,
                "mean_stop_duration": 0.0,
                "speed_autocorr_lag1": 0.85,
                "speed_fft_dominant_freq": 0.03,
                "sample_entropy": 0.8,
                "brake_count": 0.0,
                "brake_time_ratio": 0.0,
                "mean_brake_duration": 0.0,
                "vy_variance": 0.01,
                "vy_energy": 0.3,
                "density_per_lane": 6.0,
                "flow_per_lane": 240.0,
                "gap_mean": 30.0,
            },
        }
    ).encode()


def _bench(url: str, *, data: bytes | None = None, runs: int = 7) -> dict[str, float]:
    values: list[float] = []
    headers = {"Content-Type": "application/json"} if data is not None else {}
    method = "POST" if data is not None else "GET"

    for _ in range(runs):
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        values.append((time.perf_counter() - started) * 1000)

    warm = values[1:] if len(values) > 1 else values
    return {
        "cold_ms": round(values[0], 1),
        "warm_median_ms": round(statistics.median(warm), 1),
        "warm_mean_ms": round(statistics.mean(warm), 1),
        "warm_min_ms": round(min(warm), 1),
        "warm_max_ms": round(max(warm), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://traffic-estimator-gcbqhrztha-du.a.run.app")
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results = {
        "health": _bench(f"{base_url}/health?format=json", runs=args.runs),
        "map_page": _bench(f"{base_url}/map", runs=args.runs),
        "predict": _bench(f"{base_url}/predict", data=_build_predict_payload(), runs=args.runs),
        "ingest_features": _bench(
            f"{base_url}/ingest-features",
            data=_build_feature_payload(),
            runs=args.runs,
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
