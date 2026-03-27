#!/usr/bin/env python3
"""Benchmark the grid-indexed GIS link matcher on local deployment assets."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gis.link_matcher import LinkMatcher  # noqa: E402


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    matcher = LinkMatcher.from_config(args.config)
    load_ms = (time.perf_counter() - started) * 1000
    if matcher is None:
        raise SystemExit("Link matcher could not be loaded from config.")

    rng = random.Random(args.seed)
    sample_count = min(args.samples, len(matcher.links))
    sampled_links = rng.sample(matcher.links, sample_count)

    durations_ms: list[float] = []
    for link in sampled_links:
        lat = link.center_lat + rng.uniform(-1e-5, 1e-5)
        lon = link.center_lon + rng.uniform(-1e-5, 1e-5)
        started = time.perf_counter()
        result = matcher.match(lat=lat, lon=lon)
        durations_ms.append((time.perf_counter() - started) * 1000)
        if result is None:
            raise RuntimeError("Matcher returned no result for a sampled link center.")

    print(
        json.dumps(
            {
                "config": args.config,
                "samples": sample_count,
                "load_ms": round(load_ms, 1),
                "links": len(matcher.links),
                "grid_cells": len(matcher._grid),
                "median_ms": round(statistics.median(durations_ms), 3),
                "mean_ms": round(statistics.mean(durations_ms), 3),
                "p95_ms": round(_p95(durations_ms), 3),
                "min_ms": round(min(durations_ms), 3),
                "max_ms": round(max(durations_ms), 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
