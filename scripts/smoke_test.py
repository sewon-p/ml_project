"""Smoke test: run full pipeline with 5 scenarios to catch errors early."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("Generate scenarios", "scripts/generate_scenarios.py"),
    ("Run SUMO simulations", "scripts/run_simulation.py"),
    ("Extract features", "scripts/extract_features.py"),
    ("Train model", "scripts/train.py"),
    ("Evaluate model", "scripts/evaluate.py"),
]


def run_step(
    description: str,
    script: str,
    config: str,
) -> tuple[bool, float]:
    """Run a pipeline step and return (success, elapsed_seconds)."""
    cmd = [sys.executable, script, "--config", config]
    print(f"\n{'=' * 60}")
    print(f"  [{description}]")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    return result.returncode == 0, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: 5-scenario pipeline validation.",
    )
    parser.add_argument(
        "--config",
        default="configs/smoke_test.yaml",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip scenario generation and SUMO simulation (use existing FCD data).",
    )
    args = parser.parse_args()

    sim_scripts = {
        "scripts/generate_scenarios.py",
        "scripts/run_simulation.py",
    }

    results: list[tuple[str, bool, float]] = []
    total_t0 = time.time()

    for description, script in STEPS:
        if args.skip_simulation and script in sim_scripts:
            print(f"\n  [SKIP] {description}")
            results.append((description, True, 0.0))
            continue

        ok, elapsed = run_step(description, script, args.config)
        status = "PASS" if ok else "FAIL"
        print(f"\n  [{status}] {description}  ({elapsed:.1f}s)")
        results.append((description, ok, elapsed))

        if not ok:
            print(f"\nSmoke test aborted at: {description}")
            break

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'=' * 60}")
    print("  Smoke Test Summary")
    print(f"{'=' * 60}")
    for desc, ok, elapsed in results:
        tag = "PASS" if ok else "FAIL"
        if elapsed == 0.0 and ok and desc in {d for d, _ in STEPS[:2]}:
            tag = "SKIP"
        print(f"  [{tag}] {desc:30s}  {elapsed:6.1f}s")
    print(f"{'─' * 60}")
    print(f"  Total: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print("\nSmoke test PASSED.")
    else:
        print("\nSmoke test FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
