"""Validate and clean up incomplete simulation outputs.

Checks each scenario's fcd.csv for:
1. File exists and is not empty
2. Has the expected CSV header
3. Data covers the full collection period (time reaches warmup + collect_time)
4. Has a minimum number of data rows

Removes invalid scenarios so they can be re-run with --resume.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def validate_fcd(fcd_path: Path, expected_max_time: float = 900.0, min_rows: int = 1000) -> str:
    """Validate a single fcd.csv file. Returns 'ok' or error reason."""
    if not fcd_path.exists():
        return "missing"

    size = fcd_path.stat().st_size
    if size < 100:
        return "too_small"

    try:
        with open(fcd_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or "time" not in header:
                return "bad_header"

            time_idx = header.index("time")
            row_count = 0
            max_time = 0.0

            for row in reader:
                row_count += 1
                try:
                    t = float(row[time_idx])
                    if t > max_time:
                        max_time = t
                except (ValueError, IndexError):
                    continue

            if row_count < min_rows:
                return f"too_few_rows({row_count})"

            # Check if simulation reached at least 95% of expected time
            if max_time < expected_max_time * 0.95:
                return f"truncated(max_t={max_time:.0f})"

    except Exception as e:
        return f"read_error({e})"

    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and clean FCD outputs.")
    parser.add_argument("--fcd-dir", default="data/fcd", help="FCD output directory")
    parser.add_argument("--warmup", type=float, default=300.0)
    parser.add_argument("--collect", type=float, default=600.0)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument(
        "--delete-invalid", action="store_true", help="Delete invalid scenario dirs"
    )
    parser.add_argument(
        "--delete-empty-dirs", action="store_true", help="Delete dirs without fcd.csv"
    )
    args = parser.parse_args()

    fcd_dir = Path(args.fcd_dir)
    expected_max_time = args.warmup + args.collect

    if not fcd_dir.exists():
        print(f"Directory not found: {fcd_dir}")
        return

    dirs = sorted([d for d in fcd_dir.iterdir() if d.is_dir()], key=lambda d: d.name)

    stats = {"ok": 0, "missing": 0, "invalid": 0}
    invalid_dirs = []

    for d in dirs:
        fcd_path = d / "fcd.csv"
        result = validate_fcd(fcd_path, expected_max_time, args.min_rows)

        if result == "ok":
            stats["ok"] += 1
        elif result == "missing":
            stats["missing"] += 1
            if args.delete_empty_dirs:
                shutil.rmtree(d)
                print(f"  Deleted empty dir: {d.name}")
        else:
            stats["invalid"] += 1
            invalid_dirs.append((d.name, result))
            if args.delete_invalid:
                # Just delete the fcd.csv so --resume will re-run it
                fcd_path.unlink(missing_ok=True)
                print(f"  Deleted invalid fcd.csv: {d.name} ({result})")

    print("\n=== Validation Results ===")
    print(f"  Valid:       {stats['ok']}")
    print(f"  Missing:     {stats['missing']} (no fcd.csv)")
    print(f"  Invalid:     {stats['invalid']} (truncated/corrupted)")
    print(f"  Total dirs:  {len(dirs)}")
    print(f"  Usable samples: ~{stats['ok'] * 5}")

    if invalid_dirs and not args.delete_invalid:
        print("\nInvalid scenarios (run with --delete-invalid to clean):")
        for name, reason in invalid_dirs[:20]:
            print(f"  {name}: {reason}")
        if len(invalid_dirs) > 20:
            print(f"  ... and {len(invalid_dirs) - 20} more")


if __name__ == "__main__":
    main()
