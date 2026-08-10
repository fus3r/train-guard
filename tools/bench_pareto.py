#!/usr/bin/env python3
"""Check and benchmark the Pareto-front pass on an adversarial workload."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainguard.native import KernelRow
from trainguard.sweep import _pareto_flags

ParetoPass = Callable[[list[KernelRow]], list[bool]]


def _row(run: float, hot: float, low: float) -> KernelRow:
    return KernelRow(0.0, 0.0, 0.0, run, hot, low, 0, 0, "0" * 16, None)


def _adversarial_rows(count: int) -> list[KernelRow]:
    return [_row(float(index), float(index), 0.0) for index in range(count)]


def _pairwise(rows: list[KernelRow]) -> list[bool]:
    flags = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other.run_seconds >= candidate.run_seconds
            and other.hot_degc_seconds <= candidate.hot_degc_seconds
            and other.low_battery_run_seconds <= candidate.low_battery_run_seconds
            and (
                other.run_seconds > candidate.run_seconds
                or other.hot_degc_seconds < candidate.hot_degc_seconds
                or other.low_battery_run_seconds < candidate.low_battery_run_seconds
            )
            for other in rows
        )
        flags.append(not dominated)
    return flags


def _measure(
    function: ParetoPass,
    rows: list[KernelRow],
    repeat: int,
) -> tuple[list[bool], list[float]]:
    result = []
    timings = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = function(rows)
        timings.append(time.perf_counter() - started)
    return result, timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--oracle-points", type=int, default=2_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.points < 1 or not 1 <= args.oracle_points <= args.points or args.repeat < 1:
        parser.error("points and repeat must be positive, with oracle-points <= points")

    mixed = [
        _row(10.0, 5.0, 5.0),
        _row(9.0, 6.0, 6.0),
        _row(10.0, 5.0, 5.0),
        _row(10.0, 4.0, 7.0),
        _row(11.0, 8.0, 2.0),
        _row(8.0, 4.0, 8.0),
        _row(10.0, 6.0, 4.0),
    ]
    mixed_expected = [True, False, True, True, True, False, True]
    if _pairwise(mixed) != mixed_expected or _pareto_flags(mixed) != mixed_expected:
        print("Pareto result disagrees with the hand-checked mixed corpus", file=sys.stderr)
        return 1

    rows = _adversarial_rows(args.points)
    oracle_rows = rows[: args.oracle_points]
    expected, oracle_timings = _measure(_pairwise, oracle_rows, 1)
    observed, checked_timings = _measure(_pareto_flags, oracle_rows, args.repeat)
    if observed != expected:
        print("optimized result disagrees with the pairwise oracle", file=sys.stderr)
        return 1

    full, full_timings = _measure(_pareto_flags, rows, args.repeat)
    if sum(full) != args.points:
        print("adversarial workload should leave every point non-dominated", file=sys.stderr)
        return 1

    report = {
        "schema_version": 1,
        "workload": "all_points_non_dominated",
        "mixed_corpus_agreement": True,
        "oracle_agreement": True,
        "points": args.points,
        "oracle_points": args.oracle_points,
        "repeat": args.repeat,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optimized_full_seconds": {
            "median": statistics.median(full_timings),
            "minimum": min(full_timings),
            "maximum": max(full_timings),
        },
        "checked_prefix_seconds": {
            "optimized_median": statistics.median(checked_timings),
            "pairwise": oracle_timings[0],
            "speedup": oracle_timings[0] / statistics.median(checked_timings),
        },
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"workload: {args.points:,} non-dominated points; "
            f"{args.oracle_points:,}-point pairwise oracle; {args.repeat} repeats"
        )
        print(f"python: {report['python']}  platform: {report['platform']}")
        print(
            "optimized full median: "
            f"{report['optimized_full_seconds']['median']:.6f}s "
            f"(min {report['optimized_full_seconds']['minimum']:.6f}s, "
            f"max {report['optimized_full_seconds']['maximum']:.6f}s)"
        )
        checked = report["checked_prefix_seconds"]
        print(
            f"checked prefix: optimized {checked['optimized_median']:.6f}s, "
            f"pairwise {checked['pairwise']:.6f}s, {checked['speedup']:.1f}x"
        )
        print("hand-checked corpus and pairwise oracle agreement: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
