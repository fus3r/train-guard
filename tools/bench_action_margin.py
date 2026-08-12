#!/usr/bin/env python3
"""Measure the replay-only action-distance overhead on a deterministic trace.

The benchmark keeps trace generation outside the timed region, warms both
paths, verifies that adding the distance does not change any sensitivity
envelope, and reports descriptive timings. It is not a portable CI gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_sweep import synthetic_trace

from trainguard.config import PolicyConfig
from trainguard.robustness import (
    ActionChangeMargin,
    SensitivityBounds,
    SensitivityResult,
    analyze_sensitivity,
    analyze_sensitivity_with_margin,
)
from trainguard.simulation import ordered_timestamps

Result = TypeVar("Result")


def _measure(function: Callable[[], Result], repeat: int) -> tuple[Result, list[float]]:
    started = time.perf_counter()
    result = function()
    timings = [time.perf_counter() - started]
    for _ in range(repeat - 1):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    return result, timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.repeat < 1:
        parser.error("samples and repeat must be positive")

    observations = synthetic_trace(args.samples)
    timestamps = ordered_timestamps(observations)
    durations = [
        (timestamps[index + 1] - timestamps[index]).total_seconds()
        for index in range(len(timestamps) - 1)
    ] + [0.0]
    policy = PolicyConfig()
    bounds = SensitivityBounds(temperature_c=0.5, charge_pct=1.0)

    def envelope_call() -> SensitivityResult:
        return analyze_sensitivity(policy, observations, durations, bounds)

    def margin_call() -> tuple[SensitivityResult, ActionChangeMargin]:
        return analyze_sensitivity_with_margin(policy, observations, durations, bounds)

    envelope_call()
    margin_call()
    envelope, envelope_timings = _measure(envelope_call, args.repeat)
    margin_result, margin_timings = _measure(margin_call, args.repeat)
    margin_envelope, margin = margin_result
    if margin_envelope != envelope:
        print("action-distance path changed the sensitivity envelope", file=sys.stderr)
        return 1

    envelope_median = statistics.median(envelope_timings)
    margin_median = statistics.median(margin_timings)
    report = {
        "schema_version": 1,
        "workload": "deterministic_synthetic_trace",
        "samples": args.samples,
        "repeat": args.repeat,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "bounds": bounds.to_dict(),
        "envelope_seconds": {
            "median": envelope_median,
            "minimum": min(envelope_timings),
            "maximum": max(envelope_timings),
        },
        "envelope_and_action_distance_seconds": {
            "median": margin_median,
            "minimum": min(margin_timings),
            "maximum": max(margin_timings),
        },
        "overhead_ratio": margin_median / envelope_median,
        "envelope_agreement": True,
        "action_change_margin": margin.to_dict(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"workload: {args.samples:,} observations; {args.repeat} warm repeats; "
            "trace generation excluded"
        )
        print(f"python: {report['python']}  platform: {report['platform']}")
        print(f"envelope median: {envelope_median:.6f}s")
        print(f"envelope + action distance median: {margin_median:.6f}s")
        print(f"overhead: {report['overhead_ratio']:.2f}x; envelope agreement: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
