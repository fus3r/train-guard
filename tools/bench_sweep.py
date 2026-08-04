#!/usr/bin/env python3
"""Measure the native replay kernel against the Python reference.

The benchmark generates a deterministic synthetic trace, evaluates the
same policy grid with both engines, verifies that every aggregate of
every candidate matches bit for bit, and prints wall-clock timings. Run
it from a checkout with the kernel built:

    python tools/bench_sweep.py [--samples 20000] [--kernel PATH]

Timings are workload-specific: report them only with this protocol
(sample count, grid size, hardware) attached.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainguard.config import PolicyConfig
from trainguard.model import Observation, PowerSource
from trainguard.native import find_kernel, resolve_thread_count, run_kernel
from trainguard.simulation import ordered_timestamps
from trainguard.sweep import SweepGrid, _python_rows, expand_grid


def synthetic_trace(samples: int) -> list[Observation]:
    """A reproducible battery/thermal walk with 20-second sampling."""

    observations = []
    state = 0x2545F4914F6CDD1D
    percent = 65.0
    temperature = 31.0
    plugged = True
    for index in range(samples):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        noise = ((state >> 33) % 2000) / 1000.0 - 1.0  # [-1, 1)
        if (state >> 17) % 97 == 0:
            plugged = not plugged
        temperature = min(46.0, max(24.0, temperature + noise + (0.08 if plugged else -0.05)))
        percent = min(100.0, max(3.0, percent + (0.05 if plugged else -0.04) + noise / 50.0))
        minute, second = divmod(index * 20, 60)
        hour, minute = divmod(minute, 60)
        day, hour = divmod(hour, 24)
        observations.append(
            Observation(
                source=PowerSource.AC if plugged else PowerSource.BATTERY,
                percent=round(percent, 2),
                temperature_c=round(temperature, 2),
                charging=plugged and percent < 100,
                observed_at=f"2026-07-{(day % 28) + 1:02d}T{hour:02d}:{minute:02d}:{second:02d}Z",
            )
        )
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--kernel", help="kernel binary (default: TRAIN_GUARD_KERNEL or PATH)")
    parser.add_argument(
        "--kernel-threads",
        type=int,
        default=None,
        help="kernel worker threads (default: automatic, capped at 16)",
    )
    args = parser.parse_args()

    if args.kernel:
        os.environ["TRAIN_GUARD_KERNEL"] = args.kernel
    kernel = find_kernel()
    if kernel is None:
        print("no kernel found; build native/ and pass --kernel", file=sys.stderr)
        return 2

    overrides = {
        "temp_pause_c": [40.0, 41.0, 42.0, 43.0, 44.0],
        "temp_gentle_c": [34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
        "battery_floor_pct": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        "run_on_battery": [True, False],
    }
    import itertools

    keys = sorted(overrides)
    grid = SweepGrid(
        fields=tuple(keys),
        combinations=tuple(
            dict(zip(keys, values)) for values in itertools.product(*(overrides[k] for k in keys))
        ),
    )
    candidates, rejected = expand_grid(grid, PolicyConfig())
    policies = [PolicyConfig()] + [policy for _, policy in candidates]
    observations = synthetic_trace(args.samples)
    timestamps = ordered_timestamps(observations)
    durations = [0.0] * len(observations)
    for index in range(len(observations) - 1):
        durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()

    decisions = len(policies) * len(observations)
    threads = resolve_thread_count(len(policies), requested=args.kernel_threads)
    print(
        f"workload: {len(policies)} policies ({len(rejected)} grid combos rejected) x "
        f"{len(observations)} observations = {decisions} decisions, "
        f"kernel threads {threads}"
    )

    started = time.perf_counter()
    python_rows = _python_rows(
        policies, observations, durations, hot_ref_c=35.0, low_battery_ref_pct=20.0
    )
    python_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    kernel_rows = run_kernel(
        kernel,
        policies,
        observations,
        timestamps,
        hot_ref_c=35.0,
        low_battery_ref_pct=20.0,
        threads=threads,
    )
    kernel_elapsed = time.perf_counter() - started

    mismatches = 0
    for python_row, kernel_row in zip(python_rows, kernel_rows):
        for name in (
            "full_seconds",
            "gentle_seconds",
            "stop_seconds",
            "run_seconds",
            "hot_degc_seconds",
            "low_battery_run_seconds",
            "action_transitions",
            "decision_transitions",
            "checksum",
        ):
            if getattr(python_row, name) != getattr(kernel_row, name):
                mismatches += 1

    print(f"python reference: {python_elapsed:.3f}s ({decisions / python_elapsed:,.0f}/s)")
    print(
        f"native kernel:    {kernel_elapsed:.3f}s ({decisions / kernel_elapsed:,.0f}/s) "
        "including encode, spawn and decode"
    )
    print(f"speedup: {python_elapsed / kernel_elapsed:.1f}x")
    print(f"bit-exact aggregate mismatches across all rows: {mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
