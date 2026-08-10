"""Multi-objective policy sweep over one immutable observation trace.

The metrics re-weight recorded exposure under replayed actions. They do not
simulate how a different action would have changed temperature, charge or
battery lifetime.
"""

from __future__ import annotations

import itertools
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__:
    from .clairvoyant import (
        CostFrontier,
        JointCostFrontier,
        build_frontier,
        build_joint_frontier,
        clairvoyant_report,
        hot_rated_intervals,
        low_battery_rated_intervals,
    )
    from .config import ConfigError, PolicyConfig, policy_from_mapping
    from .model import Action, DecisionReason, Observation, PowerSource
    from .native import KernelError, KernelRow, find_kernel, fnv1a_decisions, run_kernel
    from .policy import PolicyEngine
    from .simulation import TraceError, _fingerprint, ordered_timestamps
else:  # Keep direct CLI execution from a checkout working.
    from clairvoyant import (
        CostFrontier,
        JointCostFrontier,
        build_frontier,
        build_joint_frontier,
        clairvoyant_report,
        hot_rated_intervals,
        low_battery_rated_intervals,
    )
    from config import ConfigError, PolicyConfig, policy_from_mapping
    from model import Action, DecisionReason, Observation, PowerSource
    from policy import PolicyEngine
    from simulation import TraceError, _fingerprint, ordered_timestamps

    from native import KernelError, KernelRow, find_kernel, fnv1a_decisions, run_kernel


DEFAULT_HOT_REFERENCE_C = 35.0
DEFAULT_LOW_BATTERY_REFERENCE_PCT = 20.0
_GRID_EXAMPLE = '{"temp_pause_c":[40,42],"run_on_battery":[true,false]}'
_ACTION_DIGITS = {Action.FULL: "0", Action.GENTLE: "1", Action.STOP: "2"}


class SweepError(ValueError):
    """Raised when a grid or engine selection cannot be evaluated."""


@dataclass(frozen=True)
class SweepGrid:
    fields: tuple[str, ...]
    combinations: tuple[dict[str, Any], ...]


def parse_grid(text: str, source: str = "grid") -> SweepGrid:
    """Parse a Cartesian policy grid into explicit override mappings."""

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SweepError(
            f"{source}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(raw, dict) or not raw:
        raise SweepError(f"{source}: the grid must be a non-empty object, e.g. {_GRID_EXAMPLE}")

    valid_fields = set(PolicyConfig().to_dict())
    axes: list[tuple[str, list[Any]]] = []
    for name in sorted(raw):
        if name not in valid_fields:
            known = ", ".join(sorted(valid_fields))
            raise SweepError(f"{source}: unknown policy field '{name}' (known fields: {known})")
        values = raw[name]
        if not isinstance(values, list) or not values:
            raise SweepError(f"{source}: {name} must map to a non-empty list of values")
        deduplicated: list[Any] = []
        for value in values:
            if isinstance(value, (dict, list)):
                raise SweepError(f"{source}: {name} values must be scalars")
            if value not in deduplicated:
                deduplicated.append(value)
        axes.append((name, deduplicated))

    field_names = tuple(name for name, _ in axes)
    combinations = tuple(
        dict(zip(field_names, values))
        for values in itertools.product(*(values for _, values in axes))
    )
    return SweepGrid(fields=field_names, combinations=combinations)


def expand_grid(
    grid: SweepGrid,
    base: PolicyConfig,
) -> tuple[list[tuple[dict[str, Any], PolicyConfig]], list[dict[str, Any]]]:
    """Merge, validate and value-deduplicate every candidate."""

    base_values = base.to_dict()
    candidates: list[tuple[dict[str, Any], PolicyConfig]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for overrides in grid.combinations:
        merged = dict(base_values)
        merged.update(overrides)
        try:
            policy = policy_from_mapping(merged)
        except ConfigError as exc:
            rejected.append({"overrides": overrides, "error": str(exc)})
            continue
        fingerprint = _fingerprint(policy.to_dict())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append((overrides, policy))
    return candidates, rejected


def _python_rows(
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    durations: Sequence[float],
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
    emit_actions: bool = False,
) -> list[KernelRow]:
    """Compute reference aggregates in the native kernel's operation order."""

    rows: list[KernelRow] = []
    for policy in policies:
        engine = PolicyEngine()
        action_seconds = {
            Action.FULL: 0.0,
            Action.GENTLE: 0.0,
            Action.STOP: 0.0,
        }
        run_seconds = 0.0
        hot_degc_seconds = 0.0
        low_battery_seconds = 0.0
        action_transitions = 0
        decision_transitions = 0
        previous: Optional[tuple[Action, DecisionReason]] = None
        decisions: list[tuple[Action, DecisionReason]] = []

        for observation, duration in zip(observations, durations):
            decision = engine.decide(policy, observation)
            action = decision.action
            signature = (action, decision.reason)
            decisions.append(signature)
            action_seconds[action] += duration
            if action is not Action.STOP:
                run_seconds += duration
                if observation.temperature_c is not None:
                    excess = observation.temperature_c - hot_ref_c
                    if excess > 0.0:
                        hot_degc_seconds += excess * duration
                if (
                    observation.source is PowerSource.BATTERY
                    and observation.percent is not None
                    and observation.percent <= low_battery_ref_pct
                ):
                    low_battery_seconds += duration
            if previous is not None and action is not previous[0]:
                action_transitions += 1
            if previous is not None and signature != previous:
                decision_transitions += 1
            previous = signature

        rows.append(
            KernelRow(
                full_seconds=action_seconds[Action.FULL],
                gentle_seconds=action_seconds[Action.GENTLE],
                stop_seconds=action_seconds[Action.STOP],
                run_seconds=run_seconds,
                hot_degc_seconds=hot_degc_seconds,
                low_battery_run_seconds=low_battery_seconds,
                action_transitions=action_transitions,
                decision_transitions=decision_transitions,
                checksum=fnv1a_decisions(decisions),
                actions=(
                    "".join(_ACTION_DIGITS[action] for action, _ in decisions)
                    if emit_actions
                    else None
                ),
            )
        )
    return rows


def _verify_kernel_row(python_row: KernelRow, kernel_row: KernelRow, kernel: Path) -> None:
    """Refuse native results if any baseline aggregate differs by one bit."""

    fields = (
        "full_seconds",
        "gentle_seconds",
        "stop_seconds",
        "run_seconds",
        "hot_degc_seconds",
        "low_battery_run_seconds",
        "action_transitions",
        "decision_transitions",
        "checksum",
    )
    mismatches = [name for name in fields if getattr(python_row, name) != getattr(kernel_row, name)]
    if mismatches:
        raise SweepError(
            f"native kernel {kernel} disagrees with the Python reference on the baseline "
            f"policy ({', '.join(mismatches)}); refusing its results"
        )


def trace_facts(
    observations: Sequence[Observation],
    durations: Sequence[float],
    *,
    hot_ref_c: float,
    high_soc_ref_pct: float = 80.0,
    cycle_deadband_pct: float = 2.0,
) -> dict[str, Any]:
    """Return policy-independent arithmetic about the recorded trace."""

    elapsed = sum(durations)
    temperature_seconds = 0.0
    hot_degc_seconds = 0.0
    percent_seconds = 0.0
    high_soc_seconds = 0.0
    hot_and_full_seconds = 0.0
    percent_travel = 0.0
    previous_percent: Optional[float] = None
    for observation, duration in zip(observations, durations):
        temperature = observation.temperature_c
        percent = observation.percent
        if temperature is not None:
            temperature_seconds += duration
            hot_degc_seconds += max(0.0, temperature - hot_ref_c) * duration
        if percent is not None:
            percent_seconds += duration
            if percent >= high_soc_ref_pct:
                high_soc_seconds += duration
                if temperature is not None and temperature >= hot_ref_c:
                    hot_and_full_seconds += duration
            if previous_percent is not None:
                step = abs(percent - previous_percent)
                if step >= cycle_deadband_pct:
                    percent_travel += step
            previous_percent = percent

    return {
        "elapsed_seconds": elapsed,
        "hot_reference_c": hot_ref_c,
        "hot_degc_seconds": hot_degc_seconds,
        "temperature_coverage": (temperature_seconds / elapsed if elapsed > 0 else None),
        "high_soc_reference_pct": high_soc_ref_pct,
        "high_soc_seconds": high_soc_seconds,
        "hot_and_full_seconds": hot_and_full_seconds,
        "percent_coverage": percent_seconds / elapsed if elapsed > 0 else None,
        "cycle_deadband_pct": cycle_deadband_pct,
        "equivalent_full_cycles": percent_travel / 200.0,
    }


def _pareto_flags(rows: Sequence[KernelRow]) -> list[bool]:
    """Mark candidates not dominated on run up, exposures down.

    Sorting by decreasing run time turns dominance into a two-dimensional
    prefix-minimum query. Equal metric triples are collapsed for the pass so
    duplicates remain co-optimal: neither is strictly better than the other.
    """

    points = [(row.run_seconds, row.hot_degc_seconds, row.low_battery_run_seconds) for row in rows]
    if not points:
        return []

    ordered = sorted(set(points), key=lambda point: (-point[0], point[1], point[2]))
    hot_values = sorted({point[1] for point in ordered})
    tree = [math.inf] * (len(hot_values) + 1)
    dominated: set[tuple[float, float, float]] = set()

    def prefix_min(index: int) -> float:
        best = math.inf
        while index > 0:
            best = min(best, tree[index])
            index -= index & -index
        return best

    def update(index: int, value: float) -> None:
        while index < len(tree):
            tree[index] = min(tree[index], value)
            index += index & -index

    for point in ordered:
        _, hot, low_battery = point
        rank = bisect_left(hot_values, hot) + 1
        if prefix_min(rank) <= low_battery:
            dominated.add(point)
        update(rank, low_battery)

    return [point not in dominated for point in points]


def _candidate_report(
    overrides: dict[str, Any],
    policy: PolicyConfig,
    row: KernelRow,
    baseline_row: KernelRow,
    elapsed: float,
    pareto_optimal: Optional[bool],
    hot_frontier: CostFrontier,
    low_frontier: CostFrontier,
    joint_frontier: JointCostFrontier,
    hot_ref_c: float,
) -> dict[str, Any]:
    def share(seconds: float) -> Optional[float]:
        return 100.0 * seconds / elapsed if elapsed > 0 else None

    return {
        "overrides": overrides,
        "policy": policy.to_dict(),
        "policy_sha256": _fingerprint(policy.to_dict()),
        "pareto_optimal": pareto_optimal,
        "clairvoyant": clairvoyant_report(
            hot_frontier,
            low_frontier,
            joint_frontier,
            run_seconds=row.run_seconds,
            hot_cost=row.hot_degc_seconds,
            low_battery_cost=row.low_battery_run_seconds,
            hot_ref_c=hot_ref_c,
        ),
        "metrics": {
            "action_seconds": {
                "full": row.full_seconds,
                "gentle": row.gentle_seconds,
                "stop": row.stop_seconds,
            },
            "run_seconds": row.run_seconds,
            "run_percent": share(row.run_seconds),
            "hot_run_degc_seconds": row.hot_degc_seconds,
            "low_battery_run_seconds": row.low_battery_run_seconds,
            "action_transitions": row.action_transitions,
            "decision_transitions": row.decision_transitions,
            "decision_checksum": row.checksum,
        },
        "delta_vs_baseline": {
            "run_seconds": row.run_seconds - baseline_row.run_seconds,
            "hot_run_degc_seconds": (row.hot_degc_seconds - baseline_row.hot_degc_seconds),
            "low_battery_run_seconds": (
                row.low_battery_run_seconds - baseline_row.low_battery_run_seconds
            ),
            "action_transitions": (row.action_transitions - baseline_row.action_transitions),
        },
    }


def run_sweep(
    base: PolicyConfig,
    grid: SweepGrid,
    observations: Sequence[Observation],
    *,
    engine: str = "auto",
    hot_ref_c: float = DEFAULT_HOT_REFERENCE_C,
    low_battery_ref_pct: float = DEFAULT_LOW_BATTERY_REFERENCE_PCT,
) -> dict[str, Any]:
    """Evaluate a validated grid with Python or a verified native kernel."""

    if engine not in {"auto", "python", "native"}:
        raise SweepError("engine must be auto, python or native")
    if not math.isfinite(float(hot_ref_c)) or not -100 <= hot_ref_c <= 200:
        raise SweepError("hot reference must be between -100 and 200")
    if not math.isfinite(float(low_battery_ref_pct)) or not 0 <= low_battery_ref_pct <= 100:
        raise SweepError("low-battery reference must be between 0 and 100")
    if not observations:
        raise TraceError("cannot sweep an empty trace")

    timestamps = ordered_timestamps(observations)
    durations = [0.0] * len(observations)
    for index in range(len(observations) - 1):
        durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()

    candidates, rejected = expand_grid(grid, base)
    if not candidates:
        raise SweepError("every grid combination was rejected; nothing to evaluate")
    policies = [base] + [policy for _, policy in candidates]

    try:
        kernel = find_kernel() if engine in {"auto", "native"} else None
        if engine == "native" and kernel is None:
            raise SweepError(
                "engine=native requires the replay kernel; build native/ and put "
                "train-guard-kernel on PATH or set TRAIN_GUARD_KERNEL"
            )

        engine_used = "python"
        kernel_verified = False
        if kernel is not None:
            kernel_rows = run_kernel(
                kernel,
                policies,
                observations,
                timestamps,
                hot_ref_c=hot_ref_c,
                low_battery_ref_pct=low_battery_ref_pct,
            )
            baseline_reference = _python_rows(
                [base],
                observations,
                durations,
                hot_ref_c=hot_ref_c,
                low_battery_ref_pct=low_battery_ref_pct,
            )[0]
            _verify_kernel_row(baseline_reference, kernel_rows[0], kernel)
            rows = kernel_rows
            engine_used = "native"
            kernel_verified = True
        else:
            rows = _python_rows(
                policies,
                observations,
                durations,
                hot_ref_c=hot_ref_c,
                low_battery_ref_pct=low_battery_ref_pct,
            )
    except KernelError as exc:
        raise SweepError(str(exc)) from exc

    baseline_row, candidate_rows = rows[0], rows[1:]
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds()

    hot_rated = hot_rated_intervals(observations, durations, hot_ref_c)
    low_rated = low_battery_rated_intervals(observations, durations, low_battery_ref_pct)
    hot_frontier = build_frontier(hot_rated)
    low_frontier = build_frontier(low_rated)
    joint_frontier = build_joint_frontier(hot_rated, low_rated)
    flags = _pareto_flags(candidate_rows)
    reports = [
        _candidate_report(
            overrides,
            policy,
            row,
            baseline_row,
            elapsed,
            pareto,
            hot_frontier,
            low_frontier,
            joint_frontier,
            hot_ref_c,
        )
        for (overrides, policy), row, pareto in zip(candidates, candidate_rows, flags)
    ]
    reports.sort(
        key=lambda item: (
            not item["pareto_optimal"],
            -item["metrics"]["run_seconds"],
            item["metrics"]["hot_run_degc_seconds"],
            item["metrics"]["low_battery_run_seconds"],
            item["policy_sha256"],
        )
    )

    return {
        "schema_version": 1,
        "engine": engine_used,
        "kernel_verified_against_reference": kernel_verified,
        "hot_reference_c": hot_ref_c,
        "low_battery_reference_pct": low_battery_ref_pct,
        "samples": len(observations),
        "elapsed_seconds": elapsed,
        "observations_sha256": _fingerprint(
            [observation.to_dict() for observation in observations]
        ),
        "trace_facts": trace_facts(observations, durations, hot_ref_c=hot_ref_c),
        "clairvoyant": {
            "total_run_seconds": hot_frontier.total_seconds,
            "hot_free_run_seconds": hot_frontier.free_seconds,
            "hot_total_cost_degc_seconds": hot_frontier.total_cost,
            "low_battery_free_run_seconds": low_frontier.free_seconds,
            "low_battery_total_cost_seconds": low_frontier.total_cost,
        },
        "grid_fields": list(grid.fields),
        "candidates_evaluated": len(candidates),
        "candidates_rejected": len(rejected),
        "rejected": rejected[:10],
        "baseline": _candidate_report(
            {},
            base,
            baseline_row,
            baseline_row,
            elapsed,
            None,
            hot_frontier,
            low_frontier,
            joint_frontier,
            hot_ref_c,
        ),
        "pareto_front_size": sum(flags),
        "candidates": reports,
    }
