"""Exact bounded sensitivity analysis for replayed policy decisions.

The live sensors expose measurements, not exact physical state. This module
does not guess a noise distribution. Instead, callers supply closed
half-widths around recorded temperature and charge values. Every value in
each interval is treated as possible, independently at every sample, while a
missing measurement remains missing.

The policy contains one bit of memory: thermal cooldown. Propagating the set
of reachable values of that bit is therefore enough to compute exact marginal
bounds for runnable time, hot exposure and low-battery exposure, plus the
duration for which every admissible trace agrees on the action. Policy
predicates and exposure rates change only at finitely many thresholds;
interval endpoints, each threshold, and the adjacent floating-point values
represent every decision region without sampling a grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

if __package__:
    from .config import PolicyConfig
    from .model import Action, Observation, PowerSource
    from .policy import PolicyEngine
else:  # Keep direct module execution from a checkout working.
    from config import PolicyConfig
    from model import Action, Observation, PowerSource
    from policy import PolicyEngine


class SensitivityError(ValueError):
    """Raised when a bounded-sensitivity request is not meaningful."""


@dataclass(frozen=True)
class SensitivityBounds:
    """User-supplied closed half-widths, with no probability interpretation."""

    temperature_c: float = 0.0
    charge_pct: float = 0.0

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("temperature uncertainty", self.temperature_c, 300.0),
            ("charge uncertainty", self.charge_pct, 100.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= maximum
            ):
                raise SensitivityError(
                    f"{name} must be a finite half-width between 0 and {maximum:g}"
                )
        object.__setattr__(self, "temperature_c", float(self.temperature_c) + 0.0)
        object.__setattr__(self, "charge_pct", float(self.charge_pct) + 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "bounded_adversarial",
            "numeric_domain": "ieee_754_binary64",
            "temperature_half_width_c": self.temperature_c,
            "charge_half_width_pct": self.charge_pct,
            "confidence_level": None,
            "missing_measurements": "remain_missing",
            "uncertainty_set": "cartesian_product",
            "temporal_correlation": "unmodeled",
        }


@dataclass(frozen=True)
class ActionChangeMargin:
    """Nearest action-changing representative path in the declared binary64 box."""

    minimum_normalized_action_distance: Optional[float]
    critical_sample: Optional[int] = None
    observed_at: Optional[str] = None
    nominal_action: Optional[Action] = None
    alternative_action: Optional[Action] = None

    def to_dict(self) -> dict[str, object]:
        critical = None
        if self.critical_sample is not None:
            if (
                self.observed_at is None
                or self.nominal_action is None
                or self.alternative_action is None
            ):
                raise AssertionError("a critical sample requires complete action-change context")
            critical = {
                "sample": self.critical_sample,
                "observed_at": self.observed_at,
                "nominal_action": self.nominal_action.value,
                "alternative_action": self.alternative_action.value,
            }
        return {
            "model": "bounded_minimum_distortion",
            "metric": "normalized_linf",
            "numeric_domain": "ieee_754_binary64",
            "minimum_normalized_action_distance": self.minimum_normalized_action_distance,
            "minimum_normalized_action_distance_hex": (
                None
                if self.minimum_normalized_action_distance is None
                else self.minimum_normalized_action_distance.hex()
            ),
            "stable_for_declared_box": self.minimum_normalized_action_distance is None,
            "critical_sample": critical,
        }


@dataclass(frozen=True)
class SensitivityResult:
    """Tight marginal objective envelopes and action stability for one policy."""

    minimum_run_seconds: float
    maximum_run_seconds: float
    minimum_hot_run_degc_seconds: float
    maximum_hot_run_degc_seconds: float
    minimum_low_battery_run_seconds: float
    maximum_low_battery_run_seconds: float
    action_stable_seconds: float
    action_ambiguous_seconds: float
    action_stable_samples: int
    action_ambiguous_samples: int

    def to_dict(
        self,
        bounds: SensitivityBounds,
        *,
        hot_ref_c: float = 35.0,
        low_battery_ref_pct: float = 20.0,
        action_change_margin: Optional[ActionChangeMargin] = None,
    ) -> dict[str, object]:
        elapsed = self.action_stable_seconds + self.action_ambiguous_seconds
        report: dict[str, object] = {
            "schema_version": 2,
            "bounds": bounds.to_dict(),
            "run_seconds": {
                "minimum": self.minimum_run_seconds,
                "maximum": self.maximum_run_seconds,
            },
            "hot_run_degc_seconds": {
                "minimum": self.minimum_hot_run_degc_seconds,
                "maximum": self.maximum_hot_run_degc_seconds,
                "reference_c": hot_ref_c,
            },
            "low_battery_run_seconds": {
                "minimum": self.minimum_low_battery_run_seconds,
                "maximum": self.maximum_low_battery_run_seconds,
                "reference_pct": low_battery_ref_pct,
            },
            "action_stability": {
                "stable_seconds": self.action_stable_seconds,
                "ambiguous_seconds": self.action_ambiguous_seconds,
                "stable_percent": (
                    100.0 * self.action_stable_seconds / elapsed if elapsed > 0.0 else None
                ),
                "stable_samples": self.action_stable_samples,
                "ambiguous_samples": self.action_ambiguous_samples,
            },
        }
        if action_change_margin is not None:
            report["schema_version"] = 3
            report["action_change_margin"] = action_change_margin.to_dict()
        return report


@dataclass(frozen=True)
class _TransitionEnvelope:
    """Marginal exposure rates for one reachable action/state transition."""

    action: Action
    cooling: bool
    minimum_hot_rate: float
    maximum_hot_rate: float
    minimum_low_battery_rate: float
    maximum_low_battery_rate: float
    minimum_action_distance: Optional[float] = None


@dataclass(frozen=True)
class _ObjectiveEnvelope:
    """Marginal cumulative extrema conditional on one cooldown state."""

    minimum_run: float
    maximum_run: float
    minimum_hot: float
    maximum_hot: float
    minimum_low_battery: float
    maximum_low_battery: float


def _representatives(
    value: Optional[float],
    half_width: float,
    thresholds: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    include_endpoints: bool = True,
    include_recorded: bool = False,
) -> tuple[Optional[float], ...]:
    if value is None:
        return (None,)

    low = max(lower_bound, float(value) - half_width)
    high = min(upper_bound, float(value) + half_width)
    points = {low, high} if include_endpoints else set()
    if include_recorded:
        points.add(value)
    for threshold in thresholds:
        if low <= threshold <= high:
            points.add(threshold)
        below = math.nextafter(threshold, -math.inf)
        above = math.nextafter(threshold, math.inf)
        if low <= below <= high:
            points.add(below)
        if low <= above <= high:
            points.add(above)
    return tuple(sorted(points))


def _normalized_axis_distance(
    recorded: Optional[float],
    candidate: Optional[float],
    half_width: float,
) -> float:
    """Return the documented binary64 distance on one admitted box axis.

    The full interval is constructed with binary64 ``recorded +/- half_width``.
    That rounded endpoint can be microscopically farther than ``half_width``
    under a second subtraction. Saturating an already-admitted endpoint at
    one keeps the metric consistent with the sensitivity box it summarizes.
    """

    if recorded is None or candidate is None:
        if recorded is not candidate:
            raise AssertionError("bounded sensitivity cannot invent a missing value")
        return 0.0
    if candidate == recorded:
        return 0.0
    if half_width == 0.0:
        raise AssertionError("a zero-width axis cannot contain another value")
    return min(1.0, abs(candidate - recorded) / half_width)


def _possible_transitions(
    config: PolicyConfig,
    observation: Observation,
    bounds: SensitivityBounds,
    cooling: bool,
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
    track_action_distance: bool = False,
) -> tuple[_TransitionEnvelope, ...]:
    temperatures = _representatives(
        observation.temperature_c,
        bounds.temperature_c,
        (
            config.temp_resume_c,
            config.temp_pause_c,
            config.temp_charge_gentle_c,
            config.temp_gentle_c,
            hot_ref_c,
        ),
        lower_bound=-100.0,
        upper_bound=200.0,
    )
    percentages = _representatives(
        observation.percent,
        bounds.charge_pct,
        (
            config.battery_floor_pct,
            config.charge_cool_until_pct,
            low_battery_ref_pct,
        ),
        lower_bound=0.0,
        upper_bound=100.0,
    )
    if track_action_distance:
        action_temperatures = _representatives(
            observation.temperature_c,
            bounds.temperature_c,
            (
                config.temp_resume_c,
                config.temp_pause_c,
                config.temp_charge_gentle_c,
                config.temp_gentle_c,
            ),
            lower_bound=-100.0,
            upper_bound=200.0,
            include_endpoints=False,
            include_recorded=True,
        )
        action_percentages = _representatives(
            observation.percent,
            bounds.charge_pct,
            (
                config.battery_floor_pct,
                config.charge_cool_until_pct,
            ),
            lower_bound=0.0,
            upper_bound=100.0,
            include_endpoints=False,
            include_recorded=True,
        )
    else:
        action_temperatures = ()
        action_percentages = ()

    # (action, next cooling) -> [min hot rate, max hot rate,
    #                            min low-battery rate, max low-battery rate]
    outcomes: dict[tuple[Action, bool], list[float]] = {}
    action_distances: dict[tuple[Action, bool], float] = {}
    for temperature in temperatures:
        for percent in percentages:
            perturbed = Observation(
                source=observation.source,
                percent=percent,
                temperature_c=temperature,
                charging=observation.charging,
                observed_at=observation.observed_at,
                warnings=observation.warnings,
            )
            decision = PolicyEngine(cooling=cooling).decide(config, perturbed)
            running = decision.action is not Action.STOP
            hot_rate = (
                max(float(temperature) - hot_ref_c, 0.0)
                if running and temperature is not None
                else 0.0
            )
            low_battery_rate = float(
                running
                and observation.source is PowerSource.BATTERY
                and percent is not None
                and percent <= low_battery_ref_pct
            )
            key = (decision.action, decision.cooling)
            previous = outcomes.get(key)
            if previous is None:
                outcomes[key] = [hot_rate, hot_rate, low_battery_rate, low_battery_rate]
            else:
                previous[0] = min(previous[0], hot_rate)
                previous[1] = max(previous[1], hot_rate)
                previous[2] = min(previous[2], low_battery_rate)
                previous[3] = max(previous[3], low_battery_rate)

    for temperature in action_temperatures:
        for percent in action_percentages:
            perturbed = Observation(
                source=observation.source,
                percent=percent,
                temperature_c=temperature,
                charging=observation.charging,
                observed_at=observation.observed_at,
                warnings=observation.warnings,
            )
            decision = PolicyEngine(cooling=cooling).decide(config, perturbed)
            key = (decision.action, decision.cooling)
            action_distance = max(
                _normalized_axis_distance(
                    observation.temperature_c,
                    temperature,
                    bounds.temperature_c,
                ),
                _normalized_axis_distance(
                    observation.percent,
                    percent,
                    bounds.charge_pct,
                ),
            )
            previous_distance = action_distances.get(key)
            if previous_distance is None or action_distance < previous_distance:
                action_distances[key] = action_distance

    return tuple(
        _TransitionEnvelope(
            action,
            next_cooling,
            minimum_hot_rate=rates[0],
            maximum_hot_rate=rates[1],
            minimum_low_battery_rate=rates[2],
            maximum_low_battery_rate=rates[3],
            minimum_action_distance=action_distances.get((action, next_cooling)),
        )
        for (action, next_cooling), rates in outcomes.items()
    )


def _reference(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise SensitivityError(f"{name} must be a finite value between {minimum:g} and {maximum:g}")
    return float(value) + 0.0


def _analyze_sensitivity(
    config: PolicyConfig,
    observations: Sequence[Observation],
    durations: Sequence[float],
    bounds: SensitivityBounds,
    *,
    hot_ref_c: float = 35.0,
    low_battery_ref_pct: float = 20.0,
    include_action_margin: bool,
) -> tuple[SensitivityResult, Optional[ActionChangeMargin]]:
    """Propagate every admissible decision path through thermal hysteresis.

    Objective intervals are marginal: each minimum and maximum is tight, but
    extrema for different objectives need not be attained by the same trace.
    """

    if not observations:
        raise SensitivityError("cannot analyze an empty trace")
    if len(observations) != len(durations):
        raise SensitivityError("one duration is required for every observation")
    if any(
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration < 0.0
        for duration in durations
    ):
        raise SensitivityError("durations must be finite and non-negative")

    hot_reference = _reference(
        hot_ref_c,
        name="hot reference",
        minimum=-100.0,
        maximum=200.0,
    )
    low_battery_reference = _reference(
        low_battery_ref_pct,
        name="low-battery reference",
        minimum=0.0,
        maximum=100.0,
    )

    reachable: dict[bool, _ObjectiveEnvelope] = {
        False: _ObjectiveEnvelope(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
    stable_seconds = 0.0
    ambiguous_seconds = 0.0
    stable_samples = 0
    ambiguous_samples = 0
    nominal_engine = PolicyEngine()
    matching_prefixes: dict[bool, float] = {False: 0.0}
    best_change: Optional[tuple[float, int, str, bool]] = None
    best_context: Optional[tuple[str, Action, Action]] = None

    for sample_index, (observation, raw_duration) in enumerate(zip(observations, durations), 1):
        duration = float(raw_duration)
        next_reachable: dict[bool, _ObjectiveEnvelope] = {}
        possible_actions: set[Action] = set()
        transitions_by_cooling: dict[bool, tuple[_TransitionEnvelope, ...]] = {}
        nominal_action = (
            nominal_engine.decide(config, observation).action if include_action_margin else None
        )

        for cooling, envelope in reachable.items():
            transitions = _possible_transitions(
                config,
                observation,
                bounds,
                cooling,
                hot_ref_c=hot_reference,
                low_battery_ref_pct=low_battery_reference,
                track_action_distance=include_action_margin,
            )
            transitions_by_cooling[cooling] = transitions
            for transition in transitions:
                possible_actions.add(transition.action)
                run_increment = 0.0 if transition.action is Action.STOP else duration
                candidate = _ObjectiveEnvelope(
                    minimum_run=envelope.minimum_run + run_increment,
                    maximum_run=envelope.maximum_run + run_increment,
                    minimum_hot=envelope.minimum_hot + transition.minimum_hot_rate * duration,
                    maximum_hot=envelope.maximum_hot + transition.maximum_hot_rate * duration,
                    minimum_low_battery=(
                        envelope.minimum_low_battery
                        + transition.minimum_low_battery_rate * duration
                    ),
                    maximum_low_battery=(
                        envelope.maximum_low_battery
                        + transition.maximum_low_battery_rate * duration
                    ),
                )
                previous = next_reachable.get(transition.cooling)
                if previous is None:
                    next_reachable[transition.cooling] = candidate
                else:
                    next_reachable[transition.cooling] = _ObjectiveEnvelope(
                        minimum_run=min(previous.minimum_run, candidate.minimum_run),
                        maximum_run=max(previous.maximum_run, candidate.maximum_run),
                        minimum_hot=min(previous.minimum_hot, candidate.minimum_hot),
                        maximum_hot=max(previous.maximum_hot, candidate.maximum_hot),
                        minimum_low_battery=min(
                            previous.minimum_low_battery,
                            candidate.minimum_low_battery,
                        ),
                        maximum_low_battery=max(
                            previous.maximum_low_battery,
                            candidate.maximum_low_battery,
                        ),
                    )

        if include_action_margin:
            if nominal_action is None:
                raise AssertionError("action-margin analysis requires a nominal action")
            next_matching: dict[bool, float] = {}
            for cooling, prefix_distance in matching_prefixes.items():
                for transition in transitions_by_cooling[cooling]:
                    local_distance = transition.minimum_action_distance
                    if local_distance is None:
                        raise AssertionError("action-margin transitions require a distance")
                    candidate_distance = max(prefix_distance, local_distance)
                    if transition.action is not nominal_action:
                        candidate_key = (
                            candidate_distance,
                            sample_index,
                            transition.action.value,
                            transition.cooling,
                        )
                        if best_change is None or candidate_key < best_change:
                            best_change = candidate_key
                            best_context = (
                                observation.observed_at,
                                nominal_action,
                                transition.action,
                            )
                        continue
                    previous_distance = next_matching.get(transition.cooling)
                    if previous_distance is None or candidate_distance < previous_distance:
                        next_matching[transition.cooling] = candidate_distance
            matching_prefixes = next_matching

        if len(possible_actions) == 1:
            stable_seconds += duration
            stable_samples += 1
        else:
            ambiguous_seconds += duration
            ambiguous_samples += 1
        reachable = next_reachable

    result = SensitivityResult(
        minimum_run_seconds=min(value.minimum_run for value in reachable.values()),
        maximum_run_seconds=max(value.maximum_run for value in reachable.values()),
        minimum_hot_run_degc_seconds=min(value.minimum_hot for value in reachable.values()),
        maximum_hot_run_degc_seconds=max(value.maximum_hot for value in reachable.values()),
        minimum_low_battery_run_seconds=min(
            value.minimum_low_battery for value in reachable.values()
        ),
        maximum_low_battery_run_seconds=max(
            value.maximum_low_battery for value in reachable.values()
        ),
        action_stable_seconds=stable_seconds,
        action_ambiguous_seconds=ambiguous_seconds,
        action_stable_samples=stable_samples,
        action_ambiguous_samples=ambiguous_samples,
    )
    if not include_action_margin:
        return result, None
    if best_change is None:
        return result, ActionChangeMargin(minimum_normalized_action_distance=None)
    if best_context is None:
        raise AssertionError("an action change requires critical context")
    observed_at, nominal_action, alternative_action = best_context
    return result, ActionChangeMargin(
        minimum_normalized_action_distance=best_change[0],
        critical_sample=best_change[1],
        observed_at=observed_at,
        nominal_action=nominal_action,
        alternative_action=alternative_action,
    )


def analyze_sensitivity(
    config: PolicyConfig,
    observations: Sequence[Observation],
    durations: Sequence[float],
    bounds: SensitivityBounds,
    *,
    hot_ref_c: float = 35.0,
    low_battery_ref_pct: float = 20.0,
) -> SensitivityResult:
    """Return tight bounded objective envelopes and local action stability."""

    result, _ = _analyze_sensitivity(
        config,
        observations,
        durations,
        bounds,
        hot_ref_c=hot_ref_c,
        low_battery_ref_pct=low_battery_ref_pct,
        include_action_margin=False,
    )
    return result


def analyze_sensitivity_with_margin(
    config: PolicyConfig,
    observations: Sequence[Observation],
    durations: Sequence[float],
    bounds: SensitivityBounds,
    *,
    hot_ref_c: float = 35.0,
    low_battery_ref_pct: float = 20.0,
) -> tuple[SensitivityResult, ActionChangeMargin]:
    """Also find the nearest admitted representative path with another action sequence."""

    result, margin = _analyze_sensitivity(
        config,
        observations,
        durations,
        bounds,
        hot_ref_c=hot_ref_c,
        low_battery_ref_pct=low_battery_ref_pct,
        include_action_margin=True,
    )
    if margin is None:
        raise AssertionError("action-margin analysis did not produce a result")
    return result, margin
