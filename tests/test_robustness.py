from __future__ import annotations

import itertools
import math

import pytest

from trainguard.config import PolicyConfig
from trainguard.model import Action, Observation, PowerSource
from trainguard.policy import PolicyEngine
from trainguard.robustness import (
    SensitivityBounds,
    SensitivityError,
    analyze_sensitivity,
    analyze_sensitivity_with_margin,
)


def _observation(second: int, temperature: float | None) -> Observation:
    return Observation(
        source=PowerSource.AC,
        percent=50.0,
        temperature_c=temperature,
        charging=False,
        observed_at=f"2026-08-08T12:00:{second:02d}Z",
    )


def test_hysteresis_sensitivity_has_a_hand_checked_tight_envelope():
    observations = [
        _observation(0, 41.8),
        _observation(10, 36.0),
        _observation(20, 35.0),
    ]
    durations = [10.0, 10.0, 0.0]

    exact = analyze_sensitivity(PolicyConfig(), observations, durations, SensitivityBounds())
    bounds = SensitivityBounds(temperature_c=0.3)
    bounded = analyze_sensitivity(PolicyConfig(), observations, durations, bounds)

    assert (exact.minimum_run_seconds, exact.maximum_run_seconds) == (20.0, 20.0)
    assert exact.action_stable_seconds == 20.0
    # At 41.8 +/- 0.3C, gentle and thermal-stop are both possible. That
    # makes cooldown reachable; at 36 +/- 0.3C it may either persist or clear.
    assert (bounded.minimum_run_seconds, bounded.maximum_run_seconds) == (0.0, 20.0)
    assert (bounded.action_stable_seconds, bounded.action_ambiguous_seconds) == (0.0, 20.0)
    assert (bounded.action_stable_samples, bounded.action_ambiguous_samples) == (1, 2)

    report = bounded.to_dict(bounds)
    assert report["schema_version"] == 2
    assert report["bounds"] == {
        "model": "bounded_adversarial",
        "numeric_domain": "ieee_754_binary64",
        "temperature_half_width_c": 0.3,
        "charge_half_width_pct": 0.0,
        "confidence_level": None,
        "missing_measurements": "remain_missing",
        "uncertainty_set": "cartesian_product",
        "temporal_correlation": "unmodeled",
    }
    assert report["run_seconds"] == {"minimum": 0.0, "maximum": 20.0}


def test_dynamic_program_matches_an_independent_enumeration_and_boundary_contracts():
    policy = PolicyConfig()
    observations = [
        _observation(0, 42.0),
        _observation(1, 36.0),
        _observation(2, 35.0),
    ]
    durations = [1.0, 1.0, 0.0]
    result = analyze_sensitivity(
        policy,
        observations,
        durations,
        SensitivityBounds(temperature_c=1.0),
    )

    run_totals = []
    hot_totals = []
    actions_by_sample = [set(), set(), set()]
    for temperatures in itertools.product(
        (41.0, math.nextafter(42.0, -math.inf), 42.0, 43.0),
        (35.0, 36.0, 37.0),
        (34.0, 35.0, 36.0),
    ):
        engine = PolicyEngine()
        run_total = 0.0
        hot_total = 0.0
        for index, (observation, temperature, duration) in enumerate(
            zip(observations, temperatures, durations)
        ):
            perturbed = Observation(
                source=observation.source,
                percent=observation.percent,
                temperature_c=temperature,
                charging=observation.charging,
                observed_at=observation.observed_at,
            )
            action = engine.decide(policy, perturbed).action
            actions_by_sample[index].add(action)
            if action is not Action.STOP:
                run_total += duration
                hot_total += max(temperature - 35.0, 0.0) * duration
        run_totals.append(run_total)
        hot_totals.append(hot_total)

    assert result.minimum_run_seconds == min(run_totals)
    assert result.maximum_run_seconds == max(run_totals)
    assert result.minimum_hot_run_degc_seconds == min(hot_totals)
    assert result.maximum_hot_run_degc_seconds == max(hot_totals)
    assert result.action_ambiguous_seconds == sum(
        duration for actions, duration in zip(actions_by_sample, durations) if len(actions) > 1
    )

    # Missing measurements remain missing even under a wide declared bound.
    missing = [_observation(0, None), _observation(1, None)]
    missing_result = analyze_sensitivity(
        policy,
        missing,
        [1.0, 0.0],
        SensitivityBounds(temperature_c=100.0),
    )
    assert (missing_result.minimum_run_seconds, missing_result.maximum_run_seconds) == (1.0, 1.0)
    assert missing_result.action_ambiguous_seconds == 0.0

    # A zero-duration sample can enter cooldown. A later missing temperature
    # cannot clear it, so implementations must not skip either sample.
    hot_then_missing = [_observation(0, 42.0), _observation(0, None), _observation(5, None)]
    stateful = analyze_sensitivity(
        policy,
        hot_then_missing,
        [0.0, 5.0, 0.0],
        SensitivityBounds(),
    )
    assert (stateful.minimum_run_seconds, stateful.maximum_run_seconds) == (0.0, 0.0)

    # Both actions permit work, so runtime alone cannot reveal this strict
    # charge-threshold ambiguity.
    warm_charging = Observation(
        source=PowerSource.AC,
        percent=80.0,
        temperature_c=35.0,
        charging=True,
        observed_at="2026-08-08T12:00:00Z",
    )
    action_only = analyze_sensitivity(
        policy,
        [warm_charging],
        [1.0],
        SensitivityBounds(charge_pct=1.0),
    )
    assert (action_only.minimum_run_seconds, action_only.maximum_run_seconds) == (1.0, 1.0)
    assert action_only.action_ambiguous_seconds == 1.0

    # The exposure reference is part of the finite partition too.
    battery = Observation(
        source=PowerSource.BATTERY,
        percent=20.0,
        temperature_c=30.0,
        charging=False,
        observed_at="2026-08-08T12:00:00Z",
    )
    low_battery = analyze_sensitivity(
        PolicyConfig(run_on_battery=True, battery_floor_pct=10.0),
        [battery],
        [1.0],
        SensitivityBounds(charge_pct=1.0),
        low_battery_ref_pct=20.0,
    )
    assert (
        low_battery.minimum_low_battery_run_seconds,
        low_battery.maximum_low_battery_run_seconds,
    ) == (0.0, 1.0)

    invalid_bounds = (
        {"temperature_c": float("nan")},
        {"temperature_c": True},
        {"charge_pct": 101.0},
    )
    for values in invalid_bounds:
        with pytest.raises(SensitivityError, match="finite half-width"):
            SensitivityBounds(**values)

    for invalid_duration in (float("nan"), -1.0, True):
        with pytest.raises(SensitivityError, match="durations must be finite"):
            analyze_sensitivity(policy, [observations[0]], [invalid_duration], SensitivityBounds())

    invalid_references = (
        ({"hot_ref_c": float("inf")}, "hot reference"),
        ({"hot_ref_c": -101.0}, "hot reference"),
        ({"low_battery_ref_pct": float("nan")}, "low-battery reference"),
        ({"low_battery_ref_pct": 101.0}, "low-battery reference"),
    )
    for references, message in invalid_references:
        with pytest.raises(SensitivityError, match=message):
            analyze_sensitivity(
                policy,
                [observations[0]],
                [0.0],
                SensitivityBounds(),
                **references,
            )

    with pytest.raises(SensitivityError, match="empty trace"):
        analyze_sensitivity(policy, [], [], SensitivityBounds())
    with pytest.raises(SensitivityError, match="one duration"):
        analyze_sensitivity(policy, observations, durations[:-1], SensitivityBounds())


def test_action_change_margin_tracks_hidden_state_and_binary64_edges():
    policy = PolicyConfig()
    battery_stop = Observation(
        source=PowerSource.BATTERY,
        percent=50.0,
        temperature_c=41.75,
        charging=False,
        observed_at="2026-08-08T12:00:00Z",
    )
    warm_ac = _observation(1, 37.0)
    bounds = SensitivityBounds(temperature_c=0.5)
    envelope, stateful = analyze_sensitivity_with_margin(
        policy,
        [battery_stop, warm_ac],
        [1.0, 0.0],
        bounds,
    )
    # At sample 1 both paths stop, but only 42C enters cooldown. The altered
    # state first changes the action at the zero-duration final sample.
    assert envelope == analyze_sensitivity(policy, [battery_stop, warm_ac], [1.0, 0.0], bounds)
    assert stateful.minimum_normalized_action_distance == 0.5
    assert stateful.critical_sample == 2
    assert (stateful.nominal_action, stateful.alternative_action) == (
        Action.FULL,
        Action.STOP,
    )
    assert stateful.to_dict() == {
        "model": "bounded_minimum_distortion",
        "metric": "normalized_linf",
        "numeric_domain": "ieee_754_binary64",
        "minimum_normalized_action_distance": 0.5,
        "minimum_normalized_action_distance_hex": "0x1.0000000000000p-1",
        "stable_for_declared_box": False,
        "critical_sample": {
            "sample": 2,
            "observed_at": warm_ac.observed_at,
            "nominal_action": "full",
            "alternative_action": "stop",
        },
    }
    report = envelope.to_dict(bounds, action_change_margin=stateful)
    assert report["schema_version"] == 3
    assert report["action_change_margin"] == stateful.to_dict()

    charging = Observation(
        source=PowerSource.AC,
        percent=80.0,
        temperature_c=35.0,
        charging=True,
        observed_at="2026-08-08T12:00:00Z",
    )
    _, strict = analyze_sensitivity_with_margin(
        policy,
        [charging],
        [0.0],
        SensitivityBounds(charge_pct=1.0),
    )
    assert strict.minimum_normalized_action_distance == math.ulp(80.0)

    # Endpoint construction can make a second quotient microscopically exceed
    # one. An endpoint already admitted to the full box saturates at one.
    _, endpoint = analyze_sensitivity_with_margin(
        PolicyConfig(temp_gentle_c=35.1),
        [_observation(0, 35.0)],
        [0.0],
        SensitivityBounds(temperature_c=0.1),
    )
    assert endpoint.minimum_normalized_action_distance == 1.0

    _, stable = analyze_sensitivity_with_margin(
        policy,
        [_observation(0, 30.0)],
        [0.0],
        SensitivityBounds(temperature_c=0.5, charge_pct=1.0),
    )
    assert stable.minimum_normalized_action_distance is None
    assert stable.to_dict()["stable_for_declared_box"] is True
    assert stable.to_dict()["critical_sample"] is None


def test_action_change_margin_matches_independent_tiny_trace_enumeration():
    policy = PolicyConfig()
    temperature_width = math.ulp(42.0)
    observations = [
        Observation(
            source=PowerSource.BATTERY,
            percent=50.0,
            temperature_c=math.nextafter(42.0, -math.inf),
            charging=False,
            observed_at="2026-08-08T12:00:00Z",
        ),
        _observation(1, 37.0),
    ]

    def all_binary64(recorded: float) -> tuple[float, ...]:
        value = recorded - temperature_width
        high = recorded + temperature_width
        values = []
        while value <= high:
            values.append(value)
            value = math.nextafter(value, math.inf)
        return tuple(values)

    nominal_engine = PolicyEngine()
    nominal_actions = [
        nominal_engine.decide(policy, observation).action for observation in observations
    ]
    brute_force_distances = []
    candidate_temperatures = []
    for observation in observations:
        if observation.temperature_c is None:
            raise AssertionError("this oracle requires recorded temperatures")
        candidate_temperatures.append(all_binary64(observation.temperature_c))
    for temperatures in itertools.product(*candidate_temperatures):
        engine = PolicyEngine()
        actions = []
        distance = 0.0
        for observation, temperature in zip(observations, temperatures):
            perturbed = Observation(
                source=observation.source,
                percent=observation.percent,
                temperature_c=temperature,
                charging=observation.charging,
                observed_at=observation.observed_at,
            )
            actions.append(engine.decide(policy, perturbed).action)
            distance = max(
                distance,
                min(
                    1.0,
                    abs(temperature - float(observation.temperature_c)) / temperature_width,
                ),
            )
        if actions != nominal_actions:
            brute_force_distances.append(distance)

    envelope, margin = analyze_sensitivity_with_margin(
        policy,
        observations,
        [1.0, 0.0],
        SensitivityBounds(temperature_c=temperature_width),
    )
    assert brute_force_distances
    assert envelope == analyze_sensitivity(
        policy,
        observations,
        [1.0, 0.0],
        SensitivityBounds(temperature_c=temperature_width),
    )
    assert margin.minimum_normalized_action_distance == min(brute_force_distances)
