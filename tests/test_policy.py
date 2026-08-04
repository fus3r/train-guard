from dataclasses import replace

import pytest

from trainguard.config import PolicyConfig
from trainguard.model import Action, DecisionReason, Observation, PowerSource, utc_now
from trainguard.policy import PolicyEngine

DEFAULTS = PolicyConfig()
BATTERY_GENTLE = replace(DEFAULTS, run_on_battery=True)
BATTERY_FULL = replace(DEFAULTS, run_on_battery=True, battery_band="full")


def sample(*, source=PowerSource.AC, percent=80.0, temperature=30.0, charging=False):
    """Build the observation the engine would receive for these readings."""
    return Observation(
        source=source,
        percent=percent,
        temperature_c=temperature,
        charging=charging,
        observed_at=utc_now(),
    )


@pytest.mark.parametrize(
    ("policy", "percent", "temperature", "expected"),
    [
        (DEFAULTS, 95, None, (Action.STOP, DecisionReason.BATTERY_DISABLED)),
        (BATTERY_GENTLE, 31, None, (Action.GENTLE, DecisionReason.BATTERY_POLICY)),
        (BATTERY_GENTLE, 30, None, (Action.STOP, DecisionReason.BATTERY_FLOOR)),
        (BATTERY_FULL, 80, 38.0, (Action.FULL, DecisionReason.BATTERY_POLICY)),
    ],
    ids=["disabled", "above-floor", "at-floor", "band-outranks-ac-warmth"],
)
def test_battery_rules_run_before_the_ac_rules(policy, percent, temperature, expected):
    decision = PolicyEngine().decide(
        policy,
        sample(source=PowerSource.BATTERY, percent=percent, temperature=temperature),
    )

    assert (decision.action, decision.reason) == expected


def test_thermal_limit_outranks_a_permissive_battery_band():
    """No configuration may buy its way past the pause threshold."""
    policy = replace(BATTERY_FULL, battery_floor_pct=0)

    decision = PolicyEngine().decide(
        policy, sample(source=PowerSource.BATTERY, percent=100, temperature=42.0)
    )

    assert (decision.action, decision.reason) == (Action.STOP, DecisionReason.THERMAL_LIMIT)


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (37.9, (Action.FULL, DecisionReason.AC_POLICY)),
        (38.0, (Action.GENTLE, DecisionReason.WARM_AC)),
        (41.9, (Action.GENTLE, DecisionReason.WARM_AC)),
        (42.0, (Action.STOP, DecisionReason.THERMAL_LIMIT)),
    ],
)
def test_ac_temperature_boundaries(temperature, expected):
    decision = PolicyEngine().decide(DEFAULTS, sample(temperature=temperature))

    assert (decision.action, decision.reason) == expected


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        (79, (Action.GENTLE, DecisionReason.WARM_CHARGING)),
        (80, (Action.FULL, DecisionReason.AC_POLICY)),
    ],
)
def test_warm_charging_rule_stops_at_the_charge_cutoff(percent, expected):
    decision = PolicyEngine().decide(
        DEFAULTS, sample(percent=percent, temperature=35.0, charging=True)
    )

    assert (decision.action, decision.reason) == expected


def test_warm_charging_is_named_before_generic_ac_warmth():
    """Both rules ask for gentle here; the specific one has to report why."""
    decision = PolicyEngine().decide(DEFAULTS, sample(percent=79, temperature=39.0, charging=True))

    assert (decision.action, decision.reason) == (Action.GENTLE, DecisionReason.WARM_CHARGING)


def test_ac_band_selects_the_cool_default():
    decision = PolicyEngine().decide(replace(DEFAULTS, ac_band="gentle"), sample())

    assert (decision.action, decision.reason) == (Action.GENTLE, DecisionReason.AC_POLICY)


def test_thermal_pause_holds_until_the_resume_threshold():
    engine = PolicyEngine()

    paused = engine.decide(DEFAULTS, sample(temperature=42.0))
    still_warm = engine.decide(DEFAULTS, sample(temperature=37.0))
    resumed = engine.decide(DEFAULTS, sample(temperature=36.0))

    assert (paused.action, paused.reason, paused.cooling) == (
        Action.STOP,
        DecisionReason.THERMAL_LIMIT,
        True,
    )
    assert (still_warm.action, still_warm.reason) == (
        Action.STOP,
        DecisionReason.THERMAL_COOLDOWN,
    )
    assert (resumed.action, resumed.reason, resumed.cooling) == (
        Action.FULL,
        DecisionReason.AC_POLICY,
        False,
    )


def test_cooldown_without_a_reading_keeps_the_job_paused():
    engine = PolicyEngine(cooling=True)

    decision = engine.decide(DEFAULTS, sample(temperature=None))

    assert (decision.action, decision.reason) == (Action.STOP, DecisionReason.THERMAL_COOLDOWN)
    assert engine.cooling is True


def test_missing_temperature_skips_only_the_thermal_rules():
    engine = PolicyEngine()

    warm_charge_path = engine.decide(DEFAULTS, sample(percent=20, temperature=None, charging=True))
    battery_path = engine.decide(
        DEFAULTS, sample(source=PowerSource.BATTERY, percent=90, temperature=None)
    )

    assert (warm_charge_path.action, warm_charge_path.reason) == (
        Action.FULL,
        DecisionReason.AC_POLICY,
    )
    assert (battery_path.action, battery_path.reason) == (
        Action.STOP,
        DecisionReason.BATTERY_DISABLED,
    )


def test_a_missing_charge_skips_only_the_rules_that_need_it():
    """Losing the percentage must not cost the job its thermal protection."""
    engine = PolicyEngine()

    on_battery = engine.decide(
        BATTERY_GENTLE, sample(source=PowerSource.BATTERY, percent=None, temperature=30.0)
    )
    while_charging = engine.decide(DEFAULTS, sample(percent=None, temperature=39.0, charging=True))

    assert (on_battery.action, on_battery.reason) == (Action.GENTLE, DecisionReason.BATTERY_POLICY)
    assert (while_charging.action, while_charging.reason) == (Action.GENTLE, DecisionReason.WARM_AC)


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (30.0, (Action.FULL, DecisionReason.NO_BATTERY)),
        (39.0, (Action.GENTLE, DecisionReason.WARM_AC)),
        (43.0, (Action.STOP, DecisionReason.THERMAL_LIMIT)),
    ],
)
def test_a_host_without_a_battery_climbs_the_ac_thermal_ladder(temperature, expected):
    """Only the reason reported at the foot of the ladder may differ from AC."""
    decision = PolicyEngine().decide(
        DEFAULTS,
        sample(
            source=PowerSource.NO_BATTERY,
            percent=None,
            temperature=temperature,
            charging=None,
        ),
    )

    assert (decision.action, decision.reason) == expected


def test_each_engine_keeps_its_own_cooldown_state():
    """Two supervisors in one process must not share a thermal pause."""
    hot, cool = PolicyEngine(), PolicyEngine()

    assert hot.decide(DEFAULTS, sample(temperature=42.0)).cooling is True

    unaffected = cool.decide(DEFAULTS, sample(temperature=37.0))
    still_cooling = hot.decide(DEFAULTS, sample(temperature=37.0))

    assert (unaffected.action, unaffected.reason) == (Action.FULL, DecisionReason.AC_POLICY)
    assert cool.cooling is False
    assert still_cooling.reason is DecisionReason.THERMAL_COOLDOWN


def test_the_log_signature_reports_the_measured_values():
    signature = sample(source=PowerSource.BATTERY, percent=55.0, temperature=36.2).signature()

    assert signature == "power=battery batt=55% temp=36.2C charging=no"
