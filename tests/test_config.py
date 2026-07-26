from __future__ import annotations

import json

import pytest

from trainguard.config import (
    ConfigError,
    ConfigWatcher,
    PolicyConfig,
    load_policy,
    parse_policy_text,
    policy_from_mapping,
)


def test_defaults_match_documented_policy():
    assert PolicyConfig().to_dict() == {
        "poll": 20.0,
        "run_on_battery": False,
        "battery_floor_pct": 30.0,
        "battery_band": "gentle",
        "ac_band": "full",
        "temp_gentle_c": 38.0,
        "temp_pause_c": 42.0,
        "temp_resume_c": 36.0,
        "charge_cool_until_pct": 80.0,
        "temp_charge_gentle_c": 35.0,
    }


def test_legacy_ecore_names_are_migrated():
    policy = policy_from_mapping({"temp_ecore_c": 39, "temp_charge_ecore_c": 34})
    assert policy.temp_gentle_c == 39
    assert policy.temp_charge_gentle_c == 34


def test_conflicting_legacy_and_current_names_are_rejected():
    with pytest.raises(ConfigError, match="conflicts"):
        policy_from_mapping({"temp_ecore_c": 39, "temp_gentle_c": 38})
    assert policy_from_mapping({"temp_ecore_c": 38, "temp_gentle_c": 38}).temp_gentle_c == 38


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"typo": 1}, "unknown configuration key"),
        ({"run_on_battery": 1}, "true or false"),
        ({"poll": True}, "poll must be a number"),
        ({"poll": 0}, "between 0.1 and 3600"),
        ({"battery_floor_pct": 101}, "between 0 and 100"),
        ({"ac_band": "fast"}, "full.*gentle"),
        ({"ac_band": []}, "full.*gentle"),
        ({"battery_band": {}}, "full.*gentle"),
        (
            {"temp_resume_c": 42, "temp_pause_c": 42},
            "temp_resume_c must be lower",
        ),
        (
            {"temp_gentle_c": 43, "temp_pause_c": 42},
            "temp_gentle_c cannot exceed",
        ),
        (
            {"temp_charge_gentle_c": 43, "temp_pause_c": 42},
            "temp_charge_gentle_c cannot exceed",
        ),
    ],
)
def test_invalid_policy_values_are_rejected(value, message):
    with pytest.raises(ConfigError, match=message):
        policy_from_mapping(value)


def test_json_errors_include_location():
    with pytest.raises(ConfigError, match=r"line 1, column 1[12]"):
        parse_policy_text('{"poll": 2,}', "example.json")


def test_top_level_value_must_be_object():
    with pytest.raises(ConfigError, match="top-level JSON"):
        parse_policy_text("[]")


def test_missing_policy_file_uses_defaults(tmp_path):
    assert load_policy(tmp_path / "missing.json") == PolicyConfig()


def test_watcher_keeps_last_valid_policy_and_recovers(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"poll": 2}), encoding="utf-8")
    initial = load_policy(path)
    watcher = ConfigWatcher(path, initial)

    policy, changed, error = watcher.poll()
    assert policy.poll == 2
    assert changed is False
    assert error is None

    path.write_text('{"poll": "fast"}', encoding="utf-8")
    policy, changed, error = watcher.poll()
    assert policy.poll == 2
    assert changed is True
    assert "poll must be a number" in (error or "")

    policy, changed, error = watcher.poll()
    assert policy.poll == 2
    assert changed is False
    assert error is not None

    path.write_text(json.dumps({"poll": 3}), encoding="utf-8")
    policy, changed, error = watcher.poll()
    assert policy.poll == 3
    assert changed is True
    assert error is None
