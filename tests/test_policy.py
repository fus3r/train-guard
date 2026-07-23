import json

import pytest

from trainguard import cli


def set_readings(monkeypatch, *, source="AC", percent=80, temp=30.0, charging=False):
    monkeypatch.setattr(cli, "power_source", lambda: source)
    monkeypatch.setattr(cli, "battery_pct", lambda: percent)
    monkeypatch.setattr(cli, "battery_temp_c", lambda: temp)
    monkeypatch.setattr(cli, "is_charging", lambda: charging)


def test_defaults_match_documented_thresholds():
    cfg = cli.load_config()
    assert cfg["temp_charge_gentle_c"] == 35
    assert cfg["temp_gentle_c"] == 38
    assert cfg["temp_pause_c"] == 42
    assert cfg["temp_resume_c"] == 36


def test_v01_ecore_config_keys_are_migrated():
    cli._ensure_dirs()
    cli.CONFIGF.write_text(json.dumps({
        "temp_ecore_c": 39,
        "temp_charge_ecore_c": 34,
    }))

    cfg = cli.load_config()

    assert cfg["temp_gentle_c"] == 39
    assert cfg["temp_charge_gentle_c"] == 34
    assert "temp_ecore_c" not in cfg
    assert "temp_charge_ecore_c" not in cfg


def test_pauses_on_battery_by_default(monkeypatch):
    set_readings(monkeypatch, source="Battery", percent=95, temp=None)
    decision, signature = cli.decide(cli.load_config())
    assert decision == "stop"
    assert "power=Battery" in signature


def test_optional_battery_mode_obeys_floor(monkeypatch):
    cfg = cli.load_config()
    cfg["run_on_battery"] = True

    set_readings(monkeypatch, source="Battery", percent=31, temp=None)
    assert cli.decide(cfg)[0] == "gentle"

    set_readings(monkeypatch, source="Battery", percent=30, temp=None)
    assert cli.decide(cfg)[0] == "stop"


def test_thermal_pause_has_hysteresis(monkeypatch):
    temperature = {"value": 42.0}
    set_readings(monkeypatch, temp=30.0)
    monkeypatch.setattr(cli, "battery_temp_c", lambda: temperature["value"])
    cfg = cli.load_config()

    assert cli.decide(cfg)[0] == "stop"
    assert cli._Cool.on is True

    temperature["value"] = 37.0
    assert cli.decide(cfg)[0] == "stop"

    temperature["value"] = 36.0
    assert cli.decide(cfg)[0] == "full"
    assert cli._Cool.on is False


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(37.9, "full"), (38.0, "gentle"), (41.9, "gentle"), (42.0, "stop")],
)
def test_ac_temperature_boundaries(monkeypatch, temperature, expected):
    set_readings(monkeypatch, temp=temperature)
    assert cli.decide(cli.load_config())[0] == expected


def test_warm_low_charge_uses_gentle_mode(monkeypatch):
    set_readings(monkeypatch, percent=79, temp=35.0, charging=True)
    assert cli.decide(cli.load_config())[0] == "gentle"


def test_missing_temperature_skips_only_thermal_rules(monkeypatch):
    set_readings(monkeypatch, source="AC", temp=None, charging=True, percent=20)
    assert cli.decide(cli.load_config())[0] == "full"

    set_readings(monkeypatch, source="Battery", temp=None, percent=90)
    assert cli.decide(cli.load_config())[0] == "stop"
