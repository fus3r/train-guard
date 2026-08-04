import re
import subprocess
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from trainguard import sensors
from trainguard.model import PowerSource

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def battery(percent=73.5, plugged=True):
    return SimpleNamespace(percent=percent, power_plugged=plugged, secsleft=0)


@pytest.fixture
def pack_temperature(monkeypatch):
    """Pin the temperature adapter so battery tests stay host independent."""

    def use(value):
        monkeypatch.setattr(sensors, "battery_temperature_c", lambda _system: value)

    return use


def test_one_snapshot_answers_every_battery_field(monkeypatch, pack_temperature):
    """Source, charge and charging have to describe the same instant.

    Separate reads could report AC from a plugged snapshot and the charge
    from the one taken after the cable was pulled.
    """
    calls = []

    def sensors_battery():
        calls.append(1)
        return battery()

    monkeypatch.setattr(sensors.psutil, "sensors_battery", sensors_battery)
    pack_temperature(36.2)

    sample = sensors.SensorReader("Darwin").sample()

    assert len(calls) == 1
    assert (sample.source, sample.percent, sample.temperature_c, sample.charging) == (
        PowerSource.AC,
        73.5,
        36.2,
        True,
    )
    assert sample.warnings == ()


def test_a_host_without_a_battery_states_its_assumption(monkeypatch, pack_temperature):
    """No battery is a missing measurement, not a full charge."""
    monkeypatch.setattr(sensors.psutil, "sensors_battery", lambda: None)
    pack_temperature(None)

    sample = sensors.SensorReader("Windows").sample()

    assert sample.source is PowerSource.NO_BATTERY
    assert (sample.percent, sample.charging, sample.temperature_c) == (None, None, None)
    assert sample.signature() == "power=no_battery batt=n/a temp=n/a charging=n/a"
    assert len(sample.warnings) == 2


def test_a_failing_sensor_becomes_a_warning(monkeypatch, pack_temperature):
    def explode():
        raise RuntimeError("driver reset")

    monkeypatch.setattr(sensors.psutil, "sensors_battery", explode)
    pack_temperature(31.0)

    sample = sensors.SensorReader("Linux").sample()

    assert sample.source is PowerSource.NO_BATTERY
    assert "driver reset" in sample.warnings[0]


@pytest.mark.parametrize("percent", [float("nan"), float("inf"), -1, 101, "bad"])
def test_an_unusable_charge_is_dropped_rather_than_guessed(monkeypatch, pack_temperature, percent):
    monkeypatch.setattr(sensors.psutil, "sensors_battery", lambda: battery(percent=percent))
    pack_temperature(30.0)

    sample = sensors.SensorReader("TestOS").sample()

    assert sample.percent is None
    assert sample.charging is None
    assert any("percentage was invalid" in warning for warning in sample.warnings)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -21, 101])
def test_an_implausible_pack_temperature_skips_the_thermal_rules(
    monkeypatch, pack_temperature, temperature
):
    monkeypatch.setattr(sensors.psutil, "sensors_battery", lambda: battery())
    pack_temperature(temperature)

    sample = sensors.SensorReader("TestOS").sample()

    assert sample.temperature_c is None
    assert any("outside the supported range" in warning for warning in sample.warnings)


def test_observations_are_stamped_to_the_millisecond(monkeypatch, pack_temperature):
    """poll may be configured below a second, so whole seconds would collide."""
    monkeypatch.setattr(sensors.psutil, "sensors_battery", lambda: battery())
    pack_temperature(30.0)

    stamp = sensors.SensorReader("Darwin").sample().observed_at

    assert TIMESTAMP.fullmatch(stamp)
    assert datetime.fromisoformat(stamp.replace("Z", "+00:00")).utcoffset() == timedelta(0)


def test_macos_reads_the_pack_through_ioreg(monkeypatch):
    result = subprocess.CompletedProcess([], 0, stdout='"Temperature" = 3650\n', stderr="")
    monkeypatch.setattr(sensors.subprocess, "run", lambda *args, **kwargs: result)

    assert sensors.battery_temperature_c("Darwin") == 36.5


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(37.25, 37.25), (372, 37.2), (37250, 37.25)],
    ids=["celsius", "tenths", "millidegrees"],
)
def test_a_psutil_reading_is_rescaled_only_when_impossible(monkeypatch, reported, expected):
    monkeypatch.setattr(
        sensors.psutil,
        "sensors_temperatures",
        lambda: {"BAT0": [SimpleNamespace(current=reported)]},
        raising=False,
    )

    assert sensors.battery_temperature_c("Linux") == expected


def test_a_cpu_sensor_is_never_taken_for_the_pack(monkeypatch):
    """A 75 C core reading would pause every job on this host."""
    monkeypatch.setattr(
        sensors.psutil,
        "sensors_temperatures",
        lambda: {"coretemp": [SimpleNamespace(current=75)]},
        raising=False,
    )
    monkeypatch.setattr(sensors.Path, "glob", lambda *_args: [])

    assert sensors.battery_temperature_c("Linux") is None


def test_the_sysfs_fallback_is_always_read_as_tenths(monkeypatch, tmp_path):
    """power_supply/*/temp has a defined unit: a cold pack is not 45 C.

    Guessing the unit here read 4.5 C as 45 C, which parked a cool pack in a
    cooldown it could never leave.
    """
    monkeypatch.setattr(sensors.psutil, "sensors_temperatures", lambda: {}, raising=False)
    temp_file = tmp_path / "BAT0" / "temp"
    temp_file.parent.mkdir()
    monkeypatch.setattr(sensors.Path, "glob", lambda *_args: [temp_file])

    for raw, expected in (("45\n", 4.5), ("372\n", 37.2)):
        temp_file.write_text(raw, encoding="utf-8")
        assert sensors.battery_temperature_c("Linux") == expected
