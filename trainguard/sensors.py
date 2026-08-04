"""Host power and battery sensor adapters.

The supervisor asks this module for one sample per cycle. Reading the battery
once and deriving every field from that snapshot is what keeps a report
consistent: with separate calls, the source could come from a plugged-in
reading and the percentage from the reading taken after the cable was pulled.
"""

from __future__ import annotations

import math
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import psutil

if __package__:
    from .model import Observation, PowerSource, utc_now
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from model import Observation, PowerSource, utc_now

# The pack temperatures a supported laptop can actually report. The policy
# validator accepts thresholds over the same range.
MIN_PLAUSIBLE_C = -20.0
MAX_PLAUSIBLE_C = 100.0


def _normalise_psutil_temperature(value: float) -> float:
    """Rescale hwmon readings from drivers that misreport their unit.

    psutil documents degrees Celsius, so a plausible value passes through
    untouched and only a magnitude no battery pack can reach is treated as
    tenths or millidegrees. The sysfs fallback below never uses this guess:
    that interface has a defined unit.
    """
    magnitude = abs(value)
    if magnitude >= 1000:
        return value / 1000.0
    if magnitude > 200:
        return value / 10.0
    return value


def battery_temperature_c(system: Optional[str] = None) -> Optional[float]:
    """Battery pack temperature in degrees Celsius, or None if unavailable."""
    system = system or platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["ioreg", "-rn", "AppleSmartBattery", "-w0"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            match = re.search(r'"Temperature"\s*=\s*(\d+)', out)
            return float(match.group(1)) / 100.0 if match else None

        if system == "Linux":
            temps: dict[str, list[Any]] = (
                getattr(psutil, "sensors_temperatures", lambda: {})() or {}
            )
            for name, entries in temps.items():
                if "bat" not in name.lower():
                    continue
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if current is not None:
                        return _normalise_psutil_temperature(float(current))

            # The power-supply class ABI defines this file as tenths of a
            # degree, so a cold pack reading of 45 is 4.5 C and never 45 C.
            # Guessing the unit here froze cool packs in a cooldown they
            # could not leave.
            for path in Path("/sys/class/power_supply").glob("BAT*/temp"):
                try:
                    raw = float(path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    continue
                return raw / 10.0
        # Windows exposes pack temperature through vendor tools only.
    except (OSError, subprocess.SubprocessError):
        return None
    return None


class SensorReader:
    """Take one battery snapshot per policy cycle."""

    def __init__(self, system: Optional[str] = None):
        self.system = system or platform.system()

    def sample(self) -> Observation:
        warnings = []
        try:
            battery: Any = psutil.sensors_battery()
        except (OSError, RuntimeError) as exc:
            battery = None
            warnings.append(f"battery sensor failed: {exc}")

        if battery is None:
            source = PowerSource.NO_BATTERY
            percent = None
            charging = None
            warnings.append("battery not exposed; treating this host as mains-powered")
        else:
            source = PowerSource.AC if battery.power_plugged else PowerSource.BATTERY
            try:
                raw_percent = float(battery.percent)
            except (TypeError, ValueError):
                raw_percent = math.nan
            if math.isfinite(raw_percent) and 0 <= raw_percent <= 100:
                percent = raw_percent
                # psutil reports the AC connection, not charge current, so
                # this is a portable inference and not a hardware reading.
                charging = bool(battery.power_plugged and percent < 100)
            else:
                percent = None
                charging = None
                warnings.append("battery percentage was invalid and has been ignored")

        temperature = battery_temperature_c(self.system)
        if temperature is not None and not (
            math.isfinite(temperature) and MIN_PLAUSIBLE_C <= temperature <= MAX_PLAUSIBLE_C
        ):
            warnings.append("battery temperature was outside the supported range and was ignored")
            temperature = None
        if temperature is None:
            warnings.append("battery temperature unavailable; thermal rules skipped")

        return Observation(
            source=source,
            percent=percent,
            temperature_c=temperature,
            charging=charging,
            observed_at=utc_now(),
            warnings=tuple(warnings),
        )
