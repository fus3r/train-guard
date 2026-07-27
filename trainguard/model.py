"""Shared domain models for train-guard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PowerSource(str, Enum):
    AC = "ac"
    BATTERY = "battery"


class Action(str, Enum):
    FULL = "full"
    GENTLE = "gentle"
    STOP = "stop"


class DecisionReason(str, Enum):
    """Why the policy chose an action, in evaluation order."""

    THERMAL_COOLDOWN = "thermal_cooldown"
    THERMAL_LIMIT = "thermal_limit"
    BATTERY_DISABLED = "battery_disabled"
    BATTERY_FLOOR = "battery_floor"
    BATTERY_POLICY = "battery_policy"
    WARM_CHARGING = "warm_charging"
    WARM_AC = "warm_ac"
    AC_POLICY = "ac_policy"


@dataclass(frozen=True)
class Observation:
    """One power and thermal sample handed to the policy.

    ``temperature_c`` is optional because several supported platforms do not
    expose a battery pack sensor.
    """

    source: PowerSource
    percent: float
    temperature_c: Optional[float]
    charging: bool

    def signature(self) -> str:
        temperature = "n/a" if self.temperature_c is None else f"{self.temperature_c:.0f}C"
        return (
            f"power={self.source.value} batt={self.percent:g}% "
            f"temp={temperature} charging={'yes' if self.charging else 'no'}"
        )


@dataclass(frozen=True)
class PolicyDecision:
    """What the job may do, why, and the cooldown state that produced it."""

    action: Action
    reason: DecisionReason
    cooling: bool
