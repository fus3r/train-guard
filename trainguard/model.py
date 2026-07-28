"""Shared domain models for train-guard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Tuple


def utc_now() -> str:
    """Return an RFC 3339 timestamp in UTC, with milliseconds.

    The documented poll interval can be set below one second, so a
    whole-second stamp would give two different observations the same time.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class PowerSource(str, Enum):
    AC = "ac"
    BATTERY = "battery"
    NO_BATTERY = "no_battery"


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
    NO_BATTERY = "no_battery"
    AC_POLICY = "ac_policy"


@dataclass(frozen=True)
class Observation:
    """One internally consistent power and thermal sample.

    Every field is read from a single battery snapshot, so the source, the
    charge and the charging flag always describe the same instant. A missing
    field is a measurement this host did not provide; it is never replaced by
    a plausible-looking default, because a substituted value would silently
    select a policy branch the hardware never justified. ``warnings`` carries
    the reason each measurement is absent.
    """

    source: PowerSource
    percent: Optional[float]
    temperature_c: Optional[float]
    charging: Optional[bool]
    observed_at: str
    warnings: Tuple[str, ...] = ()

    def signature(self) -> str:
        percent = "n/a" if self.percent is None else f"{self.percent:g}%"
        temperature = "n/a" if self.temperature_c is None else f"{self.temperature_c:g}C"
        charging = "n/a" if self.charging is None else ("yes" if self.charging else "no")
        return f"power={self.source.value} batt={percent} temp={temperature} charging={charging}"


@dataclass(frozen=True)
class PolicyDecision:
    """What the job may do, why, and the cooldown state that produced it."""

    action: Action
    reason: DecisionReason
    cooling: bool


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    """A process identifier suitable for validated on-disk state.

    Keeping both fields strict prevents ambiguous recovery data from entering
    the state directory.
    """

    pid: int
    create_time: float

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("process pid must be a positive integer")
        if (
            isinstance(self.create_time, bool)
            or not isinstance(self.create_time, (int, float))
            or not math.isfinite(float(self.create_time))
            or self.create_time <= 0
        ):
            raise ValueError("process create_time must be a positive finite number")

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "create_time": self.create_time}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProcessIdentity":
        if not isinstance(value, dict):
            raise ValueError("process identity must be an object")
        try:
            pid = value["pid"]
            create_time = value["create_time"]
        except KeyError as exc:
            raise ValueError(f"process identity is missing {exc.args[0]}") from exc
        return cls(pid=pid, create_time=create_time)
