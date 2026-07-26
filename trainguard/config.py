"""Policy configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

if __package__:
    from .model import Action
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from model import Action


class ConfigError(ValueError):
    """Raised when a policy file cannot be used safely."""


@dataclass(frozen=True)
class PolicyConfig:
    poll: float = 20.0
    run_on_battery: bool = False
    battery_floor_pct: float = 30.0
    battery_band: str = Action.GENTLE.value
    ac_band: str = Action.FULL.value
    temp_gentle_c: float = 38.0
    temp_pause_c: float = 42.0
    temp_resume_c: float = 36.0
    charge_cool_until_pct: float = 80.0
    temp_charge_gentle_c: float = 35.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FIELD_NAMES = {field.name for field in fields(PolicyConfig)}
_LEGACY_KEYS = {
    "temp_ecore_c": "temp_gentle_c",
    "temp_charge_ecore_c": "temp_charge_gentle_c",
}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _percentage(value: Any, name: str) -> float:
    number = _number(value, name)
    if not 0 <= number <= 100:
        raise ConfigError(f"{name} must be between 0 and 100")
    return number


def _temperature(value: Any, name: str) -> float:
    number = _number(value, name)
    if not -20 <= number <= 100:
        raise ConfigError(f"{name} must be between -20 and 100 degrees Celsius")
    return number


def policy_from_mapping(values: Mapping[str, Any]) -> PolicyConfig:
    """Parse a complete or partial JSON object into a validated policy."""

    migrated = dict(values)
    for old, new in _LEGACY_KEYS.items():
        if old in migrated and new in migrated and migrated[old] != migrated[new]:
            raise ConfigError(f"{old} conflicts with its replacement {new}")
        if old in migrated and new not in migrated:
            migrated[new] = migrated[old]
        migrated.pop(old, None)

    unknown = sorted(set(migrated) - _FIELD_NAMES)
    if unknown:
        raise ConfigError(f"unknown configuration key(s): {', '.join(unknown)}")

    merged = PolicyConfig().to_dict()
    merged.update(migrated)

    if not isinstance(merged["run_on_battery"], bool):
        raise ConfigError("run_on_battery must be true or false")

    poll = _number(merged["poll"], "poll")
    if not 0.1 <= poll <= 3600:
        raise ConfigError("poll must be between 0.1 and 3600 seconds")

    for name in ("battery_band", "ac_band"):
        value = merged[name]
        if not isinstance(value, str) or value not in {
            Action.FULL.value,
            Action.GENTLE.value,
        }:
            raise ConfigError(f"{name} must be 'full' or 'gentle'")

    config = PolicyConfig(
        poll=poll,
        run_on_battery=merged["run_on_battery"],
        battery_floor_pct=_percentage(merged["battery_floor_pct"], "battery_floor_pct"),
        battery_band=merged["battery_band"],
        ac_band=merged["ac_band"],
        temp_gentle_c=_temperature(merged["temp_gentle_c"], "temp_gentle_c"),
        temp_pause_c=_temperature(merged["temp_pause_c"], "temp_pause_c"),
        temp_resume_c=_temperature(merged["temp_resume_c"], "temp_resume_c"),
        charge_cool_until_pct=_percentage(
            merged["charge_cool_until_pct"], "charge_cool_until_pct"
        ),
        temp_charge_gentle_c=_temperature(
            merged["temp_charge_gentle_c"], "temp_charge_gentle_c"
        ),
    )

    if config.temp_resume_c >= config.temp_pause_c:
        raise ConfigError("temp_resume_c must be lower than temp_pause_c")
    if config.temp_gentle_c > config.temp_pause_c:
        raise ConfigError("temp_gentle_c cannot exceed temp_pause_c")
    if config.temp_charge_gentle_c > config.temp_pause_c:
        raise ConfigError("temp_charge_gentle_c cannot exceed temp_pause_c")
    return config


def parse_policy_text(text: str, source: str = "config") -> PolicyConfig:
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{source}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(values, dict):
        raise ConfigError(f"{source}: the top-level JSON value must be an object")
    try:
        return policy_from_mapping(values)
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


def load_policy(path: Path) -> PolicyConfig:
    if not path.exists():
        return PolicyConfig()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read configuration: {exc}") from exc
    return parse_policy_text(text, str(path))


class ConfigWatcher:
    """Reload a live policy while retaining the last valid version."""

    def __init__(self, path: Path, initial: Optional[PolicyConfig] = None):
        self.path = path
        self.current = initial or PolicyConfig()
        self._fingerprint: Optional[str] = None
        self.last_error: Optional[str] = None

    def poll(self) -> tuple[PolicyConfig, bool, Optional[str]]:
        try:
            raw = self.path.read_bytes() if self.path.exists() else b""
        except OSError as exc:
            error = f"{self.path}: cannot read configuration: {exc}"
            changed_error = error != self.last_error
            self.last_error = error
            return self.current, changed_error, error

        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint == self._fingerprint:
            return self.current, False, self.last_error

        self._fingerprint = fingerprint
        try:
            candidate = (
                parse_policy_text(raw.decode("utf-8"), str(self.path)) if raw else PolicyConfig()
            )
        except (UnicodeDecodeError, ConfigError) as exc:
            error = (
                f"{self.path}: configuration must be UTF-8"
                if isinstance(exc, UnicodeDecodeError)
                else str(exc)
            )
            changed_error = error != self.last_error
            self.last_error = error
            return self.current, changed_error, error

        changed = candidate != self.current or self.last_error is not None
        self.current = candidate
        self.last_error = None
        return self.current, changed, None
