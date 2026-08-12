"""Deterministic offline replay for power and thermal policy traces."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__:
    from .config import PolicyConfig
    from .model import Action, Observation, PowerSource
    from .policy import PolicyEngine
    from .robustness import SensitivityBounds, analyze_sensitivity_with_margin
else:  # Keep direct CLI execution from a checkout working.
    from config import PolicyConfig
    from model import Action, Observation, PowerSource
    from policy import PolicyEngine
    from robustness import SensitivityBounds, analyze_sensitivity_with_margin


_OBSERVATION_KEYS = frozenset(
    {"source", "percent", "temperature_c", "charging", "observed_at", "warnings"}
)
_FRACTION = re.compile(r"\.(\d+)(?=[+-]|$)")


class TraceError(ValueError):
    """Raised when a trace is malformed or temporally inconsistent."""


def _timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise TraceError(f"{context}: observed_at must be an RFC 3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    # RFC 3339 permits any fraction length. Python 3.9 does not, and replay
    # operates at microsecond resolution, so normalize and truncate explicitly.
    normalized = _FRACTION.sub(
        lambda match: "." + match.group(1)[:6].ljust(6, "0"),
        normalized,
        count=1,
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise TraceError(f"{context}: observed_at must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TraceError(f"{context}: observed_at must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _optional_number(
    value: Any,
    name: str,
    context: str,
    *,
    minimum: float,
    maximum: float,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"{context}: {name} must be a number or null")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise TraceError(f"{context}: {name} must be between {minimum:g} and {maximum:g}")
    return number


def observation_from_mapping(value: Mapping[str, Any], context: str) -> Observation:
    """Validate one raw observation or event observation payload."""

    unknown = sorted(str(key) for key in set(value) - _OBSERVATION_KEYS)
    if unknown:
        raise TraceError(f"{context}: unknown observation key(s): {', '.join(unknown)}")

    try:
        source = PowerSource(value["source"])
    except KeyError as exc:
        raise TraceError(f"{context}: observation is missing source") from exc
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in PowerSource)
        raise TraceError(f"{context}: source must be one of {choices}") from exc

    observed_at = value.get("observed_at")
    _timestamp(observed_at, context)
    percent = _optional_number(
        value.get("percent"),
        "percent",
        context,
        minimum=0,
        maximum=100,
    )
    temperature = _optional_number(
        value.get("temperature_c"),
        "temperature_c",
        context,
        minimum=-100,
        maximum=200,
    )
    charging = value.get("charging")
    if charging is not None and not isinstance(charging, bool):
        raise TraceError(f"{context}: charging must be true, false or null")

    warnings = value.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise TraceError(f"{context}: warnings must be a list of strings")
    if source is PowerSource.NO_BATTERY and (percent is not None or charging is not None):
        raise TraceError(f"{context}: no_battery observations cannot have charge data")

    return Observation(
        source=source,
        percent=percent,
        temperature_c=temperature,
        charging=charging,
        observed_at=str(observed_at),
        warnings=tuple(warnings),
    )


def load_trace(path: Path) -> tuple[Observation, ...]:
    """Read raw observation JSONL or a train-guard event journal."""

    observations: list[Observation] = []
    previous: Optional[datetime] = None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TraceError(f"{path}: cannot read trace: {exc}") from exc

    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        context = f"{path}:{line_number}"
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            # A live one-writer journal may be observed during its final append.
            # Only that final unterminated line is safe to ignore.
            if line_number == len(lines) and not text.endswith(("\n", "\r")):
                break
            raise TraceError(f"{context}: invalid JSON at column {exc.colno}") from exc
        if not isinstance(value, dict):
            raise TraceError(f"{context}: each JSON line must be an object")

        if "observation" in value:
            raw_observation = value["observation"]
        elif "event" in value:
            continue
        else:
            raw_observation = value
        if not isinstance(raw_observation, dict):
            raise TraceError(f"{context}: observation must be an object")

        observation = observation_from_mapping(raw_observation, context)
        current = _timestamp(observation.observed_at, context)
        if previous is not None and current < previous:
            raise TraceError(f"{context}: observations must be ordered by observed_at")
        observations.append(observation)
        previous = current

    if not observations:
        raise TraceError(f"{path}: trace contains no observations")
    return tuple(observations)


def ordered_timestamps(observations: Sequence[Observation]) -> list[datetime]:
    """Parse and order-check observation timestamps in one place."""

    timestamps = [
        _timestamp(observation.observed_at, f"observation {index + 1}")
        for index, observation in enumerate(observations)
    ]
    for index in range(1, len(timestamps)):
        if timestamps[index] < timestamps[index - 1]:
            raise TraceError(
                f"observation {index + 1}: observations must be ordered by observed_at"
            )
    return timestamps


def _canonical(value: Any) -> Any:
    """Normalize negative zero before canonical JSON serialization."""

    if isinstance(value, float):
        return value + 0.0
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(
        _canonical(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def simulate_policy(
    config: PolicyConfig,
    observations: Sequence[Observation],
    *,
    sensitivity: Optional[SensitivityBounds] = None,
) -> dict[str, Any]:
    """Replay one policy and return sample- and time-weighted results."""

    if not observations:
        raise TraceError("cannot simulate an empty trace")
    timestamps = ordered_timestamps(observations)
    durations = [0.0] * len(observations)
    for index in range(len(observations) - 1):
        durations[index] = (timestamps[index + 1] - timestamps[index]).total_seconds()

    engine = PolicyEngine()
    action_samples: Counter[str] = Counter()
    reason_samples: Counter[str] = Counter()
    action_seconds = {action.value: 0.0 for action in Action}
    decisions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_action: Optional[str] = None
    previous_signature: Optional[tuple[str, str]] = None
    action_transitions = 0
    decision_transitions = 0

    for index, observation in enumerate(observations):
        decision = engine.decide(config, observation)
        action = decision.action.value
        reason = decision.reason.value
        action_samples[action] += 1
        reason_samples[reason] += 1

        duration = durations[index]
        action_seconds[action] += duration
        record = {
            "sample": index + 1,
            "observed_at": observation.observed_at,
            "duration_seconds": duration,
            "observation": observation.to_dict(),
            "decision": decision.to_dict(),
        }
        decisions.append(record)

        signature = (action, reason)
        if previous_action is not None and action != previous_action:
            action_transitions += 1
        if previous_signature is not None and signature != previous_signature:
            decision_transitions += 1
        # The first transition row is the initial state. Every later row is
        # one decision transition, hence len == decision_transitions + 1.
        if signature != previous_signature:
            transitions.append(record)
        previous_action = action
        previous_signature = signature

    elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
    action_percent = {
        action.value: (100.0 * action_seconds[action.value] / elapsed if elapsed > 0 else None)
        for action in Action
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "policy": config.to_dict(),
        "policy_sha256": _fingerprint(config.to_dict()),
        "observations_sha256": _fingerprint(
            [observation.to_dict() for observation in observations]
        ),
        "samples": len(observations),
        "started_at": observations[0].observed_at,
        "ended_at": observations[-1].observed_at,
        "elapsed_seconds": elapsed,
        "action_samples": {action.value: action_samples[action.value] for action in Action},
        "action_seconds": action_seconds,
        "action_percent": action_percent,
        "reason_samples": dict(sorted(reason_samples.items())),
        "action_transitions": action_transitions,
        "decision_transitions": decision_transitions,
        "transitions": transitions,
        "decisions": decisions,
    }
    if sensitivity is not None:
        result, action_margin = analyze_sensitivity_with_margin(
            config,
            observations,
            durations,
            sensitivity,
        )
        report["schema_version"] = 3
        report["sensitivity"] = result.to_dict(
            sensitivity,
            action_change_margin=action_margin,
        )
    return report


def compare_policies(
    baseline_config: PolicyConfig,
    candidate_config: PolicyConfig,
    observations: Sequence[Observation],
    *,
    sensitivity: Optional[SensitivityBounds] = None,
) -> dict[str, Any]:
    """Replay two policies on the same observations and quantify differences."""

    baseline = simulate_policy(baseline_config, observations, sensitivity=sensitivity)
    candidate = simulate_policy(candidate_config, observations, sensitivity=sensitivity)
    disagreements: list[dict[str, Any]] = []
    action_disagreement_samples = 0
    action_disagreement_seconds = 0.0
    decision_disagreement_samples = 0

    for index, baseline_row in enumerate(baseline["decisions"]):
        candidate_row = candidate["decisions"][index]
        baseline_decision = baseline_row["decision"]
        candidate_decision = candidate_row["decision"]
        action_changed = baseline_decision["action"] != candidate_decision["action"]
        decision_changed = baseline_decision != candidate_decision
        if action_changed:
            action_disagreement_samples += 1
            action_disagreement_seconds += float(baseline_row["duration_seconds"])
        if decision_changed:
            decision_disagreement_samples += 1
            disagreements.append(
                {
                    "sample": baseline_row["sample"],
                    "observed_at": baseline_row["observed_at"],
                    "duration_seconds": baseline_row["duration_seconds"],
                    "observation": baseline_row["observation"],
                    "baseline": baseline_decision,
                    "candidate": candidate_decision,
                }
            )

    elapsed = float(baseline["elapsed_seconds"])
    return {
        "schema_version": 3 if sensitivity is not None else 1,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "action_seconds": {
                action.value: (
                    float(candidate["action_seconds"][action.value])
                    - float(baseline["action_seconds"][action.value])
                )
                for action in Action
            },
            "action_percent": {
                action.value: (
                    None
                    if baseline["action_percent"][action.value] is None
                    or candidate["action_percent"][action.value] is None
                    else (
                        float(candidate["action_percent"][action.value])
                        - float(baseline["action_percent"][action.value])
                    )
                )
                for action in Action
            },
            "action_transitions": (
                int(candidate["action_transitions"]) - int(baseline["action_transitions"])
            ),
            "decision_transitions": (
                int(candidate["decision_transitions"]) - int(baseline["decision_transitions"])
            ),
            "action_disagreement_samples": action_disagreement_samples,
            "action_disagreement_seconds": action_disagreement_seconds,
            "action_disagreement_percent": (
                100.0 * action_disagreement_seconds / elapsed if elapsed > 0 else None
            ),
            "decision_disagreement_samples": decision_disagreement_samples,
        },
        "disagreements": disagreements,
    }
