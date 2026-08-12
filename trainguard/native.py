"""Discovery and wire protocol for the optional native replay kernel.

The kernel is a standalone executable built from ``native/replay_kernel.cpp``.
Python remains the reference implementation: this module never validates
policies or traces itself, it only moves already-validated values across the
process boundary without changing a single bit. Floats therefore travel as
C99 hexadecimal literals (``float.hex`` out, ``%a`` back), and timestamps as
integer epoch microseconds, the same quantity ``timedelta.total_seconds``
divides by ``1e6``.

Nominal replay uses ``TGK 1``. Exact interval propagation uses the separate
``TGS 1`` protocol so nominal compatibility does not depend on bounded sweeps.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

if __package__:
    from .config import PolicyConfig
    from .model import Action, DecisionReason, Observation, PowerSource
    from .robustness import SensitivityBounds, SensitivityResult
else:  # Keep direct CLI execution from a checkout working.
    from config import PolicyConfig
    from model import Action, DecisionReason, Observation, PowerSource
    from robustness import SensitivityBounds, SensitivityResult

PROTOCOL_VERSION = 1
SENSITIVITY_PROTOCOL_VERSION = 1
KERNEL_BINARY = "train-guard-kernel"
# Worker threads split whole policies, never one policy's arithmetic, so
# the cap is a politeness limit on a tool meant to protect its host, not
# a correctness setting.
MAX_AUTO_THREADS = 16

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_BAND_CODES = {Action.FULL.value: 0, Action.GENTLE.value: 1}
_SOURCE_CODES = {PowerSource.AC: 0, PowerSource.BATTERY: 1, PowerSource.NO_BATTERY: 2}
_ACTION_CODES = {Action.FULL: 0, Action.GENTLE: 1, Action.STOP: 2}
_REASON_CODES = {reason: index for index, reason in enumerate(DecisionReason)}
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_UINT64 = 0xFFFFFFFFFFFFFFFF


class KernelError(RuntimeError):
    """The native kernel is unusable or broke its protocol contract."""


@dataclass(frozen=True)
class KernelRow:
    """Aggregates returned by the kernel for one candidate policy."""

    full_seconds: float
    gentle_seconds: float
    stop_seconds: float
    run_seconds: float
    hot_degc_seconds: float
    low_battery_run_seconds: float
    action_transitions: int
    decision_transitions: int
    checksum: str
    actions: Optional[str]


def find_kernel() -> Optional[Path]:
    """Locate the kernel binary, or return None to use the Python engine.

    ``TRAIN_GUARD_KERNEL`` names an explicit binary and must exist when set;
    a silent fallback would hide a broken build from its own differential
    tests. Otherwise the ``PATH`` lookup is best-effort.
    """

    configured = os.environ.get("TRAIN_GUARD_KERNEL")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise KernelError(f"TRAIN_GUARD_KERNEL does not exist: {path}")
        return path
    located = shutil.which(KERNEL_BINARY)
    return Path(located) if located else None


def resolve_thread_count(
    policy_count: int,
    *,
    emit_actions: bool = False,
    requested: Optional[int] = None,
) -> int:
    """Pick how many kernel worker threads a request should use.

    Each policy is evaluated start-to-finish by exactly one thread and
    rows are emitted in input order, so the thread count can never change
    a result bit. ``emit_actions`` responses stream one byte per
    observation per policy; the kernel handles them sequentially.
    """

    if requested is not None:
        if requested < 1:
            raise KernelError("kernel threads must be at least 1")
        if requested > 4096:
            raise KernelError("kernel threads must be at most 4096")
        count = requested
    else:
        count = min(MAX_AUTO_THREADS, os.cpu_count() or 1)
    if emit_actions:
        return 1
    return max(1, min(count, policy_count))


def epoch_microseconds(timestamp: datetime) -> int:
    """Convert an aware datetime to exact integer microseconds since 1970."""

    delta = timestamp - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def fnv1a_decisions(decisions: Iterable[tuple[Action, DecisionReason]]) -> str:
    """Fingerprint a decision sequence exactly as the kernel does."""

    value = _FNV_OFFSET
    for action, reason in decisions:
        value = ((value ^ _ACTION_CODES[action]) * _FNV_PRIME) & _UINT64
        value = ((value ^ _REASON_CODES[reason]) * _FNV_PRIME) & _UINT64
    return format(value, "016x")


def encode_request(
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    timestamps: Sequence[datetime],
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
    emit_actions: bool,
) -> str:
    lines = [
        "TGK {} {} {} {} {} {}".format(
            PROTOCOL_VERSION,
            len(policies),
            len(observations),
            int(emit_actions),
            float(hot_ref_c).hex(),
            float(low_battery_ref_pct).hex(),
        )
    ]
    lines.extend(_wire_rows(policies, observations, timestamps))
    lines.append("")
    return "\n".join(lines)


def _wire_rows(
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    timestamps: Sequence[datetime],
) -> list[str]:
    """Encode the validated policy and observation rows shared by both protocols."""

    lines: list[str] = []
    for policy in policies:
        lines.append(
            "P {} {} {} {} {} {} {} {} {}".format(
                int(policy.run_on_battery),
                policy.battery_floor_pct.hex(),
                _BAND_CODES[policy.battery_band],
                _BAND_CODES[policy.ac_band],
                policy.temp_gentle_c.hex(),
                policy.temp_pause_c.hex(),
                policy.temp_resume_c.hex(),
                policy.charge_cool_until_pct.hex(),
                policy.temp_charge_gentle_c.hex(),
            )
        )
    for observation, timestamp in zip(observations, timestamps):
        percent = observation.percent
        temperature = observation.temperature_c
        lines.append(
            "O {} {} {} {} {}".format(
                epoch_microseconds(timestamp),
                _SOURCE_CODES[observation.source],
                "-" if percent is None else float(percent).hex(),
                "-" if temperature is None else float(temperature).hex(),
                int(bool(observation.charging)),
            )
        )
    return lines


def encode_sensitivity_request(
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    timestamps: Sequence[datetime],
    bounds: SensitivityBounds,
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
) -> str:
    """Encode an exact bounded-sensitivity request for every policy."""

    lines = [
        "TGS {} {} {} {} {} {} {}".format(
            SENSITIVITY_PROTOCOL_VERSION,
            len(policies),
            len(observations),
            float(hot_ref_c).hex(),
            float(low_battery_ref_pct).hex(),
            bounds.temperature_c.hex(),
            bounds.charge_pct.hex(),
        )
    ]
    lines.extend(_wire_rows(policies, observations, timestamps))
    lines.append("")
    return "\n".join(lines)


def _parse_row(line: str, observation_count: int, actions: Optional[str]) -> KernelRow:
    fields = line.split()
    if len(fields) != 10 or fields[0] != "R":
        raise KernelError(f"kernel returned a malformed result row: {line!r}")
    try:
        floats = [float.fromhex(field) for field in fields[1:7]]
        action_transitions = int(fields[7])
        decision_transitions = int(fields[8])
    except ValueError as exc:
        raise KernelError(f"kernel returned a malformed result row: {line!r}") from exc
    checksum = fields[9]
    if len(checksum) != 16 or any(char not in "0123456789abcdef" for char in checksum):
        raise KernelError(f"kernel returned a malformed checksum: {line!r}")
    if actions is not None and (
        len(actions) != observation_count or any(char not in "012" for char in actions)
    ):
        raise KernelError("kernel returned a malformed action sequence")
    return KernelRow(
        full_seconds=floats[0],
        gentle_seconds=floats[1],
        stop_seconds=floats[2],
        run_seconds=floats[3],
        hot_degc_seconds=floats[4],
        low_battery_run_seconds=floats[5],
        action_transitions=action_transitions,
        decision_transitions=decision_transitions,
        checksum=checksum,
        actions=actions,
    )


def _parse_sensitivity_row(line: str, observation_count: int) -> SensitivityResult:
    fields = line.split()
    if len(fields) != 11 or fields[0] != "S":
        raise KernelError(f"kernel returned a malformed sensitivity row: {line!r}")
    try:
        values = [float.fromhex(field) for field in fields[1:9]]
        stable_samples = int(fields[9])
        ambiguous_samples = int(fields[10])
    except ValueError as exc:
        raise KernelError(f"kernel returned a malformed sensitivity row: {line!r}") from exc
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or values[0] > values[1]
        or values[2] > values[3]
        or values[4] > values[5]
        or stable_samples < 0
        or ambiguous_samples < 0
        or stable_samples + ambiguous_samples != observation_count
    ):
        raise KernelError(f"kernel returned invalid sensitivity bounds: {line!r}")
    return SensitivityResult(
        minimum_run_seconds=values[0],
        maximum_run_seconds=values[1],
        minimum_hot_run_degc_seconds=values[2],
        maximum_hot_run_degc_seconds=values[3],
        minimum_low_battery_run_seconds=values[4],
        maximum_low_battery_run_seconds=values[5],
        action_stable_seconds=values[6],
        action_ambiguous_seconds=values[7],
        action_stable_samples=stable_samples,
        action_ambiguous_samples=ambiguous_samples,
    )


def run_kernel(
    kernel: Path,
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    timestamps: Sequence[datetime],
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
    emit_actions: bool = False,
    threads: Optional[int] = None,
) -> list[KernelRow]:
    """Evaluate every policy against the shared trace in one kernel run."""

    resolved_threads = resolve_thread_count(
        len(policies),
        emit_actions=emit_actions,
        requested=threads,
    )
    request = encode_request(
        policies,
        observations,
        timestamps,
        hot_ref_c=hot_ref_c,
        low_battery_ref_pct=low_battery_ref_pct,
        emit_actions=emit_actions,
    )
    try:
        completed = subprocess.run(
            [str(kernel), str(resolved_threads)],
            input=request,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise KernelError(f"cannot execute kernel {kernel}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise KernelError(f"kernel {kernel} failed: {detail}")

    lines = completed.stdout.splitlines()
    if not lines or lines[0].split() != ["TGK", str(PROTOCOL_VERSION), "OK"]:
        raise KernelError(f"kernel {kernel} sent an unexpected handshake")
    expected = len(policies) * (2 if emit_actions else 1) + 2
    if len(lines) != expected or lines[-1] != "END":
        raise KernelError(f"kernel {kernel} returned a truncated response")

    rows: list[KernelRow] = []
    cursor = 1
    for _ in policies:
        actions: Optional[str] = None
        if emit_actions:
            action_line = lines[cursor + 1]
            if not action_line.startswith("A "):
                raise KernelError(f"kernel returned a malformed action row: {action_line!r}")
            actions = action_line[2:]
        rows.append(_parse_row(lines[cursor], len(observations), actions))
        cursor += 2 if emit_actions else 1
    return rows


def run_sensitivity_kernel(
    kernel: Path,
    policies: Sequence[PolicyConfig],
    observations: Sequence[Observation],
    timestamps: Sequence[datetime],
    bounds: SensitivityBounds,
    *,
    hot_ref_c: float,
    low_battery_ref_pct: float,
    threads: Optional[int] = None,
) -> list[SensitivityResult]:
    """Evaluate exact marginal uncertainty envelopes in one native run."""

    resolved_threads = resolve_thread_count(len(policies), requested=threads)
    request = encode_sensitivity_request(
        policies,
        observations,
        timestamps,
        bounds,
        hot_ref_c=hot_ref_c,
        low_battery_ref_pct=low_battery_ref_pct,
    )
    try:
        completed = subprocess.run(
            [str(kernel), str(resolved_threads)],
            input=request,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise KernelError(f"cannot execute kernel {kernel}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise KernelError(f"kernel {kernel} failed: {detail}")

    lines = completed.stdout.splitlines()
    expected_handshake = ["TGS", str(SENSITIVITY_PROTOCOL_VERSION), "OK"]
    if not lines or lines[0].split() != expected_handshake:
        raise KernelError(f"kernel {kernel} sent an unexpected sensitivity handshake")
    if len(lines) != len(policies) + 2 or lines[-1] != "END":
        raise KernelError(f"kernel {kernel} returned a truncated sensitivity response")
    return [_parse_sensitivity_row(line, len(observations)) for line in lines[1:-1]]
