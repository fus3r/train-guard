"""Identity-aware process discovery for train-guard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import psutil

if __package__:
    from .model import ProcessIdentity
    from .state import JobSpec
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from model import ProcessIdentity
    from state import JobSpec


@dataclass(frozen=True)
class TargetSnapshot:
    processes: tuple[psutil.Process, ...]
    root_alive: bool


def process_identity(process: psutil.Process) -> ProcessIdentity:
    """Capture the stable identity used to distinguish PID reuse."""

    return ProcessIdentity(process.pid, float(process.create_time()))


def _same_process(
    process: psutil.Process,
    identity: ProcessIdentity,
) -> Optional[bool]:
    """Return same, different, or unverifiable for one recorded identity."""

    if process.pid != identity.pid:
        return False
    try:
        return abs(float(process.create_time()) - identity.create_time) < 0.001
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError):
        return None


def identity_is_alive(identity: ProcessIdentity) -> bool:
    """Treat an unverifiable identity as alive so callers fail closed."""

    try:
        process = psutil.Process(identity.pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True
    return _same_process(process, identity) is not False


def is_guard_process(process: psutil.Process) -> bool:
    """Recognize the current or another train-guard supervisor."""

    if process.pid == os.getpid():
        return True
    try:
        command = " ".join(process.cmdline()).lower()
    except (psutil.Error, OSError):
        return False
    return "__supervise" in command and (
        "trainguard" in command or "train-guard" in command
    )


class ProcessController:
    """Resolve only processes whose recorded identity still matches."""

    def __init__(
        self,
        *,
        excluded_identities: Iterable[ProcessIdentity] = (),
    ):
        self._excluded_identities = set(excluded_identities)

    @property
    def excluded_identities(self) -> tuple[ProcessIdentity, ...]:
        return tuple(sorted(self._excluded_identities))

    def _is_excluded(self, process: psutil.Process) -> bool:
        if is_guard_process(process):
            return True
        return any(
            _same_process(process, identity) is not False
            for identity in self._excluded_identities
        )

    def resolve(self, spec: JobSpec) -> TargetSnapshot:
        roots: list[psutil.Process] = []
        root_alive = False
        if spec.mode == "run":
            pid = spec.root_pid
            if pid is None:
                return TargetSnapshot((), False)
            try:
                process = psutil.Process(pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return TargetSnapshot((), False)
            except psutil.AccessDenied:
                return TargetSnapshot((), True)

            if spec.root is None:
                # Legacy PID-only state must be migrated by the supervisor
                # before it can authorize any process mutation.
                return TargetSnapshot((), True)

            matches = _same_process(process, spec.root)
            if matches is False:
                return TargetSnapshot((), False)
            if matches is None:
                return TargetSnapshot((), True)
            roots.append(process)
            root_alive = True
        else:
            pattern = spec.pattern or ""
            for process in psutil.process_iter(["pid", "cmdline"]):
                try:
                    command = " ".join(process.info["cmdline"] or [])
                except (psutil.Error, OSError):
                    continue
                if pattern in command and not self._is_excluded(process):
                    roots.append(process)
            root_alive = bool(roots)

        found: dict[ProcessIdentity, psutil.Process] = {}
        for root in roots:
            candidates = [root]
            try:
                candidates.extend(root.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            for process in candidates:
                if self._is_excluded(process):
                    continue
                try:
                    found[process_identity(process)] = process
                except (psutil.Error, OSError):
                    continue
        return TargetSnapshot(tuple(found.values()), root_alive)
