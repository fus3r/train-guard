"""Identity-aware discovery and ownership-aware process control."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import psutil

if __package__:
    from .model import Action, ProcessIdentity
    from .state import JobSpec
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from model import Action, ProcessIdentity
    from state import JobSpec


@dataclass(frozen=True)
class TargetSnapshot:
    processes: tuple[psutil.Process, ...]
    root_alive: bool


@dataclass(frozen=True)
class ApplyReport:
    targeted: int
    suspended: int = 0
    resumed: int = 0
    access_denied: int = 0
    gone: int = 0
    tuned: int = 0
    restored: int = 0


@dataclass
class _BandState:
    affinity: Optional[list[int]] = None
    ionice_class: Optional[int] = None
    ionice_value: Optional[int] = None
    priority: Optional[int] = None


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


def _identity_matches(identity: ProcessIdentity) -> Optional[bool]:
    try:
        process = psutil.Process(identity.pid)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, OSError):
        return None
    return _same_process(process, identity)


def identity_is_alive(identity: ProcessIdentity) -> bool:
    """Treat an unverifiable identity as alive so callers fail closed."""

    return _identity_matches(identity) is not False


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
    """Resolve stable identities and undo only changes made by train-guard."""

    def __init__(
        self,
        system: Optional[str] = None,
        *,
        excluded_identities: Iterable[ProcessIdentity] = (),
    ):
        self.system = system or platform.system()
        self._excluded_identities = set(excluded_identities)
        self._owned_suspensions: set[ProcessIdentity] = set()
        self._band_state: dict[ProcessIdentity, _BandState] = {}

    @property
    def excluded_identities(self) -> tuple[ProcessIdentity, ...]:
        return tuple(sorted(self._excluded_identities))

    @property
    def owned_suspensions(self) -> tuple[ProcessIdentity, ...]:
        return tuple(sorted(self._owned_suspensions))

    @property
    def tuned_processes(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for identity, state in sorted(self._band_state.items()):
            record: dict[str, Any] = {
                **identity.to_dict(),
                "system": self.system,
            }
            if state.affinity is not None:
                record["affinity"] = list(state.affinity)
            if state.ionice_class is not None:
                record["ionice_class"] = state.ionice_class
                record["ionice_value"] = state.ionice_value
            if state.priority is not None:
                record["priority"] = state.priority
            records.append(record)
        return tuple(records)

    def adopt_owned(self, identities: Iterable[ProcessIdentity]) -> None:
        self._owned_suspensions.update(identities)

    def adopt_tuned(self, records: Iterable[dict[str, Any]]) -> None:
        """Adopt scheduling captures from an already validated runtime file."""

        for record in records:
            if record["system"] != self.system:
                continue
            identity = ProcessIdentity.from_dict(record)
            affinity = record.get("affinity")
            self._band_state[identity] = _BandState(
                affinity=list(affinity) if affinity is not None else None,
                ionice_class=record.get("ionice_class"),
                ionice_value=record.get("ionice_value"),
                priority=record.get("priority"),
            )

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

    def _capture_band(
        self,
        process: psutil.Process,
        identity: ProcessIdentity,
    ) -> _BandState:
        existing = self._band_state.get(identity)
        if existing is not None:
            return existing

        state = _BandState()
        if self.system == "Linux":
            cpu_affinity = getattr(process, "cpu_affinity", None)
            try:
                if cpu_affinity is not None:
                    state.affinity = list(cpu_affinity())
            except (psutil.Error, OSError, AttributeError):
                pass
            ionice = getattr(process, "ionice", None)
            try:
                if ionice is not None:
                    current = ionice()
                    state.ionice_class = int(current.ioclass)
                    state.ionice_value = int(current.value)
            except (psutil.Error, OSError, AttributeError):
                pass
        elif self.system == "Windows":
            try:
                state.priority = int(process.nice())
            except (psutil.Error, OSError, AttributeError):
                pass
        self._band_state[identity] = state
        return state

    def _set_gentle(
        self,
        process: psutil.Process,
        identity: ProcessIdentity,
    ) -> bool:
        if self.system not in {"Darwin", "Linux", "Windows"}:
            return False

        newly_owned = identity not in self._band_state
        state = self._capture_band(process, identity)
        changed = False
        try:
            if self.system == "Darwin":
                result = subprocess.run(
                    ["taskpolicy", "-b", "-p", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                if result.returncode:
                    raise OSError(
                        f"taskpolicy could not background process {process.pid}"
                    )
                changed = True
            elif self.system == "Linux":
                cpu_affinity = getattr(process, "cpu_affinity", None)
                if (
                    state.affinity
                    and len(state.affinity) > 1
                    and cpu_affinity is not None
                ):
                    cpu_affinity(state.affinity[::2])
                    changed = True
                ionice = getattr(process, "ionice", None)
                try:
                    if state.ionice_class is not None and ionice is not None:
                        ionice(getattr(psutil, "IOPRIO_CLASS_IDLE", 3))
                        changed = True
                except (psutil.Error, OSError, AttributeError):
                    pass
            elif state.priority is not None:
                process.nice(getattr(psutil, "IDLE_PRIORITY_CLASS", 64))
                changed = True
        except (
            psutil.Error,
            OSError,
            AttributeError,
            subprocess.SubprocessError,
        ):
            if newly_owned and not changed:
                self._band_state.pop(identity, None)
            raise

        if newly_owned and not changed:
            self._band_state.pop(identity, None)
        return newly_owned and changed

    def _restore_band(
        self,
        process: psutil.Process,
        identity: ProcessIdentity,
    ) -> bool:
        state = self._band_state.get(identity)
        if state is None:
            return False

        changed = False
        if self.system == "Darwin":
            result = subprocess.run(
                ["taskpolicy", "-B", "-p", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            if result.returncode:
                raise OSError(f"taskpolicy could not restore process {process.pid}")
            changed = True
        elif self.system == "Linux":
            cpu_affinity = getattr(process, "cpu_affinity", None)
            if state.affinity and cpu_affinity is not None:
                cpu_affinity(state.affinity)
                changed = True
            if state.ionice_class is not None:
                ionice = getattr(process, "ionice", None)
                if ionice is None or state.ionice_value is None:
                    raise OSError(
                        f"cannot restore I/O priority for process {process.pid}"
                    )
                ionice(state.ionice_class, state.ionice_value)
                changed = True
        elif self.system == "Windows" and state.priority is not None:
            process.nice(state.priority)
            changed = True

        self._band_state.pop(identity, None)
        return changed

    def apply(
        self,
        action: Action,
        processes: Iterable[psutil.Process],
    ) -> ApplyReport:
        process_list = list(processes)
        suspended = resumed = denied = gone = tuned = restored = 0
        resolved: list[tuple[psutil.Process, ProcessIdentity]] = []

        for process in process_list:
            try:
                resolved.append((process, process_identity(process)))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                gone += 1
            except (psutil.AccessDenied, OSError):
                denied += 1

        current = {identity for _, identity in resolved}
        departed = (self._owned_suspensions | set(self._band_state)) - current
        released = self._release_identities(departed)
        resumed += released.resumed
        restored += released.restored
        denied += released.access_denied
        gone += released.gone

        ordered: Iterable[tuple[psutil.Process, ProcessIdentity]]
        ordered = reversed(resolved) if action is Action.STOP else iter(resolved)
        for process, identity in ordered:
            try:
                if action is Action.STOP:
                    if process.status() == psutil.STATUS_STOPPED:
                        continue
                    process.suspend()
                    self._owned_suspensions.add(identity)
                    suspended += 1
                    continue

                if identity in self._owned_suspensions:
                    if process.status() == psutil.STATUS_STOPPED:
                        process.resume()
                        resumed += 1
                    self._owned_suspensions.discard(identity)

                if process.status() == psutil.STATUS_STOPPED:
                    continue
                if action is Action.GENTLE:
                    tuned += int(self._set_gentle(process, identity))
                else:
                    restored += int(self._restore_band(process, identity))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                gone += 1
            except (
                psutil.AccessDenied,
                OSError,
                AttributeError,
                subprocess.SubprocessError,
            ):
                denied += 1

        self._drop_dead_state()
        return ApplyReport(
            targeted=len(process_list),
            suspended=suspended,
            resumed=resumed,
            access_denied=denied,
            gone=gone,
            tuned=tuned,
            restored=restored,
        )

    def _drop_dead_state(self) -> None:
        for identity in tuple(self._owned_suspensions):
            if _identity_matches(identity) is False:
                self._owned_suspensions.discard(identity)
        for identity in tuple(self._band_state):
            if _identity_matches(identity) is False:
                self._band_state.pop(identity, None)

    def release_owned(self) -> ApplyReport:
        """Undo every live process change owned by this controller."""

        identities = self._owned_suspensions | set(self._band_state)
        return self._release_identities(identities)

    def _release_identities(
        self,
        identities: Iterable[ProcessIdentity],
    ) -> ApplyReport:
        resumed = restored = denied = gone = 0
        identity_list = sorted(set(identities))
        for identity in identity_list:
            try:
                process = psutil.Process(identity.pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                self._owned_suspensions.discard(identity)
                self._band_state.pop(identity, None)
                gone += 1
                continue
            except (psutil.AccessDenied, OSError):
                denied += 1
                continue

            matches = _same_process(process, identity)
            if matches is False:
                self._owned_suspensions.discard(identity)
                self._band_state.pop(identity, None)
                gone += 1
                continue
            if matches is None:
                denied += 1
                continue

            try:
                if identity in self._owned_suspensions:
                    if process.status() == psutil.STATUS_STOPPED:
                        process.resume()
                        resumed += 1
                    self._owned_suspensions.discard(identity)
                if identity in self._band_state:
                    restored += int(self._restore_band(process, identity))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                self._owned_suspensions.discard(identity)
                self._band_state.pop(identity, None)
                gone += 1
            except (
                psutil.AccessDenied,
                OSError,
                AttributeError,
                subprocess.SubprocessError,
            ):
                denied += 1

        return ApplyReport(
            targeted=len(identity_list),
            resumed=resumed,
            access_denied=denied,
            gone=gone,
            restored=restored,
        )
