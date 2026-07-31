"""Long-running supervisor lifecycle."""

from __future__ import annotations

import signal
import threading
from dataclasses import asdict, replace
from typing import Any, Optional

import psutil

if __package__:
    from .config import ConfigError, ConfigWatcher, load_policy
    from .journal import EventJournal
    from .model import (
        Action,
        Observation,
        PolicyDecision,
        ProcessIdentity,
        utc_now,
    )
    from .policy import PolicyEngine
    from .processes import ApplyReport, ProcessController, process_identity
    from .sensors import SensorReader
    from .state import AppPaths, JobSpec, JobStore, StateError
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from config import ConfigError, ConfigWatcher, load_policy
    from journal import EventJournal
    from model import Action, Observation, PolicyDecision, ProcessIdentity, utc_now
    from policy import PolicyEngine
    from processes import ApplyReport, ProcessController, process_identity
    from sensors import SensorReader
    from state import AppPaths, JobSpec, JobStore, StateError

def _runtime_payload(
    state: str,
    cooling: bool,
    pids: list[int],
    controller: ProcessController,
    observation: Any = None,
    decision: Any = None,
    config_error: Optional[str] = None,
    process_report: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "state": state,
        "cooling": cooling,
        "owned_suspensions": [
            identity.to_dict() for identity in controller.owned_suspensions
        ],
        "tuned_processes": list(controller.tuned_processes),
        "pids": sorted({int(pid) for pid in pids}),
    }
    if observation is not None:
        payload["observation"] = {
            "source": observation.source.value,
            "percent": observation.percent,
            "temperature_c": observation.temperature_c,
            "charging": observation.charging,
            "observed_at": observation.observed_at,
            "warnings": list(observation.warnings),
        }
    if decision is not None:
        payload["decision"] = {
            "action": decision.action.value,
            "reason": decision.reason.value,
            "cooling": decision.cooling,
        }
    if config_error:
        payload["config_error"] = config_error
    if process_report is not None:
        payload["process_report"] = process_report
    if error:
        payload["error"] = error
    return payload


def _migrate_legacy_identity(store: JobStore, spec: JobSpec) -> JobSpec:
    """Upgrade safe PID-only metadata without adopting a reused PID."""

    if spec.mode != "run" or spec.root is not None or spec.legacy_pid is None:
        return spec
    try:
        identity = process_identity(psutil.Process(spec.legacy_pid))
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        raise StateError(
            f"legacy job process {spec.legacy_pid} no longer exists; "
            "its metadata was preserved"
        ) from exc
    except psutil.AccessDenied as exc:
        raise StateError(
            f"cannot verify legacy job process {spec.legacy_pid}; "
            "its metadata was preserved"
        ) from exc

    try:
        recorded_at = store.spec_path(spec.name).stat().st_mtime
    except OSError as exc:
        raise StateError(
            f"cannot date legacy metadata for guard '{spec.name}'"
        ) from exc
    if identity.create_time > recorded_at:
        message = (
            f"STATE_MIGRATION_REFUSED pid={identity.pid}: current process "
            "started after the legacy metadata was written"
        )
        EventJournal(store.paths, spec.name).emit(
            "state_migration_refused",
            message,
            pid=identity.pid,
        )
        raise StateError(
            f"legacy PID {identity.pid} now belongs to a newer process; "
            "the metadata was preserved"
        )

    migrated = replace(spec, root=identity, legacy_pid=None)
    store.write_spec(migrated)
    EventJournal(store.paths, spec.name).emit(
        "state_migrated",
        f"STATE_MIGRATED pid={identity.pid} create_time={identity.create_time:g}",
        root=identity.to_dict(),
    )
    return migrated


def _report_payload(report: ApplyReport) -> dict[str, Any]:
    return asdict(report)


class Supervisor:
    """Own one guard's policy loop and release its process changes on exit."""

    def __init__(
        self,
        paths: AppPaths,
        spec: JobSpec,
        *,
        excluded_identity: Optional[ProcessIdentity] = None,
        sensors: Optional[SensorReader] = None,
        controller: Optional[ProcessController] = None,
    ):
        if spec.mode == "attach" and excluded_identity is None and controller is None:
            raise StateError(
                "cannot identify the CLI that launched this match supervisor"
            )

        self.paths = paths
        self.store = JobStore(paths)
        self.spec = spec
        self.journal = EventJournal(paths, spec.name)
        self.sensors = sensors or SensorReader()
        self.controller = controller or ProcessController(
            excluded_identities=(
                () if excluded_identity is None else (excluded_identity,)
            )
        )
        runtime = self.store.read_runtime(spec.name)
        if runtime is not None:
            self.controller.adopt_owned(
                ProcessIdentity.from_dict(value)
                for value in runtime["owned_suspensions"]
            )
            self.controller.adopt_tuned(runtime["tuned_processes"])
        self.policy = PolicyEngine(
            cooling=(runtime["cooling"] if runtime is not None else False)
        )
        self._shutdown = threading.Event()
        self._shutdown_signal: Optional[str] = None
        self._last_pids: list[int] = list(runtime["pids"]) if runtime else []
        self._config_error: Optional[str] = None
        self._last_transition: Optional[tuple[str, str, str]] = None
        self._last_waiting = False
        self._last_observation: Optional[Observation] = None
        self._last_decision: Optional[PolicyDecision] = None

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        try:
            self._shutdown_signal = signal.Signals(signum).name
        except ValueError:
            self._shutdown_signal = str(signum)
        self._shutdown.set()

    def _install_signal_handlers(self) -> None:
        for signal_name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, signal_name):
                signal.signal(getattr(signal, signal_name), self._handle_signal)

    def _write_runtime(
        self,
        state: str,
        *,
        observation: Any = None,
        decision: Any = None,
        report: Optional[ApplyReport] = None,
        error: Optional[str] = None,
        pids: Optional[list[int]] = None,
    ) -> None:
        if pids is not None:
            self._last_pids = list(pids)
        self.store.write_runtime(
            self.spec.name,
            _runtime_payload(
                state,
                self.policy.cooling,
                self._last_pids,
                self.controller,
                observation=observation,
                decision=decision,
                config_error=self._config_error,
                process_report=(None if report is None else _report_payload(report)),
                error=error,
            ),
        )

    def _retain_incomplete(
        self,
        state: str,
        report: ApplyReport,
        error: str,
    ) -> int:
        self._write_runtime(state, report=report, error=error)
        self.store.clear_stop(self.spec.name)
        self.store.remove_guard_state(self.spec.name)
        return 1

    def _terminal_trace(self) -> dict[str, Any]:
        """Close the final stable decision interval in a completed journal."""

        if self._last_observation is None:
            return {}
        observation = self._last_observation.to_dict()
        observation["observed_at"] = utc_now()
        details: dict[str, Any] = {
            "observation": observation,
            "trace_terminal": True,
        }
        if self._last_decision is not None:
            details["decision"] = self._last_decision.to_dict()
        return details

    def _finish_stop(self, how: str) -> int:
        snapshot = self.controller.resolve(self.spec)
        report = self.controller.release_owned()
        if how == "kill":
            for process in snapshot.processes:
                try:
                    process.terminate()
                except (psutil.Error, OSError):
                    pass
        event = "stop_incomplete" if report.access_denied else "stopped"
        message = (
            "STOP incomplete: owned process state preserved for recovery"
            if report.access_denied
            else (
                "STOP --kill: terminated job"
                if how == "kill"
                else "STOP: released owned process changes, detaching"
            )
        )
        self.journal.emit(
            event,
            message,
            mode=how,
            process_report=_report_payload(report),
            **self._terminal_trace(),
        )
        if report.access_denied:
            return self._retain_incomplete(
                "stop_incomplete",
                report,
                "permission denied while releasing owned process state",
            )
        self.store.remove_active_state(self.spec.name)
        return 0

    def run(self) -> int:
        self.paths.ensure()
        supervisor_identity = process_identity(psutil.Process())
        self.store.write_guard(self.spec.name, supervisor_identity)
        self._install_signal_handlers()
        self.spec = _migrate_legacy_identity(self.store, self.spec)
        config = load_policy(self.paths.config)
        watcher = ConfigWatcher(self.paths.config, config)
        detail = (
            f"pattern={self.spec.pattern}"
            if self.spec.mode == "attach"
            else f"jobpid={self.spec.root_pid}"
        )
        self.journal.emit(
            "started",
            f"START mode={self.spec.mode} {detail}",
            mode=self.spec.mode,
            root=(None if self.spec.root is None else self.spec.root.to_dict()),
            pattern=self.spec.pattern,
        )

        self._write_runtime("starting")
        self.store.write_ready(self.spec.name, supervisor_identity)
        warned: tuple[str, ...] = ()

        try:
            while not self._shutdown.is_set():
                stop_request = self.store.read_stop(self.spec.name)
                if stop_request is not None:
                    return self._finish_stop(stop_request)

                snapshot = self.controller.resolve(self.spec)
                if self.spec.mode == "run" and not snapshot.root_alive:
                    report = self.controller.release_owned()
                    self.journal.emit(
                        (
                            "job_exit_cleanup_incomplete"
                            if report.access_denied
                            else "job_exited"
                        ),
                        (
                            "job exited; owned process state preserved for recovery"
                            if report.access_denied
                            else "job exited; guard done"
                        ),
                        process_report=_report_payload(report),
                        **self._terminal_trace(),
                    )
                    if report.access_denied:
                        return self._retain_incomplete(
                            "job_exit_cleanup_incomplete",
                            report,
                            "permission denied while releasing owned process state",
                        )
                    # A short-lived job can finish before the parent consumes
                    # readiness. Keeping only that handshake lets the parent
                    # report the launch truthfully on hosts without zombies.
                    self.store.remove_active_state(
                        self.spec.name,
                        keep_ready=True,
                    )
                    return 0

                config, config_changed, self._config_error = watcher.poll()
                if config_changed:
                    if self._config_error:
                        self.journal.emit(
                            "config_rejected",
                            f"CONFIG rejected: {self._config_error}; "
                            "keeping last valid policy",
                            error=self._config_error,
                        )
                    else:
                        self.journal.emit(
                            "config_loaded",
                            "CONFIG loaded",
                            policy=config.to_dict(),
                        )

                processes = snapshot.processes
                if self.spec.mode == "attach" and not processes:
                    report = self.controller.apply(Action.FULL, ())
                    if not self._last_waiting:
                        self.journal.emit(
                            "waiting",
                            "no process matches pattern yet",
                            pattern=self.spec.pattern,
                            process_report=_report_payload(report),
                        )
                    self._last_waiting = True
                    self._last_transition = None
                    self._write_runtime("waiting", report=report, pids=[])
                    self._shutdown.wait(config.poll)
                    continue

                self._last_waiting = False
                observation = self.sensors.sample()
                if observation.warnings != warned:
                    for warning in observation.warnings:
                        self.journal.emit(
                            "sensor_warning",
                            f"SENSOR {warning}",
                        )
                    warned = observation.warnings
                decision = self.policy.decide(config, observation)
                self._last_observation = observation
                self._last_decision = decision
                report = self.controller.apply(decision.action, processes)
                pids = sorted(process.pid for process in processes)
                self._write_runtime(
                    decision.action.value,
                    observation=observation,
                    decision=decision,
                    report=report,
                    pids=pids,
                )
                transition = (
                    decision.action.value,
                    decision.reason.value,
                    observation.signature(),
                )
                if transition != self._last_transition:
                    self.journal.emit(
                        "decision",
                        f"-> {decision.action.value} "
                        f"({decision.reason.value}; {observation.signature()})",
                        observation=observation.to_dict(),
                        decision=decision.to_dict(),
                        process_report=_report_payload(report),
                        pids=pids,
                    )
                    self._last_transition = transition
                self._shutdown.wait(config.poll)

            report = self.controller.release_owned()
            self.journal.emit(
                "shutdown_incomplete" if report.access_denied else "shutdown",
                (
                    f"{self._shutdown_signal or 'signal'} cleanup incomplete"
                    if report.access_denied
                    else f"{self._shutdown_signal or 'signal'}: released owned "
                    "process changes"
                ),
                signal=self._shutdown_signal,
                process_report=_report_payload(report),
                **self._terminal_trace(),
            )
            if report.access_denied:
                return self._retain_incomplete(
                    "shutdown_incomplete",
                    report,
                    "permission denied while releasing owned process state",
                )
            self.store.remove_active_state(self.spec.name)
            return 0
        except Exception as exc:
            report = self.controller.release_owned()
            state = "error_cleanup_incomplete" if report.access_denied else "error"
            error = f"{type(exc).__name__}: {exc}"
            try:
                self._write_runtime(
                    state,
                    report=report,
                    error=error,
                )
                self.journal.emit(
                    (
                        "crash_cleanup_incomplete"
                        if report.access_denied
                        else "crashed"
                    ),
                    (
                        f"CRASH: {error}; cleanup incomplete"
                        if report.access_denied
                        else f"CRASH: {error}; owned process changes released"
                    ),
                    error=error,
                    process_report=_report_payload(report),
                    **self._terminal_trace(),
                )
            finally:
                self.store.remove_guard_state(self.spec.name)
            return 1


def run_supervisor(
    paths: AppPaths,
    name: str,
    excluded_identity: Optional[ProcessIdentity] = None,
) -> int:
    store = JobStore(paths)
    try:
        spec = store.read_spec(name)
        supervisor = Supervisor(
            paths,
            spec,
            excluded_identity=excluded_identity,
        )
        return supervisor.run()
    except (ConfigError, OSError, psutil.Error, StateError) as exc:
        store.remove_guard_state(name)
        try:
            EventJournal(paths, name).emit(
                "startup_failed",
                f"START failed: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
        except (OSError, StateError):
            pass
        return 2
