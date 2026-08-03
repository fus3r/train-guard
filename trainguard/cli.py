#!/usr/bin/env python3
"""Command-line implementation for train-guard.

The supervisor keeps a long running job paused when the battery policy says it
should not run. It resumes the same process tree, so the job is not restarted.
Process control uses psutil; battery readings come from the host OS when they
are available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.stderr.write("train-guard requires psutil:  pip install psutil\n")
    raise

if __package__:
    from .config import ConfigError, PolicyConfig, load_policy
    from .journal import EventJournal
    from .model import Action, PowerSource, ProcessIdentity
    from .platforms import (
        PlatformError,
        agent_installed,
        install_agent,
        uninstall_agent,
    )
    from .processes import (
        ProcessController,
        identity_is_alive,
        is_guard_process,
        process_identity,
    )
    from .sensors import SensorReader
    from .simulation import TraceError, compare_policies, load_trace, simulate_policy
    from .state import (
        AppPaths,
        JobSpec,
        JobStore,
        PersistenceSpec,
        StateError,
        atomic_json_write,
        validate_job_name,
    )
    from .supervisor import (
        _migrate_legacy_identity as _migrate_legacy_identity_impl,
        _runtime_payload,
        run_supervisor,
    )
    from .sweep import (
        DEFAULT_HOT_REFERENCE_C,
        DEFAULT_LOW_BATTERY_REFERENCE_PCT,
        SweepError,
        parse_grid,
        run_sweep,
    )
else:  # Keep direct execution from a checkout working.
    from config import ConfigError, PolicyConfig, load_policy
    from journal import EventJournal
    from model import Action, PowerSource, ProcessIdentity
    from platforms import PlatformError, agent_installed, install_agent, uninstall_agent
    from processes import (
        ProcessController,
        identity_is_alive,
        is_guard_process,
        process_identity,
    )
    from sensors import SensorReader
    from simulation import TraceError, compare_policies, load_trace, simulate_policy
    from state import (
        AppPaths,
        JobSpec,
        JobStore,
        PersistenceSpec,
        StateError,
        atomic_json_write,
        validate_job_name,
    )
    from supervisor import (
        _migrate_legacy_identity as _migrate_legacy_identity_impl,
        _runtime_payload,
        run_supervisor,
    )
    from sweep import (
        DEFAULT_HOT_REFERENCE_C,
        DEFAULT_LOW_BATTERY_REFERENCE_PCT,
        SweepError,
        parse_grid,
        run_sweep,
    )

__version__ = "0.2.0"
_SUPERVISOR_START_TIMEOUT_SECONDS = 5.0
SYSTEM = platform.system()  # 'Darwin' | 'Linux' | 'Windows'
HOME = Path.home()
_INITIAL_PATHS = AppPaths.from_environment()
TG_HOME = _INITIAL_PATHS.home
RUNDIR, LOGDIR, PERSIST = (
    _INITIAL_PATHS.run,
    _INITIAL_PATHS.logs,
    _INITIAL_PATHS.persist,
)
CONFIGF = _INITIAL_PATHS.config


def _sync_path_aliases(paths: AppPaths) -> None:
    """Keep the v0.2 module constants available to callers and tests."""

    global TG_HOME, RUNDIR, LOGDIR, PERSIST, CONFIGF
    TG_HOME = paths.home
    RUNDIR = paths.run
    LOGDIR = paths.logs
    PERSIST = paths.persist
    CONFIGF = paths.config


def _paths() -> AppPaths:
    paths = AppPaths.from_environment()
    # A detached child can use another cwd. Exporting the absolute value keeps
    # parent and supervisor on the same state directory.
    os.environ["TRAIN_GUARD_HOME"] = str(paths.home)
    paths.ensure()
    _sync_path_aliases(paths)
    return paths


def _ensure_dirs() -> None:
    _paths()


def _json_dump(value) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


def load_config() -> PolicyConfig:
    return load_policy(_paths().config)


def _migrate_legacy_identity(store, spec):
    """Compatibility wrapper for callers of the pre-Supervisor helper."""

    return _migrate_legacy_identity_impl(store, spec)


# How a power source is named in the status report.
_SOURCE_LABELS = {
    PowerSource.AC: "AC",
    PowerSource.BATTERY: "Battery",
    PowerSource.NO_BATTERY: "no battery exposed (treated as AC)",
}


# Process lookup.
def _pattern_running(pat):
    for p in psutil.process_iter(["cmdline"]):
        try:
            if pat in " ".join(p.info["cmdline"] or []) and not is_guard_process(p):
                return True
        except Exception:
            continue
    return False


# Filesystem helpers.
def _read_pid(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _alive(pid):
    return pid is not None and psutil.pid_exists(pid)


def _capture_identity(pid, label):
    try:
        process = psutil.Process(pid)
        return process_identity(process)
    except psutil.NoSuchProcess as exc:
        raise StateError(f"{label} process {pid} does not exist") from exc
    except psutil.AccessDenied as exc:
        raise StateError(f"cannot inspect {label} process {pid}") from exc


def _guard_status(store, name):
    identity = store.read_guard(name)
    if identity is not None:
        return identity.pid, identity_is_alive(identity)
    pid = _read_pid(store.legacy_guard_path(name))
    return pid, _alive(pid)


def _ensure_name_available(store, name):
    validate_job_name(name)
    if store.spec_path(name).exists():
        _guard_pid, guard_alive = _guard_status(store, name)
        if guard_alive:
            raise StateError(f"a guard named '{name}' is already active")
        raise StateError(
            f"guard '{name}' has stale state; inspect it with `train-guard status` "
            "before reusing this name"
        )

    _guard_pid, guard_alive = _guard_status(store, name)
    if guard_alive:
        raise StateError(
            f"guard '{name}' has a live supervisor but no job metadata; "
            "stop that process or choose another name"
        )
    if store.runtime_path(name).exists():
        raise StateError(
            f"guard '{name}' has recovery state but no job metadata; "
            f"run `train-guard recover {shlex.quote(name)}` before reusing this name"
        )
    store.remove_guard_state(name)
    store.clear_stop(name)


def _spawn_detached(args, logfile=None, cwd=None):
    out = open(logfile, "ab") if logfile else subprocess.DEVNULL
    kw = dict(stdin=subprocess.DEVNULL, stdout=out, stderr=out, cwd=cwd)
    if os.name == "nt":
        kw["creationflags"] = (
            0x00000008 | 0x00000200
        )  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        return subprocess.Popen(args, **kw)
    finally:
        if logfile:
            out.close()


def _terminate_spawned(process):
    """Best-effort rollback for a child created by the current command."""

    try:
        process.terminate()
    except OSError:
        return
    wait = getattr(process, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _supervisor_argv(name, excluded_identity=None):
    argv = [sys.executable, str(Path(__file__).resolve()), "__supervise", name]
    if excluded_identity is not None:
        argv.extend(
            [
                str(excluded_identity.pid),
                repr(float(excluded_identity.create_time)),
            ]
        )
    return argv


def _supervisor_exited(supervisor, expected):
    # A dead child remains a zombie until reaped, and its identity still
    # matches. The Popen handle is therefore authoritative when available.
    if supervisor is not None:
        poll = getattr(supervisor, "poll", None)
        if poll is not None:
            return poll() is not None
    return not identity_is_alive(expected)


def _wait_for_supervisor_ready(
    store,
    name,
    expected,
    *,
    supervisor=None,
    timeout=_SUPERVISOR_START_TIMEOUT_SECONDS,
):
    """Wait for the child-written handshake before reporting success."""

    def consume_ready():
        ready = store.read_ready(name)
        if ready is None:
            return False
        if ready != expected:
            raise StateError(
                f"supervisor identity changed while starting guard '{name}'"
            )
        # Reap a short-lived child if it has already exited.
        if supervisor is not None and hasattr(supervisor, "poll"):
            supervisor.poll()
        store.clear_ready(name)
        return True

    deadline = time.monotonic() + timeout
    exited = False
    while time.monotonic() < deadline:
        if consume_ready():
            return
        if exited:
            log_path = store.paths.logs / f"{name}.guard.log"
            raise StateError(
                f"supervisor for guard '{name}' exited before it reported "
                f"ready; inspect {log_path}"
            )
        exited = _supervisor_exited(supervisor, expected)
        # The child writes readiness before a clean immediate exit. Read it
        # once more now instead of requiring another loop iteration before
        # the deadline; a slow fsync made that timing assumption flaky.
        if exited and consume_ready():
            return
        if not exited:
            time.sleep(0.02)
    raise StateError(
        f"supervisor for guard '{name}' did not report ready within {timeout:g} seconds"
    )


def _start_supervisor(store, name, excluded_identity=None):
    store.clear_ready(name)
    supervisor = _spawn_detached(
        _supervisor_argv(name, excluded_identity),
    )
    try:
        try:
            identity = _capture_identity(supervisor.pid, "supervisor")
        except StateError:
            # On hosts without zombie processes, a guard for an immediate
            # job can write readiness and disappear before psutil observes it.
            ready = store.read_ready(name)
            if ready is None or ready.pid != supervisor.pid:
                raise
            identity = ready
        _wait_for_supervisor_ready(
            store,
            name,
            identity,
            supervisor=supervisor,
        )
        return identity
    except Exception:
        _terminate_spawned(supervisor)
        store.remove_guard_state(name)
        raise


def cmd_supervise(name, excluded_identity=None):
    return run_supervisor(_paths(), name, excluded_identity)


# CLI commands.
def _policy_line(cfg: PolicyConfig):
    batt = (
        "pause"
        if not cfg.run_on_battery
        else f"{cfg.battery_band} (floor {cfg.battery_floor_pct:g}%)"
    )
    return f"AC={cfg.ac_band}  battery={batt}"


def _restart_on_login(args):
    """Read the new flag while accepting programmatic v0.1 callers."""
    return bool(getattr(args, "restart_on_login", getattr(args, "persist", False)))


def _working_directory(value):
    path = Path(value or os.getcwd()).expanduser()
    if not path.is_dir():
        raise StateError(f"working directory does not exist: {path}")
    return str(path.resolve())


def _previous_persistence(store, name, restart):
    if not restart:
        return None, False
    path = store.persistence_path(name)
    if not path.exists():
        return None, False
    return store.read_persistence(path), True


def _rollback_persistence(store, name, previous, existed):
    if existed:
        store.write_persistence(previous)
    else:
        store.remove_persistence(name)


def cmd_run(args):
    paths = _paths()
    store = JobStore(paths)
    cfg = load_policy(paths.config)
    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.stderr.write(
            "usage: train-guard run [--name N] [--restart-on-login] [--cwd DIR] -- <command...>\n"
        )
        return 2
    name = validate_job_name(args.name or f"job-{time.strftime('%H%M%S')}")
    cwd = _working_directory(args.cwd)
    restart = _restart_on_login(args)
    log = paths.logs / f"{name}.log"
    job = None
    with store.lock_name(name):
        _ensure_name_available(store, name)
        previous, persistence_existed = _previous_persistence(
            store,
            name,
            restart,
        )
        try:
            job = _spawn_detached(cmd, logfile=log, cwd=cwd)
            root = _capture_identity(job.pid, "job")
            store.write_spec(JobSpec.launched(name, root, log))
            if restart:
                store.write_persistence(
                    PersistenceSpec(
                        mode="run",
                        name=name,
                        cwd=cwd,
                        argv=tuple(cmd),
                    )
                )
            guard = _start_supervisor(store, name)
        except Exception:
            if job is not None:
                _terminate_spawned(job)
            store.remove_active_state(name)
            if restart and not getattr(args, "_preserve_persistence", False):
                _rollback_persistence(
                    store,
                    name,
                    previous,
                    persistence_existed,
                )
            raise
    tag = "  (restarts at next login)" if restart else ""
    print(
        f"[train-guard] '{name}': job pid={job.pid}  "
        f"guard pid={guard.pid}  out={log}{tag}"
    )
    print(f"[train-guard] policy: {_policy_line(cfg)}   |   train-guard status")
    return 0


def cmd_attach(args):
    paths = _paths()
    store = JobStore(paths)
    cfg = load_policy(paths.config)
    if not args.match and not args.pid:
        sys.stderr.write(
            'usage: train-guard attach --match "<pattern>" [--name N] [--restart-on-login --start "<cmd>"]   (or --pid PID)\n'
        )
        return 2
    restart = _restart_on_login(args)
    if args.pid and restart:
        sys.stderr.write(
            "attach: a PID cannot survive a reboot; use --match with --restart-on-login and optionally --start\n"
        )
        return 2
    name = validate_job_name(args.name or f"attach-{time.strftime('%H%M%S')}")
    cwd = _working_directory(args.cwd)
    if args.pid:
        spec = JobSpec.attached_pid(name, _capture_identity(int(args.pid), "attached"))
        excluded_identity = None
    else:
        spec = JobSpec.attached_pattern(name, args.match)
        excluded_identity = _capture_identity(os.getpid(), "launcher")
    with store.lock_name(name):
        _ensure_name_available(store, name)
        previous, persistence_existed = _previous_persistence(
            store,
            name,
            restart,
        )
        try:
            store.write_spec(spec)
            if restart:
                store.write_persistence(
                    PersistenceSpec(
                        mode="attach",
                        name=name,
                        cwd=cwd,
                        pattern=args.match,
                        start=args.start,
                    )
                )
            guard = _start_supervisor(
                store,
                name,
                excluded_identity,
            )
        except Exception:
            store.remove_active_state(name)
            if restart and not getattr(args, "_preserve_persistence", False):
                _rollback_persistence(
                    store,
                    name,
                    previous,
                    persistence_existed,
                )
            raise
    tag = "  (restart/reattach configured for next login)" if restart else ""
    print(f"[train-guard] attached '{name}'  guard pid={guard.pid}{tag}")
    print(f"[train-guard] policy: {_policy_line(cfg)}   |   train-guard status")
    return 0


def _agent_installed():
    return agent_installed(AppPaths.from_environment(), SYSTEM)


def _guard_payloads(store):
    guards = []
    state_errors = store.audit_state()
    for spec in store.list_specs():
        guard_error = None
        try:
            guard_pid, guard_alive = _guard_status(store, spec.name)
        except StateError as exc:
            guard_pid, guard_alive = None, False
            guard_error = str(exc)
            if guard_error not in state_errors:
                state_errors.append(guard_error)

        runtime_error = None
        try:
            runtime = store.read_runtime(spec.name)
        except StateError as exc:
            runtime = None
            runtime_error = str(exc)
            if runtime_error not in state_errors:
                state_errors.append(runtime_error)

        guard = {
            "alive": guard_alive,
            "pid": guard_pid,
            "identity_verified": (
                guard_error is None and store.guard_path(spec.name).exists()
            ),
        }
        if guard_error:
            guard["error"] = guard_error
        item = {
            "name": spec.name,
            "mode": spec.mode,
            "guard": guard,
            "runtime": runtime,
        }
        if runtime_error:
            item["runtime_error"] = runtime_error
        guards.append(item)
    return guards, state_errors


def _status_payload(paths):
    store = JobStore(paths)
    config_error = None
    try:
        config = load_policy(paths.config)
    except ConfigError as exc:
        config = PolicyConfig()
        config_error = str(exc)
    observation = SensorReader().sample()
    guards, state_errors = _guard_payloads(store)
    restart_specs = []
    persistence_errors = []
    for path in sorted(paths.persist.glob("*.job")):
        try:
            restart_specs.append(store.read_persistence(path).name)
        except StateError as exc:
            persistence_errors.append(f"{path}: {exc}")
    return {
        "schema_version": 1,
        "version": __version__,
        "observation": observation.to_dict(),
        "policy": config.to_dict(),
        "policy_error": config_error,
        "guards": guards,
        "login_agent_installed": _agent_installed(),
        "restart_specs": restart_specs,
        "persistence_errors": persistence_errors,
        "state_errors": state_errors,
    }


def cmd_status(args):
    paths = _paths()
    payload = _status_payload(paths)
    if getattr(args, "json", False):
        _json_dump(payload)
        return 1 if payload["policy_error"] or payload["state_errors"] else 0

    cfg = PolicyConfig(**payload["policy"])
    obs = payload["observation"]
    print("power / battery")
    temp = obs["temperature_c"]
    tshow = "n/a (not exposed on this OS)" if temp is None else f"{temp:.1f}°C"
    pct = "n/a" if obs["percent"] is None else f"{obs['percent']:g}%"
    chg = "n/a" if obs["charging"] is None else ("yes" if obs["charging"] else "no")
    print(
        f"  source: {_SOURCE_LABELS[PowerSource(obs['source'])]}   charge: {pct}   "
        f"charging: {chg}   pack temp: {tshow}"
    )
    for warning in obs["warnings"]:
        print(f"  note: {warning}")
    if SYSTEM == "Darwin":
        try:
            sp = subprocess.run(
                ["system_profiler", "SPPowerDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in sp.splitlines():
                if re.search(r"Cycle Count|Condition|Maximum Capacity", line):
                    print("  " + line.strip())
        except Exception:
            pass
    print()
    print("policy (config.json)")
    if payload["policy_error"]:
        print(f"  INVALID: {payload['policy_error']}")
        print("  defaults shown; active guards keep their last valid policy")
    print(f"  {_policy_line(cfg)}")
    print(
        f"  thermal: gentle >= {cfg.temp_gentle_c:g}°C  "
        f"pause >= {cfg.temp_pause_c:g}°C  "
        f"resume <= {cfg.temp_resume_c:g}°C   "
        f"charge cool: <{cfg.charge_cool_until_pct:g}% "
        f"and >= {cfg.temp_charge_gentle_c:g}°C"
    )
    print()
    print("active guards")
    if not payload["guards"]:
        print("  (none)")
    for item in payload["guards"]:
        alive = "running" if item["guard"]["alive"] else "dead"
        runtime = item["runtime"] or {}
        state = runtime.get("state", "starting")
        pids = runtime.get("pids", [])
        print(
            f"  {item['name']} [{item['mode']}]  guard={alive} "
            f"(pid {item['guard']['pid']})  state={state}  pids={pids}"
        )
        if item["guard"].get("error"):
            print(f"    guard state invalid: {item['guard']['error']}")
        if item.get("runtime_error"):
            print(f"    runtime state invalid: {item['runtime_error']}")
        if runtime.get("decision"):
            decision = runtime["decision"]
            print(f"    last decision: {decision['action']} ({decision['reason']})")
        if runtime.get("config_error"):
            print(f"    config rejected: {runtime['config_error']}")
    if payload["state_errors"]:
        print()
        print("state issues")
        for error in payload["state_errors"]:
            print(f"  ! {error}")
    print()
    print(
        "restart after reboot (starts a new process; RAM state cannot survive a reboot)"
    )
    print(
        "  login agent: "
        + (
            "INSTALLED"
            if payload["login_agent_installed"]
            else "not installed  ->  train-guard install-agent"
        )
    )
    restart_specs = payload["restart_specs"]
    print(
        "  restart specs: "
        + (
            ", ".join(restart_specs)
            if restart_specs
            else "none (add --restart-on-login to run/attach)"
        )
    )
    for error in payload["persistence_errors"]:
        print(f"  invalid restart spec: {error}")
    return 1 if payload["policy_error"] or payload["state_errors"] else 0


def cmd_list(args):
    store = JobStore(_paths())
    guards, state_errors = _guard_payloads(store)
    if getattr(args, "json", False):
        _json_dump(
            [
                {
                    "name": item["name"],
                    "mode": item["mode"],
                    "alive": item["guard"]["alive"],
                    "state": (item["runtime"] or {}).get("state", "starting"),
                }
                for item in guards
            ]
        )
    else:
        print(
            "\n".join(item["name"] for item in guards)
            if guards
            else "(no active guards)"
        )
    for error in state_errors:
        print(f"state error: {error}", file=sys.stderr)
    return 1 if state_errors else 0


def cmd_stop(args):
    store = JobStore(_paths())
    name = validate_job_name(args.name)
    if not store.spec_path(name).exists():
        sys.stderr.write(f"no active guard named '{name}' (train-guard list)\n")
        return 1
    _guard_pid, guard_alive = _guard_status(store, name)
    if not guard_alive:
        raise StateError(
            f"guard '{name}' is not running; use "
            f"`train-guard recover {shlex.quote(name)}` to release recorded "
            "process changes"
        )
    store.remove_persistence(name)
    store.request_stop(name, args.kill)
    print(
        f"[train-guard] stop requested for '{name}'"
        + (" (--kill the job)" if args.kill else "")
        + "; releases owned changes & detaches within one poll. "
        "(login restart removed)"
    )
    return 0


def _runtime_identities(runtime):
    if runtime is None:
        return []
    return [ProcessIdentity.from_dict(value) for value in runtime["owned_suspensions"]]


def _runtime_tuning(runtime):
    if runtime is None:
        return []
    return list(runtime["tuned_processes"])


def _apply_report_dict(report):
    return {
        field: getattr(report, field)
        for field in (
            "targeted",
            "suspended",
            "resumed",
            "access_denied",
            "gone",
            "tuned",
            "restored",
        )
    }


def cmd_recover(args):
    paths = _paths()
    store = JobStore(paths)
    name = validate_job_name(args.name)
    if not store.spec_path(name).exists() and not store.runtime_path(name).exists():
        raise StateError(f"no stale guard named '{name}'")

    _guard_pid, guard_alive = _guard_status(store, name)
    if guard_alive:
        raise StateError(f"guard '{name}' is still running; use stop instead")

    runtime = store.read_runtime(name)
    controller = ProcessController()
    controller.adopt_owned(_runtime_identities(runtime))
    controller.adopt_tuned(_runtime_tuning(runtime))
    report = controller.release_owned()
    report_payload = _apply_report_dict(report)
    EventJournal(paths, name).emit(
        "recovery_incomplete" if report.access_denied else "recovered",
        (
            "could not release every process change recorded by a dead supervisor"
            if report.access_denied
            else "released process changes recorded by a dead supervisor"
        ),
        process_report=report_payload,
    )
    if report.access_denied:
        previous_pids = [] if runtime is None else runtime["pids"]
        store.write_runtime(
            name,
            _runtime_payload(
                "recovery_incomplete",
                False if runtime is None else runtime["cooling"],
                previous_pids,
                controller,
                process_report=report_payload,
                error="permission denied while releasing owned process state",
            ),
        )
        store.clear_stop(name)
        store.remove_guard_state(name)
    else:
        store.remove_active_state(name)

    print(
        f"[train-guard] recovered '{name}': resumed {report.resumed}, "
        f"restored {report.restored}, gone {report.gone}, "
        f"denied {report.access_denied}"
    )
    return 1 if report.access_denied else 0


def cmd_events(args):
    if args.limit < 1:
        raise StateError("--limit must be at least 1")
    events = EventJournal(_paths(), args.name).read(args.limit)
    if args.json:
        _json_dump(events)
        return 0
    if not events:
        print("(no events)")
        return 0
    for event in events:
        print(
            f"{event.get('timestamp', '?')}  "
            f"{event.get('event', '?'):<24}  {event.get('message', '')}"
        )
    return 0


def cmd_simulate(args):
    if args.transition_limit < 1:
        raise StateError("--transition-limit must be at least 1")
    trace_path = Path(args.trace).expanduser().resolve()
    if args.config_path:
        config_path = Path(args.config_path).expanduser().resolve()
        if not config_path.is_file():
            raise StateError(f"configuration file does not exist: {config_path}")
    else:
        # Deliberately do not call _paths(): replay must not create state.
        config_path = AppPaths.from_environment().config

    policy = load_policy(config_path)
    observations = load_trace(trace_path)
    if args.compare_config_path:
        candidate_path = Path(args.compare_config_path).expanduser().resolve()
        if not candidate_path.is_file():
            raise StateError(
                f"comparison configuration file does not exist: {candidate_path}"
            )
        candidate = load_policy(candidate_path)
        comparison = compare_policies(policy, candidate, observations)
        comparison["trace"] = str(trace_path)
        comparison["baseline_config_source"] = (
            str(config_path) if config_path.exists() else "defaults"
        )
        comparison["candidate_config_source"] = str(candidate_path)
        if args.json:
            _json_dump(comparison)
            return 0

        delta = comparison["delta"]
        disagreement_percent = delta["action_disagreement_percent"]
        disagreement_share = (
            "n/a" if disagreement_percent is None else f"{disagreement_percent:.1f}%"
        )
        print(
            f"trace: {trace_path}  samples={comparison['baseline']['samples']}  "
            f"elapsed={comparison['baseline']['elapsed_seconds']:g}s"
        )
        print(f"baseline:  {_policy_line(policy)}")
        print(f"candidate: {_policy_line(candidate)}")
        print("candidate minus baseline")
        for action in Action:
            seconds = delta["action_seconds"][action.value]
            percent = delta["action_percent"][action.value]
            percentage_points = "n/a" if percent is None else f"{percent:+.1f} pp"
            print(f"  {action.value:<7} {seconds:+8g}s  {percentage_points:>9}")
        print(
            f"action disagreement: {delta['action_disagreement_seconds']:g}s "
            f"({disagreement_share}), {delta['action_disagreement_samples']} "
            "samples"
        )
        print(
            f"transition delta: {delta['action_transitions']:+d} action, "
            f"{delta['decision_transitions']:+d} decision"
        )
        for record in comparison["disagreements"][: args.transition_limit]:
            baseline_decision = record["baseline"]
            candidate_decision = record["candidate"]
            print(
                f"  {record['observed_at']}  "
                f"{baseline_decision['action']}/{baseline_decision['reason']} -> "
                f"{candidate_decision['action']}/{candidate_decision['reason']}"
            )
        remaining = len(comparison["disagreements"]) - args.transition_limit
        if remaining > 0:
            print(f"  ... {remaining} more; use --json for the complete comparison")
        return 0

    report = simulate_policy(policy, observations)
    report["trace"] = str(trace_path)
    report["config_source"] = str(config_path) if config_path.exists() else "defaults"
    if args.json:
        _json_dump(report)
        return 0

    print(
        f"trace: {trace_path}  samples={report['samples']}  "
        f"elapsed={report['elapsed_seconds']:g}s"
    )
    print(f"policy: {_policy_line(policy)}")
    print("actions")
    for action in Action:
        seconds = report["action_seconds"][action.value]
        percent = report["action_percent"][action.value]
        share = "n/a" if percent is None else f"{percent:.1f}%"
        samples = report["action_samples"][action.value]
        print(f"  {action.value:<7} {seconds:>8g}s  {share:>6}  {samples} samples")
    print(
        f"transitions: {report['action_transitions']} action, "
        f"{report['decision_transitions']} decision"
    )
    for record in report["transitions"][: args.transition_limit]:
        observation = record["observation"]
        temperature = observation["temperature_c"]
        temperature_text = "n/a" if temperature is None else f"{temperature:g}C"
        decision = record["decision"]
        print(
            f"  {record['observed_at']}  {decision['action']:<7} "
            f"{decision['reason']:<18} temp={temperature_text}"
        )
    remaining = len(report["transitions"]) - args.transition_limit
    if remaining > 0:
        print(f"  ... {remaining} more; use --json for the complete replay")
    return 0


def cmd_sweep(args):
    if args.top < 1:
        raise StateError("--top must be at least 1")
    trace_path = Path(args.trace).expanduser().resolve()
    grid_path = Path(args.grid).expanduser().resolve()
    if not grid_path.is_file():
        raise StateError(f"grid file does not exist: {grid_path}")
    if args.config_path:
        config_path = Path(args.config_path).expanduser().resolve()
        if not config_path.is_file():
            raise StateError(f"configuration file does not exist: {config_path}")
    else:
        # Like simulate, a sweep is a pure offline command.
        config_path = AppPaths.from_environment().config

    base = load_policy(config_path)
    try:
        grid_text = grid_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StateError(f"{grid_path}: cannot read grid: {exc}") from exc
    grid = parse_grid(grid_text, str(grid_path))
    observations = load_trace(trace_path)
    report = run_sweep(
        base,
        grid,
        observations,
        engine=args.engine,
        hot_ref_c=args.hot_ref,
        low_battery_ref_pct=args.low_battery_ref,
    )
    report["trace"] = str(trace_path)
    report["config_source"] = str(config_path) if config_path.exists() else "defaults"
    report["grid_source"] = str(grid_path)
    if args.json:
        _json_dump(report)
        return 0

    facts = report["trace_facts"]
    print(
        f"trace: {trace_path}  samples={report['samples']}  "
        f"elapsed={report['elapsed_seconds']:g}s  engine={report['engine']}"
    )

    def coverage(value):
        return "n/a" if value is None else f"{100 * value:.0f}%"

    print(
        f"trace facts: {facts['hot_degc_seconds']:g} degC*s above "
        f"{facts['hot_reference_c']:g}C (temp coverage "
        f"{coverage(facts['temperature_coverage'])}), "
        f"{facts['high_soc_seconds']:g}s at "
        f">={facts['high_soc_reference_pct']:g}% charge, "
        f"{facts['equivalent_full_cycles']:.3f} equivalent full cycles"
    )
    print(
        f"candidates: {report['candidates_evaluated']} evaluated, "
        f"{report['candidates_rejected']} invalid, "
        f"pareto front {report['pareto_front_size']}"
    )
    baseline = report["baseline"]["metrics"]
    print(
        f"baseline: run {baseline['run_seconds']:g}s  "
        f"hot {baseline['hot_run_degc_seconds']:g} degC*s  "
        f"low-battery {baseline['low_battery_run_seconds']:g}s  "
        f"transitions {baseline['action_transitions']}"
    )

    def efficiency_text(value):
        return "n/a" if value is None else f"{100 * value:.0f}%"

    bound = report["baseline"]["clairvoyant"]
    threshold = bound["hot_only_hindsight_threshold_c"]
    threshold_text = "not binding" if threshold is None else f"{threshold:g}C"
    print(
        f"joint clairvoyant bound: {bound['bound_run_seconds']:g}s runnable at "
        "the baseline's own exposure "
        f"(efficiency {efficiency_text(bound['efficiency'])}, "
        f"gap {bound['gap_seconds']:g}s, hot-only cutoff {threshold_text})"
    )
    print("   run_s     hot_degC*s  lowbatt_s  trans   eff  overrides")
    for candidate in report["candidates"][: args.top]:
        metrics = candidate["metrics"]
        marker = "*" if candidate["pareto_optimal"] else " "
        overrides = json.dumps(candidate["overrides"], sort_keys=True)
        efficiency = efficiency_text(candidate["clairvoyant"]["efficiency"])
        print(
            f" {marker} {metrics['run_seconds']:>7g}  "
            f"{metrics['hot_run_degc_seconds']:>10g}  "
            f"{metrics['low_battery_run_seconds']:>9g}  "
            f"{metrics['action_transitions']:>5}  {efficiency:>4}  {overrides}"
        )
    remaining = len(report["candidates"]) - args.top
    if remaining > 0:
        print(f"  ... {remaining} more; use --json for the complete report")
    print(
        "* pareto-optimal on (run seconds up, hot exposure down, low-battery run down)"
    )
    print(
        "note: metrics re-weight the recorded trace under each policy's "
        "actions; they do not simulate temperature or charge and are not "
        "battery-life predictions. Efficiency compares permitted work with "
        "the exact joint fractional clairvoyant schedule at the same recorded "
        "exposure"
    )
    return 0


def cmd_config(args):
    paths = _paths()
    if getattr(args, "init", False):
        if paths.config.exists() and not getattr(args, "force", False):
            raise StateError(
                f"{paths.config} already exists; pass --force to replace it"
            )
        atomic_json_write(paths.config, PolicyConfig().to_dict())
        print(f"wrote {paths.config}")
    cfg = load_policy(paths.config)
    if getattr(args, "check", False):
        print(f"{paths.config}: valid")
        return 0
    print(f"policy: {paths.config}\n")
    _json_dump(cfg.to_dict())
    if not paths.config.exists():
        print("(defaults shown; use `train-guard config --init` to write them)")
    return 0


def _doctor_payload(paths):
    checks = []
    try:
        policy = load_policy(paths.config)
        checks.append({"name": "config", "ok": True, "detail": policy.to_dict()})
    except ConfigError as exc:
        checks.append({"name": "config", "ok": False, "detail": str(exc)})

    try:
        with tempfile.NamedTemporaryFile(dir=paths.home):
            pass
        checks.append(
            {
                "name": "state_directory",
                "ok": True,
                "detail": str(paths.home),
            }
        )
    except OSError as exc:
        checks.append({"name": "state_directory", "ok": False, "detail": str(exc)})

    store = JobStore(paths)
    state_errors = store.audit_state()
    checks.append(
        {
            "name": "state_files",
            "ok": not state_errors,
            "detail": {"errors": state_errors},
        }
    )

    observation = SensorReader().sample()
    checks.append(
        {
            "name": "sensors",
            "ok": True,
            "detail": observation.to_dict(),
            "advisory": bool(observation.warnings),
        }
    )

    stale = []
    for spec in store.list_specs():
        try:
            _pid, alive = _guard_status(store, spec.name)
        except StateError:
            continue
        if not alive:
            stale.append(spec.name)
    checks.append(
        {
            "name": "supervisors",
            "ok": not stale,
            "detail": {"stale": stale},
        }
    )
    login_agent_present = _agent_installed()
    checks.append(
        {
            "name": "login_agent",
            "ok": True,
            "detail": {"installed": login_agent_present},
            "advisory": not login_agent_present,
        }
    )
    return {
        "schema_version": 1,
        "ok": all(check["ok"] for check in checks),
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": checks,
    }


def cmd_doctor(args):
    payload = _doctor_payload(_paths())
    if args.json:
        _json_dump(payload)
    else:
        print(
            f"train-guard {payload['version']}  Python {payload['python']}  "
            f"{payload['platform']}"
        )
        for check in payload["checks"]:
            label = "ok" if check["ok"] else "FAIL"
            if check.get("advisory"):
                label = "note"
            detail = check["detail"]
            if check["name"] == "sensors":
                detail = ", ".join(detail["warnings"]) or "available"
            print(f"  {label:<4} {check['name']}: {detail}")
    return 0 if payload["ok"] else 1


def cmd_restart_persisted(args):
    """Recreate configured jobs after login.

    This deliberately does not claim process-state restoration: an OS reboot
    destroys RAM, so a ``run`` spec starts the command as a fresh process. Jobs
    that can checkpoint may continue from their own last durable checkpoint.
    """
    paths = _paths()
    store = JobStore(paths)
    load_policy(paths.config)
    persisted = list(store.list_persistence())
    failures = 0
    for _path, spec in persisted:
        try:
            name = spec.name
            if store.spec_path(name).exists():
                if _guard_status(store, name)[1]:
                    print(f"[restart] '{name}' already active; skipping")
                else:
                    print(
                        f"[restart] '{name}' has stale state; "
                        "run recover before restarting"
                    )
                    failures += 1
                continue
            if spec.mode == "run":
                result = cmd_run(
                    argparse.Namespace(
                        name=name,
                        restart_on_login=True,
                        cwd=spec.cwd,
                        cmd=list(spec.argv),
                        _preserve_persistence=True,
                    )
                )
            else:
                pattern, start = spec.pattern, spec.start or ""
                started_process = None
                if start and not _pattern_running(pattern):
                    shell = (
                        ["cmd", "/c", start]
                        if os.name == "nt"
                        else ["/bin/sh", "-c", start]
                    )
                    started_process = _spawn_detached(
                        shell,
                        logfile=paths.logs / f"{name}.start.log",
                        cwd=spec.cwd,
                    )
                    print(f"[restart] started job for '{name}': {start}")
                result = 1
                try:
                    result = cmd_attach(
                        argparse.Namespace(
                            name=name,
                            restart_on_login=True,
                            cwd=spec.cwd,
                            match=pattern,
                            pid=None,
                            start=start,
                            _preserve_persistence=True,
                        )
                    )
                finally:
                    if result != 0 and started_process is not None:
                        _terminate_spawned(started_process)
            failures += int(result != 0)
        except (ConfigError, OSError, StateError) as exc:
            failures += 1
            print(f"[restart] {spec.name}: {exc}", file=sys.stderr)
    print(
        "[restart] done (commands were restarted or reattached; no RAM state was restored)"
    )
    return 1 if failures else 0


# Compatibility for login agents installed by v0.1.
cmd_resume = cmd_restart_persisted


def cmd_unpersist(args):
    store = JobStore(_paths())
    name = validate_job_name(args.name)
    store.remove_persistence(name)
    print(f"[train-guard] '{name}' will no longer restart or reattach at login")
    return 0


def cmd_install_agent(args):
    paths = _paths()
    restart_argv = [sys.executable, str(Path(__file__).resolve()), "restart-persisted"]
    target = install_agent(paths, restart_argv, SYSTEM)
    print(f"[train-guard] login restart agent installed: {target}")
    if SYSTEM == "Linux":
        print(
            "  tip: `loginctl enable-linger $USER` to also restart without an active GUI session."
        )
    return 0


def cmd_uninstall_agent(args):
    uninstall_agent(_paths(), SYSTEM)
    print("[train-guard] login agent removed.")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="train-guard",
        description="Battery and thermal supervisor for long compute jobs.",
    )
    p.add_argument("--version", action="version", version=f"train-guard {__version__}")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="launch + supervise a new job")
    r.add_argument("--name")
    rg = r.add_mutually_exclusive_group()
    rg.add_argument(
        "--restart-on-login",
        action="store_true",
        help="start this command as a new process after reboot/login",
    )
    rg.add_argument(
        "--persist",
        action="store_true",
        dest="restart_on_login",
        help="deprecated alias for --restart-on-login",
    )
    r.add_argument("--cwd")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("attach", help="supervise a running job")
    a.add_argument("--name")
    a.add_argument("--match")
    a.add_argument("--pid", type=int)
    ag = a.add_mutually_exclusive_group()
    ag.add_argument(
        "--restart-on-login",
        action="store_true",
        help="reattach by pattern after login; --start may launch a new process",
    )
    ag.add_argument(
        "--persist",
        action="store_true",
        dest="restart_on_login",
        help="deprecated alias for --restart-on-login",
    )
    a.add_argument("--cwd")
    a.add_argument("--start")
    a.set_defaults(fn=cmd_attach)

    status = sub.add_parser("status", help="power/battery/temp/guards")
    status.add_argument("--json", action="store_true")
    status.set_defaults(fn=cmd_status)

    listing = sub.add_parser("list", help="list active guards")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(fn=cmd_list)

    for nm, fn, hlp in [
        (
            "restart-persisted",
            cmd_restart_persisted,
            "restart or reattach configured jobs after login",
        ),
        (
            "resume",
            cmd_resume,
            "compatibility alias for restart-persisted (does not restore RAM)",
        ),
        (
            "install-agent",
            cmd_install_agent,
            "restart configured jobs after reboot/login",
        ),
        ("uninstall-agent", cmd_uninstall_agent, "remove the login agent"),
    ]:
        s = sub.add_parser(nm, help=hlp)
        s.set_defaults(fn=fn)

    st = sub.add_parser(
        "stop",
        help="stop supervising (release owned changes); --kill ends job",
    )
    st.add_argument("name")
    st.add_argument("--kill", action="store_true")
    st.set_defaults(fn=cmd_stop)

    recover = sub.add_parser(
        "recover",
        help="release process changes recorded by a dead supervisor",
    )
    recover.add_argument("name")
    recover.set_defaults(fn=cmd_recover)

    events = sub.add_parser("events", help="show the structured job event log")
    events.add_argument("name")
    events.add_argument("--limit", type=int, default=50)
    events.add_argument("--json", action="store_true")
    events.set_defaults(fn=cmd_events)

    simulate = sub.add_parser(
        "simulate",
        help="replay a policy against observation or event JSONL",
    )
    simulate.add_argument("trace")
    simulate.add_argument("--config", dest="config_path")
    simulate.add_argument(
        "--compare-config",
        dest="compare_config_path",
        help="replay a candidate and report its delta from the baseline",
    )
    simulate.add_argument(
        "--transition-limit",
        type=int,
        default=20,
        help="maximum transition rows in human output (default: 20)",
    )
    simulate.add_argument("--json", action="store_true")
    simulate.set_defaults(fn=cmd_simulate)

    sweep = sub.add_parser(
        "sweep",
        help="evaluate a grid of policies against a recorded trace",
    )
    sweep.add_argument("trace")
    sweep.add_argument(
        "--grid",
        required=True,
        help="JSON file mapping policy fields to candidate value lists",
    )
    sweep.add_argument("--config", dest="config_path")
    sweep.add_argument(
        "--engine",
        choices=("auto", "python", "native"),
        default="auto",
        help="evaluation engine; native is unavailable until its kernel is built",
    )
    sweep.add_argument(
        "--hot-ref",
        dest="hot_ref",
        type=float,
        default=DEFAULT_HOT_REFERENCE_C,
        help="temperature reference for hot exposure (default: 35)",
    )
    sweep.add_argument(
        "--low-battery-ref",
        dest="low_battery_ref",
        type=float,
        default=DEFAULT_LOW_BATTERY_REFERENCE_PCT,
        help="charge percentage treated as low battery (default: 20)",
    )
    sweep.add_argument(
        "--top",
        type=int,
        default=10,
        help="candidate rows in human output (default: 10)",
    )
    sweep.add_argument("--json", action="store_true")
    sweep.set_defaults(fn=cmd_sweep)

    up = sub.add_parser("unpersist", help="forget a persisted job")
    up.add_argument("name")
    up.set_defaults(fn=cmd_unpersist)
    cf = sub.add_parser("config", help="show, create or validate policy")
    cf.add_argument("--init", action="store_true")
    cf.add_argument("--force", action="store_true")
    cf.add_argument("--check", action="store_true")
    cf.set_defaults(fn=cmd_config)

    doctor = sub.add_parser("doctor", help="check the local installation")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(fn=cmd_doctor)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "__supervise":  # internal
            if len(argv) not in {2, 4}:
                raise StateError("invalid internal supervisor arguments")
            excluded_identity = None
            if len(argv) == 4:
                try:
                    excluded_identity = ProcessIdentity(
                        pid=int(argv[2]),
                        create_time=float(argv[3]),
                    )
                except (TypeError, ValueError) as exc:
                    raise StateError("invalid launching CLI identity") from exc
            return cmd_supervise(argv[1], excluded_identity)
        parser = build_parser()
        args = parser.parse_args(argv)
        if not getattr(args, "fn", None):
            return cmd_status(args)
        return args.fn(args)
    except (
        ConfigError,
        OSError,
        PlatformError,
        StateError,
        SweepError,
        TraceError,
    ) as exc:
        sys.stderr.write(f"train-guard: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("train-guard: interrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
