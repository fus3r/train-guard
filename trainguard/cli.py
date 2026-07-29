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
import plistlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.stderr.write("train-guard requires psutil:  pip install psutil\n")
    raise

if __package__:
    from .config import ConfigError, ConfigWatcher, PolicyConfig, load_policy
    from .model import PowerSource, ProcessIdentity, utc_now
    from .policy import PolicyEngine
    from .processes import (
        ProcessController,
        identity_is_alive,
        is_guard_process,
        process_identity,
    )
    from .sensors import SensorReader
    from .state import (
        AppPaths,
        JobSpec,
        JobStore,
        PersistenceSpec,
        StateError,
        atomic_json_write,
        validate_job_name,
    )
else:  # Keep direct execution from a checkout working.
    from config import ConfigError, ConfigWatcher, PolicyConfig, load_policy
    from model import PowerSource, ProcessIdentity, utc_now
    from policy import PolicyEngine
    from processes import (
        ProcessController,
        identity_is_alive,
        is_guard_process,
        process_identity,
    )
    from sensors import SensorReader
    from state import (
        AppPaths,
        JobSpec,
        JobStore,
        PersistenceSpec,
        StateError,
        atomic_json_write,
        validate_job_name,
    )

__version__ = "0.2.0"
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

# Linux niceness cannot normally be raised back to its original value by an
# unprivileged process. Keep a reversible CPU-affinity snapshot instead.
_LINUX_AFFINITY = {}


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


def load_config() -> PolicyConfig:
    return load_policy(_paths().config)


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# How a power source is named in the status report.
_SOURCE_LABELS = {
    PowerSource.AC: "AC",
    PowerSource.BATTERY: "Battery",
    PowerSource.NO_BATTERY: "no battery exposed (treated as AC)",
}


# Process control.
def _set_band(proc, band):
    """Apply a best-effort scheduling band, never a CPU-core guarantee.

    On macOS, ``taskpolicy -b`` marks a process as background work and leaves
    placement to the kernel scheduler. Linux and Windows use their respective
    low-priority controls. None of these mechanisms pins a job to a particular
    physical core class.
    """
    try:
        if SYSTEM == "Darwin":
            flag = "-b" if band == "gentle" else "-B"
            subprocess.run(["taskpolicy", flag, "-p", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        elif SYSTEM == "Linux":
            if band == "gentle":
                allowed = proc.cpu_affinity()
                _LINUX_AFFINITY.setdefault(proc.pid, allowed)
                if len(allowed) > 1:
                    proc.cpu_affinity(allowed[::2])
            else:
                original = _LINUX_AFFINITY.pop(proc.pid, None)
                if original:
                    proc.cpu_affinity(original)
            try:
                proc.ionice(psutil.IOPRIO_CLASS_IDLE if band == "gentle" else psutil.IOPRIO_CLASS_NONE)
            except Exception:
                pass
        elif SYSTEM == "Windows":
            proc.nice(psutil.IDLE_PRIORITY_CLASS if band == "gentle" else psutil.NORMAL_PRIORITY_CLASS)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def _apply(decision, procs):
    for p in procs:
        try:
            if decision == "stop":
                p.suspend()
            else:
                p.resume()
                _set_band(p, decision)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


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
            "inspect it with `train-guard status` before reusing this name"
        )
    store.remove_guard_state(name)


def _glog(name, msg):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(LOGDIR / f"{name}.guard.log", "a") as f:
        f.write(f"[guard] {_now()} {msg}\n")


def _migrate_legacy_identity(store, spec):
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
        _glog(spec.name, message)
        raise StateError(
            f"legacy PID {identity.pid} now belongs to a newer process; "
            "the metadata was preserved"
        )

    migrated = replace(spec, root=identity, legacy_pid=None)
    store.write_spec(migrated)
    _glog(
        spec.name,
        f"STATE_MIGRATED pid={identity.pid} create_time={identity.create_time:g}",
    )
    return migrated


def _spawn_detached(args, logfile=None, cwd=None):
    out = open(logfile, "ab") if logfile else subprocess.DEVNULL
    kw = dict(stdin=subprocess.DEVNULL, stdout=out, stderr=out, cwd=cwd)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        return subprocess.Popen(args, **kw)
    finally:
        if logfile:
            out.close()


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


def _runtime_payload(state, cooling, pids, observation=None, decision=None, config_error=None):
    payload = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "state": state,
        "cooling": cooling,
        # Empty, explicit ownership fields keep the recovery schema unambiguous
        # without claiming ownership the current controller cannot yet prove.
        "owned_suspensions": [],
        "tuned_processes": [],
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
    return payload


# Supervisor loop.
def cmd_supervise(name, excluded_identity=None):
    paths = _paths()
    store = JobStore(paths)
    spec = _migrate_legacy_identity(store, store.read_spec(name))
    if spec.mode == "attach" and excluded_identity is None:
        raise StateError(
            "cannot identify the CLI that launched this match supervisor"
        )
    controller = ProcessController(
        excluded_identities=(
            () if excluded_identity is None else (excluded_identity,)
        )
    )
    detail = (
        f"pattern={spec.pattern}"
        if spec.mode == "attach"
        else f"jobpid={spec.root_pid}"
    )
    cfg = load_policy(paths.config)
    watcher = ConfigWatcher(paths.config, cfg)
    engine = PolicyEngine()  # cooldown state belongs to this supervisor alone
    reader = SensorReader()
    _glog(name, f"START mode={spec.mode} {detail}")
    last, warned, miss = None, (), 0
    store.write_runtime(name, _runtime_payload("starting", False, []))
    while True:
        how = store.read_stop(name)
        if how is not None:
            procs = controller.resolve(spec).processes
            _apply("full", procs)
            if how == "kill":
                for p in procs:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                _glog(name, "STOP --kill: terminated job")
            else:
                _glog(name, "STOP: unfroze job, detaching")
            store.remove_active_state(name)
            return
        snapshot = controller.resolve(spec)
        if spec.mode == "run" and not snapshot.root_alive:
            _glog(name, "job exited; guard done")
            store.remove_active_state(name)
            return
        cfg, config_changed, config_error = watcher.poll()
        if config_changed:
            if config_error:
                _glog(name, f"CONFIG rejected: {config_error}; keeping last valid policy")
            else:
                _glog(name, "CONFIG loaded")
        procs = snapshot.processes
        if spec.mode == "attach" and not procs:
            miss += 1
            if miss % 15 == 1:
                _glog(name, "no process matches pattern yet (waiting)")
            store.write_runtime(
                name,
                _runtime_payload(
                    "waiting",
                    engine.cooling,
                    [],
                    config_error=config_error,
                ),
            )
            time.sleep(cfg.poll)
            continue
        miss = 0
        observation = reader.sample()
        if observation.warnings != warned:
            for warning in observation.warnings:
                _glog(name, f"SENSOR {warning}")
            warned = observation.warnings
        decision = engine.decide(cfg, observation)
        action, sig = decision.action.value, observation.signature()
        _apply(action, procs)
        store.write_runtime(
            name,
            _runtime_payload(
                action,
                decision.cooling,
                [proc.pid for proc in procs],
                observation=observation,
                decision=decision,
                config_error=config_error,
            ),
        )
        line = f"{action} {decision.reason.value} | {sig}"
        if line != last:
            _glog(name, f"-> {action}   ({decision.reason.value}; {sig})")
            last = line
        time.sleep(cfg.poll)


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


def cmd_run(args):
    paths = _paths()
    store = JobStore(paths)
    cfg = load_policy(paths.config)
    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.stderr.write("usage: train-guard run [--name N] [--restart-on-login] [--cwd DIR] -- <command...>\n")
        return 2
    name = validate_job_name(args.name or f"job-{time.strftime('%H%M%S')}")
    cwd = args.cwd or os.getcwd()
    restart = _restart_on_login(args)
    log = paths.logs / f"{name}.log"
    with store.lock_name(name):
        _ensure_name_available(store, name)
        if restart:
            store.write_persistence(
                PersistenceSpec(
                    mode="run",
                    name=name,
                    cwd=cwd,
                    argv=tuple(cmd),
                )
            )
        job = _spawn_detached(cmd, logfile=log, cwd=cwd)
        root = _capture_identity(job.pid, "job")
        store.write_spec(JobSpec.launched(name, root, log))
        sup = _spawn_detached(_supervisor_argv(name))
        guard = _capture_identity(sup.pid, "supervisor")
        store.write_guard(name, guard)
    tag = "  (restarts at next login)" if restart else ""
    print(f"[train-guard] '{name}': job pid={job.pid}  guard pid={sup.pid}  out={log}{tag}")
    print(f"[train-guard] policy: {_policy_line(cfg)}   |   train-guard status")
    return 0


def cmd_attach(args):
    paths = _paths()
    store = JobStore(paths)
    cfg = load_policy(paths.config)
    if not args.match and not args.pid:
        sys.stderr.write('usage: train-guard attach --match "<pattern>" [--name N] [--restart-on-login --start "<cmd>"]   (or --pid PID)\n')
        return 2
    restart = _restart_on_login(args)
    if args.pid and restart:
        sys.stderr.write("attach: a PID cannot survive a reboot; use --match with --restart-on-login and optionally --start\n")
        return 2
    name = validate_job_name(args.name or f"attach-{time.strftime('%H%M%S')}")
    cwd = args.cwd or os.getcwd()
    if args.pid:
        spec = JobSpec.attached_pid(name, _capture_identity(int(args.pid), "attached"))
        excluded_identity = None
    else:
        spec = JobSpec.attached_pattern(name, args.match)
        excluded_identity = _capture_identity(os.getpid(), "launcher")
    with store.lock_name(name):
        _ensure_name_available(store, name)
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
        sup = _spawn_detached(_supervisor_argv(name, excluded_identity))
        guard = _capture_identity(sup.pid, "supervisor")
        store.write_guard(name, guard)
    tag = "  (restart/reattach configured for next login)" if restart else ""
    print(f"[train-guard] attached '{name}'  guard pid={sup.pid}{tag}")
    print(f"[train-guard] policy: {_policy_line(cfg)}   |   train-guard status")
    return 0


def _agent_installed():
    if SYSTEM == "Darwin":
        return (HOME / "Library/LaunchAgents/com.trainguard.resume.plist").exists()
    if SYSTEM == "Linux":
        return (HOME / ".config/systemd/user/trainguard-resume.service").exists()
    if SYSTEM == "Windows":
        try:
            for task in ("TrainGuardRestart", "TrainGuardResume"):
                if subprocess.run(
                    ["schtasks", "/Query", "/TN", task],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).returncode == 0:
                    return True
            return False
        except (OSError, subprocess.SubprocessError):
            return False
    return False


def cmd_status(args):
    paths = _paths()
    store = JobStore(paths)
    cfg = load_policy(paths.config)
    obs = SensorReader().sample()
    print("power / battery")
    temp = obs.temperature_c
    tshow = "n/a (not exposed on this OS)" if temp is None else f"{temp:.1f}°C"
    pct = "n/a" if obs.percent is None else f"{obs.percent:g}%"
    chg = "n/a" if obs.charging is None else ("yes" if obs.charging else "no")
    print(
        f"  source: {_SOURCE_LABELS[obs.source]}   charge: {pct}   "
        f"charging: {chg}   pack temp: {tshow}"
    )
    for warning in obs.warnings:
        print(f"  note: {warning}")
    if SYSTEM == "Darwin":
        try:
            sp = subprocess.run(["system_profiler", "SPPowerDataType"], capture_output=True, text=True, timeout=10).stdout
            for line in sp.splitlines():
                if re.search(r"Cycle Count|Condition|Maximum Capacity", line):
                    print("  " + line.strip())
        except Exception:
            pass
    print()
    print("policy (config.json)")
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
    specs = store.list_specs()
    if not specs:
        print("  (none)")
    for spec in specs:
        try:
            gp, guard_alive = _guard_status(store, spec.name)
        except StateError:
            gp, guard_alive = None, False
        alive = "running" if guard_alive else "dead"
        procs = ProcessController().resolve(spec).processes
        if procs:
            st = "PAUSED/frozen" if any(p.status() == psutil.STATUS_STOPPED for p in procs) else "running"
            worker = f"{st}  ({len(procs)} proc)"
        else:
            worker = "none right now"
        print(
            f"  {spec.name} [{spec.mode}]  guard={alive} (pid {gp})   "
            f"worker: {worker}"
        )
    state_errors = store.audit_state()
    if state_errors:
        print()
        print("state issues")
        for error in state_errors:
            print(f"  ! {error}")
    print()
    print("restart after reboot (starts a new process; RAM state cannot survive a reboot)")
    print(f"  login agent: {'INSTALLED' if _agent_installed() else 'not installed  ->  train-guard install-agent'}")
    pj = sorted(paths.persist.glob("*.job"))
    print("  restart specs: " + (", ".join(p.stem for p in pj) if pj else "none (add --restart-on-login to run/attach)"))
    return 0


def cmd_list(args):
    store = JobStore(_paths())
    specs = store.list_specs()
    print("\n".join(spec.name for spec in specs) if specs else "(no active guards)")
    return 0


def cmd_stop(args):
    store = JobStore(_paths())
    name = validate_job_name(args.name)
    if not store.spec_path(name).exists():
        sys.stderr.write(f"no active guard named '{name}' (train-guard list)\n")
        return 1
    store.request_stop(name, args.kill)
    store.remove_persistence(name)
    print(f"[train-guard] stop requested for '{name}'" + (" (--kill the job)" if args.kill else "") +
          "; unfreezes & detaches within one poll. (login restart removed)")
    return 0


def cmd_config(args):
    paths = _paths()
    defaults = PolicyConfig().to_dict()
    if getattr(args, "init", False):
        atomic_json_write(paths.config, defaults)
    cfg = load_policy(paths.config)
    print(f"policy: {paths.config}\n")
    print(json.dumps(cfg.to_dict(), indent=2))
    if not paths.config.exists():
        print("\n(no config.json yet; defaults shown. run `train-guard config --init` to write it)")
    if getattr(args, "init", False):
        print(f"\nwrote {paths.config}")
    return 0


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
    for _path, spec in persisted:
        name = spec.name
        if store.spec_path(name).exists() and _guard_status(store, name)[1]:
            print(f"[restart] '{name}' already active; skipping")
            continue
        if spec.mode == "run":
            cmd_run(argparse.Namespace(name=name, restart_on_login=True,
                                       cwd=spec.cwd, cmd=list(spec.argv)))
        else:
            pat, start = spec.pattern, spec.start or ""
            if start and not _pattern_running(pat):
                shell = ["cmd", "/c", start] if os.name == "nt" else ["/bin/sh", "-c", start]
                _spawn_detached(shell, logfile=paths.logs / f"{name}.start.log", cwd=spec.cwd)
                print(f"[restart] started job for '{name}': {start}")
            cmd_attach(argparse.Namespace(name=name, restart_on_login=True, cwd=spec.cwd,
                                          match=pat, pid=None, start=start))
    print("[restart] done (commands were restarted or reattached; no RAM state was restored)")
    return 0


# Compatibility for login agents installed by v0.1.
cmd_resume = cmd_restart_persisted


def cmd_unpersist(args):
    store = JobStore(_paths())
    name = validate_job_name(args.name)
    store.remove_persistence(name)
    print(f"[train-guard] '{name}' will no longer restart or reattach at login")
    return 0


def cmd_install_agent(args):
    _ensure_dirs()
    restart_argv = [sys.executable, str(Path(__file__).resolve()), "restart-persisted"]
    if SYSTEM == "Darwin":
        plist = HOME / "Library/LaunchAgents/com.trainguard.resume.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_bytes(plistlib.dumps({
            "Label": "com.trainguard.resume",
            "ProgramArguments": restart_argv,
            "RunAtLoad": True,
            "ProcessType": "Background",
            "StandardOutPath": str(LOGDIR / "restart.log"),
            "StandardErrorPath": str(LOGDIR / "restart.log"),
        }))
        subprocess.run(["launchctl", "unload", str(plist)], stderr=subprocess.DEVNULL)
        subprocess.run(["launchctl", "load", "-w", str(plist)], stderr=subprocess.DEVNULL)
        print(f"[train-guard] login restart agent installed (launchd): {plist}")
    elif SYSTEM == "Linux":
        unit = HOME / ".config/systemd/user/trainguard-resume.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        exec_start = shlex.join(restart_argv)
        unit.write_text(f"""[Unit]
Description=train-guard: restart configured jobs at login
[Service]
Type=oneshot
ExecStart={exec_start}
[Install]
WantedBy=default.target
""")
        subprocess.run(["systemctl", "--user", "daemon-reload"], stderr=subprocess.DEVNULL)
        subprocess.run(["systemctl", "--user", "enable", "trainguard-resume.service"], stderr=subprocess.DEVNULL)
        print(f"[train-guard] login restart service installed (systemd --user): {unit}")
        print("  tip: `loginctl enable-linger $USER` to also restart without an active GUI session.")
    elif SYSTEM == "Windows":
        task = "TrainGuardRestart"
        cmdline = subprocess.list2cmdline(restart_argv)
        subprocess.run(["schtasks", "/Create", "/TN", task, "/SC", "ONLOGON", "/TR", cmdline, "/F"])
        print(f"[train-guard] login restart task installed (schtasks): {task}")
    else:
        sys.stderr.write("install-agent: unsupported OS\n")
        return 2
    return 0


def cmd_uninstall_agent(args):
    if SYSTEM == "Darwin":
        plist = HOME / "Library/LaunchAgents/com.trainguard.resume.plist"
        subprocess.run(["launchctl", "unload", str(plist)], stderr=subprocess.DEVNULL)
        try:
            plist.unlink()
        except FileNotFoundError:
            pass
    elif SYSTEM == "Linux":
        subprocess.run(["systemctl", "--user", "disable", "trainguard-resume.service"], stderr=subprocess.DEVNULL)
        try:
            (HOME / ".config/systemd/user/trainguard-resume.service").unlink()
        except FileNotFoundError:
            pass
    elif SYSTEM == "Windows":
        subprocess.run(["schtasks", "/Delete", "/TN", "TrainGuardRestart", "/F"])
        # Clean up the v0.1 task name too, if present.
        subprocess.run(["schtasks", "/Delete", "/TN", "TrainGuardResume", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[train-guard] login agent removed.")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="train-guard", description="Battery and thermal supervisor for long compute jobs.")
    p.add_argument("--version", action="version", version=f"train-guard {__version__}")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="launch + supervise a new job")
    r.add_argument("--name")
    rg = r.add_mutually_exclusive_group()
    rg.add_argument("--restart-on-login", action="store_true",
                    help="start this command as a new process after reboot/login")
    rg.add_argument("--persist", action="store_true", dest="restart_on_login",
                    help="deprecated alias for --restart-on-login")
    r.add_argument("--cwd")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("attach", help="supervise a running job")
    a.add_argument("--name")
    a.add_argument("--match")
    a.add_argument("--pid", type=int)
    ag = a.add_mutually_exclusive_group()
    ag.add_argument("--restart-on-login", action="store_true",
                    help="reattach by pattern after login; --start may launch a new process")
    ag.add_argument("--persist", action="store_true", dest="restart_on_login",
                    help="deprecated alias for --restart-on-login")
    a.add_argument("--cwd")
    a.add_argument("--start")
    a.set_defaults(fn=cmd_attach)

    for nm, fn, hlp in [("status", cmd_status, "power/battery/temp/guards"),
                        ("list", cmd_list, "list active guards"),
                        ("restart-persisted", cmd_restart_persisted,
                         "restart or reattach configured jobs after login"),
                        ("resume", cmd_resume,
                         "compatibility alias for restart-persisted (does not restore RAM)"),
                        ("install-agent", cmd_install_agent, "restart configured jobs after reboot/login"),
                        ("uninstall-agent", cmd_uninstall_agent, "remove the login agent")]:
        s = sub.add_parser(nm, help=hlp)
        s.set_defaults(fn=fn)

    st = sub.add_parser("stop", help="stop supervising (unfreeze); --kill ends job")
    st.add_argument("name")
    st.add_argument("--kill", action="store_true")
    st.set_defaults(fn=cmd_stop)

    up = sub.add_parser("unpersist", help="forget a persisted job")
    up.add_argument("name")
    up.set_defaults(fn=cmd_unpersist)
    cf = sub.add_parser("config", help="show/init policy")
    cf.add_argument("--init", action="store_true")
    cf.set_defaults(fn=cmd_config)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "__supervise":   # internal
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
                    raise StateError(
                        "invalid launching CLI identity"
                    ) from exc
            return cmd_supervise(argv[1], excluded_identity)
        parser = build_parser()
        args = parser.parse_args(argv)
        if not getattr(args, "fn", None):
            return cmd_status(args)
        return args.fn(args)
    except (ConfigError, StateError) as exc:
        sys.stderr.write(f"train-guard: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
