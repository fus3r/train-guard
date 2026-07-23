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
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.stderr.write("train-guard requires psutil:  pip install psutil\n")
    raise

__version__ = "0.2.0"
SYSTEM = platform.system()  # 'Darwin' | 'Linux' | 'Windows'
HOME = Path.home()
TG_HOME = Path(os.environ.get("TRAIN_GUARD_HOME", HOME / ".train-guard"))
RUNDIR, LOGDIR, PERSIST = TG_HOME / "run", TG_HOME / "logs", TG_HOME / "persist"
CONFIGF = TG_HOME / "config.json"

DEFAULTS = {
    "poll": 20,                  # seconds between checks
    "run_on_battery": False,     # False = pause when unplugged (no cycling/deep discharge)
    "battery_floor_pct": 30,     # if run_on_battery, stop below this %
    "battery_band": "gentle",    # on battery: 'gentle' | 'full'
    "ac_band": "full",           # on AC and cool: 'full' | 'gentle'
    "temp_gentle_c": 38,         # on AC, at/above this battery temp -> gentle
    "temp_pause_c": 42,          # at/above this -> pause to cool (hysteresis)
    "temp_resume_c": 36,         # resume from thermal pause once cooled to/below this
    "charge_cool_until_pct": 80, # while charging below this %...
    "temp_charge_gentle_c": 35,  # ...and >= this battery temp -> gentle (charge cool)
}

# Linux niceness cannot normally be raised back to its original value by an
# unprivileged process. Keep a reversible CPU-affinity snapshot instead.
_LINUX_AFFINITY = {}


def _ensure_dirs():
    for d in (RUNDIR, LOGDIR, PERSIST):
        d.mkdir(parents=True, exist_ok=True)


def load_config():
    cfg = dict(DEFAULTS)
    try:
        if CONFIGF.exists():
            user_cfg = json.loads(CONFIGF.read_text())
            # v0.1 called this threshold "ecore", but taskpolicy only supplies a
            # scheduling hint; it cannot pin work to Apple efficiency cores.
            if "temp_gentle_c" not in user_cfg and "temp_ecore_c" in user_cfg:
                user_cfg["temp_gentle_c"] = user_cfg["temp_ecore_c"]
            if "temp_charge_gentle_c" not in user_cfg and "temp_charge_ecore_c" in user_cfg:
                user_cfg["temp_charge_gentle_c"] = user_cfg["temp_charge_ecore_c"]
            user_cfg.pop("temp_ecore_c", None)
            user_cfg.pop("temp_charge_ecore_c", None)
            cfg.update(user_cfg)
    except Exception:
        pass
    return cfg


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# Power and battery readings.
def power_source():
    b = psutil.sensors_battery()
    if b is None:
        return "AC"            # desktop or unknown, so never pause
    return "AC" if b.power_plugged else "Battery"


def battery_pct():
    b = psutil.sensors_battery()
    return int(b.percent) if b else 100


def is_charging():
    b = psutil.sensors_battery()
    return bool(b and b.power_plugged and b.percent < 100)


def battery_temp_c():
    """Battery pack temperature in °C, or None if unavailable on this platform."""
    try:
        if SYSTEM == "Darwin":
            out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery", "-w0"],
                                 capture_output=True, text=True, timeout=5).stdout
            m = re.search(r'"Temperature"\s*=\s*(\d+)', out)
            if m:
                return int(m.group(1)) / 100.0
        elif SYSTEM == "Linux":
            temps = getattr(psutil, "sensors_temperatures", lambda: {})() or {}
            for name, entries in temps.items():
                if "bat" in name.lower():
                    for e in entries:
                        if e.current:
                            return float(e.current)
            for p in Path("/sys/class/power_supply").glob("BAT*/temp"):
                try:
                    return int(p.read_text().strip()) / 10.0
                except Exception:
                    pass
        # Windows usually needs vendor tools for battery temperature.
    except Exception:
        return None
    return None


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


# Policy decision.
class _Cool:
    on = False  # thermal cooldown hysteresis state (per supervisor process)


def decide(cfg):
    src, pct, temp, chg = power_source(), battery_pct(), battery_temp_c(), is_charging()
    tshow = "n/a" if temp is None else round(temp)
    sig = f"power={src} batt={pct}% temp={tshow}C charging={'yes' if chg else 'no'}"
    have_t = temp is not None

    if _Cool.on:
        if have_t and temp <= cfg["temp_resume_c"]:
            _Cool.on = False
        else:
            return "stop", sig
    if have_t and temp >= cfg["temp_pause_c"]:
        _Cool.on = True
        return "stop", sig

    if src == "Battery":
        if not cfg["run_on_battery"]:
            return "stop", sig
        if pct <= cfg["battery_floor_pct"]:
            return "stop", sig
        return ("gentle" if cfg["battery_band"] == "gentle" else "full"), sig

    # on AC
    if have_t and temp >= cfg["temp_gentle_c"]:
        return "gentle", sig
    if chg and pct < cfg["charge_cool_until_pct"] and have_t and temp >= cfg["temp_charge_gentle_c"]:
        return "gentle", sig
    return ("gentle" if cfg["ac_band"] == "gentle" else "full"), sig


# Process lookup.
def _is_guard_proc(p):
    try:
        return any(tok in " ".join(p.cmdline()) for tok in ("trainguard", "train_guard"))
    except Exception:
        return False


def _pattern_running(pat):
    for p in psutil.process_iter(["cmdline"]):
        try:
            if pat in " ".join(p.info["cmdline"] or []) and not _is_guard_proc(p):
                return True
        except Exception:
            continue
    return False


def _targets(meta):
    roots = []
    if meta["mode"] == "run":
        try:
            roots = [psutil.Process(meta["jobpid"])]
        except psutil.NoSuchProcess:
            return []
    else:
        pat = meta["pattern"]
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(p.info["cmdline"] or [])
            except Exception:
                continue
            if pat in cl and p.pid != os.getpid() and not _is_guard_proc(p):
                roots.append(p)
    out = {}
    for r in roots:
        try:
            out[r.pid] = r
            for c in r.children(recursive=True):
                out[c.pid] = c
        except psutil.NoSuchProcess:
            continue
    return list(out.values())


# Filesystem helpers.
def _read_pid(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _alive(pid):
    return pid is not None and psutil.pid_exists(pid)


def _glog(name, msg):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(LOGDIR / f"{name}.guard.log", "a") as f:
        f.write(f"[guard] {_now()} {msg}\n")


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


def _supervisor_argv(meta_path):
    return [sys.executable, str(Path(__file__).resolve()), "__supervise", str(meta_path)]


# Supervisor loop.
def cmd_supervise(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    name = meta["name"]
    detail = f"pattern={meta.get('pattern')}" if meta["mode"] == "attach" else f"jobpid={meta.get('jobpid')}"
    _glog(name, f"START mode={meta['mode']} {detail}")
    last, miss = None, 0
    while True:
        cfg = load_config()
        stopf = RUNDIR / f"{name}.stop"
        if stopf.exists():
            how = stopf.read_text().strip()
            procs = _targets(meta)
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
            for f in (stopf, RUNDIR / f"{name}.meta.json", RUNDIR / f"{name}.gpid"):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
            return
        if meta["mode"] == "run" and not psutil.pid_exists(meta["jobpid"]):
            _glog(name, "job exited; guard done")
            for f in (RUNDIR / f"{name}.meta.json", RUNDIR / f"{name}.gpid"):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
            return
        procs = _targets(meta)
        if meta["mode"] == "attach" and not procs:
            miss += 1
            if miss % 15 == 1:
                _glog(name, "no process matches pattern yet (waiting)")
            time.sleep(cfg["poll"])
            continue
        miss = 0
        decision, sig = decide(cfg)
        _apply(decision, procs)
        line = f"{decision} | {sig}"
        if line != last:
            _glog(name, f"-> {decision}   ({sig})")
            last = line
        time.sleep(cfg["poll"])


# CLI commands.
def _policy_line(cfg):
    batt = "pause" if not cfg["run_on_battery"] else f"{cfg['battery_band']} (floor {cfg['battery_floor_pct']}%)"
    return f"AC={cfg['ac_band']}  battery={batt}"


def _restart_on_login(args):
    """Read the new flag while accepting programmatic v0.1 callers."""
    return bool(getattr(args, "restart_on_login", getattr(args, "persist", False)))


def cmd_run(args):
    _ensure_dirs()
    cfg = load_config()
    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.stderr.write("usage: train-guard run [--name N] [--restart-on-login] [--cwd DIR] -- <command...>\n")
        return 2
    name = args.name or f"job-{time.strftime('%H%M%S')}"
    metaf, gpidf = RUNDIR / f"{name}.meta.json", RUNDIR / f"{name}.gpid"
    if metaf.exists() and _alive(_read_pid(gpidf)):
        sys.stderr.write(f"a guard named '{name}' is already active\n")
        return 1
    cwd = args.cwd or os.getcwd()
    restart = _restart_on_login(args)
    if restart:
        (PERSIST / f"{name}.job").write_text(json.dumps(
            {"mode": "run", "name": name, "cwd": cwd, "argv": cmd,
             "reboot_semantics": "restart-command"}))
    log = LOGDIR / f"{name}.log"
    job = _spawn_detached(cmd, logfile=log, cwd=cwd)
    metaf.write_text(json.dumps({"mode": "run", "name": name, "jobpid": job.pid, "log": str(log)}))
    sup = _spawn_detached(_supervisor_argv(metaf))
    gpidf.write_text(str(sup.pid))
    tag = "  (restarts at next login)" if restart else ""
    print(f"[train-guard] '{name}': job pid={job.pid}  guard pid={sup.pid}  out={log}{tag}")
    print(f"[train-guard] policy: {_policy_line(cfg)}   |   train-guard status")
    return 0


def cmd_attach(args):
    _ensure_dirs()
    cfg = load_config()
    if not args.match and not args.pid:
        sys.stderr.write('usage: train-guard attach --match "<pattern>" [--name N] [--restart-on-login --start "<cmd>"]   (or --pid PID)\n')
        return 2
    restart = _restart_on_login(args)
    if args.pid and restart:
        sys.stderr.write("attach: a PID cannot survive a reboot; use --match with --restart-on-login and optionally --start\n")
        return 2
    name = args.name or f"attach-{time.strftime('%H%M%S')}"
    metaf, gpidf = RUNDIR / f"{name}.meta.json", RUNDIR / f"{name}.gpid"
    if metaf.exists() and _alive(_read_pid(gpidf)):
        sys.stderr.write(f"a guard named '{name}' is already active\n")
        return 1
    cwd = args.cwd or os.getcwd()
    if args.pid:
        meta = {"mode": "run", "name": name, "jobpid": int(args.pid)}
    else:
        meta = {"mode": "attach", "name": name, "pattern": args.match}
        if restart:
            (PERSIST / f"{name}.job").write_text(json.dumps(
                {"mode": "attach", "name": name, "cwd": cwd, "pattern": args.match,
                 "start": args.start or "", "reboot_semantics": "restart-or-reattach"}))
    metaf.write_text(json.dumps(meta))
    sup = _spawn_detached(_supervisor_argv(metaf))
    gpidf.write_text(str(sup.pid))
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
    _ensure_dirs()
    cfg = load_config()
    src, pct, temp, chg = power_source(), battery_pct(), battery_temp_c(), is_charging()
    print("power / battery")
    tshow = "n/a (not exposed on this OS)" if temp is None else f"{temp:.1f}°C"
    print(f"  source: {src}   charge: {pct}%   charging: {'yes' if chg else 'no'}   pack temp: {tshow}")
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
    print(f"  thermal: gentle >= {cfg['temp_gentle_c']}°C  pause >= {cfg['temp_pause_c']}°C  "
          f"resume <= {cfg['temp_resume_c']}°C   charge cool: <{cfg['charge_cool_until_pct']}% and >= {cfg['temp_charge_gentle_c']}°C")
    print()
    print("active guards")
    metas = sorted(RUNDIR.glob("*.meta.json"))
    if not metas:
        print("  (none)")
    for mf in metas:
        try:
            meta = json.loads(mf.read_text())
        except Exception:
            continue
        name = meta["name"]
        gp = _read_pid(RUNDIR / f"{name}.gpid")
        alive = "running" if _alive(gp) else "dead"
        procs = _targets(meta)
        if procs:
            st = "PAUSED/frozen" if any(p.status() == psutil.STATUS_STOPPED for p in procs) else "running"
            worker = f"{st}  ({len(procs)} proc)"
        else:
            worker = "none right now"
        print(f"  {name} [{meta['mode']}]  guard={alive} (pid {gp})   worker: {worker}")
    print()
    print("restart after reboot (starts a new process; RAM state cannot survive a reboot)")
    print(f"  login agent: {'INSTALLED' if _agent_installed() else 'not installed  ->  train-guard install-agent'}")
    pj = sorted(PERSIST.glob("*.job"))
    print("  restart specs: " + (", ".join(p.stem for p in pj) if pj else "none (add --restart-on-login to run/attach)"))
    return 0


def cmd_list(args):
    metas = sorted(RUNDIR.glob("*.meta.json"))
    print("\n".join(m.stem.replace(".meta", "") for m in metas) if metas else "(no active guards)")
    return 0


def cmd_stop(args):
    name = args.name
    if not (RUNDIR / f"{name}.meta.json").exists():
        sys.stderr.write(f"no active guard named '{name}' (train-guard list)\n")
        return 1
    (RUNDIR / f"{name}.stop").write_text("kill" if args.kill else "soft")
    try:
        (PERSIST / f"{name}.job").unlink()
    except FileNotFoundError:
        pass
    print(f"[train-guard] stop requested for '{name}'" + (" (--kill the job)" if args.kill else "") +
          "; unfreezes & detaches within one poll. (login restart removed)")
    return 0


def cmd_config(args):
    _ensure_dirs()
    print(f"policy: {CONFIGF}\n")
    if CONFIGF.exists():
        print(CONFIGF.read_text())
    else:
        print(json.dumps(DEFAULTS, indent=2))
        print("\n(no config.json yet; defaults shown. run `train-guard config --init` to write it)")
    if getattr(args, "init", False):
        CONFIGF.write_text(json.dumps(DEFAULTS, indent=2))
        print(f"\nwrote {CONFIGF}")
    return 0


def cmd_restart_persisted(args):
    """Recreate configured jobs after login.

    This deliberately does not claim process-state restoration: an OS reboot
    destroys RAM, so a ``run`` spec starts the command as a fresh process. Jobs
    that can checkpoint may continue from their own last durable checkpoint.
    """
    _ensure_dirs()
    for jf in sorted(PERSIST.glob("*.job")):
        try:
            j = json.loads(jf.read_text())
        except Exception:
            continue
        name = j.get("name")
        if not name:
            continue
        if (RUNDIR / f"{name}.meta.json").exists() and _alive(_read_pid(RUNDIR / f"{name}.gpid")):
            print(f"[restart] '{name}' already active; skipping")
            continue
        if j["mode"] == "run":
            cmd_run(argparse.Namespace(name=name, restart_on_login=True,
                                       cwd=j.get("cwd"), cmd=j["argv"]))
        else:
            pat, start = j["pattern"], j.get("start") or ""
            if start and not _pattern_running(pat):
                shell = ["cmd", "/c", start] if os.name == "nt" else ["/bin/sh", "-c", start]
                _spawn_detached(shell, logfile=LOGDIR / f"{name}.start.log", cwd=j.get("cwd"))
                print(f"[restart] started job for '{name}': {start}")
            cmd_attach(argparse.Namespace(name=name, restart_on_login=True, cwd=j.get("cwd"),
                                          match=pat, pid=None, start=start))
    print("[restart] done (commands were restarted or reattached; no RAM state was restored)")
    return 0


# Compatibility for login agents installed by v0.1.
cmd_resume = cmd_restart_persisted


def cmd_unpersist(args):
    for ext in (".job",):
        try:
            (PERSIST / f"{args.name}{ext}").unlink()
        except FileNotFoundError:
            pass
    print(f"[train-guard] '{args.name}' will no longer restart or reattach at login")
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
    a.add_argument("--name"); a.add_argument("--match"); a.add_argument("--pid")
    ag = a.add_mutually_exclusive_group()
    ag.add_argument("--restart-on-login", action="store_true",
                    help="reattach by pattern after login; --start may launch a new process")
    ag.add_argument("--persist", action="store_true", dest="restart_on_login",
                    help="deprecated alias for --restart-on-login")
    a.add_argument("--cwd"); a.add_argument("--start")
    a.set_defaults(fn=cmd_attach)

    for nm, fn, hlp in [("status", cmd_status, "power/battery/temp/guards"),
                        ("list", cmd_list, "list active guards"),
                        ("restart-persisted", cmd_restart_persisted,
                         "restart or reattach configured jobs after login"),
                        ("resume", cmd_resume,
                         "compatibility alias for restart-persisted (does not restore RAM)"),
                        ("install-agent", cmd_install_agent, "restart configured jobs after reboot/login"),
                        ("uninstall-agent", cmd_uninstall_agent, "remove the login agent")]:
        s = sub.add_parser(nm, help=hlp); s.set_defaults(fn=fn)

    st = sub.add_parser("stop", help="stop supervising (unfreeze); --kill ends job")
    st.add_argument("name"); st.add_argument("--kill", action="store_true"); st.set_defaults(fn=cmd_stop)

    up = sub.add_parser("unpersist", help="forget a persisted job"); up.add_argument("name"); up.set_defaults(fn=cmd_unpersist)
    cf = sub.add_parser("config", help="show/init policy"); cf.add_argument("--init", action="store_true"); cf.set_defaults(fn=cmd_config)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "__supervise":   # internal
        return cmd_supervise(argv[1])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        return cmd_status(args)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
