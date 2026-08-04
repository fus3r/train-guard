from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, processes
from trainguard import state as state_module
from trainguard.model import Action, Observation, PowerSource, ProcessIdentity
from trainguard.processes import ApplyReport, ProcessController, TargetSnapshot, process_identity
from trainguard.state import JobSpec, JobStore, PersistenceSpec, StateError, atomic_json_write
from trainguard.supervisor import Supervisor, run_supervisor

EXAMPLE_TRACE = Path(__file__).parents[1] / "examples" / "power-trace.jsonl"
CANDIDATE_POLICY = Path(__file__).parents[1] / "examples" / "battery-enabled-policy.json"


class MutableSensors:
    def __init__(self, *, warning: str = "") -> None:
        self.warning = warning

    def sample(self) -> Observation:
        return Observation(
            source=PowerSource.AC,
            percent=80.0,
            temperature_c=30.0,
            charging=False,
            observed_at="2026-08-04T12:00:00Z",
            warnings=((self.warning,) if self.warning else ()),
        )


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def _run_in_thread(supervisor: Supervisor) -> tuple[threading.Thread, list[int]]:
    supervisor._install_signal_handlers = lambda: None  # type: ignore[method-assign]
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(supervisor.run()))
    thread.start()
    return thread, result


def test_simulate_human_reports_preserve_the_hand_checked_replay(capsys):
    assert cli.main(["simulate", str(EXAMPLE_TRACE), "--transition-limit", "1"]) == 0
    replay = capsys.readouterr().out
    assert "full         600s" in replay
    assert "gentle       300s" in replay
    assert "stop         900s" in replay
    assert "... 6 more" in replay

    assert (
        cli.main(
            [
                "simulate",
                str(EXAMPLE_TRACE),
                "--compare-config",
                str(CANDIDATE_POLICY),
                "--transition-limit",
                "1",
            ]
        )
        == 0
    )
    comparison = capsys.readouterr().out
    assert "candidate minus baseline" in comparison
    assert "action disagreement: 300s" in comparison
    assert "stop        -300s" in comparison
    assert "stop/battery_disabled -> gentle/battery_policy" in comparison


@pytest.mark.parametrize(
    "arguments",
    (
        ("simulate", str(EXAMPLE_TRACE), "--transition-limit", "0"),
        ("simulate", str(EXAMPLE_TRACE), "--config", "missing.json"),
        ("simulate", str(EXAMPLE_TRACE), "--compare-config", "missing.json"),
    ),
)
def test_simulate_rejects_invalid_cli_inputs(arguments, capsys):
    assert cli.main(list(arguments)) == 2
    assert capsys.readouterr().err


def test_attach_waiting_reconciles_an_empty_target(app_paths):
    atomic_json_write(app_paths.config, {"poll": 0.1})
    spec = JobSpec.attached_pattern("waiting", "pattern-that-does-not-exist")
    store = JobStore(app_paths)
    store.write_spec(spec)

    class TrackingController:
        owned_suspensions: tuple[ProcessIdentity, ...] = ()
        tuned_processes: tuple[dict[str, object], ...] = ()

        def __init__(self) -> None:
            self.actions: list[tuple[Action, tuple[object, ...]]] = []

        def adopt_owned(self, _identities) -> None:
            pass

        def adopt_tuned(self, _values) -> None:
            pass

        def resolve(self, _spec) -> TargetSnapshot:
            return TargetSnapshot((), False)

        def apply(self, action, processes) -> ApplyReport:
            self.actions.append((action, tuple(processes)))
            return ApplyReport(targeted=0, resumed=1, restored=1)

        def release_owned(self) -> ApplyReport:
            return ApplyReport(targeted=0)

    controller = TrackingController()
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=MutableSensors(),
        controller=controller,  # type: ignore[arg-type]
    )
    thread, result = _run_in_thread(supervisor)
    try:
        _wait_until(lambda: (store.read_runtime("waiting") or {}).get("state") == "waiting")
        runtime = store.read_runtime("waiting")
        assert runtime is not None
        assert runtime["process_report"]["restored"] == 1
        assert supervisor.journal.read()[-1]["event"] == "waiting"
    finally:
        store.request_stop("waiting", False)
        thread.join(timeout=5)

    assert result == [0]
    assert controller.actions
    assert all(action is Action.FULL and not processes for action, processes in controller.actions)


def test_supervisor_reports_sensor_warning_and_valid_reload(app_paths):
    atomic_json_write(app_paths.config, {"poll": 0.1, "ac_band": "full"})
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    process = psutil.Process(worker.pid)
    spec = JobSpec.launched("reload", process_identity(process), app_paths.logs / "reload.log")
    store = JobStore(app_paths)
    store.write_spec(spec)
    sensors = MutableSensors(warning="pack sensor is intermittent")
    supervisor = Supervisor(app_paths, spec, sensors=sensors)
    thread, result = _run_in_thread(supervisor)

    try:
        _wait_until(lambda: (store.read_runtime("reload") or {}).get("state") == "full")
        atomic_json_write(app_paths.config, {"poll": 0.1, "ac_band": "gentle"})
        _wait_until(lambda: (store.read_runtime("reload") or {}).get("state") == "gentle")
        store.request_stop("reload", False)
        thread.join(timeout=5)
        events = [event["event"] for event in supervisor.journal.read()]
        assert "sensor_warning" in events
        assert "config_loaded" in events
        assert result == [0]
    finally:
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=5)


def test_incomplete_kill_retains_recovery_state(app_paths):
    class BrokenProcess:
        pid = 321

        def terminate(self) -> None:
            raise psutil.AccessDenied(pid=self.pid)

    class DeniedController:
        owned_suspensions: tuple[ProcessIdentity, ...] = ()
        tuned_processes: tuple[dict[str, object], ...] = ()

        def adopt_owned(self, _identities) -> None:
            pass

        def adopt_tuned(self, _values) -> None:
            pass

        def resolve(self, _spec) -> TargetSnapshot:
            return TargetSnapshot((BrokenProcess(),), True)  # type: ignore[arg-type]

        def release_owned(self) -> ApplyReport:
            return ApplyReport(targeted=1, access_denied=1)

    spec = JobSpec.launched("denied", ProcessIdentity(123, 1.0), app_paths.logs / "denied.log")
    store = JobStore(app_paths)
    store.write_spec(spec)
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=MutableSensors(),
        controller=DeniedController(),  # type: ignore[arg-type]
    )

    assert supervisor._finish_stop("kill") == 1
    runtime = store.read_runtime("denied")
    assert runtime is not None
    assert runtime["state"] == "stop_incomplete"
    assert runtime["process_report"]["access_denied"] == 1
    assert supervisor.journal.read()[-1]["event"] == "stop_incomplete"
    supervisor._handle_signal(999, None)
    assert supervisor._shutdown_signal == "999"


def test_run_supervisor_reports_a_missing_job(app_paths):
    assert run_supervisor(app_paths, "missing") == 2
    events = (app_paths.logs / "missing.events.jsonl").read_text(encoding="utf-8")
    assert "startup_failed" in events


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "JSON object"),
        (
            {"schema_version": True, "name": "job", "mode": "run", "jobpid": 1},
            "schema version",
        ),
        ({"mode": "run", "jobpid": 1}, "missing name"),
        ({"name": "job", "jobpid": 1}, "missing mode"),
        ({"name": 1, "mode": "run", "jobpid": 1}, "name must be a string"),
        ({"name": "job", "mode": 1, "jobpid": 1}, "mode must be a string"),
        ({"name": "job", "mode": "other", "jobpid": 1}, "unsupported job mode"),
        (
            {"name": "job", "mode": "run", "jobpid": 1, "created_at": ""},
            "created_at",
        ),
        ({"name": "job", "mode": "run", "jobpid": True}, "positive integer jobpid"),
        (
            {
                "name": "job",
                "mode": "run",
                "jobpid": 1,
                "job_create_time": 0,
            },
            "invalid root process identity",
        ),
        ({"name": "job", "mode": "attach", "pattern": " "}, "non-empty pattern"),
        (
            {"name": "job", "mode": "attach", "pattern": "python", "log": ""},
            "log must be a non-empty string",
        ),
    ),
)
def test_job_metadata_schema_rejects_ambiguous_records(value, message):
    with pytest.raises(StateError, match=message):
        JobSpec.from_dict(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "JSON object"),
        (
            {
                "schema_version": False,
                "mode": "run",
                "name": "job",
                "cwd": "/tmp",
                "argv": ["python"],
            },
            "schema version",
        ),
        ({"name": "job", "cwd": "/tmp"}, "missing mode"),
        ({"mode": "run", "cwd": "/tmp"}, "missing name"),
        ({"mode": "run", "name": "job"}, "missing cwd"),
        (
            {"mode": "run", "name": 1, "cwd": "/tmp", "argv": ["python"]},
            "name must be a string",
        ),
        (
            {"mode": "run", "name": "job", "cwd": "", "argv": ["python"]},
            "cwd must be a non-empty string",
        ),
        (
            {"mode": "run", "name": "job", "cwd": "/tmp", "argv": []},
            "non-empty argv",
        ),
        (
            {"mode": "attach", "name": "job", "cwd": "/tmp", "pattern": " "},
            "non-empty pattern",
        ),
        (
            {
                "mode": "attach",
                "name": "job",
                "cwd": "/tmp",
                "pattern": "python",
                "start": 1,
            },
            "start must be a string or null",
        ),
        (
            {"mode": "other", "name": "job", "cwd": "/tmp"},
            "unsupported persistence mode",
        ),
    ),
)
def test_persistence_schema_rejects_ambiguous_records(value, message):
    with pytest.raises(StateError, match=message):
        PersistenceSpec.from_dict(value)  # type: ignore[arg-type]


def _runtime(**overrides):
    value = {
        "schema_version": 1,
        "updated_at": "2026-08-04T12:00:00Z",
        "state": "full",
        "cooling": False,
        "owned_suspensions": [],
        "tuned_processes": [],
        "pids": [],
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "expected a JSON object"),
        (_runtime(schema_version=True), "runtime schema version"),
        (_runtime(updated_at=""), "updated_at"),
        (_runtime(state=""), "runtime state"),
        (_runtime(cooling=0), "runtime cooling"),
        (_runtime(owned_suspensions={}), "owned_suspensions must be a list"),
        (_runtime(tuned_processes={}), "tuned_processes must be a list"),
        (_runtime(owned_suspensions=["bad"]), r"owned_suspensions\[0\].*object"),
        (_runtime(owned_suspensions=[{"pid": 1}]), "invalid process identity"),
        (
            _runtime(tuned_processes=[{"pid": 1, "create_time": 1.0}]),
            "system must be a non-empty string",
        ),
        (
            _runtime(
                tuned_processes=[
                    {"pid": 1, "create_time": 1.0, "system": "Linux", "affinity": [True]}
                ]
            ),
            "affinity must be a list",
        ),
        (
            _runtime(
                tuned_processes=[
                    {"pid": 1, "create_time": 1.0, "system": "Linux", "ionice_class": 3}
                ]
            ),
            "must appear together",
        ),
        (
            _runtime(
                tuned_processes=[
                    {
                        "pid": 1,
                        "create_time": 1.0,
                        "system": "Windows",
                        "priority": "idle",
                    }
                ]
            ),
            "priority must be an integer",
        ),
        (_runtime(pids=[True]), "positive integers"),
        (_runtime(observation=[]), "observation must be an object"),
        (_runtime(error=""), "error must be a non-empty string"),
    ),
)
def test_runtime_schema_rejects_ambiguous_recovery_records(tmp_path, value, message):
    with pytest.raises(StateError, match=message):
        state_module._validate_runtime_state(tmp_path / "runtime.json", value)


def test_process_identity_helpers_fail_closed(monkeypatch):
    identity = ProcessIdentity(123, 10.0)
    monkeypatch.setattr(
        processes.psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.AccessDenied(123)),
    )
    assert processes.identity_is_alive(identity) is True

    missing = SimpleNamespace(
        pid=124,
        create_time=lambda: (_ for _ in ()).throw(psutil.NoSuchProcess(124)),
    )
    assert processes._same_process(missing, ProcessIdentity(124, 10.0)) is False

    broken_guard = SimpleNamespace(
        pid=999_999,
        cmdline=lambda: (_ for _ in ()).throw(psutil.AccessDenied(999_999)),
    )
    assert processes.is_guard_process(broken_guard) is False


@pytest.mark.parametrize(
    ("outcome", "denied", "gone"),
    (
        ("missing", 0, 1),
        ("denied", 1, 0),
        ("reused", 0, 1),
        ("unverifiable", 1, 0),
    ),
)
def test_release_owned_classifies_identity_failures(monkeypatch, outcome, denied, gone):
    identity = ProcessIdentity(123, 10.0)
    controller = ProcessController("Other")
    controller.adopt_owned([identity])

    created = 11.0 if outcome == "reused" else None

    class Candidate:
        pid = 123

        def create_time(self):
            if created is None:
                raise psutil.AccessDenied(123)
            return created

    def replacement(_pid):
        if outcome == "missing":
            raise psutil.NoSuchProcess(123)
        if outcome == "denied":
            raise psutil.AccessDenied(123)
        return Candidate()

    monkeypatch.setattr(processes.psutil, "Process", replacement)
    report = controller.release_owned()
    assert (report.access_denied, report.gone) == (denied, gone)
    assert bool(controller.owned_suspensions) is bool(denied)


def test_human_status_and_doctor_show_a_stale_guard(app_paths, monkeypatch, capsys):
    store = JobStore(app_paths)
    store.write_spec(JobSpec.attached_pattern("stale", "python train.py"))
    store.write_guard("stale", ProcessIdentity(4_194_000, 1.0))
    store.write_runtime("stale", _runtime(state="waiting"))
    observation = MutableSensors(warning="pack sensor unavailable").sample()
    monkeypatch.setattr(cli, "SensorReader", lambda: SimpleNamespace(sample=lambda: observation))
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)

    assert cli.main(["status"]) == 0
    status = capsys.readouterr().out
    assert "stale [attach]" in status
    assert "guard=dead" in status
    assert "state=waiting" in status

    assert cli.main(["doctor"]) == 1
    doctor = capsys.readouterr().out
    assert "FAIL supervisors" in doctor
    assert "note sensors" in doctor


def test_empty_events_and_agent_wrappers_have_explicit_output(
    app_paths,
    monkeypatch,
    capsys,
):
    assert cli.main(["events", "empty"]) == 0
    assert "(no events)" in capsys.readouterr().out

    monkeypatch.setattr(cli, "install_agent", lambda *_args: app_paths.home / "agent")
    monkeypatch.setattr(cli, "uninstall_agent", lambda *_args: None)
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    assert cli.main(["install-agent"]) == 0
    assert "login restart agent installed" in capsys.readouterr().out
    assert cli.main(["uninstall-agent"]) == 0
    assert "login agent removed" in capsys.readouterr().out


def test_restart_attach_starts_then_reattaches(app_paths, monkeypatch, capsys):
    store = JobStore(app_paths)
    store.write_persistence(
        PersistenceSpec(
            mode="attach",
            name="attach",
            cwd=str(app_paths.home),
            pattern="python worker.py",
            start="python worker.py",
        )
    )
    started = SimpleNamespace(pid=91)
    attached = []
    monkeypatch.setattr(cli, "_pattern_running", lambda _pattern: False)
    monkeypatch.setattr(cli, "_spawn_detached", lambda *_args, **_kwargs: started)
    monkeypatch.setattr(cli, "cmd_attach", lambda args: attached.append(args) or 0)

    assert cli.main(["restart-persisted"]) == 0
    assert attached[0].match == "python worker.py"
    assert "started job for 'attach'" in capsys.readouterr().out


def test_internal_supervisor_dispatch_validates_identity(monkeypatch, capsys):
    assert cli.main(["__supervise"]) == 2
    assert "invalid internal supervisor arguments" in capsys.readouterr().err
    assert cli.main(["__supervise", "job", "bad", "identity"]) == 2
    assert "invalid launching CLI identity" in capsys.readouterr().err

    captured = []
    monkeypatch.setattr(
        cli,
        "run_supervisor",
        lambda _paths, name, identity: captured.append((name, identity)) or 7,
    )
    assert cli.main(["__supervise", "job", "123", "10.5"]) == 7
    assert captured == [("job", ProcessIdentity(123, 10.5))]


def test_legacy_pid_helpers_reject_unreadable_values(tmp_path, monkeypatch):
    pid_file = tmp_path / "guard.pid"
    pid_file.write_text("not-a-pid", encoding="utf-8")
    assert cli._read_pid(pid_file) is None
    assert cli._read_pid(tmp_path / "missing.pid") is None
    monkeypatch.setattr(cli.psutil, "pid_exists", lambda pid: pid == 123)
    assert cli._alive(123) is True
    assert cli._alive(None) is False
