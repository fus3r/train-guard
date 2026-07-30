import signal
import subprocess
import sys
import threading
import time

import psutil
import pytest

from trainguard import cli
from trainguard.model import Action, Observation, PowerSource, ProcessIdentity, utc_now
from trainguard.processes import (
    ApplyReport,
    ProcessController,
    TargetSnapshot,
    process_identity,
)
from trainguard.state import (
    JobSpec,
    JobStore,
    PersistenceSpec,
    StateError,
    atomic_json_write,
)
from trainguard.supervisor import Supervisor


class MutableSensors:
    def __init__(self, temperature_c):
        self.temperature_c = temperature_c
        self.failure = None

    def sample(self):
        if self.failure is not None:
            raise self.failure
        return Observation(
            source=PowerSource.AC,
            percent=80.0,
            temperature_c=self.temperature_c,
            charging=False,
            observed_at=utc_now(),
        )


def _start_worker(seconds=30):
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"]
    )


def _wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition was not reached before timeout")


def _run_in_thread(supervisor):
    supervisor._install_signal_handlers = lambda: None
    result = []
    thread = threading.Thread(target=lambda: result.append(supervisor.run()))
    thread.start()
    return thread, result


def _resume_and_terminate(worker):
    try:
        psutil.Process(worker.pid).resume()
    except psutil.Error:
        pass
    if worker.poll() is None:
        worker.terminate()
    worker.wait(timeout=5)


def test_readiness_uses_popen_and_rechecks_after_observed_exit(
    app_paths,
    monkeypatch,
):
    store = JobStore(app_paths)
    expected = ProcessIdentity(123, 10.0)
    polls = []

    def short_lived_poll():
        polls.append(True)
        if len(polls) == 1:
            store.write_ready("quick", expected)
        return 0

    monkeypatch.setattr(
        cli,
        "identity_is_alive",
        lambda _identity: (_ for _ in ()).throw(
            AssertionError("the Popen handle must be authoritative")
        ),
    )
    cli._wait_for_supervisor_ready(
        store,
        "quick",
        expected,
        supervisor=type("Child", (), {"poll": staticmethod(short_lived_poll)})(),
        timeout=0.2,
    )
    assert store.read_ready("quick") is None

    def exited_before_capture(*_args, **_kwargs):
        store.write_ready("captured-from-handshake", expected)
        return type(
            "Child",
            (),
            {"pid": expected.pid, "poll": staticmethod(lambda: 0)},
        )()

    monkeypatch.setattr(cli, "_spawn_detached", exited_before_capture)
    monkeypatch.setattr(
        cli,
        "_capture_identity",
        lambda *_args: (_ for _ in ()).throw(StateError("process already exited")),
    )
    assert cli._start_supervisor(store, "captured-from-handshake") == expected

    dead = type("Child", (), {"poll": staticmethod(lambda: 2)})()
    with pytest.raises(StateError, match="exited before it reported ready"):
        cli._wait_for_supervisor_ready(
            store,
            "failed",
            expected,
            supervisor=dead,
            timeout=0.2,
        )


def test_failed_start_escalates_children_and_restores_previous_restart_spec(
    app_paths,
    monkeypatch,
    capsys,
):
    store = JobStore(app_paths)
    previous = PersistenceSpec(
        mode="run",
        name="rollback",
        cwd=str(app_paths.home),
        argv=("python", "old.py"),
    )
    store.write_persistence(previous)

    class Child:
        def __init__(self, pid):
            self.pid = pid
            self.calls = []

        def terminate(self):
            self.calls.append("terminate")

        def wait(self, timeout):
            self.calls.append(("wait", timeout))
            if self.calls.count(("wait", timeout)) == 1:
                raise subprocess.TimeoutExpired(["child"], timeout)

        def kill(self):
            self.calls.append("kill")

    job = Child(91)
    supervisor = Child(92)
    children = iter((job, supervisor))
    monkeypatch.setattr(
        cli,
        "_spawn_detached",
        lambda *_args, **_kwargs: next(children),
    )
    monkeypatch.setattr(
        cli,
        "_capture_identity",
        lambda pid, _label: ProcessIdentity(pid, float(pid)),
    )
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StateError("not ready")),
    )

    assert (
        cli.main(
            [
                "run",
                "--name",
                "rollback",
                "--restart-on-login",
                "--",
                "python",
                "new.py",
            ]
        )
        == 2
    )
    expected_calls = ["terminate", ("wait", 2), "kill", ("wait", 2)]
    assert supervisor.calls == expected_calls
    assert job.calls == expected_calls
    assert not store.spec_path("rollback").exists()
    assert not store.runtime_path("rollback").exists()
    assert store.read_persistence(store.persistence_path("rollback")) == previous
    assert "not ready" in capsys.readouterr().err


def test_stop_of_dead_guard_preserves_restart_spec(app_paths, capsys):
    store = JobStore(app_paths)
    store.write_spec(JobSpec.attached_pattern("stale", "python train.py"))
    store.write_guard("stale", ProcessIdentity(4_194_000, 1.0))
    persistence = PersistenceSpec(
        mode="attach",
        name="stale",
        cwd=str(app_paths.home),
        pattern="python train.py",
    )
    store.write_persistence(persistence)

    assert cli.main(["stop", "stale"]) == 2
    assert "not running" in capsys.readouterr().err
    assert store.read_persistence(store.persistence_path("stale")) == persistence
    assert not store.stop_path("stale").exists()


def test_soft_stop_resumes_worker_and_restores_gentle_state(app_paths):
    atomic_json_write(
        app_paths.config,
        {
            "poll": 0.1,
            "ac_band": "gentle",
        },
    )
    worker = _start_worker()
    process = psutil.Process(worker.pid)
    sensors = MutableSensors(30.0)
    identity = process_identity(process)
    spec = JobSpec.launched(
        "soft-stop",
        identity,
        app_paths.logs / "soft-stop.log",
    )

    class LifecycleController:
        def __init__(self):
            self.owned = False
            self.tuned = False

        @property
        def owned_suspensions(self):
            return (identity,) if self.owned else ()

        @property
        def tuned_processes(self):
            if not self.tuned:
                return ()
            return (
                {
                    **identity.to_dict(),
                    "system": "Test",
                    "priority": 1,
                },
            )

        def resolve(self, _spec):
            return TargetSnapshot((process,), True)

        def apply(self, action, _processes):
            if action is Action.GENTLE:
                self.tuned = True
                return ApplyReport(targeted=1, tuned=1)
            if action is Action.STOP and not self.owned:
                process.suspend()
                self.owned = True
                return ApplyReport(targeted=1, suspended=1)
            return ApplyReport(targeted=1)

        def release_owned(self):
            resumed = restored = 0
            if self.owned:
                process.resume()
                self.owned = False
                resumed = 1
            if self.tuned:
                self.tuned = False
                restored = 1
            return ApplyReport(
                targeted=1,
                resumed=resumed,
                restored=restored,
            )

    controller = LifecycleController()
    store = JobStore(app_paths)
    store.write_spec(spec)
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=sensors,
        controller=controller,
    )
    thread, result = _run_in_thread(supervisor)
    try:
        _wait_until(
            lambda: (
                bool(supervisor.controller.tuned_processes)
                and (store.read_runtime("soft-stop") or {}).get("state") == "gentle"
            )
        )
        sensors.temperature_c = 45.0
        _wait_until(
            lambda: (
                process.status() == psutil.STATUS_STOPPED
                and (store.read_runtime("soft-stop") or {}).get("state") == "stop"
            )
        )

        store.request_stop("soft-stop", False)
        thread.join(timeout=5)

        assert result == [0]
        assert worker.poll() is None
        _wait_until(lambda: process.status() != psutil.STATUS_STOPPED)
        assert supervisor.controller.owned_suspensions == ()
        assert supervisor.controller.tuned_processes == ()
        assert not store.spec_path("soft-stop").exists()
    finally:
        _resume_and_terminate(worker)


def test_signal_shutdown_releases_owned_suspension(app_paths):
    atomic_json_write(app_paths.config, {"poll": 0.1})
    worker = _start_worker()
    process = psutil.Process(worker.pid)
    identity = process_identity(process)
    spec = JobSpec.launched(
        "signal",
        identity,
        app_paths.logs / "signal.log",
    )
    store = JobStore(app_paths)
    store.write_spec(spec)
    process.suspend()
    _wait_until(lambda: process.status() == psutil.STATUS_STOPPED)
    store.write_runtime(
        "signal",
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "state": "stop",
            "cooling": True,
            "owned_suspensions": [identity.to_dict()],
            "tuned_processes": [],
            "pids": [worker.pid],
        },
    )
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=MutableSensors(None),
    )
    thread, result = _run_in_thread(supervisor)
    try:
        _wait_until(
            lambda: (
                (store.read_runtime("signal") or {}).get("decision", {}).get("reason")
                == "thermal_cooldown"
            )
        )
        supervisor._handle_signal(signal.SIGTERM, None)
        thread.join(timeout=5)

        assert result == [0]
        assert worker.poll() is None
        _wait_until(lambda: process.status() != psutil.STATUS_STOPPED)
        assert not store.runtime_path("signal").exists()
    finally:
        _resume_and_terminate(worker)


def test_root_exit_keeps_readiness_until_parent_consumes_it(app_paths):
    atomic_json_write(app_paths.config, {"poll": 0.1})
    worker = _start_worker(0.1)
    spec = JobSpec.launched(
        "quick",
        process_identity(psutil.Process(worker.pid)),
        app_paths.logs / "quick.log",
    )
    store = JobStore(app_paths)
    store.write_spec(spec)
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=MutableSensors(30.0),
    )
    thread, result = _run_in_thread(supervisor)

    worker.wait(timeout=5)
    thread.join(timeout=5)

    assert result == [0]
    assert store.read_ready("quick") == process_identity(psutil.Process())
    assert not store.spec_path("quick").exists()
    assert not store.runtime_path("quick").exists()
    assert store.read_guard("quick") is None
    store.clear_ready("quick")


def test_invalid_reload_keeps_policy_and_crash_leaves_recoverable_error(
    app_paths,
):
    atomic_json_write(
        app_paths.config,
        {
            "poll": 0.1,
            "temp_gentle_c": 39,
            "temp_pause_c": 40,
            "temp_resume_c": 35,
        },
    )
    worker = _start_worker()
    process = psutil.Process(worker.pid)
    sensors = MutableSensors(41.0)
    spec = JobSpec.launched(
        "crash",
        process_identity(process),
        app_paths.logs / "crash.log",
    )
    store = JobStore(app_paths)
    store.write_spec(spec)
    supervisor = Supervisor(app_paths, spec, sensors=sensors)
    thread, result = _run_in_thread(supervisor)
    try:
        _wait_until(lambda: process.status() == psutil.STATUS_STOPPED)
        app_paths.config.write_text('{"poll": "fast"}', encoding="utf-8")
        _wait_until(lambda: "config_error" in (store.read_runtime("crash") or {}))
        assert (store.read_runtime("crash") or {})["decision"]["action"] == "stop"

        sensors.failure = RuntimeError("sensor bug")
        thread.join(timeout=5)

        assert result == [1]
        runtime = store.read_runtime("crash")
        assert runtime is not None
        assert runtime["state"] == "error"
        assert "sensor bug" in runtime["error"]
        assert store.spec_path("crash").exists()
        assert store.read_guard("crash") is None
        assert worker.poll() is None
        _wait_until(lambda: process.status() != psutil.STATUS_STOPPED)
    finally:
        _resume_and_terminate(worker)


def test_orphan_recovery_keeps_denied_identity_for_retry(
    app_paths,
    monkeypatch,
):
    worker = _start_worker()
    process = psutil.Process(worker.pid)
    identity = process_identity(process)
    process.suspend()
    _wait_until(lambda: process.status() == psutil.STATUS_STOPPED)
    store = JobStore(app_paths)
    store.write_runtime(
        "orphan",
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "state": "stop",
            "cooling": True,
            "owned_suspensions": [identity.to_dict()],
            "tuned_processes": [],
            "pids": [worker.pid],
        },
    )
    store.write_guard("orphan", ProcessIdentity(4_194_000, 1.0))

    class DeniedController(ProcessController):
        def release_owned(self):
            return ApplyReport(targeted=1, access_denied=1)

    controllers = iter((DeniedController(), ProcessController()))
    monkeypatch.setattr(cli, "ProcessController", lambda: next(controllers))
    try:
        assert cli.main(["recover", "orphan"]) == 1
        runtime = store.read_runtime("orphan")
        assert runtime is not None
        assert runtime["state"] == "recovery_incomplete"
        assert runtime["owned_suspensions"] == [identity.to_dict()]
        assert store.read_guard("orphan") is None
        assert process.status() == psutil.STATUS_STOPPED

        assert cli.main(["recover", "orphan"]) == 0
        _wait_until(lambda: process.status() != psutil.STATUS_STOPPED)
        assert not store.runtime_path("orphan").exists()
    finally:
        _resume_and_terminate(worker)
