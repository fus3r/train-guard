from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, processes
from trainguard.model import Action, ProcessIdentity
from trainguard.processes import ProcessController
from trainguard.state import JobSpec, JobStore, StateError


class FakeProcess:
    def __init__(
        self,
        pid: int,
        *,
        created: float,
        command: tuple[str, ...] = ("python", "train.py"),
        children: tuple["FakeProcess", ...] = (),
        deny_creation_time: bool = False,
        status: str = psutil.STATUS_RUNNING,
    ):
        self.pid = pid
        self.created = created
        self.command = command
        self.child_processes = children
        self.deny_creation_time = deny_creation_time
        self.current_status = status
        self.suspend_calls = 0
        self.resume_calls = 0
        self.affinity = [0, 1, 2, 3]
        self.affinity_calls = []
        self.ionice_value = SimpleNamespace(ioclass=2, value=4)
        self.ionice_calls = []
        self.priority = 32
        self.nice_calls = []
        self.info = {"cmdline": list(command)}

    def create_time(self):
        if self.deny_creation_time:
            raise psutil.AccessDenied(self.pid)
        return self.created

    def cmdline(self):
        return list(self.command)

    def children(self, recursive=False):
        assert recursive is True
        return list(self.child_processes)

    def status(self):
        return self.current_status

    def suspend(self):
        self.suspend_calls += 1
        self.current_status = psutil.STATUS_STOPPED

    def resume(self):
        self.resume_calls += 1
        self.current_status = psutil.STATUS_RUNNING

    def cpu_affinity(self, value=None):
        if value is None:
            return list(self.affinity)
        self.affinity = list(value)
        self.affinity_calls.append(list(value))

    def ionice(self, *value):
        if not value:
            return self.ionice_value
        self.ionice_calls.append(value)
        self.ionice_value = value

    def nice(self, value=None):
        if value is None:
            return self.priority
        self.nice_calls.append(value)
        self.priority = value


@pytest.mark.parametrize(
    ("created", "denied", "root_alive", "targeted"),
    [
        (100.0, False, True, [12]),
        (101.0, False, False, []),
        (100.0, True, True, []),
    ],
)
def test_root_resolution_distinguishes_same_reused_and_unverifiable_pid(
    monkeypatch,
    created,
    denied,
    root_alive,
    targeted,
):
    process = FakeProcess(
        12,
        created=created,
        deny_creation_time=denied,
    )
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: process)
    spec = JobSpec.launched(
        "training",
        ProcessIdentity(12, 100.0),
        Path("/tmp/training.log"),
    )

    snapshot = ProcessController().resolve(spec)

    assert snapshot.root_alive is root_alive
    assert [item.pid for item in snapshot.processes] == targeted


def test_match_resolution_excludes_launcher_and_supervisor_but_keeps_children(
    monkeypatch,
):
    pattern = "python"
    launcher = FakeProcess(
        20,
        created=200.0,
        command=("train-guard", "attach", "--match", pattern),
        deny_creation_time=True,
    )
    guard = FakeProcess(
        22,
        created=220.0,
        command=("python", "trainguard/cli.py", "__supervise", "training"),
    )
    child = FakeProcess(
        23,
        created=230.0,
        command=("python", "data-loader.py"),
    )
    worker = FakeProcess(
        21,
        created=210.0,
        command=("python", "train.py"),
        children=(guard, child),
    )
    monkeypatch.setattr(
        processes.psutil,
        "process_iter",
        lambda _attrs: [launcher, worker, guard],
    )
    controller = ProcessController(excluded_identities=(ProcessIdentity(20, 200.0),))

    snapshot = controller.resolve(JobSpec.attached_pattern("training", pattern))

    assert snapshot.root_alive is True
    assert [item.pid for item in snapshot.processes] == [21, 23]


def test_match_attach_passes_the_launcher_identity_to_its_supervisor(
    app_paths,
    monkeypatch,
):
    calls = []

    def capture(pid, label):
        created = 123.5 if label == "launcher" else float(pid)
        return ProcessIdentity(pid, created)

    def spawn(argv, logfile=None, cwd=None):
        calls.append(list(argv))
        return SimpleNamespace(pid=5001)

    monkeypatch.setattr(cli, "_capture_identity", capture)
    monkeypatch.setattr(cli, "_spawn_detached", spawn)
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_ready",
        lambda *_args, **_kwargs: None,
    )

    assert (
        cli.main(
            [
                "attach",
                "--match",
                "python train.py",
                "--name",
                "training",
            ]
        )
        == 0
    )
    assert calls[0][-2:] == [str(os.getpid()), "123.5"]


def test_match_attach_fails_before_spawn_when_launcher_is_unverifiable(
    app_paths,
    monkeypatch,
    capsys,
):
    spawned = []

    def capture(pid, label):
        if label == "launcher":
            raise StateError(f"cannot inspect launcher process {pid}")
        return ProcessIdentity(pid, float(pid))

    monkeypatch.setattr(cli, "_capture_identity", capture)
    monkeypatch.setattr(
        cli,
        "_spawn_detached",
        lambda *args, **kwargs: spawned.append(args),
    )

    result = cli.main(
        [
            "attach",
            "--match",
            "python train.py",
            "--name",
            "training",
        ]
    )

    assert result == 2
    assert spawned == []
    assert not JobStore(app_paths).spec_path("training").exists()
    assert "cannot inspect launcher process" in capsys.readouterr().err


def test_legacy_migration_refuses_a_process_born_after_its_metadata(
    app_paths,
    monkeypatch,
):
    store = JobStore(app_paths)
    processes_by_pid = {
        111: FakeProcess(111, created=100.0),
        222: FakeProcess(222, created=300.0),
    }
    monkeypatch.setattr(
        cli.psutil,
        "Process",
        lambda pid: processes_by_pid[pid],
    )

    safe = JobSpec(
        name="safe",
        mode="run",
        created_at="unknown",
        legacy_pid=111,
    )
    store.write_spec(safe)
    os.utime(store.spec_path("safe"), (200.0, 200.0))
    migrated = cli._migrate_legacy_identity(store, safe)

    assert migrated.root == ProcessIdentity(111, 100.0)
    assert migrated.legacy_pid is None
    assert store.read_spec("safe") == migrated

    reused = JobSpec(
        name="reused",
        mode="run",
        created_at="unknown",
        legacy_pid=222,
    )
    store.write_spec(reused)
    os.utime(store.spec_path("reused"), (200.0, 200.0))

    with pytest.raises(StateError, match="newer process"):
        cli._migrate_legacy_identity(store, reused)

    preserved = store.read_spec("reused")
    assert preserved.root is None
    assert preserved.legacy_pid == 222
    assert "STATE_MIGRATION_REFUSED pid=222" in (app_paths.logs / "reused.guard.log").read_text(
        encoding="utf-8"
    )


def test_only_owned_suspensions_are_reasserted_serialized_and_released(
    app_paths,
    monkeypatch,
):
    owned = FakeProcess(31, created=310.0)
    external = FakeProcess(
        32,
        created=320.0,
        status=psutil.STATUS_STOPPED,
    )
    by_pid = {process.pid: process for process in (owned, external)}
    monkeypatch.setattr(processes.psutil, "Process", lambda pid: by_pid[pid])
    controller = ProcessController("Other")

    first = controller.apply(Action.STOP, [owned, external])
    owned.current_status = psutil.STATUS_RUNNING  # An external SIGCONT.
    second = controller.apply(Action.STOP, [owned, external])

    payload = cli._runtime_payload("stop", False, [31, 32], controller)
    store = JobStore(app_paths)
    store.write_runtime("training", payload)
    runtime = store.read_runtime("training")
    assert runtime is not None
    recovered = ProcessController("Other")
    recovered.adopt_owned(
        ProcessIdentity.from_dict(value) for value in runtime["owned_suspensions"]
    )
    released = recovered.apply(Action.FULL, [])

    assert (first.suspended, second.suspended, released.resumed) == (1, 1, 1)
    assert owned.suspend_calls == 2
    assert owned.resume_calls == 1
    assert external.resume_calls == 0
    assert recovered.owned_suspensions == ()


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_tuning_is_restored_by_a_new_controller_after_target_becomes_empty(
    app_paths,
    monkeypatch,
    system,
):
    monkeypatch.setattr(processes.psutil, "IOPRIO_CLASS_IDLE", 3, raising=False)
    monkeypatch.setattr(
        processes.psutil,
        "IDLE_PRIORITY_CLASS",
        64,
        raising=False,
    )
    process = FakeProcess(41, created=410.0)
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: process)
    first = ProcessController(system)

    applied = first.apply(Action.GENTLE, [process])
    payload = cli._runtime_payload("gentle", False, [41], first)
    store = JobStore(app_paths)
    store.write_runtime(system.lower(), payload)
    runtime = store.read_runtime(system.lower())
    assert runtime is not None
    recovered = ProcessController(system)
    recovered.adopt_tuned(runtime["tuned_processes"])
    released = recovered.apply(Action.GENTLE, [])

    assert applied.tuned == 1
    assert released.restored == 1
    assert recovered.tuned_processes == ()
    if system == "Linux":
        assert process.affinity_calls == [[0, 2], [0, 1, 2, 3]]
        assert process.ionice_calls == [(3,), (2, 4)]
    else:
        assert process.nice_calls == [64, 32]


def test_failed_gentle_and_release_keep_only_recoverable_ownership(
    monkeypatch,
):
    denied = FakeProcess(51, created=510.0)

    def deny_priority(value=None):
        if value is None:
            return denied.priority
        raise psutil.AccessDenied(denied.pid)

    denied.nice = deny_priority
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: denied)
    controller = ProcessController("Windows")

    report = controller.apply(Action.GENTLE, [denied])

    assert report.access_denied == 1
    assert controller.tuned_processes == ()
    assert controller.release_owned().access_denied == 0

    unknown = FakeProcess(52, created=520.0)

    def unknown_affinity(value=None):
        if value is None:
            raise psutil.AccessDenied(unknown.pid)
        raise AssertionError("affinity with no captured baseline must not change")

    def unknown_io(*value):
        if not value:
            raise psutil.AccessDenied(unknown.pid)
        raise AssertionError("I/O priority with no captured baseline must not change")

    unknown.cpu_affinity = unknown_affinity
    unknown.ionice = unknown_io
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: unknown)
    controller = ProcessController("Linux")

    assert controller.apply(Action.GENTLE, [unknown]).tuned == 0
    assert controller.tuned_processes == ()

    partial = FakeProcess(53, created=530.0)

    def deny_io(*value):
        if not value:
            return partial.ionice_value
        raise psutil.AccessDenied(partial.pid)

    partial.ionice = deny_io
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: partial)
    controller = ProcessController("Linux")
    assert controller.apply(Action.GENTLE, [partial]).tuned == 1

    partial.cpu_affinity = lambda _value=None: (_ for _ in ()).throw(
        psutil.AccessDenied(partial.pid)
    )
    release = controller.release_owned()

    assert release.access_denied == 1
    assert len(controller.tuned_processes) == 1
