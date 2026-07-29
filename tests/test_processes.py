from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, processes
from trainguard.model import ProcessIdentity
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
    ):
        self.pid = pid
        self.created = created
        self.command = command
        self.child_processes = children
        self.deny_creation_time = deny_creation_time
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
    assert "STATE_MIGRATION_REFUSED pid=222" in (
        app_paths.logs / "reused.guard.log"
    ).read_text(encoding="utf-8")
