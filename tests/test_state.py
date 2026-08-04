from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainguard import cli, state
from trainguard.model import Observation, PowerSource, ProcessIdentity
from trainguard.state import (
    AppPaths,
    JobSpec,
    JobStore,
    PersistenceSpec,
    StateError,
    atomic_json_write,
    read_json,
    validate_job_name,
)


def _runtime(**overrides):
    value = {
        "schema_version": 1,
        "updated_at": "2026-07-28T12:00:00Z",
        "state": "stop",
        "cooling": True,
        "owned_suspensions": [],
        "tuned_processes": [],
        "pids": [],
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--name", "../outside", "--", "python", "train.py"],
        ["run", "--name", "con", "--", "python", "train.py"],
        ["run", "--name", "x" * 65, "--", "python", "train.py"],
        ["attach", "--name", "training", "--match", "   "],
    ],
)
def test_unsafe_names_and_blank_patterns_fail_before_spawn(argv, monkeypatch, capsys):
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda *args, **kwargs: spawned.append(args))

    assert cli.main(argv) == 2
    assert spawned == []
    assert capsys.readouterr().err
    assert validate_job_name("model.v2-run") == "model.v2-run"


def test_state_home_is_absolute_and_exported_for_detached_children(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIN_GUARD_HOME", "relative-state")

    paths = cli._paths()

    assert paths.home == tmp_path / "relative-state"
    assert paths.home.is_absolute()
    assert os.environ["TRAIN_GUARD_HOME"] == str(paths.home)

    monkeypatch.setenv("TRAIN_GUARD_HOME", "")
    assert AppPaths.from_environment().home == (Path.home() / ".train-guard").resolve()


def test_job_guard_and_persistence_records_are_separate_and_filename_bound(app_paths):
    store = JobStore(app_paths)
    root = ProcessIdentity(123, 456.75)
    spec = JobSpec.launched("training", root, app_paths.logs / "training.log")
    persistence = PersistenceSpec(
        mode="run",
        name="training",
        cwd="/tmp/work",
        argv=("python", "train.py"),
    )

    store.write_spec(spec)
    store.write_guard("training", ProcessIdentity(456, 789.25))
    store.write_persistence(persistence)

    assert store.read_spec("training") == spec
    assert store.read_guard("training") == ProcessIdentity(456, 789.25)
    assert store.read_persistence(store.persistence_path("training")) == persistence
    assert store.spec_path("training") != store.guard_path("training")

    store.spec_path("training").replace(store.spec_path("claimed"))
    with pytest.raises(StateError, match="metadata name"):
        store.read_spec("claimed")
    with pytest.raises(StateError, match="schema version"):
        JobSpec.from_dict(
            {
                "schema_version": 2,
                "name": "training",
                "mode": "run",
                "jobpid": 123,
            }
        )
    with pytest.raises(StateError, match="schema version"):
        PersistenceSpec.from_dict(
            {
                "schema_version": 2,
                "mode": "run",
                "name": "training",
                "cwd": "/tmp",
                "argv": ["python"],
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        {"pid": True, "create_time": 1.0},
        {"pid": 0, "create_time": 1.0},
        {"pid": 1, "create_time": float("nan")},
        {"pid": 1, "create_time": float("inf")},
        {"pid": 1, "create_time": 0},
    ],
)
def test_process_identity_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        ProcessIdentity.from_dict(value)


def test_atomic_json_write_replaces_fully_and_retries_windows_reader(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.json"
    atomic_json_write(path, {"version": 1})

    real_replace = os.replace
    attempts = []

    def sharing_violation_then_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(state, "_WINDOWS", True)
    monkeypatch.setattr(state.os, "replace", sharing_violation_then_replace)
    monkeypatch.setattr(state.time, "sleep", lambda _seconds: None)
    atomic_json_write(path, {"version": 2})

    assert read_json(path) == {"version": 2}
    assert len(attempts) == 3
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(StateError, match="valid JSON"):
        atomic_json_write(path, {"value": float("nan")})
    assert read_json(path) == {"version": 2}
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(StateError, match="non-finite"):
        read_json(path)


def test_runtime_validation_is_indivisible_and_preserves_bad_evidence(app_paths):
    store = JobStore(app_paths)
    store.write_spec(JobSpec.attached_pattern("training", "python train.py"))
    path = store.runtime_path("training")
    invalid = _runtime(
        owned_suspensions=[
            ProcessIdentity(123, 456.75).to_dict(),
            {"pid": 999},
        ]
    )
    atomic_json_write(path, invalid)
    before = path.read_bytes()

    with pytest.raises(StateError, match=r"owned_suspensions\[1\]"):
        store.read_runtime("training")

    assert path.read_bytes() == before
    assert any("owned_suspensions[1]" in error for error in store.audit_state())


def test_orphan_runtime_is_reported_and_blocks_spawn(app_paths, monkeypatch, capsys):
    store = JobStore(app_paths)
    store.write_runtime("orphan", _runtime())
    observation = Observation(
        source=PowerSource.AC,
        percent=80,
        temperature_c=30,
        charging=False,
        observed_at="2026-07-28T12:00:00Z",
    )
    monkeypatch.setattr(
        cli,
        "SensorReader",
        lambda: SimpleNamespace(sample=lambda: observation),
    )
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)

    assert cli.main(["status"]) == 1
    assert "recovery state has no job metadata" in capsys.readouterr().out

    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda *args, **kwargs: spawned.append(args))
    assert cli.main(["run", "--name", "orphan", "--", "python", "train.py"]) == 2
    assert spawned == []
    assert store.runtime_path("orphan").exists()
    assert "recovery state" in capsys.readouterr().err


def test_job_name_lock_serializes_creation_before_spawn_and_is_reusable(
    app_paths,
    monkeypatch,
    capsys,
):
    store = JobStore(app_paths)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from trainguard.state import AppPaths, JobStore\n"
        "root=Path(sys.argv[1])\n"
        "paths=AppPaths(root,root/'run',root/'logs',root/'persist',root/'config.json')\n"
        "paths.ensure()\n"
        "with JobStore(paths).lock_name('training'):\n"
        " print('acquired', flush=True)\n"
        " sys.stdin.read(1)\n"
    )

    holder = subprocess.Popen(
        [sys.executable, "-c", script, str(app_paths.home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "acquired"
    spawned = []
    monkeypatch.setattr(cli, "_spawn_detached", lambda *args, **kwargs: spawned.append(args))
    try:
        assert cli.main(["run", "--name", "training", "--", "python", "train.py"]) == 2
    finally:
        holder.communicate(input="x", timeout=5)

    assert spawned == []
    assert "another command" in capsys.readouterr().err
    acquired = subprocess.run(
        [sys.executable, "-c", script, str(app_paths.home)],
        input="x",
        capture_output=True,
        text=True,
        check=False,
    )

    assert acquired.returncode == 0
    assert acquired.stdout.strip() == "acquired"
    assert store.lock_path("training").exists()
