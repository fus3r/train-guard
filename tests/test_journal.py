from __future__ import annotations

import json
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, journal as journal_module
from trainguard import supervisor as supervisor_module
from trainguard.journal import EventJournal
from trainguard.model import Observation, PowerSource, ProcessIdentity
from trainguard.processes import ApplyReport, TargetSnapshot, process_identity
from trainguard.state import JobSpec, JobStore, atomic_json_write
from trainguard.supervisor import Supervisor


def test_journal_appends_both_formats_and_limits_valid_events(app_paths, monkeypatch):
    monkeypatch.setattr(
        journal_module,
        "utc_now",
        lambda: "2026-08-03T10:00:00.000Z",
    )
    journal = EventJournal(app_paths, "training")

    first = journal.emit("started", "supervisor started", mode="run")
    with journal.json_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":\n{"event":"nonstandard","value":NaN}\n')
    second = journal.emit("decision", "gentle: warm_ac", action="gentle")
    third = journal.emit("stopped", "released owned process changes")

    assert first == {
        "timestamp": "2026-08-03T10:00:00.000Z",
        "job": "training",
        "event": "started",
        "message": "supervisor started",
        "mode": "run",
    }
    assert [event["event"] for event in journal.read(limit=2)] == [
        second["event"],
        third["event"],
    ]
    assert [event["event"] for event in journal.read()] == [
        "started",
        "decision",
        "stopped",
    ]
    assert "decision: gentle: warm_ac" in journal.text_path.read_text(encoding="utf-8")

    line_count = len(journal.json_path.read_text(encoding="utf-8").splitlines())
    with pytest.raises(ValueError, match=r"[Oo]ut of range"):
        journal.emit("sensor", "invalid reading", temperature_c=float("nan"))
    assert len(journal.json_path.read_text(encoding="utf-8").splitlines()) == line_count


def test_supervisor_records_changed_decision_and_closes_terminal_interval(
    app_paths,
    monkeypatch,
):
    atomic_json_write(
        app_paths.config,
        {
            "poll": 0.1,
            "ac_band": "gentle",
        },
    )
    observations = iter(
        (
            Observation(
                PowerSource.AC,
                80.0,
                35.0,
                False,
                "2026-08-03T12:00:00.000Z",
            ),
            Observation(
                PowerSource.AC,
                80.0,
                35.0,
                False,
                "2026-08-03T12:10:00.000Z",
            ),
        )
    )

    class Controller:
        def __init__(self):
            self.resolve_calls = 0

        @property
        def owned_suspensions(self):
            return ()

        @property
        def tuned_processes(self):
            return ()

        def resolve(self, _spec):
            self.resolve_calls += 1
            if self.resolve_calls <= 2:
                return TargetSnapshot((SimpleNamespace(pid=321),), True)
            return TargetSnapshot((), False)

        def apply(self, _action, processes):
            return ApplyReport(targeted=len(tuple(processes)), tuned=1)

        def release_owned(self):
            return ApplyReport(targeted=1, restored=1)

    monkeypatch.setattr(
        supervisor_module,
        "utc_now",
        lambda: "2026-08-03T13:00:00.000Z",
    )
    sensors = SimpleNamespace(sample=lambda: next(observations))
    spec = JobSpec.launched(
        "training",
        ProcessIdentity(321, 1.0),
        app_paths.logs / "training.log",
    )
    store = JobStore(app_paths)
    store.write_spec(spec)
    supervisor = Supervisor(
        app_paths,
        spec,
        sensors=sensors,
        controller=Controller(),
    )
    supervisor._install_signal_handlers = lambda: None

    assert supervisor.run() == 0

    events = EventJournal(app_paths, "training").read()
    decisions = [event for event in events if event["event"] == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["observation"]["observed_at"] == ("2026-08-03T12:00:00.000Z")
    assert decisions[0]["process_report"]["tuned"] == 1
    terminal = events[-1]
    assert terminal["event"] == "job_exited"
    assert terminal["trace_terminal"] is True
    assert terminal["observation"]["observed_at"] == "2026-08-03T13:00:00.000Z"
    assert terminal["process_report"]["restored"] == 1


def test_events_status_and_list_have_machine_readable_output(
    app_paths,
    monkeypatch,
    capsys,
):
    observation = Observation(
        PowerSource.AC,
        88.0,
        33.0,
        True,
        "2026-08-03T14:00:00.000Z",
    )
    monkeypatch.setattr(
        cli,
        "SensorReader",
        lambda: SimpleNamespace(sample=lambda: observation),
    )
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)
    store = JobStore(app_paths)
    store.write_spec(JobSpec.attached_pattern("training", "python train.py"))
    store.write_guard("training", process_identity(psutil.Process()))
    store.write_runtime(
        "training",
        {
            "schema_version": 1,
            "updated_at": "2026-08-03T14:00:00.000Z",
            "state": "gentle",
            "cooling": False,
            "owned_suspensions": [],
            "tuned_processes": [],
            "pids": [321],
        },
    )
    EventJournal(app_paths, "training").emit("started", "supervisor started")

    assert cli.main(["events", "training", "--limit", "1"]) == 0
    assert "started" in capsys.readouterr().out

    assert cli.main(["events", "training", "--limit", "1", "--json"]) == 0
    events = json.loads(capsys.readouterr().out)
    assert [event["event"] for event in events] == ["started"]

    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["schema_version"] == 1
    assert status["observation"]["percent"] == 88.0
    assert status["guards"][0]["runtime"]["state"] == "gentle"

    assert cli.main(["list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing == [
        {
            "alive": True,
            "mode": "attach",
            "name": "training",
            "state": "gentle",
        }
    ]


def test_events_rejects_a_non_positive_limit(capsys):
    assert cli.main(["events", "training", "--limit", "0"]) == 2
    assert "--limit must be at least 1" in capsys.readouterr().err
