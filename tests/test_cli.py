import json
from types import SimpleNamespace

import pytest

from trainguard import cli
from trainguard.model import Observation, PowerSource, ProcessIdentity, utc_now
from trainguard.state import JobSpec, JobStore, atomic_json_write


def pin_sensors(monkeypatch, **readings):
    """Report one fixed observation instead of this machine's battery."""
    observation = Observation(
        source=readings.get("source", PowerSource.AC),
        percent=readings.get("percent", 100.0),
        temperature_c=readings.get("temperature_c"),
        charging=readings.get("charging", False),
        observed_at=utc_now(),
        warnings=readings.get("warnings", ()),
    )
    reader = SimpleNamespace(sample=lambda: observation)
    monkeypatch.setattr(cli, "SensorReader", lambda *args, **kwargs: reader)


@pytest.fixture
def fake_spawn(monkeypatch):
    calls = []

    def spawn(argv, logfile=None, cwd=None):
        calls.append({"argv": list(argv), "logfile": logfile, "cwd": cwd})
        return SimpleNamespace(pid=5000 + len(calls))

    monkeypatch.setattr(cli, "_spawn_detached", spawn)
    monkeypatch.setattr(
        cli,
        "_capture_identity",
        lambda pid, _label: ProcessIdentity(pid, float(pid)),
    )
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_ready",
        lambda *_args, **_kwargs: None,
    )
    return calls


@pytest.mark.parametrize("flag", ["--restart-on-login", "--persist"])
def test_run_records_explicit_restart_semantics(flag, fake_spawn, capsys, tmp_path):
    result = cli.main(
        [
            "run",
            "--name",
            "training",
            flag,
            "--cwd",
            str(tmp_path),
            "--",
            "python",
            "train.py",
        ]
    )

    spec = json.loads((cli.PERSIST / "training.job").read_text())
    assert result == 0
    assert spec["argv"] == ["python", "train.py"]
    assert spec["reboot_semantics"] == "restart-command"
    assert "restarts at next login" in capsys.readouterr().out


def test_attach_pid_rejects_login_restart(capsys):
    result = cli.main(
        [
            "attach",
            "--pid",
            "123",
            "--name",
            "training",
            "--restart-on-login",
        ]
    )

    assert result == 2
    assert "PID cannot survive a reboot" in capsys.readouterr().err
    assert not (cli.PERSIST / "training.job").exists()


def test_attach_pattern_records_reattach_semantics(fake_spawn, tmp_path):
    result = cli.main(
        [
            "attach",
            "--match",
            "python train.py",
            "--name",
            "training",
            "--restart-on-login",
            "--start",
            "python train.py",
            "--cwd",
            str(tmp_path),
        ]
    )

    spec = json.loads((cli.PERSIST / "training.job").read_text())
    assert result == 0
    assert spec["pattern"] == "python train.py"
    assert spec["reboot_semantics"] == "restart-or-reattach"


def test_restart_persisted_relaunches_as_a_fresh_process(fake_spawn, capsys, tmp_path):
    cli._ensure_dirs()
    (cli.PERSIST / "training.job").write_text(
        json.dumps(
            {
                "mode": "run",
                "name": "training",
                "cwd": str(tmp_path),
                "argv": ["python", "train.py", "--checkpoint", "latest"],
                "reboot_semantics": "restart-command",
            }
        )
    )

    result = cli.main(["restart-persisted"])

    assert result == 0
    assert fake_spawn[0]["argv"] == ["python", "train.py", "--checkpoint", "latest"]
    output = capsys.readouterr().out
    assert "restarts at next login" in output
    assert "no RAM state was restored" in output


def test_status_names_restart_semantics(monkeypatch, capsys):
    pin_sensors(monkeypatch)
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)

    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "starts a new process" in output
    assert "RAM state cannot survive a reboot" in output
    assert "35°C" in output and "38°C" in output and "42°C" in output


def test_status_shows_missing_readings_instead_of_defaults(monkeypatch, capsys):
    """A host with no battery used to be reported as a full charge."""
    pin_sensors(
        monkeypatch,
        source=PowerSource.NO_BATTERY,
        percent=None,
        charging=None,
        warnings=("battery not exposed; treating this host as mains-powered",),
    )
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)

    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "charge: n/a" in output
    assert "charging: n/a" in output
    assert "note: battery not exposed" in output


def test_config_init_creates_its_parent(capsys):
    assert cli.main(["config", "--init"]) == 0
    cfg = json.loads(cli.CONFIGF.read_text())
    assert cfg["temp_gentle_c"] == 38
    assert "temp_ecore_c" not in cfg


def test_run_rejects_invalid_config_before_spawning(fake_spawn, capsys):
    cli._ensure_dirs()
    cli.CONFIGF.write_text('{"poll": "fast"}', encoding="utf-8")

    result = cli.main(["run", "--name", "training", "--", "python", "train.py"])

    assert result == 2
    assert fake_spawn == []
    assert "poll must be a number" in capsys.readouterr().err


def _runtime_state(**overrides):
    value = {
        "schema_version": 1,
        "updated_at": "2026-08-04T12:00:00Z",
        "state": "stop",
        "cooling": False,
        "owned_suspensions": [],
        "tuned_processes": [],
        "pids": [],
    }
    value.update(overrides)
    return value


def test_config_check_and_force_have_explicit_outcomes(app_paths, capsys):
    app_paths.config.write_text('{"typo": 1}', encoding="utf-8")
    assert cli.main(["config", "--check"]) == 2
    assert "unknown configuration key" in capsys.readouterr().err

    assert cli.main(["config", "--init"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert cli.main(["config", "--init", "--force", "--check"]) == 0
    assert f"{app_paths.config}: valid" in capsys.readouterr().out


def test_status_and_doctor_json_report_corrupt_state_without_erasing_it(
    app_paths,
    monkeypatch,
    capsys,
):
    pin_sensors(monkeypatch, temperature_c=33.0)
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)
    store = JobStore(app_paths)
    app_paths.config.write_text('{"typo": 1}', encoding="utf-8")
    store.write_spec(JobSpec.attached_pattern("broken", "python train.py"))
    store.runtime_path("broken").write_text(
        '{"schema_version":1,"state":"stop"}',
        encoding="utf-8",
    )
    store.write_runtime("orphan", _runtime_state())
    (app_paths.persist / "bad.job").write_text("[]", encoding="utf-8")
    before = store.runtime_path("broken").read_bytes()

    assert cli.main(["status", "--json"]) == 1
    status = json.loads(capsys.readouterr().out)
    assert status["schema_version"] == 1
    assert "unknown configuration key" in status["policy_error"]
    assert status["guards"][0]["runtime"] is None
    assert "runtime_error" in status["guards"][0]
    assert any("recovery state has no job metadata" in error for error in status["state_errors"])
    assert status["persistence_errors"]

    assert cli.main(["doctor", "--json"]) == 1
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["schema_version"] == 1
    assert doctor["ok"] is False
    state_check = next(check for check in doctor["checks"] if check["name"] == "state_files")
    assert state_check["ok"] is False
    assert any("orphan" in error for error in state_check["detail"]["errors"])
    assert store.runtime_path("broken").read_bytes() == before


def test_recover_refuses_ambiguous_runtime_without_erasing_evidence(
    app_paths,
    capsys,
):
    store = JobStore(app_paths)
    store.write_spec(JobSpec.attached_pattern("unsafe", "python train.py"))
    atomic_json_write(
        store.runtime_path("unsafe"),
        _runtime_state(owned_suspensions=[{"pid": 123}]),
    )
    before = store.runtime_path("unsafe").read_bytes()

    assert cli.main(["recover", "unsafe"]) == 2
    assert "owned_suspensions[0]" in capsys.readouterr().err
    assert store.runtime_path("unsafe").read_bytes() == before


def test_run_reports_a_missing_working_directory_without_partial_state(
    app_paths,
    capsys,
):
    missing = app_paths.home / "missing"
    assert (
        cli.main(
            [
                "run",
                "--name",
                "training",
                "--cwd",
                str(missing),
                "--",
                "echo",
                "ok",
            ]
        )
        == 2
    )
    assert "working directory does not exist" in capsys.readouterr().err
    assert JobStore(app_paths).list_specs() == []


def test_keyboard_interrupt_has_a_distinct_exit_code(monkeypatch, capsys):
    def interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_doctor", interrupt)
    assert cli.main(["doctor"]) == 130
    assert "interrupted" in capsys.readouterr().err
