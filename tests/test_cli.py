import json
from types import SimpleNamespace

import pytest

from trainguard import cli


@pytest.fixture
def fake_spawn(monkeypatch):
    calls = []

    def spawn(argv, logfile=None, cwd=None):
        calls.append({"argv": list(argv), "logfile": logfile, "cwd": cwd})
        return SimpleNamespace(pid=5000 + len(calls))

    monkeypatch.setattr(cli, "_spawn_detached", spawn)
    return calls


@pytest.mark.parametrize("flag", ["--restart-on-login", "--persist"])
def test_run_records_explicit_restart_semantics(flag, fake_spawn, capsys, tmp_path):
    result = cli.main([
        "run", "--name", "training", flag, "--cwd", str(tmp_path),
        "--", "python", "train.py",
    ])

    spec = json.loads((cli.PERSIST / "training.job").read_text())
    assert result == 0
    assert spec["argv"] == ["python", "train.py"]
    assert spec["reboot_semantics"] == "restart-command"
    assert "restarts at next login" in capsys.readouterr().out


def test_attach_pid_rejects_login_restart(capsys):
    result = cli.main([
        "attach", "--pid", "123", "--name", "training", "--restart-on-login",
    ])

    assert result == 2
    assert "PID cannot survive a reboot" in capsys.readouterr().err
    assert not (cli.PERSIST / "training.job").exists()


def test_attach_pattern_records_reattach_semantics(fake_spawn, tmp_path):
    result = cli.main([
        "attach", "--match", "python train.py", "--name", "training",
        "--restart-on-login", "--start", "python train.py", "--cwd", str(tmp_path),
    ])

    spec = json.loads((cli.PERSIST / "training.job").read_text())
    assert result == 0
    assert spec["pattern"] == "python train.py"
    assert spec["reboot_semantics"] == "restart-or-reattach"


def test_restart_persisted_relaunches_as_a_fresh_process(fake_spawn, capsys, tmp_path):
    cli._ensure_dirs()
    (cli.PERSIST / "training.job").write_text(json.dumps({
        "mode": "run",
        "name": "training",
        "cwd": str(tmp_path),
        "argv": ["python", "train.py", "--checkpoint", "latest"],
        "reboot_semantics": "restart-command",
    }))

    result = cli.main(["restart-persisted"])

    assert result == 0
    assert fake_spawn[0]["argv"] == ["python", "train.py", "--checkpoint", "latest"]
    output = capsys.readouterr().out
    assert "restarts at next login" in output
    assert "no RAM state was restored" in output


def test_status_names_restart_semantics(monkeypatch, capsys):
    monkeypatch.setattr(cli, "power_source", lambda: "AC")
    monkeypatch.setattr(cli, "battery_pct", lambda: 100)
    monkeypatch.setattr(cli, "battery_temp_c", lambda: None)
    monkeypatch.setattr(cli, "is_charging", lambda: False)
    monkeypatch.setattr(cli, "_agent_installed", lambda: False)

    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "starts a new process" in output
    assert "RAM state cannot survive a reboot" in output
    assert "35°C" in output and "38°C" in output and "42°C" in output


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
