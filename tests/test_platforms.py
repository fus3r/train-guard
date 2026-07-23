import subprocess
import sys
import time
import plistlib
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli


class FakeProcess:
    def __init__(self):
        self.pid = 123
        self.nice_calls = []
        self.ionice_calls = []
        self.affinity = [0, 1, 2, 3]
        self.affinity_calls = []

    def nice(self, value):
        self.nice_calls.append(value)

    def ionice(self, value):
        self.ionice_calls.append(value)

    def cpu_affinity(self, value=None):
        if value is None:
            return list(self.affinity)
        self.affinity = list(value)
        self.affinity_calls.append(list(value))


def test_macos_gentle_is_a_background_hint(monkeypatch):
    proc = FakeProcess()
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Darwin")
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kwargs: calls.append(argv))

    cli._set_band(proc, "gentle")
    cli._set_band(proc, "full")

    assert calls == [
        ["taskpolicy", "-b", "-p", "123"],
        ["taskpolicy", "-B", "-p", "123"],
    ]


def test_linux_gentle_uses_reversible_affinity_and_idle_io(monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    monkeypatch.setattr(cli.psutil, "IOPRIO_CLASS_IDLE", 3, raising=False)
    monkeypatch.setattr(cli.psutil, "IOPRIO_CLASS_NONE", 0, raising=False)

    cli._set_band(proc, "gentle")
    cli._set_band(proc, "full")

    assert proc.nice_calls == []
    assert proc.affinity_calls == [[0, 2], [0, 1, 2, 3]]
    assert proc.ionice_calls == [3, 0]


def test_windows_gentle_uses_idle_priority(monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(cli, "SYSTEM", "Windows")
    monkeypatch.setattr(cli.psutil, "IDLE_PRIORITY_CLASS", 64, raising=False)

    cli._set_band(proc, "gentle")

    assert proc.nice_calls == [64]


def test_macos_battery_temperature_adapter(monkeypatch):
    monkeypatch.setattr(cli, "SYSTEM", "Darwin")
    result = subprocess.CompletedProcess([], 0, stdout='"Temperature" = 3650\n', stderr="")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: result)
    assert cli.battery_temp_c() == 36.5


def test_linux_battery_temperature_adapter(monkeypatch):
    sensor = SimpleSensor(current=37.25)
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    monkeypatch.setattr(
        cli.psutil, "sensors_temperatures", lambda: {"battery": [sensor]}, raising=False
    )
    assert cli.battery_temp_c() == 37.25


class SimpleSensor:
    def __init__(self, current):
        self.current = current


def wait_for_counter(path, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = int(path.read_text())
        except (OSError, ValueError):
            value = -1
        if predicate(value):
            return value
        time.sleep(0.03)
    pytest.fail("worker counter did not reach the expected state")


def test_real_child_process_can_be_suspended_and_resumed(tmp_path):
    """Portable smoke test for the core psutil operation used on all three OSes."""
    counter = tmp_path / "counter.txt"
    code = (
        "import pathlib,time,sys\n"
        "p=pathlib.Path(sys.argv[1])\n"
        "tmp=p.with_suffix('.tmp')\n"
        "for i in range(500):\n"
        " tmp.write_text(str(i))\n"
        " while True:\n"
        "  try:\n"
        "   tmp.replace(p)\n"
        "   break\n"
        "  except PermissionError:\n"
        "   time.sleep(0.001)\n"
        " time.sleep(0.02)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(counter)])
    proc = psutil.Process(child.pid)
    try:
        wait_for_counter(counter, lambda value: value >= 2)
        cli._apply("stop", [proc])
        time.sleep(0.12)
        stopped_at = int(counter.read_text())
        time.sleep(0.18)
        assert int(counter.read_text()) == stopped_at

        cli._apply("full", [proc])
        wait_for_counter(counter, lambda value: value > stopped_at)
    finally:
        try:
            proc.resume()
        except psutil.Error:
            pass
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()


def test_macos_login_agent_runs_restart_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Darwin")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    plist_path = cli.HOME / "Library/LaunchAgents/com.trainguard.resume.plist"
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload["ProgramArguments"][-1] == "restart-persisted"
    assert payload["StandardOutPath"].endswith("restart.log")
    assert calls[-1][:3] == ["launchctl", "load", "-w"]


def test_linux_login_service_runs_restart_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    unit = cli.HOME / ".config/systemd/user/trainguard-resume.service"
    assert "restart-persisted" in unit.read_text()
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "trainguard-resume.service"] in calls


def test_windows_login_task_runs_restart_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Windows")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    create = calls[-1]
    assert create[:5] == ["schtasks", "/Create", "/TN", "TrainGuardRestart", "/SC"]
    assert "restart-persisted" in create[create.index("/TR") + 1]
