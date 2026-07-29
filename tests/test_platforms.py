import subprocess
import sys
import time
import plistlib
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, processes
from trainguard.model import Action
from trainguard.processes import ProcessController


class FakeProcess:
    def __init__(self):
        self.pid = 123
        self.created = 100.0
        self.current_status = psutil.STATUS_RUNNING
        self.priority = 32
        self.ionice_value = SimpleNamespace(ioclass=2, value=4)
        self.nice_calls = []
        self.ionice_calls = []
        self.affinity = [0, 1, 2, 3]
        self.affinity_calls = []

    def create_time(self):
        return self.created

    def status(self):
        return self.current_status

    def suspend(self):
        self.current_status = psutil.STATUS_STOPPED

    def resume(self):
        self.current_status = psutil.STATUS_RUNNING

    def nice(self, value=None):
        if value is None:
            return self.priority
        self.nice_calls.append(value)
        self.priority = value

    def ionice(self, *value):
        if not value:
            return self.ionice_value
        self.ionice_calls.append(value)
        self.ionice_value = value

    def cpu_affinity(self, value=None):
        if value is None:
            return list(self.affinity)
        self.affinity = list(value)
        self.affinity_calls.append(list(value))


def test_macos_gentle_is_a_background_hint(monkeypatch):
    proc = FakeProcess()
    calls = []
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: proc)
    monkeypatch.setattr(
        processes.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    controller = ProcessController("Darwin")

    controller.apply(Action.GENTLE, [proc])
    controller.apply(Action.FULL, [proc])

    assert calls == [
        ["taskpolicy", "-b", "-p", "123"],
        ["taskpolicy", "-B", "-p", "123"],
    ]


def test_linux_gentle_uses_reversible_affinity_and_idle_io(monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: proc)
    monkeypatch.setattr(psutil, "IOPRIO_CLASS_IDLE", 3, raising=False)
    controller = ProcessController("Linux")

    controller.apply(Action.GENTLE, [proc])
    controller.apply(Action.FULL, [proc])

    assert proc.nice_calls == []
    assert proc.affinity_calls == [[0, 2], [0, 1, 2, 3]]
    assert proc.ionice_calls == [(3,), (2, 4)]


def test_windows_gentle_uses_idle_priority(monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: proc)
    monkeypatch.setattr(psutil, "IDLE_PRIORITY_CLASS", 64, raising=False)
    controller = ProcessController("Windows")

    controller.apply(Action.GENTLE, [proc])
    controller.apply(Action.FULL, [proc])

    assert proc.nice_calls == [64, 32]


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
    controller = ProcessController()
    try:
        wait_for_counter(counter, lambda value: value >= 2)
        controller.apply(Action.STOP, [proc])
        time.sleep(0.12)
        stopped_at = int(counter.read_text())
        time.sleep(0.18)
        assert int(counter.read_text()) == stopped_at

        controller.apply(Action.FULL, [proc])
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
