import subprocess
import sys
import time
import plistlib
from types import SimpleNamespace

import psutil
import pytest

from trainguard import cli, platforms, processes
from trainguard.model import Action
from trainguard.processes import ProcessController
from trainguard.state import JobStore, PersistenceSpec, StateError


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


@pytest.fixture
def login_home(monkeypatch, tmp_path):
    monkeypatch.setattr(platforms.Path, "home", lambda: tmp_path)
    return tmp_path


def test_macos_login_agent_replaces_legacy_agent(
    app_paths,
    login_home,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Darwin")
    legacy = login_home / "Library/LaunchAgents/com.trainguard.resume.plist"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(
        platforms.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    plist_path = login_home / "Library/LaunchAgents/com.trainguard.restart.plist"
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload["Label"] == "com.trainguard.restart"
    assert payload["ProgramArguments"][-1] == "restart-persisted"
    assert payload["StandardOutPath"].endswith("restart.log")
    assert not legacy.exists()
    assert calls == [
        ["launchctl", "unload", str(plist_path)],
        ["launchctl", "unload", str(legacy)],
        ["launchctl", "load", "-w", str(plist_path)],
    ]


def test_linux_login_service_replaces_legacy_and_stays_active(
    app_paths,
    login_home,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Linux")
    legacy = login_home / ".config/systemd/user/trainguard-resume.service"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(
        platforms.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    unit = login_home / ".config/systemd/user/trainguard-restart.service"
    contents = unit.read_text(encoding="utf-8")
    assert "RemainAfterExit=yes" in contents
    assert "restart-persisted" in contents
    assert not legacy.exists()
    assert calls == [
        ["systemctl", "--user", "disable", "trainguard-restart.service"],
        ["systemctl", "--user", "disable", "trainguard-resume.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "trainguard-restart.service"],
    ]


def test_windows_login_task_deletes_legacy_before_replace(app_paths, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "SYSTEM", "Windows")
    monkeypatch.setattr(
        platforms.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert cli.cmd_install_agent(SimpleNamespace()) == 0

    assert calls[0] == [
        "schtasks",
        "/Delete",
        "/TN",
        "TrainGuardResume",
        "/F",
    ]
    create = calls[1]
    assert create[:5] == ["schtasks", "/Create", "/TN", "TrainGuardRestart", "/SC"]
    assert "restart-persisted" in create[create.index("/TR") + 1]


@pytest.mark.parametrize("system", ["Darwin", "Linux", "Windows"])
def test_install_agent_reports_registration_failure(
    app_paths,
    login_home,
    monkeypatch,
    capsys,
    system,
):
    monkeypatch.setattr(cli, "SYSTEM", system)
    monkeypatch.setattr(
        platforms.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    assert cli.main(["install-agent"]) == 2
    assert "train-guard:" in capsys.readouterr().err


def test_agent_detection_recognizes_legacy_names(
    app_paths,
    login_home,
    monkeypatch,
):
    mac = login_home / "Library/LaunchAgents/com.trainguard.resume.plist"
    mac.parent.mkdir(parents=True)
    mac.write_text("legacy", encoding="utf-8")
    assert platforms.agent_installed(app_paths, "Darwin") is True
    mac.unlink()

    linux = login_home / ".config/systemd/user/trainguard-resume.service"
    linux.parent.mkdir(parents=True)
    linux.write_text("legacy", encoding="utf-8")
    assert platforms.agent_installed(app_paths, "Linux") is True

    calls = []

    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0 if argv[3] == "TrainGuardResume" else 1)

    monkeypatch.setattr(platforms.subprocess, "run", query)
    assert platforms.agent_installed(app_paths, "Windows") is True
    assert [call[3] for call in calls] == ["TrainGuardRestart", "TrainGuardResume"]


@pytest.mark.parametrize(
    ("system", "current_relative", "legacy_relative"),
    [
        (
            "Darwin",
            "Library/LaunchAgents/com.trainguard.restart.plist",
            "Library/LaunchAgents/com.trainguard.resume.plist",
        ),
        (
            "Linux",
            ".config/systemd/user/trainguard-restart.service",
            ".config/systemd/user/trainguard-resume.service",
        ),
        ("Windows", None, None),
    ],
)
def test_uninstall_agent_removes_current_and_legacy_generations(
    app_paths,
    login_home,
    monkeypatch,
    system,
    current_relative,
    legacy_relative,
):
    files = []
    for relative in (current_relative, legacy_relative):
        if relative is not None:
            path = login_home / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("agent", encoding="utf-8")
            files.append(path)
    calls = []
    monkeypatch.setattr(
        platforms.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    platforms.uninstall_agent(app_paths, system)

    assert all(not path.exists() for path in files)
    if system == "Darwin":
        assert calls == [
            ["launchctl", "unload", str(login_home / current_relative)],
            ["launchctl", "unload", str(login_home / legacy_relative)],
        ]
    elif system == "Linux":
        assert calls == [
            ["systemctl", "--user", "disable", "trainguard-restart.service"],
            ["systemctl", "--user", "disable", "trainguard-resume.service"],
            ["systemctl", "--user", "daemon-reload"],
        ]
    else:
        assert calls == [
            ["schtasks", "/Delete", "/TN", "TrainGuardRestart", "/F"],
            ["schtasks", "/Delete", "/TN", "TrainGuardResume", "/F"],
        ]


def test_login_retry_validates_policy_before_starting_command(
    app_paths,
    monkeypatch,
):
    store = JobStore(app_paths)
    persistence = PersistenceSpec(
        mode="attach",
        name="retry",
        cwd=str(app_paths.home),
        pattern="worker.py",
        start="python worker.py",
    )
    store.write_persistence(persistence)
    app_paths.config.write_text('{"poll": 0}', encoding="utf-8")
    spawned = []
    monkeypatch.setattr(
        cli,
        "_spawn_detached",
        lambda *args, **kwargs: spawned.append((args, kwargs)),
    )

    assert cli.main(["restart-persisted"]) == 2
    assert spawned == []
    assert store.read_persistence(store.persistence_path("retry")) == persistence


def test_failed_attach_retry_preserves_spec_and_terminates_started_command(
    app_paths,
    monkeypatch,
):
    store = JobStore(app_paths)
    persistence = PersistenceSpec(
        mode="attach",
        name="retry",
        cwd=str(app_paths.home),
        pattern="worker.py",
        start="python worker.py",
    )
    store.write_persistence(persistence)
    monkeypatch.setattr(cli, "_pattern_running", lambda _pattern: False)
    monkeypatch.setattr(
        cli,
        "_start_supervisor",
        lambda *args, **kwargs: (_ for _ in ()).throw(StateError("not ready")),
    )

    class StartedCommand:
        def __init__(self):
            self.calls = []

        def terminate(self):
            self.calls.append("terminate")

        def wait(self, timeout):
            self.calls.append(("wait", timeout))

    started = StartedCommand()
    monkeypatch.setattr(cli, "_spawn_detached", lambda *args, **kwargs: started)

    assert cli.main(["restart-persisted"]) == 1
    assert started.calls == ["terminate", ("wait", 2)]
    assert store.read_persistence(store.persistence_path("retry")) == persistence
