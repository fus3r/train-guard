from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from trainguard.model import ProcessIdentity
from trainguard.state import JobStore, atomic_json_write

PROJECT_ROOT = Path(__file__).parents[1]


def _run_cli(environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "trainguard", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def _terminate_identity(identity: ProcessIdentity | None) -> None:
    if identity is None:
        return
    try:
        process = psutil.Process(identity.pid)
        if abs(float(process.create_time()) - identity.create_time) >= 0.001:
            return
        process.terminate()
        process.wait(timeout=3)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def test_detached_cli_lifecycle_reports_ready_before_success(app_paths):
    atomic_json_write(
        app_paths.config,
        {
            "poll": 0.1,
            "run_on_battery": True,
            "battery_floor_pct": 0,
            "battery_band": "full",
            "ac_band": "full",
            "temp_charge_gentle_c": 100,
            "temp_gentle_c": 100,
            "temp_pause_c": 100,
            "temp_resume_c": 99,
        },
    )
    environment = os.environ.copy()
    environment["TRAIN_GUARD_HOME"] = str(app_paths.home)
    store = JobStore(app_paths)
    worker_identity: ProcessIdentity | None = None
    guard_identity: ProcessIdentity | None = None

    try:
        launched = _run_cli(
            environment,
            "run",
            "--name",
            "e2e",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        )
        assert launched.returncode == 0, launched.stderr
        assert "guard pid=" in launched.stdout
        assert not store.ready_path("e2e").exists()

        spec = store.read_spec("e2e")
        worker_identity = spec.root
        guard_identity = store.read_guard("e2e")
        assert worker_identity is not None
        assert guard_identity is not None

        _wait_until(
            lambda: (store.read_runtime("e2e") or {}).get("decision", {}).get("action") == "full"
        )

        status = _run_cli(environment, "status", "--json")
        assert status.returncode == 0, status.stderr
        payload = json.loads(status.stdout)
        guard = next(item for item in payload["guards"] if item["name"] == "e2e")
        assert guard["guard"]["alive"] is True
        assert guard["runtime"]["decision"]["action"] == "full"

        events = _run_cli(environment, "events", "e2e", "--json")
        assert events.returncode == 0, events.stderr
        event_names = [event["event"] for event in json.loads(events.stdout)]
        assert "started" in event_names
        assert "decision" in event_names

        stopped = _run_cli(environment, "stop", "e2e")
        assert stopped.returncode == 0, stopped.stderr
        _wait_until(lambda: not store.spec_path("e2e").exists())
        assert psutil.pid_exists(worker_identity.pid)
    finally:
        if store.spec_path("e2e").exists():
            _run_cli(environment, "stop", "e2e", "--kill")
        _terminate_identity(guard_identity)
        _terminate_identity(worker_identity)
