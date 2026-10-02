import subprocess
import sys
import threading
import time
from dataclasses import replace

import psutil
import pytest

from trainguard.agents import global_ignored
from trainguard.model import Observation, PowerSource, utc_now
from trainguard.processes import process_identity
from trainguard.state import JobSpec, JobStore, atomic_json_write
from trainguard.supervisor import Supervisor


def wait_for(predicate):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("supervisor did not reach the expected state")


def test_global_override_expiry_resumes_policy_for_current_and_future_jobs(app_paths):
    atomic_json_write(app_paths.config, {"poll": 0.1})
    store = JobStore(app_paths)
    override = app_paths.home / "global-override.json"

    class HotBattery:
        def sample(self):
            return Observation(PowerSource.BATTERY, 80, 45, False, utc_now())

    workers = []
    supervisors = []
    threads = []

    def start(name, agent=None):
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        workers.append(worker)
        spec = replace(
            JobSpec.attached_pid(name, process_identity(psutil.Process(worker.pid))), agent=agent
        )
        store.write_spec(spec)
        supervisor = Supervisor(app_paths, spec, sensors=HotBattery())
        supervisor._install_signal_handlers = lambda: None
        supervisors.append(supervisor)
        thread = threading.Thread(target=supervisor.run)
        threads.append(thread)
        thread.start()

    def reason(name):
        return (store.read_runtime(name) or {}).get("decision", {}).get("reason")

    try:
        start("current", "chosen")
        wait_for(lambda: psutil.Process(workers[0].pid).status() == psutil.STATUS_STOPPED)
        atomic_json_write(override, {"schema_version": 1, "enabled": True, "expires_at": None})
        wait_for(lambda: reason("current") == "global_ignored")
        assert psutil.Process(workers[0].pid).status() != psutil.STATUS_STOPPED
        start("future")  # No agent owner, launched after the global exception began.
        wait_for(lambda: reason("future") == "global_ignored")
        assert psutil.Process(workers[1].pid).status() != psutil.STATUS_STOPPED
        (app_paths.home / "ignored-agents").write_text("chosen  # owner selection\n")
        override.unlink()
        wait_for(lambda: reason("current") == "agent_ignored")
        wait_for(lambda: reason("future") == "thermal_cooldown")
        assert psutil.Process(workers[1].pid).status() == psutil.STATUS_STOPPED
        atomic_json_write(
            override, {"schema_version": 1, "enabled": True, "expires_at": time.time() + 1}
        )
        wait_for(lambda: reason("future") == "global_ignored")
        # No Warden process or timer removes the file. Each guard independently enforces expiry.
        wait_for(lambda: reason("future") == "thermal_cooldown")
        assert override.exists()
        assert not global_ignored(app_paths.home)
        # Each supervisor polls independently after expiry.
        wait_for(lambda: reason("current") == "agent_ignored")
    finally:
        for supervisor in supervisors:
            supervisor._shutdown.set()
        for thread in threads:
            thread.join(timeout=6)
        for worker in workers:
            try:
                psutil.Process(worker.pid).resume()
            except psutil.NoSuchProcess:
                pass
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=5)
