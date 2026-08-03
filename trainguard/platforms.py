"""Login persistence adapters for supported operating systems."""

from __future__ import annotations

import os
import platform
import plistlib
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

if __package__:
    from .state import AppPaths
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from state import AppPaths


class PlatformError(RuntimeError):
    """Raised when a host login integration cannot be installed."""


def _system(system: Optional[str]) -> str:
    return system or platform.system()


def login_agent_path(paths: AppPaths, system: Optional[str] = None) -> Path:
    current = _system(system)
    if current == "Darwin":
        return Path.home() / "Library/LaunchAgents/com.trainguard.restart.plist"
    if current == "Linux":
        return Path.home() / ".config/systemd/user/trainguard-restart.service"
    if current == "Windows":
        return paths.home / "TrainGuardRestart.task"
    raise PlatformError(f"login persistence is not supported on {current}")


def _legacy_agent_paths(paths: AppPaths, system: str) -> tuple[Path, ...]:
    if system == "Darwin":
        return (Path.home() / "Library/LaunchAgents/com.trainguard.resume.plist",)
    if system == "Linux":
        return (Path.home() / ".config/systemd/user/trainguard-resume.service",)
    if system == "Windows":
        return (paths.home / "TrainGuardResume.task",)
    return ()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def agent_installed(paths: AppPaths, system: Optional[str] = None) -> bool:
    current = _system(system)
    if current in {"Darwin", "Linux"}:
        candidates = (
            login_agent_path(paths, current),
            *_legacy_agent_paths(paths, current),
        )
        return any(candidate.exists() for candidate in candidates)
    if current == "Windows":
        for task_name in ("TrainGuardRestart", "TrainGuardResume"):
            try:
                result = subprocess.run(
                    ["schtasks", "/Query", "/TN", task_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            if result.returncode == 0:
                return True
        return False
    return False


def install_agent(
    paths: AppPaths,
    restart_argv: Sequence[str],
    system: Optional[str] = None,
) -> Path:
    current = _system(system)
    if current == "Darwin":
        target = login_agent_path(paths, current)
        legacy_targets = _legacy_agent_paths(paths, current)
        # An absent or already-unloaded plist is normal during migration. The
        # authoritative load of the new agent below must succeed.
        for candidate in (target, *legacy_targets):
            subprocess.run(
                ["launchctl", "unload", str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        for candidate in legacy_targets:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

        payload = {
            "Label": "com.trainguard.restart",
            "ProgramArguments": list(restart_argv),
            "RunAtLoad": True,
            "ProcessType": "Background",
            "StandardOutPath": str(paths.logs / "restart.log"),
            "StandardErrorPath": str(paths.logs / "restart.log"),
        }
        _atomic_write(target, plistlib.dumps(payload))
        result = subprocess.run(
            ["launchctl", "load", "-w", str(target)],
            check=False,
        )
        if result.returncode:
            raise PlatformError(f"launchctl could not load {target}")
        return target

    if current == "Linux":
        target = login_agent_path(paths, current)
        legacy_targets = _legacy_agent_paths(paths, current)
        # Disabling an absent unit is expected to fail; enabling the new unit
        # after daemon-reload is the checked migration boundary.
        for candidate in (target, *legacy_targets):
            subprocess.run(
                ["systemctl", "--user", "disable", candidate.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        for candidate in legacy_targets:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

        unit = "\n".join(
            [
                "[Unit]",
                "Description=train-guard: restart configured jobs at login",
                "",
                "[Service]",
                "Type=oneshot",
                "RemainAfterExit=yes",
                f"ExecStart={shlex.join(restart_argv)}",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        _atomic_write(target, unit.encode("utf-8"))
        result = subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=False,
        )
        if result.returncode:
            raise PlatformError("systemctl --user daemon-reload failed")
        result = subprocess.run(
            ["systemctl", "--user", "enable", target.name],
            check=False,
        )
        if result.returncode:
            raise PlatformError(f"systemctl could not enable {target.name}")
        return target

    if current == "Windows":
        subprocess.run(
            ["schtasks", "/Delete", "/TN", "TrainGuardResume", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        command = subprocess.list2cmdline(list(restart_argv))
        result = subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                "TrainGuardRestart",
                "/SC",
                "ONLOGON",
                "/TR",
                command,
                "/F",
            ],
            check=False,
        )
        if result.returncode:
            raise PlatformError("schtasks could not create TrainGuardRestart")
        return login_agent_path(paths, current)

    raise PlatformError(f"login persistence is not supported on {current}")


def uninstall_agent(
    paths: AppPaths,
    system: Optional[str] = None,
) -> Optional[Path]:
    current = _system(system)
    if current == "Darwin":
        target = login_agent_path(paths, current)
        removed_path: Optional[Path] = None
        for candidate in (target, *_legacy_agent_paths(paths, current)):
            subprocess.run(
                ["launchctl", "unload", str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                candidate.unlink()
                removed_path = removed_path or candidate
            except FileNotFoundError:
                pass
        return removed_path

    if current == "Linux":
        target = login_agent_path(paths, current)
        removed_path = None
        for candidate in (target, *_legacy_agent_paths(paths, current)):
            subprocess.run(
                ["systemctl", "--user", "disable", candidate.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                candidate.unlink()
                removed_path = removed_path or candidate
            except FileNotFoundError:
                pass
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return removed_path

    if current == "Windows":
        removed = False
        for task_name in ("TrainGuardRestart", "TrainGuardResume"):
            result = subprocess.run(
                ["schtasks", "/Delete", "/TN", task_name, "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            removed = removed or result.returncode == 0
        return login_agent_path(paths, current) if removed else None

    raise PlatformError(f"login persistence is not supported on {current}")
