"""Validated and atomic local state for train-guard."""

from __future__ import annotations

import errno
import importlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

if __package__:
    from .model import ProcessIdentity, utc_now
else:  # Keep ``python trainguard/cli.py`` working from a checkout.
    from model import ProcessIdentity, utc_now


class StateError(ValueError):
    """Raised when local state is unsafe or malformed."""


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)
_WINDOWS = os.name == "nt"


def validate_job_name(name: str) -> str:
    """Return a safe state-file stem or reject it before any process starts."""

    if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise StateError(
            "job names must be 1-64 characters using letters, digits, '.', '_' or '-'"
        )
    if name.split(".", 1)[0].lower() in _RESERVED_NAMES:
        raise StateError(f"job name '{name}' is a reserved Windows device name")
    return name


def _replace_atomic(temporary: Path, destination: Path) -> None:
    if not _WINDOWS:
        os.replace(temporary, destination)
        return

    # A status reader can briefly hold the old file open on Windows. Retrying
    # that sharing violation is bounded; other failures still surface.
    for attempt in range(3):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.01)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        _replace_atomic(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json_write(path: Path, value: Any) -> None:
    try:
        text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as exc:
        raise StateError(f"{path}: state is not valid JSON: {exc}") from exc
    _atomic_write(path, text)


def read_json(path: Path) -> Any:
    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_non_finite,
        )
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise StateError(f"{path}: invalid state file: {exc}") from exc


@dataclass(frozen=True)
class AppPaths:
    home: Path
    run: Path
    logs: Path
    persist: Path
    config: Path

    @classmethod
    def from_environment(cls) -> "AppPaths":
        configured = os.environ.get("TRAIN_GUARD_HOME")
        root = Path(configured or Path.home() / ".train-guard").expanduser()
        # Python 3.9 on Windows can leave a nonexistent relative path
        # unresolved, so make the cwd relationship explicit first.
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()
        return cls(
            home=root,
            run=root / "run",
            logs=root / "logs",
            persist=root / "persist",
            config=root / "config.json",
        )

    def ensure(self) -> None:
        for directory in (self.home, self.run, self.logs, self.persist):
            directory.mkdir(parents=True, exist_ok=True)


class JobNameLock:
    """A cross-platform advisory lock for one job name."""

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        self._handle: Optional[BinaryIO] = None

    def __enter__(self) -> "JobNameLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            os.chmod(self.path, 0o600)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if _WINDOWS:
                msvcrt = importlib.import_module("msvcrt")
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise StateError(
                    f"another command is already creating job '{self.name}'"
                ) from exc
            raise
        self._handle = handle
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if _WINDOWS:
                msvcrt = importlib.import_module("msvcrt")
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass(frozen=True)
class JobSpec:
    name: str
    mode: str
    created_at: str
    root: Optional[ProcessIdentity] = None
    legacy_pid: Optional[int] = None
    pattern: Optional[str] = None
    log_path: Optional[str] = None

    @classmethod
    def launched(cls, name: str, root: ProcessIdentity, log_path: Path) -> "JobSpec":
        return cls(
            name=validate_job_name(name),
            mode="run",
            created_at=utc_now(),
            root=root,
            log_path=str(log_path),
        )

    @classmethod
    def attached_pid(cls, name: str, root: ProcessIdentity) -> "JobSpec":
        return cls(
            name=validate_job_name(name),
            mode="run",
            created_at=utc_now(),
            root=root,
        )

    @classmethod
    def attached_pattern(cls, name: str, pattern: str) -> "JobSpec":
        if not isinstance(pattern, str) or not pattern.strip():
            raise StateError("attach patterns cannot be empty")
        return cls(
            name=validate_job_name(name),
            mode="attach",
            created_at=utc_now(),
            pattern=pattern,
        )

    @property
    def root_pid(self) -> Optional[int]:
        if self.root is not None:
            return self.root.pid
        return self.legacy_pid

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "name": self.name,
            "mode": self.mode,
            "created_at": self.created_at,
        }
        if self.root is not None:
            value["jobpid"] = self.root.pid
            value["job_create_time"] = self.root.create_time
        elif self.legacy_pid is not None:
            value["jobpid"] = self.legacy_pid
        if self.pattern is not None:
            value["pattern"] = self.pattern
        if self.log_path is not None:
            value["log"] = self.log_path
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobSpec":
        if not isinstance(value, dict):
            raise StateError("job metadata must be a JSON object")
        schema_version = value.get("schema_version", 1)
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise StateError(
                f"unsupported job metadata schema version: {schema_version!r}"
            )
        try:
            raw_name = value["name"]
            raw_mode = value["mode"]
        except KeyError as exc:
            raise StateError(f"job metadata is missing {exc.args[0]}") from exc
        if not isinstance(raw_name, str):
            raise StateError("job metadata name must be a string")
        if not isinstance(raw_mode, str):
            raise StateError("job metadata mode must be a string")
        if raw_mode not in {"run", "attach"}:
            raise StateError(f"unsupported job mode: {raw_mode!r}")
        name = validate_job_name(raw_name)

        created_at = value.get("created_at", "unknown")
        if not isinstance(created_at, str) or not created_at:
            raise StateError("job metadata created_at must be a non-empty string")

        root: Optional[ProcessIdentity] = None
        legacy_pid: Optional[int] = None
        pattern: Optional[str] = None
        if raw_mode == "run":
            pid = value.get("jobpid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise StateError("run metadata needs a positive integer jobpid")
            if "job_create_time" in value:
                try:
                    root = ProcessIdentity(pid, value["job_create_time"])
                except ValueError as exc:
                    raise StateError(f"invalid root process identity: {exc}") from exc
            else:
                legacy_pid = pid
        else:
            pattern_value = value.get("pattern")
            if not isinstance(pattern_value, str) or not pattern_value.strip():
                raise StateError("attach metadata needs a non-empty pattern")
            pattern = pattern_value

        log_path = value.get("log")
        if log_path is not None and (not isinstance(log_path, str) or not log_path):
            raise StateError("job metadata log must be a non-empty string")
        return cls(
            name=name,
            mode=raw_mode,
            created_at=created_at,
            root=root,
            legacy_pid=legacy_pid,
            pattern=pattern,
            log_path=log_path,
        )


@dataclass(frozen=True)
class PersistenceSpec:
    mode: str
    name: str
    cwd: str
    argv: tuple[str, ...] = ()
    pattern: Optional[str] = None
    start: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "mode": self.mode,
            "name": validate_job_name(self.name),
            "cwd": self.cwd,
            "reboot_semantics": (
                "restart-command" if self.mode == "run" else "restart-or-reattach"
            ),
        }
        if self.mode == "run":
            value["argv"] = list(self.argv)
        elif self.mode == "attach":
            value["pattern"] = self.pattern
            value["start"] = self.start or ""
        else:
            raise StateError(f"unsupported persistence mode: {self.mode!r}")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PersistenceSpec":
        if not isinstance(value, dict):
            raise StateError("persistence spec must be a JSON object")
        schema_version = value.get("schema_version", 1)
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise StateError(
                f"unsupported persistence schema version: {schema_version!r}"
            )
        try:
            mode = value["mode"]
            raw_name = value["name"]
            cwd = value["cwd"]
        except KeyError as exc:
            raise StateError(f"persistence spec is missing {exc.args[0]}") from exc
        if not isinstance(raw_name, str):
            raise StateError("persistence name must be a string")
        name = validate_job_name(raw_name)
        if not isinstance(cwd, str) or not cwd:
            raise StateError("persistence cwd must be a non-empty string")

        if mode == "run":
            argv = value.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(item, str) or not item for item in argv)
            ):
                raise StateError("run persistence spec needs a non-empty argv list")
            return cls(mode="run", name=name, cwd=cwd, argv=tuple(argv))
        if mode == "attach":
            pattern = value.get("pattern")
            if not isinstance(pattern, str) or not pattern.strip():
                raise StateError("attach persistence spec needs a non-empty pattern")
            start = value.get("start")
            if start is not None and not isinstance(start, str):
                raise StateError("attach persistence start must be a string or null")
            return cls(
                mode="attach",
                name=name,
                cwd=cwd,
                pattern=pattern,
                start=start or None,
            )
        raise StateError(f"unsupported persistence mode: {mode!r}")


def _validate_runtime_state(path: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateError(f"{path}: expected a JSON object")
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise StateError(
            f"{path}: unsupported runtime schema version: {schema_version!r}"
        )
    if not isinstance(value.get("updated_at"), str) or not value["updated_at"]:
        raise StateError(f"{path}: runtime updated_at must be a non-empty string")
    if not isinstance(value.get("state"), str) or not value["state"]:
        raise StateError(f"{path}: runtime state must be a non-empty string")
    if not isinstance(value.get("cooling"), bool):
        raise StateError(f"{path}: runtime cooling must be true or false")

    for field_name in ("owned_suspensions", "tuned_processes"):
        records = value.get(field_name)
        if not isinstance(records, list):
            raise StateError(f"{path}: runtime {field_name} must be a list")
        for index, record in enumerate(records):
            context = f"{path}: runtime {field_name}[{index}]"
            if not isinstance(record, dict):
                raise StateError(f"{context} must be an object")
            try:
                ProcessIdentity.from_dict(record)
            except ValueError as exc:
                raise StateError(
                    f"{context} has an invalid process identity: {exc}"
                ) from exc
            if field_name == "tuned_processes":
                system = record.get("system")
                if not isinstance(system, str) or not system:
                    raise StateError(f"{context} system must be a non-empty string")
                affinity = record.get("affinity")
                if affinity is not None and (
                    not isinstance(affinity, list)
                    or any(
                        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                        for cpu in affinity
                    )
                ):
                    raise StateError(
                        f"{context} affinity must be a list of non-negative integers"
                    )
                ionice_class = record.get("ionice_class")
                ionice_value = record.get("ionice_value")
                if (ionice_class is None) != (ionice_value is None):
                    raise StateError(
                        f"{context} ionice_class and ionice_value must appear together"
                    )
                for integer_name in ("ionice_class", "ionice_value", "priority"):
                    integer_value = record.get(integer_name)
                    if integer_value is not None and (
                        isinstance(integer_value, bool)
                        or not isinstance(integer_value, int)
                    ):
                        raise StateError(f"{context} {integer_name} must be an integer")

    pids = value.get("pids")
    if not isinstance(pids, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
    ):
        raise StateError(f"{path}: runtime pids must be a list of positive integers")
    for field_name in ("observation", "decision", "process_report"):
        field_value = value.get(field_name)
        if field_value is not None and not isinstance(field_value, dict):
            raise StateError(f"{path}: runtime {field_name} must be an object")
    for field_name in ("config_error", "error"):
        field_value = value.get(field_name)
        if field_value is not None and (
            not isinstance(field_value, str) or not field_value
        ):
            raise StateError(f"{path}: runtime {field_name} must be a non-empty string")
    return value


class JobStore:
    def __init__(self, paths: AppPaths):
        self.paths = paths

    def _name_path(self, name: str, suffix: str) -> Path:
        return self.paths.run / f"{validate_job_name(name)}{suffix}"

    def spec_path(self, name: str) -> Path:
        return self._name_path(name, ".meta.json")

    def guard_path(self, name: str) -> Path:
        return self._name_path(name, ".guard.json")

    def legacy_guard_path(self, name: str) -> Path:
        return self._name_path(name, ".gpid")

    def runtime_path(self, name: str) -> Path:
        return self._name_path(name, ".runtime.json")

    def stop_path(self, name: str) -> Path:
        return self._name_path(name, ".stop")

    def lock_path(self, name: str) -> Path:
        return self._name_path(name, ".lock")

    def lock_name(self, name: str) -> JobNameLock:
        return JobNameLock(self.lock_path(name), validate_job_name(name))

    def write_spec(self, spec: JobSpec) -> None:
        value = spec.to_dict()
        JobSpec.from_dict(value)
        atomic_json_write(self.spec_path(spec.name), value)

    def read_spec(self, name: str) -> JobSpec:
        expected_name = validate_job_name(name)
        spec = JobSpec.from_dict(read_json(self.spec_path(expected_name)))
        if spec.name != expected_name:
            raise StateError(
                f"{self.spec_path(expected_name)}: metadata name is {spec.name!r}, "
                f"expected {expected_name!r}"
            )
        return spec

    def write_guard(self, name: str, identity: ProcessIdentity) -> None:
        atomic_json_write(self.guard_path(name), identity.to_dict())
        _atomic_write(self.legacy_guard_path(name), f"{identity.pid}\n")

    def read_guard(self, name: str) -> Optional[ProcessIdentity]:
        try:
            value = read_json(self.guard_path(name))
        except FileNotFoundError:
            return None
        try:
            return ProcessIdentity.from_dict(value)
        except ValueError as exc:
            raise StateError(
                f"{self.guard_path(name)}: invalid process identity: {exc}"
            ) from exc

    def write_runtime(self, name: str, value: dict[str, Any]) -> None:
        path = self.runtime_path(name)
        atomic_json_write(path, _validate_runtime_state(path, value))

    def read_runtime(self, name: str) -> Optional[dict[str, Any]]:
        path = self.runtime_path(name)
        try:
            value = read_json(path)
        except FileNotFoundError:
            return None
        return _validate_runtime_state(path, value)

    def request_stop(self, name: str, kill: bool) -> None:
        _atomic_write(self.stop_path(name), "kill\n" if kill else "soft\n")

    def read_stop(self, name: str) -> Optional[str]:
        try:
            value = self.stop_path(name).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if value not in {"soft", "kill"}:
            raise StateError(f"{self.stop_path(name)}: invalid stop request")
        return value

    def remove_active_state(self, name: str) -> None:
        for path in (
            self.stop_path(name),
            self.spec_path(name),
            self.guard_path(name),
            self.legacy_guard_path(name),
            self.runtime_path(name),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def remove_guard_state(self, name: str) -> None:
        for path in (self.guard_path(name), self.legacy_guard_path(name)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def list_specs(self) -> list[JobSpec]:
        specs: list[JobSpec] = []
        for path in sorted(self.paths.run.glob("*.meta.json")):
            name = path.name[: -len(".meta.json")]
            try:
                specs.append(self.read_spec(name))
            except (FileNotFoundError, StateError):
                continue
        return specs

    def persistence_path(self, name: str) -> Path:
        return self.paths.persist / f"{validate_job_name(name)}.job"

    def write_persistence(self, spec: PersistenceSpec) -> None:
        value = spec.to_dict()
        PersistenceSpec.from_dict(value)
        atomic_json_write(self.persistence_path(spec.name), value)

    def read_persistence(self, path: Path) -> PersistenceSpec:
        value = read_json(path)
        spec = PersistenceSpec.from_dict(value)
        expected_filename = f"{spec.name}.job"
        if path.name != expected_filename:
            raise StateError(
                f"{path}: persistence name is {spec.name!r}, "
                f"but the filename is not {expected_filename!r}"
            )
        return spec

    def list_persistence(self) -> Iterable[tuple[Path, PersistenceSpec]]:
        for path in sorted(self.paths.persist.glob("*.job")):
            yield path, self.read_persistence(path)

    def remove_persistence(self, name: str) -> None:
        try:
            self.persistence_path(name).unlink()
        except FileNotFoundError:
            pass

    def audit_state(self) -> list[str]:
        """Report malformed or orphaned state without modifying any file."""

        errors: list[str] = []
        metadata_names = {
            path.name[: -len(".meta.json")]
            for path in self.paths.run.glob("*.meta.json")
        }
        for path in sorted(self.paths.run.glob("*.meta.json")):
            name = path.name[: -len(".meta.json")]
            try:
                spec = self.read_spec(name)
            except (FileNotFoundError, StateError) as exc:
                if isinstance(exc, StateError):
                    errors.append(str(exc))
                continue
            for reader in (self.read_guard, self.read_runtime, self.read_stop):
                try:
                    reader(spec.name)
                except FileNotFoundError:
                    continue
                except StateError as exc:
                    errors.append(str(exc))

        for path in sorted(self.paths.run.glob("*.runtime.json")):
            name = path.name[: -len(".runtime.json")]
            if name in metadata_names:
                continue
            errors.append(
                f"{path}: recovery state has no job metadata; "
                f"inspect it before reusing job name {name!r}"
            )
            try:
                self.read_runtime(name)
            except StateError as exc:
                errors.append(str(exc))

        for path in sorted(self.paths.persist.glob("*.job")):
            try:
                self.read_persistence(path)
            except FileNotFoundError:
                continue
            except StateError as exc:
                errors.append(str(exc))
        return errors
