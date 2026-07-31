"""Structured and human-readable supervisor event logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

if __package__:
    from .model import utc_now
    from .state import AppPaths, validate_job_name
else:  # Keep direct execution from a checkout working.
    from model import utc_now
    from state import AppPaths, validate_job_name


def _reject_non_standard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


class EventJournal:
    """Append and read independent JSON Lines events for one job."""

    def __init__(self, paths: AppPaths, name: str):
        self.name = validate_job_name(name)
        self.json_path = paths.logs / f"{self.name}.events.jsonl"
        self.text_path = paths.logs / f"{self.name}.guard.log"

    @staticmethod
    def _append(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)

    def emit(self, event: str, message: str, **details: Any) -> dict[str, Any]:
        payload = {
            **details,
            "timestamp": utc_now(),
            "job": self.name,
            "event": event,
            "message": message,
        }
        line = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._append(self.json_path, line + "\n")
        self._append(
            self.text_path,
            f"[guard] {payload['timestamp']} {event}: {message}\n",
        )
        return payload

    def read(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last valid object events, oldest first."""

        if limit < 1:
            return []
        try:
            lines = self.json_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            return []

        events: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_non_standard_number,
                )
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                events.append(value)
                if len(events) == limit:
                    break
        events.reverse()
        return events
