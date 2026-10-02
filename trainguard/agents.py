"""Owner-controlled exceptions shared with Warden."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path


def global_ignored(home: Path) -> bool:
    """An owner exception covers every job and expires without Warden running."""
    try:
        with (home / "global-override.json").open(encoding="utf-8") as source:
            value = json.load(source)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("enabled") is not True
        ):
            return False
        expiry = value.get("expires_at", "missing")
        return expiry is None or (
            isinstance(expiry, (int, float))
            and not isinstance(expiry, bool)
            and math.isfinite(expiry)
            and expiry > time.time()
        )
    except (OSError, UnicodeError, ValueError):
        return False
