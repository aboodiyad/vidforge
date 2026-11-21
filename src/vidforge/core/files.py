"""File helpers."""

from __future__ import annotations

import time
from pathlib import Path


def remove_quietly(path: Path, timeout: float = 5.0) -> bool:
    """Delete *path*, retrying while Windows still holds it open (a just-killed process can
    keep its handle for a moment). Never raises; returns True if the file is gone."""
    end = time.monotonic() + timeout
    while True:
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if time.monotonic() >= end:
                return False
            time.sleep(0.1)
        except OSError:
            return False
