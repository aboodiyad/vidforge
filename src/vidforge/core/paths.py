"""Where VidForge keeps its tools, models, temp files and logs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VidForge"
PORTABLE_DIR = "vidforge-data"


def portable_root() -> Path:
    """A `vidforge-data` folder next to the Python environment (the project folder for a dev
    checkout, the install folder for a portable build)."""
    return Path(sys.prefix).resolve().parent / PORTABLE_DIR


def data_root() -> Path:
    """VIDFORGE_HOME if set, else the portable folder if it exists, else %LOCALAPPDATA%\\VidForge.

    The portable folder also avoids MSIX file-system virtualization: when a packaged app (for
    example a terminal inside a Store app) starts VidForge, writes under %LOCALAPPDATA% are
    redirected into that app's private storage and are invisible to other programs."""
    override = os.environ.get("VIDFORGE_HOME")
    if override:
        return Path(override)
    portable = portable_root()
    if portable.is_dir():
        return portable
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


def tools_dir() -> Path:
    return data_root() / "tools"


def models_dir() -> Path:
    return data_root() / "models"


def temp_dir() -> Path:
    return data_root() / "temp"


def logs_dir() -> Path:
    return data_root() / "logs"


def settings_file() -> Path:
    return data_root() / "settings.json"


def ensure_dirs() -> None:
    for d in (tools_dir(), models_dir(), temp_dir(), logs_dir()):
        d.mkdir(parents=True, exist_ok=True)
