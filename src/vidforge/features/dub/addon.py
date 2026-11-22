"""The voice add-on: a separate Python environment with PyTorch, Demucs and Chatterbox.

It lives in the data folder (addons\\voice), not in VidForge's own environment: Chatterbox needs
numpy 1.x and a fixed PyTorch, and PyTorch with CUDA is about 3 GB that people who only upscale
never need. The voice work runs there as a separate process (voice_worker.py), which also hands
the graphics memory back as soon as it ends.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

from ...core import paths
from ...core.gpu import GpuInfo
from ...core.process import CancelToken, run_process

VERSION = 1  # bump when the requirements change, so old installs are redone
TORCH = ("torch==2.6.0", "torchaudio==2.6.0")
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu124", "cpu": "https://download.pytorch.org/whl/cpu"}
CHATTERBOX = "chatterbox-tts==0.1.7"
INSTALL_MB = {"cuda": 5200, "cpu": 1600}  # disk space after install, roughly


def root() -> Path:
    return paths.data_root() / "addons" / "voice"


def python() -> Path:
    return root() / "Scripts" / "python.exe" if sys.platform == "win32" else root() / "bin" / "python"


def _marker() -> Path:
    return root() / "vidforge-addon.json"


def status() -> dict | None:
    """The install record ({version, device}) if the add-on is installed and current."""
    try:
        data = json.loads(_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("version") != VERSION or not python().is_file():
        return None
    return data


def is_installed() -> bool:
    return status() is not None


def device_for(gpu: GpuInfo | None) -> str:
    return "cuda" if gpu and gpu.vendor == "nvidia" else "cpu"


def requirements_file() -> Path:
    return Path(__file__).with_name("requirements-voice.txt")


def base_python() -> str:
    """The Python that VidForge's own environment was made from (a venv cannot copy a venv)."""
    return getattr(sys, "_base_executable", None) or sys.executable


def install(gpu: GpuInfo | None, *, cancel: CancelToken, on_log: Callable[[str], None],
            on_step: Callable[[int, int, str], None]) -> Path:
    device = device_for(gpu)
    target = root()
    if _marker().exists():
        _marker().unlink()
    steps = [
        ("environment", [base_python(), "-m", "venv", str(target)]),
        ("pip", [str(python()), "-m", "pip", "install", "--upgrade", "pip"]),
        ("torch", [str(python()), "-m", "pip", "install", *TORCH, "--index-url", TORCH_INDEX[device]]),
        ("packages", [str(python()), "-m", "pip", "install", "-r", str(requirements_file())]),
        ("voice", [str(python()), "-m", "pip", "install", "--no-deps", CHATTERBOX]),
    ]
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONIOENCODING="utf-8")
    for i, (name, cmd) in enumerate(steps):
        on_step(i, len(steps), name)
        on_log("$ " + " ".join(cmd))
        run_process(cmd, on_line=on_log, cancel=cancel, env=env, tail_lines=200)
    _marker().write_text(json.dumps({"version": VERSION, "device": device}), encoding="utf-8")
    on_step(len(steps), len(steps), "done")
    return target


def remove() -> None:
    shutil.rmtree(root(), ignore_errors=True)
