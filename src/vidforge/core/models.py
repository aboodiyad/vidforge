"""Registry of AI model files, downloaded on first use and verified by pinned SHA-256.

Features register their models here in their own milestone (M1: Real-ESRGAN, M2: YuNet/GFPGAN/
CodeFormer, M3: Whisper, M4: LLM/TTS/Demucs). A model without a pinned hash cannot be registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paths
from .download import download, sha256_file
from .process import CancelToken


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    url: str
    sha256: str
    filename: str
    size_mb: float
    license: str
    feature: str
    commercial_ok: bool = True
    folder: str | None = None  # shared folder for models made of several files (default: key)

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError(f"model {self.key}: a pinned 64-hex SHA-256 is required")


MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.key in MODELS:
        raise ValueError(f"duplicate model key {spec.key}")
    MODELS[spec.key] = spec
    return spec


def model_path(key: str) -> Path:
    spec = MODELS[key]
    return paths.models_dir() / (spec.folder or spec.key) / spec.filename


def is_installed(key: str, verify: bool = False) -> bool:
    p = model_path(key)
    if not p.is_file():
        return False
    return sha256_file(p) == MODELS[key].sha256 if verify else True


def ensure(
    key: str,
    *,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    p = model_path(key)
    if p.is_file():
        return p
    spec = MODELS[key]
    return download(spec.url, p, sha256=spec.sha256, progress=progress, cancel=cancel)
