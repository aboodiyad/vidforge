"""Face models, downloaded on first use and checked against pinned SHA-256 hashes.

The ONNX files come from FaceFusion's public model releases (facefusion/facefusion-assets).
YuNet's hash matches the digest GitHub publishes; GFPGAN's and CodeFormer's were pinned on
first download and the GFPGAN conversion was checked against the official PyTorch weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import models
from ...core.process import CancelToken

_BASE = "https://github.com/facefusion/facefusion-assets/releases/download"


@dataclass(frozen=True)
class FaceModel:
    key: str  # key in core.models
    licence: str
    size_mb: float
    commercial_ok: bool


FACE_MODELS = {
    "yunet": FaceModel("yunet_2023_mar", "MIT", 0.2, True),
    "gfpgan": FaceModel("gfpgan_1.4", "Apache-2.0", 340.3, True),
    "codeformer": FaceModel("codeformer", "S-Lab License 1.0 (non-commercial)", 377.0, False),
}

for _name, _rel, _sha in [
    ("yunet_2023_mar", "models-3.4.0", "490d79bc15c0e958d91c999c566c6a338b310ead96e0274dd004e59dd6abecb9"),
    ("gfpgan_1.4", "models-3.0.0", "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"),
    ("codeformer", "models-3.0.0", "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f"),
]:
    _fm = next(m for m in FACE_MODELS.values() if m.key == _name)
    models.register(models.ModelSpec(
        key=_name, display=_name, url=f"{_BASE}/{_rel}/{_name}.onnx", sha256=_sha, filename=f"{_name}.onnx",
        size_mb=_fm.size_mb, license=_fm.licence, feature="face", commercial_ok=_fm.commercial_ok,
    ))


def is_installed(name: str) -> bool:
    return models.is_installed(FACE_MODELS[name].key)


def ensure(name: str, *, progress: Callable[[int, int | None], None] | None = None,
           cancel: CancelToken | None = None) -> Path:
    return models.ensure(FACE_MODELS[name].key, progress=progress, cancel=cancel)
