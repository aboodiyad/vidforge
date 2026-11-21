"""Real-ESRGAN models for Quality mode: official weights, converted to ONNX on first use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import models
from ...core.process import CancelToken
from . import esrgan_convert

_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
# Bump when the converter's output changes, so cached ONNX files are rebuilt.
CONVERTER_VERSION = 1


@dataclass(frozen=True)
class QualityModel:
    key: str  # short name used in the UI / CLI
    model_key: str  # key in core.models
    label: str
    tile: int  # input tile edge; frames up to this size run in one piece


QUALITY_MODELS = {
    m.key: m
    for m in [
        QualityModel("general", "realesr-general-x4v3", "Balanced: general video (recommended)", 1920),
        QualityModel("max", "RealESRGAN_x4plus", "Max detail: strongest restoration, slow", 512),
        QualityModel("anime", "realesr-animevideov3", "Anime and cartoons", 1920),
    ]
}

for _key, _file, _sha, _size in [
    ("realesr-general-x4v3", "v0.2.5.0/realesr-general-x4v3.pth",
     "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292", 4.9),
    ("RealESRGAN_x4plus", "v0.1.0/RealESRGAN_x4plus.pth",
     "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", 67.0),
    ("realesr-animevideov3", "v0.2.5.0/realesr-animevideov3.pth",
     "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d", 2.5),
]:
    models.register(models.ModelSpec(
        key=_key, display=_key, url=f"{_BASE}/{_file}", sha256=_sha, filename=_file.split("/")[-1],
        size_mb=_size, license="BSD-3-Clause", feature="upscale",
    ))


def onnx_path(key: str) -> Path:
    spec = QUALITY_MODELS[key]
    return models.model_path(spec.model_key).with_name(f"{spec.model_key}.fp16-u8.v{CONVERTER_VERSION}.onnx")


def ensure_onnx(
    key: str,
    *,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    out = onnx_path(key)
    if out.is_file():
        return out
    pth = models.ensure(QUALITY_MODELS[key].model_key, progress=progress, cancel=cancel)
    esrgan_convert.convert(pth, out, fp16=True, uint8_io=True)
    return out
