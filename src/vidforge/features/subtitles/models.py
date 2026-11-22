"""Whisper speech-recognition models (whisper.cpp "ggml" format) and the Silero voice detector,
from the official Hugging Face repositories, pinned by the SHA-256 that Hugging Face publishes.
The Silero voice detector comes as ONNX from onnx-community/silero-vad. Both are MIT-licensed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import models
from ...core.process import CancelToken

_HF = "https://huggingface.co/{repo}/resolve/main/{file}"


@dataclass(frozen=True)
class SpeechModel:
    key: str  # short name used by the UI / CLI
    file: str
    size_mb: float
    dtw: str  # whisper.cpp alignment-head preset for precise word timings


SPEECH_MODELS = {
    m.key: m
    for m in [
        SpeechModel("quick", "ggml-small-q5_1.bin", 190.1, "small"),
        SpeechModel("balanced", "ggml-large-v3-turbo-q5_0.bin", 574.0, "large.v3.turbo"),
        SpeechModel("best", "ggml-large-v3-turbo-q8_0.bin", 874.2, "large.v3.turbo"),
    ]
}
VAD_KEY = "silero-vad"

for _repo, _file, _sha, _mb in [
    ("ggerganov/whisper.cpp", "ggml-small-q5_1.bin",
     "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb", 190.1),
    ("ggerganov/whisper.cpp", "ggml-large-v3-turbo-q5_0.bin",
     "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", 574.0),
    ("ggerganov/whisper.cpp", "ggml-large-v3-turbo-q8_0.bin",
     "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1", 874.2),
]:
    models.register(models.ModelSpec(
        key=_file.removesuffix(".bin"), display=_file, url=_HF.format(repo=_repo, file=_file), sha256=_sha,
        filename=_file, size_mb=_mb, license="MIT", feature="subtitles",
    ))


models.register(models.ModelSpec(
    key=VAD_KEY, display="Silero VAD", url=_HF.format(repo="onnx-community/silero-vad", file="onnx/model.onnx"),
    sha256="a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808", filename="silero_vad.onnx",
    size_mb=2.2, license="MIT", feature="subtitles",
))


def _key(file: str) -> str:
    return file.removesuffix(".bin")


def is_installed(level: str) -> bool:
    return models.is_installed(_key(SPEECH_MODELS[level].file))


def vad_installed() -> bool:
    return models.is_installed(VAD_KEY)


def ensure(level: str, *, progress: Callable[[int, int | None], None] | None = None,
           cancel: CancelToken | None = None) -> Path:
    return models.ensure(_key(SPEECH_MODELS[level].file), progress=progress, cancel=cancel)


def ensure_vad(*, cancel: CancelToken | None = None) -> Path:
    return models.ensure(VAD_KEY, cancel=cancel)


def download_mb(level: str) -> float:
    mb = 0.0 if is_installed(level) else SPEECH_MODELS[level].size_mb
    return mb + (0.0 if vad_installed() else 2.2)
