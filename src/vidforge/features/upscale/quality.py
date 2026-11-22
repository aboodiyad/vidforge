"""Quality mode: Real-ESRGAN on any DirectML GPU (NVIDIA, AMD, Intel), CPU as a last resort.

Decoding, upscaling and encoding run at the same time (see core.framepipe.run_frames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import encoders
from ...core.errors import FriendlyError
from ...core.framepipe import FrameReader, FrameWriter, run_frames
from ...core.media import MediaInfo
from ...core.process import CancelToken
from .esrgan_onnx import EsrganEngine
from .models import QUALITY_MODELS, ensure_onnx



@dataclass
class QualityOptions:
    width: int
    height: int
    model: str = "general"
    codec: str = "hevc"
    quality: str = "balanced"
    device_id: int = 0


def run(
    src: Path,
    dst: Path,
    info: MediaInfo,
    opts: QualityOptions,
    vendor: str,
    *,
    on_progress: Callable[[float | None, str], None],
    on_log: Callable[[str], None],
    cancel: CancelToken,
) -> None:
    v = info.video
    if v is None:
        raise ValueError("the source has no video stream")
    if v.hdr:
        raise FriendlyError("hdr_quality", "Quality mode works in 8-bit SDR and would clip this HDR video; use Fast mode")
    spec = QUALITY_MODELS[opts.model]

    def dl(done: int, total: int | None) -> None:
        on_progress(None, f"downloading model {done / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB")

    model_path = ensure_onnx(opts.model, progress=dl, cancel=cancel)
    on_progress(None, "loading model")
    engine = EsrganEngine(model_path, device_id=opts.device_id, tile=spec.tile)
    on_log(f"model {spec.model_key} on {engine.provider}, tile {spec.tile}")
    enc = encoders.choose(opts.codec, opts.quality, vendor)
    on_log(f"encoder {enc.name} ({'hardware' if enc.hardware else 'software'})")

    reader = FrameReader(src, info)
    s = engine.scale
    writer = FrameWriter(
        src, dst, info,
        in_size=(reader.width * s, reader.height * s),
        out_size=(opts.width, opts.height),
        fps=reader.fps,
        encoder_args=enc.ffmpeg_args(),
        pix_fmt=enc.pix_fmt,
    )
    on_log("$ " + " ".join(writer.command))
    run_frames(reader, writer, engine.upscale, info.frame_count(), on_progress, cancel)
