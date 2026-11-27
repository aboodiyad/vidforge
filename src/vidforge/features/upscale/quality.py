"""Quality mode: Real-ESRGAN on any DirectML GPU (NVIDIA, AMD, Intel), CPU as a last resort;
on RTX cards with the NVIDIA speed-up add-on, TensorRT (about 2x faster on an RTX 2060 SUPER).

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

# TensorRT tile edge per model. Bigger tiles mean less overlap work, but "max" got slower above
# 768 (720p frames: 512 -> 4.1 s, 768 -> 2.4 s, whole frame -> 13.5 s per frame on a 2060 SUPER).
TRT_TILE = {"general": 1920, "anime": 1920, "max": 768}


class AutoEngine:
    """TensorRT when it works, else DirectML. TensorRT prepares itself on the first frame, so a
    failure there (or later) switches to DirectML and redoes that frame; the video continues."""

    def __init__(self, key: str, model_path: Path, *, device_id: int, gpu, on_log: Callable[[str], None]):
        from . import speedup

        self.key, self.model_path, self.device_id, self.on_log = key, model_path, device_id, on_log
        self.engine = None
        self.on_trt = False
        if speedup.supported(gpu) and speedup.is_installed():
            try:
                from .trt_engine import TrtEsrganEngine

                self.engine = TrtEsrganEngine(model_path, speedup.root(), tile=TRT_TILE.get(key, 512), on_log=on_log)
                self.on_trt = True
            except Exception as e:  # noqa: BLE001 - any problem: use DirectML
                on_log(f"NVIDIA speed-up not usable ({e}); using DirectML")
        if self.engine is None:
            self._directml()

    def _directml(self) -> None:
        self.on_trt = False
        self.engine = EsrganEngine(self.model_path, device_id=self.device_id, tile=QUALITY_MODELS[self.key].tile)

    @property
    def scale(self) -> int:
        return self.engine.scale

    @property
    def provider(self) -> str:
        return self.engine.provider

    def upscale(self, frame):
        try:
            return self.engine.upscale(frame)
        except Exception as e:  # noqa: BLE001
            if not self.on_trt:
                raise
            self.on_log(f"NVIDIA speed-up stopped working ({e}); continuing with DirectML")
            self._directml()
            return self.engine.upscale(frame)



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
    gpu=None,
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
    engine = AutoEngine(opts.model, model_path, device_id=opts.device_id, gpu=gpu, on_log=on_log)
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
