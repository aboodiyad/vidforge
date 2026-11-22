"""Fix faces in a whole video, optionally making it bigger at the same time.

Background options:
- keep the size: faces are restored in the original frames;
- Best quality: Real-ESRGAN upscales each frame in the same pass, then faces are restored;
- Fast: NVEncC RTX VSR first writes a high-quality upscaled copy, then faces are restored in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ...core import encoders, media
from ...core.errors import FriendlyError
from ...core.files import remove_quietly
from ...core.framepipe import FrameReader, FrameWriter, run_frames
from ...core.gpu import GpuInfo
from ...core.media import MediaInfo
from ...core.process import CancelToken
from ..upscale import nvencc_vsr
from ..upscale.esrgan_onnx import EsrganEngine
from ..upscale.models import QUALITY_MODELS, ensure_onnx
from .detect import FaceDetector
from .enhancer import EnhanceSettings, FaceEnhancer, FrameStats
from .models import FACE_MODELS, ensure as ensure_model
from .restore import FaceRestorer

FAST_SHARE = 0.25  # share of the progress bar for the Fast upscaling pass


@dataclass
class FaceOptions:
    width: int  # output size (display orientation)
    height: int
    model: str = "gfpgan"
    background: str = "keep"  # keep / fast / quality
    esrgan_model: str = "general"
    vsr_quality: int = 4
    codec: str = "hevc"
    quality: str = "balanced"
    device_id: int = 0
    enhance: EnhanceSettings = field(default_factory=EnhanceSettings)


def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    interp = cv2.INTER_AREA if w < fw else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (w, h), interpolation=interp)


def run(
    src: Path,
    dst: Path,
    info: MediaInfo,
    opts: FaceOptions,
    gpu: GpuInfo | None,
    *,
    on_progress: Callable[[float | None, str], None],
    on_log: Callable[[str], None],
    cancel: CancelToken,
) -> FrameStats:
    v = info.video
    if v is None:
        raise FriendlyError("no_video", "the source has no video stream")
    if v.hdr:
        raise FriendlyError("hdr_faces", "face fixing works in 8-bit SDR and would clip this HDR video")

    def dl(done: int, total: int | None) -> None:
        on_progress(None, f"downloading model {done / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB")

    detector_path = ensure_model("yunet", progress=dl, cancel=cancel)
    restorer_path = ensure_model(opts.model, progress=dl, cancel=cancel)
    on_progress(None, "loading model")
    restorer = FaceRestorer(restorer_path, device_id=opts.device_id)
    on_log(f"face model {FACE_MODELS[opts.model].key} on {restorer.provider}")
    vendor = gpu.vendor if gpu else "other"

    face_src, face_info, temp = src, info, None
    progress_base, progress_span = 0.0, 1.0
    engine = None
    if opts.background == "fast":
        temp = dst.with_name(f"{dst.stem}.upscaled-tmp.mkv")
        fast = nvencc_vsr.FastOptions(opts.width, opts.height, "hevc", "high", opts.vsr_quality)

        def fast_progress(frac, msg):
            on_progress(None if frac is None else frac * FAST_SHARE, f"upscaling: {msg}")

        nvencc_vsr.run(src, temp, info, fast, gpu, on_progress=fast_progress, on_log=on_log, cancel=cancel)
        face_src, face_info = temp, media.probe(temp)
        progress_base, progress_span = FAST_SHARE, 1.0 - FAST_SHARE
    elif opts.background == "quality":
        spec = QUALITY_MODELS[opts.esrgan_model]
        engine = EsrganEngine(ensure_onnx(opts.esrgan_model, progress=dl, cancel=cancel),
                              device_id=opts.device_id, tile=spec.tile)
        on_log(f"background model {spec.model_key} on {engine.provider}")

    try:
        reader = FrameReader(face_src, face_info)
        detector = FaceDetector(detector_path)
        enhancer = FaceEnhancer(detector, restorer, opts.enhance, face_info.video.fps or 30.0)
        w, h = opts.width, opts.height

        def process(frame: np.ndarray) -> np.ndarray:
            if engine is not None:
                out = _resize(engine.upscale(frame), w, h)
            else:
                out = _resize(frame, w, h).copy()
            return enhancer.process(frame, out)

        enc = encoders.choose(opts.codec, opts.quality, vendor)
        on_log(f"encoder {enc.name} ({'hardware' if enc.hardware else 'software'})")
        writer = FrameWriter(face_src, dst, face_info, in_size=(w, h), out_size=(w, h), fps=reader.fps,
                             encoder_args=enc.ffmpeg_args(), pix_fmt=enc.pix_fmt)
        on_log("$ " + " ".join(writer.command))

        def faces_progress(frac, msg):
            on_progress(None if frac is None else progress_base + frac * progress_span, f"faces: {msg}")

        run_frames(reader, writer, process, face_info.frame_count(), faces_progress, cancel)
        on_log(f"restored {enhancer.stats.faces} faces in {enhancer.stats.frames} frames")
        return enhancer.stats
    finally:
        if temp is not None:
            remove_quietly(temp)
