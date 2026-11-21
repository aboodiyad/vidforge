"""Quality mode: Real-ESRGAN on any DirectML GPU (NVIDIA, AMD, Intel), CPU as a last resort.

Three stages run at once: FFmpeg decodes frames (reader thread), the model upscales them (this
thread; ONNX Runtime releases the GIL while the GPU works), and FFmpeg encodes them (writer
thread). Bounded queues between the stages keep memory use flat.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import encoders
from ...core.framepipe import FrameReader, FrameWriter
from ...core.media import MediaInfo
from ...core.process import Cancelled, CancelToken
from .esrgan_onnx import EsrganEngine
from .models import QUALITY_MODELS, ensure_onnx

_END = object()


@dataclass
class QualityOptions:
    width: int
    height: int
    model: str = "general"
    codec: str = "hevc"
    quality: str = "balanced"
    device_id: int = 0


class _Stage(threading.Thread):
    def __init__(self, fn, name: str):
        super().__init__(name=name, daemon=True)
        self.fn = fn
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.fn()
        except BaseException as e:  # reported by the main thread
            self.error = e


def _put(q: queue.Queue, item, cancel: CancelToken, failed: Callable[[], bool]) -> None:
    while True:
        if cancel.cancelled or failed():
            raise Cancelled()
        try:
            q.put(item, timeout=0.2)
            return
        except queue.Full:
            continue


def _get(q: queue.Queue, cancel: CancelToken, failed: Callable[[], bool]):
    while True:
        if cancel.cancelled or failed():
            raise Cancelled()
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            continue


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
        raise ValueError("Quality mode works in 8-bit SDR and would clip this HDR video; use Fast mode")
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
    total = info.frame_count()
    q_in: queue.Queue = queue.Queue(maxsize=3)
    q_out: queue.Queue = queue.Queue(maxsize=3)
    stages: list[_Stage] = []
    failed = lambda: any(st.error is not None for st in stages)  # noqa: E731

    def read() -> None:
        for frame in reader:
            _put(q_in, frame, cancel, failed)
        _put(q_in, _END, cancel, failed)

    def write() -> None:
        while (item := _get(q_out, cancel, failed)) is not _END:
            writer.write(item)
        writer.close()

    stages += [_Stage(read, "decode"), _Stage(write, "encode")]
    for st in stages:
        st.start()
    done = 0
    try:
        while (frame := _get(q_in, cancel, failed)) is not _END:
            _put(q_out, engine.upscale(frame), cancel, failed)
            done += 1
            frac = min(done / total, 0.999) if total else None
            on_progress(frac, f"{done}/{total or '?'} frames")
        _put(q_out, _END, cancel, failed)
        stages[1].join()
    except Cancelled:
        pass
    finally:
        if cancel.cancelled or failed() or stages[1].is_alive():
            reader.kill()
            writer.kill()
        for st in stages:
            st.join(timeout=5)
    if cancel.cancelled:
        raise Cancelled()  # killed processes report errors too; cancellation wins
    errors = [st.error for st in stages if st.error is not None and not isinstance(st.error, Cancelled)]
    if errors:
        raise errors[0]
