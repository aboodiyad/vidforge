"""Fix faces in one video as a job: choose the background mode, size the output, run, learn."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from ...core import media
from ...core.errors import FriendlyError
from ...core.files import remove_quietly
from ...core.jobs import JobContext
from ..upscale.service import CONTAINERS, _dml_device, output_size, pick_gpu, resolve_mode, unique_path
from . import estimate, pipeline
from .enhancer import EnhanceSettings

KEEP = "keep"


@dataclass
class FaceRequest:
    input: Path
    output: Path | None = None
    output_dir: Path | None = None
    size: str = KEEP  # "keep" or an upscale size preset / WxH
    background: str = "auto"  # auto / fast / quality (only used when the size changes)
    model: str = "gfpgan"  # gfpgan / codeformer
    strength: float = 1.0
    fidelity: float = 0.7
    esrgan_model: str = "general"
    vsr_quality: int = 4
    codec: str = "hevc"
    quality: str = "balanced"
    container: str = "mp4"
    gpu_index: int = 0
    overwrite: bool = False


def keep_size(info: media.MediaInfo) -> tuple[int, int]:
    """Same picture size, with square pixels (anamorphic video is widened)."""
    dw, dh = info.video.display_size
    return max(2, dw // 2 * 2), max(2, dh // 2 * 2)


def target(info: media.MediaInfo, req: FaceRequest, gpu) -> tuple[int, int, str]:
    if info.video is None or info.video.display_size is None:
        raise FriendlyError("no_video", "the source has no video stream")
    if req.size == KEEP:
        w, h = keep_size(info)
        return w, h, KEEP
    w, h = output_size(info, req.size)
    return w, h, resolve_mode(req.background, gpu)


def run_faces(req: FaceRequest, ctx: JobContext) -> Path:
    src = req.input.resolve()
    if not src.is_file():
        raise FriendlyError("file_missing", f"file not found: {src}", name=src.name)
    ctx.report(None, "reading source", phase="preparing")
    info = media.probe(src)
    gpu = pick_gpu(req.gpu_index)
    w, h, background = target(info, req, gpu)
    folder = req.output_dir or src.parent
    suffix = "faces" if background == KEEP else f"faces_{w}x{h}_{background}"
    dst = req.output.resolve() if req.output else folder / f"{src.stem}_{suffix}.{req.container}"
    if dst == src:
        raise FriendlyError("same_file", "the output file must differ from the input")
    if dst.suffix.lower().lstrip(".") not in CONTAINERS:
        raise ValueError(f"output must be .{' or .'.join(CONTAINERS)}")
    if not req.overwrite:
        dst = unique_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
    ctx.log(f"{src.name}: {info.video.display_size} -> {w}x{h}, background {background}, model {req.model}")

    def progress(frac, msg):
        phase = ("downloading" if msg.startswith("downloading")
                 else "preparing" if frac is None else "working")
        ctx.report(frac, msg, phase=phase)

    opts = pipeline.FaceOptions(
        width=w, height=h, model=req.model, background=background, esrgan_model=req.esrgan_model,
        vsr_quality=req.vsr_quality, codec=req.codec, quality=req.quality, device_id=_dml_device(gpu),
        enhance=EnhanceSettings(strength=req.strength, fidelity=req.fidelity),
    )
    started = time.monotonic()
    try:
        stats = pipeline.run(src, partial, info, opts, gpu, on_progress=progress, on_log=ctx.log, cancel=ctx.cancel)
        work = time.monotonic() - started
        partial.replace(dst)
    except BaseException:
        if not remove_quietly(partial):
            ctx.log(f"could not delete the unfinished file {partial}")
        raise
    try:
        estimate.learn(info, req, background, w, h, work, stats, dst)
    except Exception:
        pass  # estimates are a convenience
    ctx.report(1.0, f"done: {dst.name}", phase="done")
    return dst


__all__ = ["FaceRequest", "run_faces", "target", "KEEP"]
