"""Upscale one video: choose Fast or Quality mode, size the output, run it as a job."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ...core import gpu as gpu_mod
from ...core import media, sizing, tools
from ...core.errors import FriendlyError
from ...core.files import remove_quietly
from ...core.jobs import JobContext
from ...core.process import Cancelled, ProcessError
from . import estimate, nvencc_vsr, quality

MODES = ("auto", "fast", "quality")
CODECS = ("hevc", "h264", "av1")
CONTAINERS = ("mp4", "mkv")


@dataclass
class UpscaleRequest:
    input: Path
    output: Path | None = None  # None = next to the input (or in output_dir)
    output_dir: Path | None = None
    size: str = "2160p"  # a sizing preset, or WxH
    mode: str = "auto"
    codec: str = "hevc"
    quality: str = "balanced"
    container: str = "mp4"
    vsr_quality: int = 4
    model: str = "general"
    gpu_index: int = 0
    overwrite: bool = False
    extra: dict = field(default_factory=dict)


def pick_gpu(index: int) -> gpu_mod.GpuInfo | None:
    gpus = gpu_mod.detect_gpus()
    if not gpus:
        return None
    return next((g for g in gpus if g.index == index), gpus[0])


def resolve_mode(mode: str, gpu: gpu_mod.GpuInfo | None) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    fast_ok = bool(gpu and gpu.vsr_capable) and tools.locate("nvencc")[0] is not None
    if mode == "fast":
        if not (gpu and gpu.vsr_capable):
            raise FriendlyError("fast_needs_rtx", "Fast mode needs an NVIDIA RTX GPU")
        if tools.locate("nvencc")[0] is None:
            raise FriendlyError("fast_needs_nvencc", "Fast mode needs NVEncC (install it on the System page)")
        return "fast"
    if mode == "quality":
        return "quality"
    return "fast" if fast_ok else "quality"


def output_size(info: media.MediaInfo, size: str) -> tuple[int, int]:
    v = info.video
    if v is None or v.display_size is None:
        raise FriendlyError("no_video", "the source has no video stream")
    if "x" in size.lower() and size.lower() not in sizing.FACTORS:
        return sizing.parse_custom(size)
    return sizing.target_size(*v.display_size, size)


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    for i in range(1, 1000):
        c = p.with_name(f"{p.stem} ({i}){p.suffix}")
        if not c.exists():
            return c
    raise FileExistsError(p)


def default_output(req: UpscaleRequest, mode: str, w: int, h: int) -> Path:
    folder = req.output_dir or req.input.parent
    return folder / f"{req.input.stem}_{w}x{h}_{mode}.{req.container}"


def run_upscale(req: UpscaleRequest, ctx: JobContext) -> Path:
    src = req.input.resolve()
    if not src.is_file():
        raise FriendlyError("file_missing", f"file not found: {src}", name=src.name)
    ctx.report(None, "reading source", phase="preparing")
    info = media.probe(src)
    w, h = output_size(info, req.size)
    gpu = pick_gpu(req.gpu_index)
    mode = resolve_mode(req.mode, gpu)
    dst = req.output.resolve() if req.output else default_output(req, mode, w, h)
    if dst == src:
        raise FriendlyError("same_file", "the output file must differ from the input")
    if dst.suffix.lower().lstrip(".") not in CONTAINERS:
        raise ValueError(f"output must be .{' or .'.join(CONTAINERS)}")
    if not req.overwrite:
        dst = unique_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
    ctx.log(f"{src.name}: {info.video.display_size} -> {w}x{h}, {mode} mode, output {dst}")

    def progress(frac, msg):
        phase = ("downloading" if msg.startswith("downloading")
                 else "preparing" if frac is None else "working")
        ctx.report(frac, f"{mode}: {msg}", phase=phase)

    def do(m: str) -> None:
        if m == "fast":
            nvencc_vsr.run(
                src, partial, info,
                nvencc_vsr.FastOptions(w, h, req.codec, req.quality, req.vsr_quality),
                gpu, on_progress=progress, on_log=ctx.log, cancel=ctx.cancel,
            )
        else:
            quality.run(
                src, partial, info,
                quality.QualityOptions(w, h, req.model, req.codec, req.quality,
                                       device_id=_dml_device(gpu)),
                gpu.vendor if gpu else "other",
                on_progress=progress, on_log=ctx.log, cancel=ctx.cancel,
            )

    started = time.monotonic()
    try:
        try:
            do(mode)
        except ProcessError as e:
            # In auto mode, fall back to Quality if the RTX path fails (driver/NGX problems).
            if req.mode != "auto" or mode != "fast":
                raise
            ctx.log(f"Fast mode failed ({e.returncode}); retrying in Quality mode")
            remove_quietly(partial)
            mode = "quality"
            started = time.monotonic()
            do(mode)
        work = time.monotonic() - started
        partial.replace(dst)
    except BaseException:
        if not remove_quietly(partial):
            ctx.log(f"could not delete the unfinished file {partial}")
        raise
    _learn(info, req, mode, w, h, dst, work)
    ctx.report(1.0, f"done: {dst.name}", phase="done")
    return dst


def _learn(info, req: UpscaleRequest, mode: str, w: int, h: int, dst: Path, work: float) -> None:
    """Teach the estimator how fast this PC is and how big files really come out."""
    try:
        v = info.video
        predicted, _ = estimate.video_bytes(info, w, h, mode, req.codec, req.quality, stats={})
        actual = dst.stat().st_size - estimate.audio_bytes(info, req.container)
        estimate.learn(
            mode=mode, model=req.model, codec=req.codec, quality=req.quality,
            frames=info.frame_count() or 0, in_pixels=(v.width or 0) * (v.height or 0), out_pixels=w * h,
            work_seconds=work - estimate.STARTUP_S[mode],
            predicted_video_bytes=int(predicted), actual_video_bytes=int(actual),
        )
    except Exception:
        pass  # estimates are a convenience; never fail a finished job over them


def _dml_device(gpu: gpu_mod.GpuInfo | None) -> int:
    # DirectML enumerates adapters in DXGI order, which puts the primary GPU first. Our GPU
    # index follows nvidia-smi first, then other adapters, which matches for single-GPU systems.
    return gpu.index if gpu else 0


__all__ = ["UpscaleRequest", "run_upscale", "resolve_mode", "output_size", "MODES", "CODECS", "Cancelled"]
