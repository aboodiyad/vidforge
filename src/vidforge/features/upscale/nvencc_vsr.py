"""Fast mode: NVIDIA RTX Video Super Resolution through NVEncC.

One NVEncC process does everything on the GPU: decode (NVDEC, or libavcodec for codecs the
GPU cannot decode), NGX VSR upscaling, NVENC encoding, and muxing of the copied audio,
subtitles and chapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ...core import tools
from ...core.gpu import GpuInfo
from ...core.media import MediaInfo, StreamInfo
from ...core.process import CancelToken, run_process_console
from ...core.streams import container_of, plan_audio, plan_subtitles

# NVEncC --qvbr targets per quality level and codec (lower = better quality, bigger file).
QVBR = {
    "high": {"hevc": 20, "h264": 20, "av1": 26},
    "balanced": {"hevc": 24, "h264": 23, "av1": 32},
    "small": {"hevc": 28, "h264": 27, "av1": 38},
}

_NVDEC_CODECS = {"h264", "hevc", "mpeg1video", "mpeg2video", "mpeg4", "vc1", "vp8", "vp9"}
# Harmless message NVEncC prints for every resize; it only matters for streams that change
# resolution mid-way. Kept out of the job log to avoid confusion.
_NOISE = ("normalization resize falls back",)


@dataclass
class FastOptions:
    width: int
    height: int
    codec: str = "hevc"  # hevc / h264 / av1
    quality: str = "balanced"
    vsr_quality: int = 4  # NGX VSR quality level 1..4
    extra_args: list[str] = field(default_factory=list)


def can_hw_decode(v: StreamInfo, gpu: GpuInfo | None) -> bool:
    if v.codec == "av1":
        return bool(gpu and gpu.compute_cap and gpu.compute_cap >= (8, 6))
    if v.codec not in _NVDEC_CODECS:
        return False
    fmt = v.pix_fmt or ""
    if "422" in fmt or "444" in fmt:
        return False
    if v.codec == "h264" and v.bit_depth > 8:
        return False  # NVDEC has no H.264 High 10
    return True


def supports_av1_encode(gpu: GpuInfo | None) -> bool:
    return bool(gpu and gpu.compute_cap and gpu.compute_cap >= (8, 9))


def build_command(
    nvencc: Path, src: Path, dst: Path, info: MediaInfo, opts: FastOptions, gpu: GpuInfo | None
) -> list[str]:
    v = info.video
    if v is None:
        raise ValueError("the source has no video stream")
    if not 1 <= opts.vsr_quality <= 4:
        raise ValueError("VSR quality must be between 1 and 4")
    if opts.codec == "av1" and not supports_av1_encode(gpu):
        raise ValueError("this GPU cannot encode AV1 (needs an RTX 40-series or newer)")
    hw = can_hw_decode(v, gpu)
    cmd = [str(nvencc), "--avhw" if hw else "--avsw", "-i", str(src), "-o", str(dst)]
    cmd += ["--codec", opts.codec]
    if opts.codec == "h264":
        cmd += ["--output-depth", "8"]
    else:
        cmd += ["--output-depth", "10"]
        if opts.codec == "hevc":
            cmd += ["--profile", "main10"]
    cmd += ["--preset", "quality", "--qvbr", str(QVBR[opts.quality][opts.codec])]
    if v.interlaced:
        cmd += ["--vpp-deinterlace", "normal"] if hw else ["--vpp-yadif"]
    # opts.width/height are in display orientation. NVEncC resizes the stored frames and keeps
    # the rotation flag, so for 90/270-degree video the stored size is the swapped one.
    out_w, out_h = (opts.height, opts.width) if abs(v.rotation) % 180 == 90 else (opts.width, opts.height)
    cmd += [
        "--vpp-resize", f"algo=ngx-vsr,vsr-quality={opts.vsr_quality}",
        "--output-res", f"{out_w}x{out_h}",
        "--sar", "1:1",
        # Keep the source's colour description (and HDR metadata when present).
        "--colormatrix", "auto", "--colorprim", "auto", "--transfer", "auto",
        "--colorrange", "auto", "--chromaloc", "auto",
    ]
    if v.hdr:
        cmd += ["--max-cll", "copy", "--master-display", "copy"]

    container = container_of(str(dst))
    audio = plan_audio(info, container)
    copy = [str(i + 1) for i, p in enumerate(audio) if p.action == "copy"]
    if copy:
        cmd += ["--audio-copy", ",".join(copy)]
    for i, p in enumerate(audio):
        if p.action == "transcode":
            cmd += ["--audio-codec", f"{i + 1}?{p.codec}", "--audio-bitrate", f"{i + 1}?192"]
    subs = plan_subtitles(info, container)
    keep = [str(i + 1) for i, p in enumerate(subs) if p.action != "drop"]
    if keep:
        cmd += ["--sub-copy", ",".join(keep)]
        if container == "mp4":
            cmd += ["--sub-codec", "mov_text"]
    cmd += ["--chapter-copy", "--metadata", "copy", "--video-metadata", "copy",
            "--audio-metadata", "copy", "--sub-metadata", "copy"]
    cmd += opts.extra_args
    return cmd


# NVEncC's live line, e.g. "[52.7%] 68 frames: 35.38 fps, 4198 kbps, remain 0:00:01, ...".
# It is printed only to a console (and not with --perf-monitor). Its percentage is an early
# estimate, so progress is computed from the frame count instead.
_PROGRESS = re.compile(r"^\[\s*[\d.]+%\]\s+(\d+) frames: ([\d.]+) fps")


def parse_progress(line: str) -> tuple[int, float] | None:
    m = _PROGRESS.match(line.strip())
    return (int(m.group(1)), float(m.group(2))) if m else None


def run(
    src: Path,
    dst: Path,
    info: MediaInfo,
    opts: FastOptions,
    gpu: GpuInfo | None,
    *,
    on_progress: Callable[[float | None, str], None],
    on_log: Callable[[str], None],
    cancel: CancelToken,
) -> None:
    nvencc = tools.exe_path("nvencc")
    cmd = build_command(nvencc, src, dst, info, opts, gpu)
    on_log("$ " + " ".join(cmd))
    total = info.frame_count()

    def handle(line: str) -> None:
        p = parse_progress(line)
        if p:
            n, fps = p
            frac = min(n / total, 0.999) if total else None
            on_progress(frac, f"{n}/{total or '?'} frames, {fps:.1f} fps")
        elif not any(s in line for s in _NOISE):
            on_log(line)

    run_process_console(cmd, on_line=handle, cancel=cancel)
