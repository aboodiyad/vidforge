"""Pick an FFmpeg video encoder: the GPU's hardware encoder when it works, else software.

Used whenever VidForge encodes frames itself (Quality upscaling, face enhancement).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from . import tools
from .process import capture

QUALITY_LEVELS = ("high", "balanced", "small")


@dataclass(frozen=True)
class Encoder:
    name: str  # ffmpeg encoder name
    codec: str  # hevc / h264 / av1
    pix_fmt: str
    hardware: bool
    args: tuple[str, ...]  # rate-control and preset args (quality already filled in)

    def ffmpeg_args(self) -> list[str]:
        return ["-c:v", self.name, *self.args, "-pix_fmt", self.pix_fmt]


def _q(level: str, high: int, balanced: int, small: int) -> str:
    return str({"high": high, "balanced": balanced, "small": small}[level])


def _candidates(codec: str, level: str, vendor: str) -> list[Encoder]:
    hw: list[Encoder] = []
    if vendor == "nvidia":
        name = {"hevc": "hevc_nvenc", "h264": "h264_nvenc", "av1": "av1_nvenc"}[codec]
        q = _q(level, 20, 24, 28) if codec != "av1" else _q(level, 26, 32, 38)
        args = ("-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", q, "-b:v", "0")
        if codec == "hevc":
            args += ("-profile:v", "main10")
        hw.append(Encoder(name, codec, "yuv420p" if codec == "h264" else "p010le", True, args))
    elif vendor == "amd":
        name = {"hevc": "hevc_amf", "h264": "h264_amf", "av1": "av1_amf"}[codec]
        q = _q(level, 20, 24, 28)
        args = ("-quality", "quality", "-rc", "cqp", "-qp_i", q, "-qp_p", q)
        hw.append(Encoder(name, codec, "yuv420p" if codec == "h264" else "p010le", True, args))
    elif vendor == "intel":
        name = {"hevc": "hevc_qsv", "h264": "h264_qsv", "av1": "av1_qsv"}[codec]
        args = ("-preset", "slower", "-global_quality", _q(level, 20, 24, 28))
        hw.append(Encoder(name, codec, "nv12" if codec == "h264" else "p010le", True, args))
    sw = {
        "hevc": Encoder("libx265", "hevc", "yuv420p10le", False,
                        ("-preset", "medium", "-crf", _q(level, 18, 21, 25))),
        "h264": Encoder("libx264", "h264", "yuv420p", False,
                        ("-preset", "slow", "-crf", _q(level, 17, 20, 24))),
        "av1": Encoder("libsvtav1", "av1", "yuv420p10le", False,
                       ("-preset", "6", "-crf", _q(level, 26, 32, 38))),
    }[codec]
    return [*hw, sw]


@functools.lru_cache(maxsize=None)
def encoder_works(name: str, pix_fmt: str) -> bool:
    """Encode one tiny frame to check the encoder exists and the GPU accepts it."""
    try:
        ffmpeg = tools.exe_path("ffmpeg")
    except FileNotFoundError:
        return False
    out = capture(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=256x256:r=1",
         "-frames:v", "1", "-c:v", name, "-pix_fmt", pix_fmt, "-f", "null", "-"],
        timeout=60,
    )
    return out.returncode == 0


def choose(codec: str, level: str, vendor: str) -> Encoder:
    if level not in QUALITY_LEVELS:
        raise ValueError(f"quality must be one of {', '.join(QUALITY_LEVELS)}")
    for enc in _candidates(codec, level, vendor):
        if not enc.hardware or encoder_works(enc.name, enc.pix_fmt):
            return enc
    raise RuntimeError(f"no working {codec} encoder")  # unreachable: software is always last
