"""Before-you-start estimates for an upscale: output size in pixels, file size, and time.

Both estimates start from simple models and then learn from the user's own finished jobs
(stored in stats.json in the data folder), so they get closer the more VidForge is used.

File size: constant-quality encoders spend bits according to content, which we cannot know
up front. The best cheap hint is how many bits per pixel the *source* needed: busy, noisy
video needs many, smooth video few. The output's bits per pixel is modelled as a fraction of
the source's (adjusted for codecs, the quality choice and the mode), times its pixel count.

Time: frames x pixels / speed. Fast mode's cost follows output pixels (VSR + NVENC at the
output size); Quality mode's follows input pixels (the network runs at input size). Default
speeds were measured on an RTX 2060 SUPER.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

from ...core import models as model_registry
from ...core import paths
from ...core.media import MediaInfo
from .models import QUALITY_MODELS, onnx_path

# Pixels per second (output pixels for fast, input pixels for quality) and fixed start-up time.
DEFAULT_SPEED = {"fast": 3.2e8, "quality/general": 4.5e6, "quality/anime": 7.9e6, "quality/max": 2.0e5,
                 # with the NVIDIA speed-up (TensorRT), measured the same way
                 "quality-trt/general": 8.4e6, "quality-trt/anime": 1.5e7, "quality-trt/max": 4.0e5}
STARTUP_S = {"fast": 3.5, "quality": 5.0}

# How much picture information a bit carries, relative to H.264 (higher = more efficient).
CODEC_EFFICIENCY = {
    "h264": 1.0, "hevc": 1.5, "av1": 1.8, "vp9": 1.4, "vp8": 0.9, "mpeg2video": 0.5,
    "mpeg1video": 0.45, "mpeg4": 0.7, "msmpeg4v3": 0.6, "wmv3": 0.7, "vc1": 0.8, "prores": 0.3,
}
QUALITY_FACTOR = {"high": 1.6, "balanced": 1.0, "small": 0.6}
MODE_FACTOR = {"fast": 1.3, "quality": 1.0}  # VSR output keeps more fine texture
BPP_RATIO = 0.63  # HEVC output bpp / H.264-equivalent source bpp (balanced, quality mode)
BPP_MIN, BPP_MAX = 0.008, 0.30


@dataclass
class Estimate:
    width: int
    height: int
    mode: str
    seconds: float
    seconds_low: float
    seconds_high: float
    bytes: int
    bytes_low: int
    bytes_high: int
    download_mb: float = 0.0  # model download needed before the first Quality run
    learned: bool = False  # True when the user's own past jobs shaped the numbers


# --- learned statistics ---------------------------------------------------------------

_lock = threading.Lock()


def _stats_file() -> Path:
    return paths.data_root() / "stats.json"


def load_stats() -> dict:
    try:
        data = json.loads(_stats_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_stats(data: dict) -> None:
    f = _stats_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(f)


def _trt() -> bool:
    try:
        from . import speedup

        return speedup.is_installed()
    except Exception:  # noqa: BLE001 - estimates never fail
        return False


def speed_key(mode: str, model: str, trt: bool | None = None) -> str:
    """Speeds are kept apart for DirectML and TensorRT (the NVIDIA speed-up), which differ ~2x."""
    if mode == "fast":
        return "fast"
    return f"quality{'-trt' if (_trt() if trt is None else trt) else ''}/{model}"


def size_key(mode: str, codec: str, quality: str) -> str:
    return f"{mode}/{codec}/{quality}"


def _ema(old: float | None, new: float, weight: float = 0.5) -> float:
    return new if old is None else old * (1 - weight) + new * weight


def learn(*, mode: str, model: str, codec: str, quality: str, frames: int, in_pixels: int,
          out_pixels: int, work_seconds: float, predicted_video_bytes: int, actual_video_bytes: int) -> None:
    """Record a finished job so later estimates match this PC and this kind of video."""
    if frames <= 0 or work_seconds <= 0.5:
        return
    px = out_pixels if mode == "fast" else in_pixels
    rate = frames * px / work_seconds
    with _lock:
        data = load_stats()
        speeds = data.setdefault("speed", {})
        k = speed_key(mode, model)
        speeds[k] = _ema(speeds.get(k), rate)
        if predicted_video_bytes > 0 and actual_video_bytes > 0:
            sizes = data.setdefault("size", {})
            k = size_key(mode, codec, quality)
            factor = min(3.0, max(0.3, actual_video_bytes / predicted_video_bytes))
            sizes[k] = _ema(sizes.get(k), factor)
        _save_stats(data)


# --- the models -----------------------------------------------------------------------

def source_video_bitrate(info: MediaInfo) -> float | None:
    v = info.video
    if v is None:
        return None
    if v.bit_rate:
        return float(v.bit_rate)
    audio = sum(a.bit_rate or 0 for a in info.audio)
    if info.bit_rate:
        return max(float(info.bit_rate) - audio, info.bit_rate * 0.5)
    if info.duration and info.path and Path(info.path).is_file():
        return Path(info.path).stat().st_size * 8 / info.duration - audio
    return None


def audio_bytes(info: MediaInfo, container: str) -> float:
    from ...core.streams import plan_audio

    if not info.duration:
        return 0.0
    total = 0.0
    for p in plan_audio(info, container):
        if p.action == "copy":
            rate = p.stream.bit_rate or (128_000 * max(1, (p.stream.channels or 2)) / 2)
        else:
            rate = 192_000
        total += rate * info.duration / 8
    return total


def video_bytes(info: MediaInfo, out_w: int, out_h: int, mode: str, codec: str, quality: str,
                stats: dict | None = None) -> tuple[float, bool]:
    v = info.video
    fps = v.fps or 30.0
    duration = info.duration or 0.0
    src_px = (v.width or 1) * (v.height or 1)
    src_rate = source_video_bitrate(info)
    if src_rate and src_rate > 0:
        src_bpp = src_rate / (src_px * fps) * CODEC_EFFICIENCY.get(v.codec, 1.0)
    else:
        src_bpp = 0.08  # unknown: assume ordinary compressed video
    # BPP_RATIO was fitted on HEVC outputs, so other output codecs scale relative to HEVC.
    codec_scale = CODEC_EFFICIENCY["hevc"] / CODEC_EFFICIENCY.get(codec, 1.0)
    bpp = src_bpp * BPP_RATIO * QUALITY_FACTOR[quality] * MODE_FACTOR[mode] * codec_scale
    bpp = min(BPP_MAX, max(BPP_MIN, bpp))
    learned = False
    factor = (stats or {}).get("size", {}).get(size_key(mode, codec, quality))
    if factor:
        bpp *= factor
        learned = True
    return bpp * out_w * out_h * fps * duration / 8, learned


def seconds(info: MediaInfo, out_w: int, out_h: int, mode: str, model: str,
            stats: dict | None = None) -> tuple[float, bool]:
    v = info.video
    frames = info.frame_count() or 0
    learned_rate = (stats or {}).get("speed", {}).get(speed_key(mode, model))
    rate = learned_rate or DEFAULT_SPEED[speed_key(mode, model)]
    px = out_w * out_h if mode == "fast" else (v.width or 1) * (v.height or 1)
    return STARTUP_S[mode] + frames * px / rate, bool(learned_rate)


def needed_download_mb(mode: str, model: str) -> float:
    if mode != "quality" or onnx_path(model).is_file():
        return 0.0
    key = QUALITY_MODELS[model].model_key
    if model_registry.is_installed(key):
        return 0.0
    return model_registry.MODELS[key].size_mb


def estimate(info: MediaInfo, out_w: int, out_h: int, *, mode: str, model: str, codec: str,
             quality: str, container: str, stats: dict | None = None) -> Estimate:
    stats = load_stats() if stats is None else stats
    vbytes, size_learned = video_bytes(info, out_w, out_h, mode, codec, quality, stats)
    total = vbytes + audio_bytes(info, container)
    secs, time_learned = seconds(info, out_w, out_h, mode, model, stats)
    s_lo, s_hi = (0.85, 1.2) if time_learned else (0.75, 1.4)
    b_lo, b_hi = (0.75, 1.35) if size_learned else (0.6, 1.7)
    return Estimate(
        width=out_w, height=out_h, mode=mode,
        seconds=secs, seconds_low=secs * s_lo, seconds_high=secs * s_hi,
        bytes=int(total), bytes_low=int(total * b_lo), bytes_high=int(total * b_hi),
        download_mb=needed_download_mb(mode, model),
        learned=size_learned or time_learned,
    )


def round_nice(value: float) -> float:
    """Round to 2 significant digits so estimates don't pretend to be exact."""
    if value <= 0:
        return 0.0
    mag = 10 ** (math.floor(math.log10(value)) - 1)
    return round(value / mag) * mag
