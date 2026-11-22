"""Time and size estimates for fixing faces, learned from finished jobs like upscaling's.

Time = optional background upscaling (from the upscale estimator) + the face pass, which costs
about the same per frame whatever the resolution (faces are always restored at 512x512).
Default face-pass speeds are frames per second with one face, measured on an RTX 2060 SUPER.
"""

from __future__ import annotations

from ...core.media import MediaInfo
from ..upscale import estimate as up
from .models import FACE_MODELS, is_installed

DEFAULT_FACE_FPS = {"gfpgan": 9.0, "codeformer": 5.0}
STARTUP_S = 6.0


def _speed_key(model: str) -> str:
    return f"faces/{model}"


def _size_key(codec: str, quality: str) -> str:
    return f"faces/{codec}/{quality}"


def background_seconds(info: MediaInfo, w: int, h: int, background: str, esrgan_model: str, stats: dict) -> float:
    if background in ("fast", "quality"):
        return up.seconds(info, w, h, background, esrgan_model, stats)[0]
    return 0.0


def estimate(info: MediaInfo, w: int, h: int, *, background: str, model: str, esrgan_model: str,
             codec: str, quality: str, container: str, stats: dict | None = None) -> up.Estimate:
    stats = up.load_stats() if stats is None else stats
    frames = info.frame_count() or 0
    learned_fps = stats.get("speed", {}).get(_speed_key(model))
    face_s = STARTUP_S + frames / (learned_fps or DEFAULT_FACE_FPS[model])
    secs = face_s + background_seconds(info, w, h, background, esrgan_model, stats)

    vbytes, size_learned = up.video_bytes(info, w, h, "quality", codec, quality, stats={})
    factor = stats.get("size", {}).get(_size_key(codec, quality))
    if factor:
        vbytes *= factor
    total = vbytes + up.audio_bytes(info, container)
    learned = bool(learned_fps or factor)
    s_lo, s_hi = (0.8, 1.3) if learned_fps else (0.6, 1.8)  # depends on how many faces appear
    b_lo, b_hi = (0.75, 1.35) if factor else (0.6, 1.7)
    download = sum(FACE_MODELS[k].size_mb for k in ("yunet", model) if not is_installed(k))
    if background == "quality":
        download += up.needed_download_mb("quality", esrgan_model)
    return up.Estimate(
        width=w, height=h, mode="faces", seconds=secs, seconds_low=secs * s_lo, seconds_high=secs * s_hi,
        bytes=int(total), bytes_low=int(total * b_lo), bytes_high=int(total * b_hi),
        download_mb=download, learned=learned,
    )


def learn(info: MediaInfo, req, background: str, w: int, h: int, work_seconds: float, stats, dst) -> None:
    frames = info.frame_count() or 0
    known = up.load_stats()
    face_seconds = work_seconds - STARTUP_S - background_seconds(info, w, h, background, req.esrgan_model, known)
    if frames <= 0 or face_seconds <= 1:
        return
    predicted, _ = up.video_bytes(info, w, h, "quality", req.codec, req.quality, stats={})
    actual = dst.stat().st_size - up.audio_bytes(info, req.container)
    with up._lock:
        data = up.load_stats()
        speeds = data.setdefault("speed", {})
        k = _speed_key(req.model)
        speeds[k] = up._ema(speeds.get(k), frames / face_seconds)
        if predicted > 0 and actual > 0:
            sizes = data.setdefault("size", {})
            k = _size_key(req.codec, req.quality)
            sizes[k] = up._ema(sizes.get(k), min(3.0, max(0.3, actual / predicted)))
        up._save_stats(data)
