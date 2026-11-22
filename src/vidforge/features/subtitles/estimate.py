"""How long writing the words takes: Whisper's speed is measured in seconds of audio per
second of work, and depends on the model and on GPU vs CPU. Defaults below were measured on an
RTX 2060 SUPER (GPU) and are rough guesses for CPU; finished jobs refine them."""

from __future__ import annotations

from ...core.gpu import GpuInfo
from ..upscale import estimate as up
from . import models

# audio seconds processed per wall-clock second
DEFAULT_SPEED = {
    ("quick", "gpu"): 60.0, ("balanced", "gpu"): 25.0, ("best", "gpu"): 20.0,
    ("quick", "cpu"): 6.0, ("balanced", "cpu"): 1.2, ("best", "cpu"): 0.9,
}
STARTUP_S = 4.0


def _device(gpu: GpuInfo | None) -> str:
    return "gpu" if gpu and gpu.vendor == "nvidia" else "cpu"


def _key(level: str, gpu: GpuInfo | None) -> str:
    return f"subtitles/{level}/{_device(gpu)}"


def seconds(audio_seconds: float, level: str, gpu: GpuInfo | None, stats: dict | None = None) -> tuple[float, bool]:
    stats = up.load_stats() if stats is None else stats
    learned = stats.get("speed", {}).get(_key(level, gpu))
    speed = learned or DEFAULT_SPEED[(level, _device(gpu))]
    return STARTUP_S + audio_seconds / speed + audio_seconds / 400, bool(learned)  # + audio extraction


def estimate(info, *, level: str, gpu: GpuInfo | None, add_to_video: bool, stats: dict | None = None) -> up.Estimate:
    dur = info.duration or 0.0
    secs, learned = seconds(dur, level, gpu, stats)
    lo, hi = (0.8, 1.3) if learned else (0.6, 1.8)
    srt_bytes = dur * 15  # roughly 15 bytes of subtitle text per second of speech
    size = srt_bytes
    if add_to_video and info.path:
        from pathlib import Path

        p = Path(info.path)
        size += p.stat().st_size if p.is_file() else 0
    # width/height 0: the picture does not change, so the list shows no "new size".
    return up.Estimate(width=0, height=0, mode="subtitles", seconds=secs, seconds_low=secs * lo,
                       seconds_high=secs * hi, bytes=int(size), bytes_low=int(size * 0.9),
                       bytes_high=int(size * 1.2), download_mb=models.download_mb(level), learned=learned)


def learn(level: str, audio_seconds: float, listen_seconds: float, gpu: GpuInfo | None) -> None:
    work = listen_seconds - STARTUP_S
    if audio_seconds <= 5 or work <= 0.5:
        return
    with up._lock:
        data = up.load_stats()
        speeds = data.setdefault("speed", {})
        k = _key(level, gpu)
        speeds[k] = up._ema(speeds.get(k), audio_seconds / work)
        up._save_stats(data)
