"""How long translating and dubbing takes, and how big the result is.

The job has up to five parts: writing down the speech (skipped when the video was done before),
translating, separating the voices from the music, saying every line in the new language, and
saving the video. Each part's speed starts from a value measured on an RTX 2060 SUPER and is
then learned from the user's own finished jobs.
"""

from __future__ import annotations

from pathlib import Path

from ...core import tools
from ...core.gpu import GpuInfo
from ..subtitles import burn
from ..subtitles import estimate as sub_est
from ..upscale import estimate as up
from . import addon, models

# characters of the original speech translated per second
LLM_SPEED = {("best", "gpu"): 20.0, ("quick", "gpu"): 100.0, ("best", "cpu"): 3.0, ("quick", "cpu"): 12.0}
LLM_STARTUP = 20.0
SEPARATE_SPEED = {"gpu": 8.0, "cpu": 0.8}  # seconds of sound per second
# seconds of new speech per second (measured on the test clip: Chatterbox ~0.7x real time, XTTS ~1.9x)
TTS_SPEED = {("chatterbox", "gpu"): 0.75, ("chatterbox", "cpu"): 0.07, ("xtts", "gpu"): 1.8, ("xtts", "cpu"): 0.15}
WORKER_STARTUP = 35.0  # starting PyTorch and loading the voice
LLAMA_TOOL_MB = 750.0
AUDIO_BITRATE = 192_000


def device(gpu: GpuInfo | None) -> str:
    return "gpu" if gpu and gpu.vendor == "nvidia" else "cpu"


def _speed(stats: dict, key: str, default: float) -> tuple[float, bool]:
    v = stats.get("speed", {}).get(key)
    return (float(v), True) if v else (default, False)


def learned_cps(stats: dict, voice: str = "chatterbox") -> dict:
    """{language: characters per second} this voice was measured to speak."""
    prefix = f"dub/cps/{voice}/"
    return {k[len(prefix):]: v for k, v in stats.get("speed", {}).items() if k.startswith(prefix)}


def estimate(info, *, level: str, dub: bool, add_to_video: bool, target: str, gpu: GpuInfo | None,
             transcript=None, stats: dict | None = None, voice: str = "chatterbox",
             burn_in: bool = False) -> up.Estimate:
    stats = up.load_stats() if stats is None else stats
    dev = device(gpu)
    dur = info.duration or 0.0
    learned_any = False
    secs = 0.0
    if transcript is None:
        s, ln = sub_est.seconds(dur, "balanced", gpu, stats)
        secs += s
        learned_any |= ln
        chars, spoken = dur * 12.0, dur * 0.7  # typical talking video
    else:
        chars = float(len(transcript.text))
        spoken = sum(w.end - w.start for w in transcript.words) * 1.25  # word times miss small gaps

    v, ln = _speed(stats, f"dub/llm/{level}/{dev}", LLM_SPEED[(level, dev)])
    secs += LLM_STARTUP + chars / v
    learned_any |= ln
    if dub:
        v, ln = _speed(stats, f"dub/separate/{dev}", SEPARATE_SPEED[dev])
        secs += WORKER_STARTUP + dur / v
        learned_any |= ln
        v, ln = _speed(stats, f"dub/tts/{voice}/{dev}", TTS_SPEED[(voice, dev)])
        secs += spoken / v
        learned_any |= ln
        secs += dur / 150 + 5  # fitting, mixing and saving
    elif add_to_video and not burn_in:
        secs += dur / 400 + 2
    if burn_in:
        secs += burn.seconds(info, gpu, stats)

    lo, hi = (0.75, 1.35) if learned_any else (0.5, 2.0)
    size = dur * 15.0  # the .srt
    if (dub or add_to_video or burn_in) and info.path:
        p = Path(info.path)
        size += (p.stat().st_size if p.is_file() else 0) * (1.2 if burn_in else 1.0)  # burned: encoded again
        if dub:
            size += dur * AUDIO_BITRATE / 8
    download = models.download_mb(level, voice=voice if dub else None)
    if tools.locate("llama")[0] is None:
        download += LLAMA_TOOL_MB
    return up.Estimate(width=0, height=0, mode="dub", seconds=secs, seconds_low=secs * lo, seconds_high=secs * hi,
                       bytes=int(size), bytes_low=int(size * 0.9), bytes_high=int(size * 1.2),
                       download_mb=download, learned=learned_any)


def addon_mb(gpu: GpuInfo | None) -> int:
    return addon.INSTALL_MB[addon.device_for(gpu)]


def learn(key: str, value: float) -> None:
    """Fold one measured speed into the stored average."""
    if not value or value <= 0:
        return
    with up._lock:
        data = up.load_stats()
        speeds = data.setdefault("speed", {})
        speeds[key] = up._ema(speeds.get(key), value)
        up._save_stats(data)
