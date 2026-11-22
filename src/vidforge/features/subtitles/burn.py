"""Writing subtitles onto the picture ("burning in"), so they show in every player and app.

FFmpeg's subtitles filter (libass, with HarfBuzz and FriBidi for Arabic and other scripts) draws
the .srt onto each frame, and the video is encoded again. Arabic and Latin text use VidForge's
own IBM Plex Sans Arabic; scripts it lacks use a Windows font that has them.

The filter reads the .srt by a name inside the filter text, where ':' and quotes are special, so
the file is copied into the work folder as `subs.srt` and FFmpeg runs in that folder.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ...core import encoders
from ...core.errors import FriendlyError
from ...core.gpu import GpuInfo
from ...core.media import MediaInfo

APP_FONT = "IBM Plex Sans Arabic"  # bundled with VidForge; Arabic and Latin letters
WINDOWS_FONT = {
    "zh": "Microsoft YaHei", "ja": "Yu Gothic", "ko": "Malgun Gothic", "hi": "Nirmala UI", "bn": "Nirmala UI",
    "ta": "Nirmala UI", "th": "Leelawadee UI", "he": "Arial", "ru": "Arial", "uk": "Arial", "el": "Arial",
}
FILE_NAME = "subs.srt"


def fonts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "gui" / "assets" / "fonts"


def font_for(lang: str) -> str:
    return WINDOWS_FONT.get(lang, APP_FONT)


def quote(text: str) -> str:
    """A value for a filter option: in single quotes, ':' and '\\' escaped, quotes closed around."""
    text = text.replace("\\", "/").replace(":", "\\:")
    return "'" + text.replace("'", "'\\''") + "'"


def style(lang: str) -> str:
    # Sizes are in libass's 288-line canvas, so they scale with the video's height.
    return (f"FontName={font_for(lang)},FontSize=22,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,BorderStyle=1,Outline=1.6,Shadow=0.6,MarginV=16")


def video_filter(lang: str) -> str:
    return f"subtitles={FILE_NAME}:fontsdir={quote(str(fonts_dir()))}:force_style={quote(style(lang))}"


@dataclass
class Burn:
    """How to draw a subtitle file onto the main video stream."""
    srt: Path
    lang: str
    encoder: encoders.Encoder

    def prepare(self, work: Path) -> Path:
        """Copy the subtitles where FFmpeg will look; returns the folder to run FFmpeg in."""
        work.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.srt, work / FILE_NAME)
        return work

    def args(self) -> list[str]:
        return ["-filter:v:0", video_filter(self.lang), *self.encoder.ffmpeg_args()]


# Pixels per second when drawing and encoding again: the graphics card's encoder, or the processor.
DEFAULT_SPEED = {True: 1.5e8, False: 2.5e7}
STARTUP_S = 2.0


def _pixels(info: MediaInfo) -> float:
    v = info.video
    if v is None or not v.width or not v.height:
        return 0.0
    fps = v.fps or 25.0
    return (info.duration or 0.0) * fps * v.width * v.height


def _key(hardware: bool) -> str:
    return f"burn/{'hw' if hardware else 'sw'}"


def seconds(info: MediaInfo, gpu: GpuInfo | None, stats: dict | None = None) -> float:
    """About how long writing the subtitles onto the picture takes (learned from past jobs)."""
    from ..upscale import estimate as up

    hardware = bool(gpu and gpu.vendor in ("nvidia", "amd", "intel"))
    stats = up.load_stats() if stats is None else stats
    speed = stats.get("speed", {}).get(_key(hardware)) or DEFAULT_SPEED[hardware]
    return STARTUP_S + _pixels(info) / speed


def learn(info: MediaInfo, b: "Burn", took: float) -> None:
    from ..upscale import estimate as up

    work = took - STARTUP_S
    if work <= 1 or _pixels(info) <= 0:
        return
    with up._lock:
        data = up.load_stats()
        speeds = data.setdefault("speed", {})
        k = _key(b.encoder.hardware)
        speeds[k] = up._ema(speeds.get(k), _pixels(info) / work)
        up._save_stats(data)


def check(info: MediaInfo) -> None:
    """Raise the friendly error if subtitles can't be written onto this video."""
    v = info.video
    if v is None:
        raise FriendlyError("no_video", "this file has no video in it")
    if v.hdr:
        raise FriendlyError("hdr_burn", "subtitles can't be written onto HDR video yet")


def make(info: MediaInfo, srt: Path, lang: str, gpu: GpuInfo | None) -> Burn:
    """Burn settings for this video: H.265 if it already is H.265, else H.264 (plays everywhere),
    on the graphics card's encoder when it works, at high quality."""
    check(info)
    v = info.video
    codec = "hevc" if v.codec == "hevc" else "h264"
    return Burn(srt, lang, encoders.choose(codec, "high", gpu.vendor if gpu else "other"))
