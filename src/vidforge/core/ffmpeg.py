"""Run FFmpeg with machine-readable progress (-progress pipe:1)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from . import tools
from .process import CancelToken, run_process


class ProgressParser:
    """Feeds on `-progress` key=value lines and reports a 0..1 fraction of *duration*."""

    def __init__(self, duration: float | None, on_progress: Callable[[float | None, str], None]):
        self.duration = duration if duration and duration > 0 else None
        self.on_progress = on_progress
        self.out_time = 0.0
        self.speed = ""
        self.ended = False

    def feed(self, line: str) -> bool:
        """Returns True if the line was a progress line (so callers can skip logging it)."""
        key, sep, value = line.partition("=")
        if not sep or " " in key:
            return False
        key, value = key.strip(), value.strip()
        if key in ("out_time_us", "out_time_ms"):  # both are microseconds in practice
            try:
                self.out_time = max(self.out_time, int(value) / 1_000_000)
            except ValueError:
                pass
        elif key == "speed":
            self.speed = value
        elif key == "progress":
            self.ended = value == "end"
            frac = 1.0 if self.ended else (
                min(self.out_time / self.duration, 1.0) if self.duration else None
            )
            self.on_progress(frac, self.speed)
        elif key not in (
            "frame", "fps", "bitrate", "total_size", "out_time", "dup_frames", "drop_frames",
        ) and not key.startswith("stream_"):
            return False
        return True


def ffmpeg_cmd(*args: str | Path) -> list[str]:
    exe = tools.exe_path("ffmpeg")
    return [str(exe), "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *map(str, args)]


def run_ffmpeg(
    args: Sequence[str | Path],
    *,
    duration: float | None = None,
    on_progress: Callable[[float | None, str], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    cancel: CancelToken | None = None,
    cwd: str | Path | None = None,
) -> None:
    parser = ProgressParser(duration, on_progress or (lambda f, s: None))

    def handle(line: str) -> None:
        if not parser.feed(line) and on_log:
            on_log(line)

    run_process(ffmpeg_cmd(*args), on_line=handle, cancel=cancel, cwd=cwd)
