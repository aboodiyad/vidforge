"""SubRip (.srt) subtitles: the cue type, reading and writing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str  # may contain one line break

    @property
    def duration(self) -> float:
        return self.end - self.start


def timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{c.text.strip()}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)


def write(cues: list[Cue], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # UTF-8 with BOM: Windows players and editors then detect Arabic and other scripts reliably.
    tmp.write_text(to_srt(cues), encoding="utf-8-sig", newline="\r\n")
    tmp.replace(path)
    return path


_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def parse(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip().lstrip("﻿")):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = _TIME.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append(Cue(start, end, "\n".join(lines[i + 1:]).strip()))
                break
    return cues


def read(path: Path) -> list[Cue]:
    return parse(path.read_text(encoding="utf-8-sig", errors="replace"))
