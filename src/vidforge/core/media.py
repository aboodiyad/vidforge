"""ffprobe wrapper: stream and format information as plain dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import tools
from .process import capture


@dataclass
class StreamInfo:
    index: int
    kind: str  # video / audio / subtitle / data / attachment
    codec: str
    language: str | None = None
    title: str | None = None
    default: bool = False
    bit_rate: int | None = None  # bits/s when the container reports it
    # video
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    fps_str: str | None = None  # exact rational, e.g. "24000/1001"
    nb_frames: int | None = None
    pix_fmt: str | None = None
    attached_pic: bool = False
    field_order: str | None = None  # progressive / tt / bb / tb / bt
    color_space: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_range: str | None = None
    rotation: int = 0  # degrees from the display matrix (as ffprobe reports it)
    sar: tuple[int, int] = (1, 1)

    @property
    def interlaced(self) -> bool:
        return self.field_order in ("tt", "bb", "tb", "bt")

    @property
    def hdr(self) -> bool:
        return self.color_transfer in ("smpte2084", "arib-std-b67")

    @property
    def bit_depth(self) -> int:
        f = self.pix_fmt or ""
        for depth in (16, 14, 12, 10):
            if str(depth) in f:
                return depth
        return 8

    @property
    def display_size(self) -> tuple[int, int] | None:
        """Width x height as shown: SAR applied, rotated if the stream is turned 90 degrees."""
        if not self.width or not self.height:
            return None
        w, h = self.width * self.sar[0] / self.sar[1], float(self.height)
        if abs(self.rotation) % 180 == 90:
            w, h = h, w
        return round(w), round(h)
    # audio
    channels: int | None = None
    sample_rate: int | None = None


@dataclass
class MediaInfo:
    path: str
    format_name: str
    duration: float | None
    bit_rate: int | None
    streams: list[StreamInfo] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == kind]

    @property
    def video(self) -> StreamInfo | None:
        main = [s for s in self.of_kind("video") if not s.attached_pic]
        return main[0] if main else None

    @property
    def audio(self) -> list[StreamInfo]:
        return self.of_kind("audio")

    @property
    def subtitles(self) -> list[StreamInfo]:
        return self.of_kind("subtitle")

    def frame_count(self) -> int | None:
        v = self.video
        if v is None:
            return None
        if v.nb_frames:
            return v.nb_frames
        if v.fps and self.duration:
            return max(1, round(v.fps * self.duration))
        return None


def _rate(text: str | None) -> float | None:
    if not text or text in ("0/0", "0"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value or None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sar(text: str | None) -> tuple[int, int]:
    try:
        n, d = (int(v) for v in (text or "").split(":"))
    except ValueError:
        return (1, 1)
    return (n, d) if n > 0 and d > 0 else (1, 1)


def _rotation(s: dict, tags: dict) -> int:
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    try:
        return -int(tags.get("rotate", 0))  # legacy tag is clockwise; display matrix is not
    except ValueError:
        return 0


def parse_ffprobe(data: dict, path: str = "") -> MediaInfo:
    fmt = data.get("format", {})
    streams = []
    for s in data.get("streams", []):
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        rate = s.get("avg_frame_rate") if _rate(s.get("avg_frame_rate")) else s.get("r_frame_rate")
        streams.append(
            StreamInfo(
                index=s["index"],
                kind=s.get("codec_type", "data"),
                codec=s.get("codec_name", "unknown"),
                language=tags.get("language"),
                title=tags.get("title"),
                default=bool(disp.get("default")),
                bit_rate=_int(s.get("bit_rate")) or _int(tags.get("bps")),
                width=_int(s.get("width")),
                height=_int(s.get("height")),
                fps=_rate(rate),
                fps_str=rate if _rate(rate) else None,
                nb_frames=_int(s.get("nb_frames")),
                pix_fmt=s.get("pix_fmt"),
                attached_pic=bool(disp.get("attached_pic")),
                field_order=s.get("field_order"),
                color_space=s.get("color_space"),
                color_primaries=s.get("color_primaries"),
                color_transfer=s.get("color_transfer"),
                color_range=s.get("color_range"),
                rotation=_rotation(s, tags),
                sar=_sar(s.get("sample_aspect_ratio")),
                channels=_int(s.get("channels")),
                sample_rate=_int(s.get("sample_rate")),
            )
        )
    return MediaInfo(
        path=path or fmt.get("filename", ""),
        format_name=fmt.get("format_name", ""),
        duration=_float(fmt.get("duration")),
        bit_rate=_int(fmt.get("bit_rate")),
        streams=streams,
    )


def probe(path: str | Path) -> MediaInfo:
    ffprobe = tools.exe_path("ffmpeg", "ffprobe.exe")
    out = capture(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {out.stderr.strip()}")
    return parse_ffprobe(json.loads(out.stdout), str(path))
