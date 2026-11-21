"""Which audio/subtitle streams can be carried into an output container, and how.

Shared by every feature that re-muxes the source's audio and subtitles.
"""

from __future__ import annotations

from dataclasses import dataclass

from .media import MediaInfo, StreamInfo

TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
# Audio codecs FFmpeg can put in MP4/MOV without transcoding.
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac", "dts", "mp2"}


@dataclass
class StreamPlan:
    stream: StreamInfo
    action: str  # "copy", "transcode" or "drop"
    codec: str | None = None  # target codec when transcoding
    reason: str = ""


def container_of(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1]
    return {"m4v": "mp4", "mov": "mp4"}.get(ext, ext)


def plan_audio(info: MediaInfo, container: str) -> list[StreamPlan]:
    plans = []
    for s in info.audio:
        if container == "mp4" and s.codec not in MP4_AUDIO:
            plans.append(StreamPlan(s, "transcode", "aac", f"{s.codec} is not supported in MP4"))
        else:
            plans.append(StreamPlan(s, "copy"))
    return plans


def plan_subtitles(info: MediaInfo, container: str) -> list[StreamPlan]:
    plans = []
    for s in info.subtitles:
        if container == "mp4":
            if s.codec in TEXT_SUBS:
                plans.append(StreamPlan(s, "transcode" if s.codec != "mov_text" else "copy", "mov_text"))
            else:
                plans.append(StreamPlan(s, "drop", reason=f"{s.codec} subtitles are not supported in MP4"))
        else:
            plans.append(StreamPlan(s, "copy"))
    return plans
