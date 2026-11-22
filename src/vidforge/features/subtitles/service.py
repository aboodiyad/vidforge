"""Write the words: speech in a video -> an .srt file (and optionally a copy of the video with
the subtitles added as a track you can turn on and off)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ...core import media, paths
from ...core.errors import FriendlyError
from ...core.ffmpeg import run_ffmpeg
from ...core.files import remove_quietly
from ...core.jobs import JobContext
from ...core.media import MediaInfo
from ...core.streams import plan_subtitles
from ..upscale.service import pick_gpu, unique_path
from . import estimate, models, srt
from .segment import Rules, cues_from_words
from .whisper import Transcript, extract_audio, transcribe

ISO3 = {
    "en": "eng", "ar": "ara", "fr": "fra", "es": "spa", "de": "deu", "it": "ita", "pt": "por", "ru": "rus",
    "ja": "jpn", "ko": "kor", "zh": "zho", "tr": "tur", "hi": "hin", "ur": "urd", "fa": "fas", "nl": "nld",
    "sv": "swe", "pl": "pol", "id": "ind", "he": "heb", "uk": "ukr", "el": "ell", "cs": "ces", "ro": "ron",
    "hu": "hun", "th": "tha", "vi": "vie", "ms": "msa", "bn": "ben", "ta": "tam", "fil": "fil",
}


@dataclass
class SubtitleRequest:
    input: Path
    output_dir: Path | None = None
    audio_track: int = 0  # 0 = first audio track
    language: str = "auto"  # what is spoken: "auto" or an ISO 639-1 code
    translate: bool = False  # True = English subtitles whatever the spoken language
    level: str = "balanced"  # quick / balanced / best
    add_to_video: bool = False  # also save a copy of the video with the subtitles inside
    gpu_index: int = 0
    overwrite: bool = False


def transcript_path(src: Path) -> Path:
    """Where the word-level transcript is kept, so translation (M4) can reuse it."""
    st = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return paths.data_root() / "transcripts" / f"{src.stem}-{key}.json"


def mux_subtitles(src: Path, info: MediaInfo, srt_path: Path, dst: Path, lang3: str, *, cancel, on_progress) -> None:
    container = "mp4" if dst.suffix.lower() in (".mp4", ".m4v", ".mov") else "mkv"
    args: list = ["-i", src, "-i", srt_path, "-map", "0:v", "-map", "0:a?"]
    kept = [p for p in plan_subtitles(info, container) if p.action != "drop"]
    for p in kept:
        args += ["-map", f"0:{p.stream.index}"]
    if container == "mkv":
        args += ["-map", "0:t?"]
    args += ["-map", "1:0", "-c", "copy"]
    for i, p in enumerate(kept):
        if p.action == "transcode":
            args += [f"-c:s:{i}", p.codec]
        args += [f"-disposition:s:{i}", "0"]
    n = len(kept)
    args += [f"-c:s:{n}", "mov_text" if container == "mp4" else "srt",
             f"-metadata:s:s:{n}", f"language={lang3}", f"-disposition:s:{n}", "default"]
    if container == "mp4":
        args += ["-movflags", "+faststart"]
    args.append(dst)
    run_ffmpeg(args, duration=info.duration, on_progress=on_progress, cancel=cancel)


def run_subtitles(req: SubtitleRequest, ctx: JobContext) -> Path:
    src = req.input.resolve()
    if not src.is_file():
        raise FriendlyError("file_missing", f"file not found: {src}", name=src.name)
    ctx.report(None, "reading source", phase="preparing")
    info = media.probe(src)
    if not info.audio:
        raise FriendlyError("no_audio", "this video has no sound")
    gpu = pick_gpu(req.gpu_index)
    work = paths.temp_dir() / f"subs-{uuid.uuid4().hex[:12]}"
    wav = work / "audio.wav"
    started = time.monotonic()
    try:
        extract_audio(src, info, req.audio_track, wav, cancel=ctx.cancel,
                      on_progress=lambda f, m: ctx.report(None if f is None else f * 0.04, m, phase="preparing"))

        def dl(done: int, total: int | None) -> None:
            ctx.report(None, f"downloading model {done / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB", phase="downloading")

        model = models.ensure(req.level, progress=dl, cancel=ctx.cancel)
        vad = models.ensure_vad(cancel=ctx.cancel)
        listen_started = time.monotonic()
        tr: Transcript = transcribe(
            wav, work / "words", gpu=gpu, model=model, vad_model=vad, dtw=models.SPEECH_MODELS[req.level].dtw,
            language=None if req.language == "auto" else req.language, translate=req.translate,
            cancel=ctx.cancel, on_log=ctx.log,
            on_progress=lambda f, m: ctx.report(None if f is None else 0.05 + f * 0.9, m, phase="working"),
        )
        listen_seconds = time.monotonic() - listen_started
    finally:
        for f in work.glob("*"):
            remove_quietly(f)
        try:
            work.rmdir()
        except OSError:
            pass
    if not tr.words:
        raise FriendlyError("no_speech", "no speech was found in this video")

    lang = "en" if req.translate else tr.language
    ctx.log(f"language {tr.language}{' -> en' if req.translate else ''}, {len(tr.words)} words")
    tp = transcript_path(src)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(json.dumps(tr.to_json(), ensure_ascii=False), encoding="utf-8")

    cues = cues_from_words(tr.words, Rules())
    folder = req.output_dir or src.parent
    srt_path = folder / f"{src.stem}.{lang}.srt"
    if not req.overwrite:
        srt_path = unique_path(srt_path)
    srt.write(cues, srt_path)
    result = srt_path

    if req.add_to_video:
        ext = src.suffix.lower() if src.suffix.lower() in (".mp4", ".mkv", ".mov", ".m4v") else ".mkv"
        dst = folder / f"{src.stem}_subtitled{ext}"
        if not req.overwrite:
            dst = unique_path(dst)
        partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
        try:
            mux_subtitles(src, info, srt_path, partial, ISO3.get(lang, "und"), cancel=ctx.cancel,
                          on_progress=lambda f, s: ctx.report(None if f is None else 0.96 + f * 0.04, "adding subtitles", phase="working"))
            partial.replace(dst)
        except BaseException:
            remove_quietly(partial)
            raise
        result = dst

    try:
        estimate.learn(req.level, info.duration or 0.0, listen_seconds, gpu)
    except Exception:
        pass
    ctx.log(f"done in {time.monotonic() - started:.0f}s: {len(cues)} subtitles")
    ctx.report(1.0, f"done: {result.name}", phase="done")
    return result
