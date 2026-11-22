"""Write the words: speech in a video -> an .srt file (and optionally a copy of the video with
the subtitles added as a track you can turn on and off)."""

from __future__ import annotations

import hashlib
import json
import shutil
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
from . import burn
from .burn import Burn
from .segment import Rules, cues_from_words
from .whisper import Transcript, extract_audio, transcribe

ISO3 = {
    "en": "eng", "ar": "ara", "fr": "fra", "es": "spa", "de": "deu", "it": "ita", "pt": "por", "ru": "rus",
    "ja": "jpn", "ko": "kor", "zh": "zho", "tr": "tur", "hi": "hin", "ur": "urd", "fa": "fas", "nl": "nld",
    "sv": "swe", "pl": "pol", "id": "ind", "he": "heb", "uk": "ukr", "el": "ell", "cs": "ces", "ro": "ron",
    "hu": "hun", "th": "tha", "vi": "vie", "ms": "msa", "bn": "ben", "ta": "tam", "fil": "fil",
    "da": "dan", "fi": "fin", "no": "nor", "sw": "swa",
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
    burn_in: bool = False  # in that copy, write them onto the picture (always shown) instead
    gpu_index: int = 0
    overwrite: bool = False


def transcript_path(src: Path) -> Path:
    """Where the word-level transcript is kept, so translation (M4) can reuse it."""
    st = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return paths.data_root() / "transcripts" / f"{src.stem}-{key}.json"


def load_transcript(src: Path) -> Transcript | None:
    """The saved transcript of *src*, if it was transcribed before (and not changed since)."""
    try:
        return Transcript.from_json(json.loads(transcript_path(src).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def transcribe_video(src: Path, info: MediaInfo, *, level: str, language: str, translate: bool,
                     audio_track: int, gpu, ctx: JobContext, span: tuple[float, float] = (0.0, 1.0)) -> Transcript:
    """Speech -> timed words (whisper.cpp + voice detection). The result is saved as the video's
    transcript (unless it was translated to English) and the speed is learned for estimates.
    Progress is reported within *span* of the job."""
    a, b = span
    work = paths.temp_dir() / f"subs-{uuid.uuid4().hex[:12]}"
    wav = work / "audio.wav"
    try:
        extract_audio(src, info, audio_track, wav, cancel=ctx.cancel,
                      on_progress=lambda f, m: ctx.report(None if f is None else a + f * (b - a) * 0.04, m, phase="preparing"))

        def dl(done: int, total: int | None) -> None:
            ctx.report(None, f"downloading model {done / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB", phase="downloading")

        model = models.ensure(level, progress=dl, cancel=ctx.cancel)
        vad = models.ensure_vad(cancel=ctx.cancel)
        listen_started = time.monotonic()
        tr: Transcript = transcribe(
            wav, work / "words", gpu=gpu, model=model, vad_model=vad, dtw=models.SPEECH_MODELS[level].dtw,
            language=None if language == "auto" else language, translate=translate,
            cancel=ctx.cancel, on_log=ctx.log,
            on_progress=lambda f, m: ctx.report(None if f is None else a + (b - a) * (0.05 + f * 0.95), m, phase="working"),
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
    if not translate:
        tp = transcript_path(src)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(tr.to_json(), ensure_ascii=False), encoding="utf-8")
    try:
        estimate.learn(level, info.duration or 0.0, listen_seconds, gpu)
    except Exception:
        pass
    return tr


def mux_subtitles(src: Path, info: MediaInfo, srt_path: Path, dst: Path, lang3: str, *, cancel, on_progress,
                  burn: Burn | None = None) -> None:
    """A copy of the video with the subtitles added: as a track that can be turned on and off, or
    with *burn*, written onto the picture (the video is then encoded again)."""
    src, srt_path, dst = Path(src).resolve(), Path(srt_path).resolve(), Path(dst).resolve()  # FFmpeg may run elsewhere
    container = "mp4" if dst.suffix.lower() in (".mp4", ".m4v", ".mov") else "mkv"
    video = ["-map", f"0:{info.video.index}"] if burn and info.video else ["-map", "0:v"]
    args: list = ["-i", src, *([] if burn else ["-i", srt_path]), *video, "-map", "0:a?"]
    kept = [p for p in plan_subtitles(info, container) if p.action != "drop"]
    for p in kept:
        args += ["-map", f"0:{p.stream.index}"]
    if container == "mkv":
        args += ["-map", "0:t?"]
    if not burn:
        args += ["-map", "1:0"]
    args += ["-c", "copy"]
    if burn:
        args += burn.args()
    for i, p in enumerate(kept):
        if p.action == "transcode":
            args += [f"-c:s:{i}", p.codec]
        args += [f"-disposition:s:{i}", "0"]
    if not burn:
        n = len(kept)
        args += [f"-c:s:{n}", "mov_text" if container == "mp4" else "srt",
                 f"-metadata:s:s:{n}", f"language={lang3}", f"-disposition:s:{n}", "default"]
    if container == "mp4":
        args += ["-movflags", "+faststart"]
    args.append(dst)
    work = paths.temp_dir() / f"burn-{uuid.uuid4().hex[:12]}"
    try:
        cwd = burn.prepare(work) if burn else None
        run_ffmpeg(args, duration=info.duration, on_progress=on_progress, cancel=cancel, cwd=cwd)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_subtitles(req: SubtitleRequest, ctx: JobContext) -> Path:
    src = req.input.resolve()
    if not src.is_file():
        raise FriendlyError("file_missing", f"file not found: {src}", name=src.name)
    ctx.report(None, "reading source", phase="preparing")
    info = media.probe(src)
    if not info.audio:
        raise FriendlyError("no_audio", "this video has no sound")
    gpu = pick_gpu(req.gpu_index)
    started = time.monotonic()
    to_video = req.add_to_video or req.burn_in
    if req.burn_in:
        burn.check(info)  # before the long listening step
    # Share of the progress bar for listening; writing onto the picture takes a real part of it.
    listen_part = 0.95
    if req.burn_in:
        listen = estimate.seconds(info.duration or 0.0, req.level, gpu)[0]
        listen_part = listen / (listen + burn.seconds(info, gpu))
    tr = transcribe_video(src, info, level=req.level, language=req.language, translate=req.translate,
                          audio_track=req.audio_track, gpu=gpu, ctx=ctx, span=(0.0, listen_part))
    lang = "en" if req.translate else tr.language
    ctx.log(f"language {tr.language}{' -> en' if req.translate else ''}, {len(tr.words)} words")

    cues = cues_from_words(tr.words, Rules())
    folder = req.output_dir or src.parent
    srt_path = folder / f"{src.stem}.{lang}.srt"
    if not req.overwrite:
        srt_path = unique_path(srt_path)
    srt.write(cues, srt_path)
    result = srt_path

    if to_video:
        ext = src.suffix.lower() if src.suffix.lower() in (".mp4", ".mkv", ".mov", ".m4v") else ".mkv"
        dst = folder / f"{src.stem}_subtitled{ext}"
        if not req.overwrite:
            dst = unique_path(dst)
        partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
        try:
            b = burn.make(info, srt_path, lang, gpu) if req.burn_in else None
            a = listen_part if req.burn_in else 0.96
            t0 = time.monotonic()
            mux_subtitles(src, info, srt_path, partial, ISO3.get(lang, "und"), cancel=ctx.cancel, burn=b,
                          on_progress=lambda f, s: ctx.report(None if f is None else a + f * (1 - a),
                                                              "writing subtitles onto the picture" if b else "adding subtitles",
                                                              phase="working"))
            partial.replace(dst)
            if b:
                burn.learn(info, b, time.monotonic() - t0)
        except BaseException:
            remove_quietly(partial)
            raise
        result = dst

    ctx.log(f"done in {time.monotonic() - started:.0f}s: {len(cues)} subtitles")
    ctx.report(1.0, f"done: {result.name}", phase="done")
    return result
