"""Translate & dub: speech in a video -> translated subtitles, and optionally a new voice track.

Steps: transcript (reused from "Write the words" when the video was done before) -> lines ->
translation by a local language model -> (dub only) voices separated from the background,
speakers told apart, every line said in the speaker's own voice in the new language, lines that
run too long made shorter and said again -> fitted into their time and mixed over the background
-> saved as a copy of the video: the dub as the main sound, the original sound as a second
track, the translated subtitles as a track that can be turned on.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import media, paths, tools
from ...core.errors import FriendlyError
from ...core.ffmpeg import run_ffmpeg
from ...core.files import remove_quietly
from ...core.jobs import JobContext
from ...core.media import MediaInfo
from ...core.process import run_process
from ...core.streams import plan_audio, plan_subtitles
from ..subtitles import srt
from ..subtitles.segment import Rules, cues_from_words
from ..subtitles import burn
from ..subtitles.burn import Burn
from ..subtitles.service import ISO3, load_transcript, mux_subtitles, transcribe_video
from ..upscale import estimate as up_est
from ..upscale.service import pick_gpu, unique_path
from . import addon, audio, estimate, llm, models
from . import translate as tl

LEVELS = ("best", "quick")
MP4_EXT = (".mp4", ".m4v", ".mov")


@dataclass
class DubRequest:
    input: Path
    target: str = "ar"  # language to translate into (ISO 639-1)
    dub: bool = True  # False = translated subtitles only
    add_to_video: bool = False  # subtitles only: also save a copy of the video with them inside
    burn_in: bool = False  # write the subtitles onto the picture of the new copy (always shown)
    level: str = "best"  # translator: best (Gemma 3 12B) or quick (Qwen3 8B)
    keep_original: bool = True  # dub: keep the original sound as a second track
    language: str = "auto"  # spoken language, used only if the video has no transcript yet
    audio_track: int = 0
    output_dir: Path | None = None
    gpu_index: int = 0
    overwrite: bool = False
    keep_temp: bool = False
    voice: str = "chatterbox"  # chatterbox (MIT) or xtts (XTTS-v2, personal use only)


class Spans:
    """Shares the job's progress bar among its steps, in proportion to their expected time."""

    def __init__(self, ctx: JobContext, weights: dict[str, float]):
        self.ctx = ctx
        total = sum(weights.values()) or 1.0
        self.span: dict[str, tuple[float, float]] = {}
        acc = 0.0
        for k, w in weights.items():
            self.span[k] = (acc / total, (acc + w) / total)
            acc += w

    def report(self, step: str, frac: float | None, msg: str, phase: str = "working") -> None:
        a, b = self.span.get(step, (0.0, 0.0))
        self.ctx.report(None if frac is None else a + (b - a) * max(0.0, min(1.0, frac)), msg, phase=phase)

    def range(self, step: str) -> tuple[float, float]:
        return self.span.get(step, (0.0, 0.0))


def _check(req: DubRequest) -> None:
    if req.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if req.target not in tl.LANGUAGE_NAMES:
        raise FriendlyError("bad_language", f"unknown language {req.target}", lang=req.target)
    if req.dub:
        if req.voice not in models.ENGINES:
            raise ValueError(f"voice must be one of {models.ENGINES}")
        if req.target not in models.VOICE_LANGUAGES[req.voice]:
            raise FriendlyError("voice_language", f"the voice can't speak {tl.language_name(req.target)} yet",
                                lang=tl.NATIVE_NAMES.get(req.target, req.target))
        if not addon.is_installed(req.voice):
            raise FriendlyError("voice_addon", "the voice add-on is not installed (vidforge addon install)")


def _downloader(spans: Spans, what: str) -> Callable[[int, int | None], None]:
    def cb(done: int, total: int | None) -> None:
        spans.ctx.report(None, f"downloading {what} {done / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB", phase="downloading")
    return cb


# --- the voice worker -------------------------------------------------------------------------

def run_worker(job: dict, work: Path, *, ctx: JobContext, on_stage: Callable[[str, float], None]) -> dict:
    job = dict(job, result=str(work / f"result-{uuid.uuid4().hex[:6]}.json"))
    job_file = work / f"job-{uuid.uuid4().hex[:6]}.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    script = Path(__file__).with_name("voice_worker.py")
    cmd = [str(addon.python()), "-I", "-X", "utf8", str(script), str(job_file)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def handle(line: str) -> None:
        if line.startswith("@@"):
            try:
                msg = json.loads(line[2:])
                on_stage(msg.get("stage", ""), float(msg.get("progress") or 0.0))
            except ValueError:
                pass
        elif "it/s]" in line or "s/it]" in line or "Warning" in line or "deprecat" in line:
            pass  # progress bars and library warnings
        else:
            ctx.log(line)

    ctx.log("$ " + " ".join(cmd))
    run_process(cmd, on_line=handle, cancel=ctx.cancel, env=env, cwd=work, tail_lines=80)
    return json.loads(Path(job["result"]).read_text(encoding="utf-8"))


# --- mux ----------------------------------------------------------------------------------------

def mux_dub(src: Path, info: MediaInfo, dub_wav: Path, srt_path: Path, dst: Path, *, target: str,
            keep_original: bool, cancel, on_progress, burn_in: Burn | None = None, work: Path | None = None) -> None:
    """The dubbed copy. With *burn_in* the translated subtitles are written onto the picture (and
    not added as a track too, which players might show on top of them)."""
    src, dub_wav, srt_path, dst = (Path(x).resolve() for x in (src, dub_wav, srt_path, dst))  # FFmpeg may run elsewhere
    container = "mp4" if dst.suffix.lower() in MP4_EXT else "mkv"
    lang3 = ISO3.get(target, "und")
    name = tl.language_name(target)
    video = ["-map", f"0:{info.video.index}"] if burn_in and info.video else ["-map", "0:v"]
    args: list = ["-i", src, "-i", dub_wav, *([] if burn_in else ["-i", srt_path]), *video, "-map", "1:a"]
    originals = plan_audio(info, container) if keep_original else []
    for p in originals:
        args += ["-map", f"0:{p.stream.index}"]
    subs = [p for p in plan_subtitles(info, container) if p.action != "drop"]
    for p in subs:
        args += ["-map", f"0:{p.stream.index}"]
    if container == "mkv":
        args += ["-map", "0:t?"]
    if not burn_in:
        args += ["-map", "2:0"]
    args += ["-c", "copy", *(burn_in.args() if burn_in else []),
             "-c:a:0", "aac", "-b:a:0", "192k", "-ar:a:0", "48000", "-filter:a:0", "alimiter=limit=0.95:level=0",
             "-metadata:s:a:0", f"language={lang3}", "-metadata:s:a:0", f"title={name} (dub)",
             "-disposition:a:0", "default"]
    for i, p in enumerate(originals, start=1):
        if p.action == "transcode":
            args += [f"-c:a:{i}", p.codec, f"-b:a:{i}", "192k"]
        args += [f"-disposition:a:{i}", "0"]
        if not p.stream.title:
            args += [f"-metadata:s:a:{i}", "title=Original"]
    for i, p in enumerate(subs):
        if p.action == "transcode":
            args += [f"-c:s:{i}", p.codec]
        args += [f"-disposition:s:{i}", "0"]
    n = len(subs)
    if not burn_in:
        args += [f"-c:s:{n}", "mov_text" if container == "mp4" else "srt", f"-metadata:s:s:{n}",
                 f"language={lang3}", f"-metadata:s:s:{n}", f"title={name}", f"-disposition:s:{n}", "0"]
    if container == "mp4":
        args += ["-movflags", "+faststart"]
    args.append(dst)
    cwd = burn_in.prepare(work or paths.temp_dir() / f"burn-{uuid.uuid4().hex[:12]}") if burn_in else None
    try:
        run_ffmpeg(args, duration=info.duration, on_progress=on_progress, cancel=cancel, cwd=cwd)
    finally:
        if cwd is not None and work is None:
            shutil.rmtree(cwd, ignore_errors=True)


# --- the job ------------------------------------------------------------------------------------

def run_dub(req: DubRequest, ctx: JobContext) -> Path:
    src = req.input.resolve()
    if not src.is_file():
        raise FriendlyError("file_missing", f"file not found: {src}", name=src.name)
    _check(req)
    ctx.report(None, "reading source", phase="preparing")
    info = media.probe(src)
    if not info.audio:
        raise FriendlyError("no_audio", "this video has no sound")
    if req.burn_in:
        burn.check(info)  # before the long steps
    gpu = pick_gpu(req.gpu_index)
    dev = estimate.device(gpu)
    stats = up_est.load_stats()
    started = time.monotonic()
    dur = info.duration or 0.0

    transcript = load_transcript(src)
    spoken_guess = dur * 0.7 if transcript is None else sum(w.end - w.start for w in transcript.words) * 1.25
    chars_guess = dur * 12 if transcript is None else len(transcript.text)
    weights = {
        "listen": 0.0 if transcript is not None else 4 + dur / 25,
        "translate": estimate.LLM_STARTUP + chars_guess / estimate.LLM_SPEED[(req.level, dev)],
    }
    if req.dub:
        weights.update(separate=estimate.WORKER_STARTUP + dur / estimate.SEPARATE_SPEED[dev],
                       voice=spoken_guess / estimate.TTS_SPEED[(req.voice, dev)], fit=3 + dur / 100)
    weights["save"] = 1 + dur / 150 + (burn.seconds(info, gpu, stats) if req.burn_in else 0.0)
    spans = Spans(ctx, weights)

    # 1. what is said
    if transcript is None:
        ctx.log("no saved transcript: writing down the speech first")
        transcript = transcribe_video(src, info, level="balanced", language=req.language, translate=False,
                                      audio_track=req.audio_track, gpu=gpu, ctx=ctx, span=spans.range("listen"))
    else:
        ctx.log(f"reusing the saved transcript ({len(transcript.words)} words)")
    if not transcript.words:
        raise FriendlyError("no_speech", "no speech was found in this video")
    src_lang = transcript.language
    if src_lang == req.target:
        raise FriendlyError("same_language", f"the video is already in {tl.language_name(req.target)}",
                            lang=tl.NATIVE_NAMES.get(req.target, req.target))
    cues = cues_from_words(transcript.words, Rules())
    lines = tl.build_lines(cues, duration=dur)
    ctx.log(f"{src_lang} -> {req.target}: {len(cues)} subtitles in {len(lines)} lines")

    # 2. translate
    if tools.locate("llama")[0] is None:
        tools.fetch("llama", progress=_downloader(spans, "translator program"), cancel=ctx.cancel)
    model = models.ensure_llm(req.level, progress=_downloader(spans, "translator"), cancel=ctx.cancel)
    rate = tl.cps(req.target, estimate.learned_cps(stats, req.voice))
    spans.report("translate", 0.0, "starting the translator")
    t0 = time.monotonic()
    with llm.LlamaServer(model, cancel=ctx.cancel, on_log=ctx.log, gpu=dev == "gpu") as server:
        spans.report("translate", 0.05, "translating")
        tl.translate_lines(lines, server.chat, src=src_lang, tgt=req.target, rate=rate, dub=req.dub,
                           on_progress=lambda f: spans.report("translate", 0.05 + 0.85 * f, "translating"))
        if req.dub:
            long = tl.too_long(lines, rate)
            if long:
                ctx.log(f"making {len(long)} lines shorter so they fit")
                spans.report("translate", 0.92, "making long lines shorter")
                tl.shorten_lines(long, server.chat, tgt=req.target,
                                 limits={ln.id: tl.char_limit(ln, req.target, rate, stretch=1.05) for ln in long})
    translate_seconds = time.monotonic() - t0
    estimate.learn(f"dub/llm/{req.level}/{dev}", sum(len(ln.source) for ln in lines) / max(1.0, translate_seconds - 5))

    folder = req.output_dir or src.parent
    folder.mkdir(parents=True, exist_ok=True)
    srt_path = folder / f"{src.stem}.{req.target}.srt"
    if not req.overwrite:
        srt_path = unique_path(srt_path)
    ext = src.suffix.lower() if src.suffix.lower() in MP4_EXT + (".mkv",) else ".mkv"
    dst = folder / f"{src.stem}_{req.target}{ext}"
    if not req.overwrite:
        dst = unique_path(dst)

    if not req.dub:
        srt.write(tl.spread(lines, cues, req.target), srt_path)
        result = srt_path
        if req.add_to_video or req.burn_in:
            partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
            try:
                b = burn.make(info, srt_path, req.target, gpu) if req.burn_in else None
                t0 = time.monotonic()
                mux_subtitles(src, info, srt_path, partial, ISO3.get(req.target, "und"), cancel=ctx.cancel, burn=b,
                              on_progress=lambda f, s: spans.report(
                                  "save", f, "writing subtitles onto the picture" if b else "adding subtitles"))
                partial.replace(dst)
                if b:
                    burn.learn(info, b, time.monotonic() - t0)
            except BaseException:
                remove_quietly(partial)
                raise
            result = dst
        ctx.log(f"done in {time.monotonic() - started:.0f}s")
        ctx.report(1.0, f"done: {result.name}", phase="done")
        return result

    # 3. voices
    voice_files = models.ensure_voice(req.voice, progress=_downloader(spans, "voice"), cancel=ctx.cancel)
    work = paths.temp_dir() / f"dub-{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        result = _dub_audio_and_save(req, ctx, spans, src, info, work, lines, cues, model, voice_files,
                                     rate, dev, srt_path, dst, gpu)
    finally:
        if not req.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            ctx.log(f"work files kept in {work}")
    ctx.log(f"done in {time.monotonic() - started:.0f}s")
    ctx.report(1.0, f"done: {result.name}", phase="done")
    return result


def _dub_audio_and_save(req: DubRequest, ctx: JobContext, spans: Spans, src: Path, info: MediaInfo, work: Path,
                        lines: list[tl.Line], cues, model: Path, voice_files: dict[str, Path], rate: float,
                        dev: str, srt_path: Path, dst: Path, gpu) -> Path:
    dur = info.duration or 0.0
    track = req.audio_track if 0 <= req.audio_track < len(info.audio) else 0
    mix_wav = work / "mix.wav"
    spans.report("separate", None, "reading the sound", phase="preparing")
    run_ffmpeg(["-i", src, "-map", f"0:a:{track}", "-vn", "-ac", "2", "-ar", str(audio.SR), "-c:a", "pcm_f32le", mix_wav],
               duration=dur, cancel=ctx.cancel)
    device = "cuda" if dev == "gpu" else "cpu"
    stage_msgs = {"separate": ("separate", "separating voices from music"),
                  "speakers": ("separate", "finding who speaks"),
                  "tts": ("voice", "making the new voices")}
    t_worker = time.monotonic()

    def on_stage(stage: str, frac: float) -> None:
        step, msg = stage_msgs.get(stage, ("voice", "working"))
        if stage == "speakers":
            frac = 0.9 + 0.1 * frac
        elif stage == "separate":
            frac = 0.15 + 0.75 * frac
        spans.report(step, frac, msg)

    def tts_lines(todo: list[tl.Line]) -> list[dict]:
        return [{"id": ln.id, "text": ln.text, "expect": min(ln.slot, len(ln.text) / rate)} for ln in todo]

    # cfg_weight 0: Chatterbox carries less of the reference language's accent over.
    voice_job = {"engine": req.voice, "model_dir": str(voice_files["voice"]), "encoder_dir": str(voice_files["encoder"]),
                 "language": req.target, "codes": models.XTTS_CODES, "cfg_weight": 0.0, "exaggeration": 0.5}
    spans.report("separate", 0.02, "starting the voice helper")
    result = run_worker({
        "device": device,
        "separate": {"input": str(mix_wav), "model": str(voice_files["demucs"]), "vocals": str(work / "vocals.wav"),
                     "background": str(work / "background.wav")},
        "speakers": {"vocals": str(work / "vocals.wav"), "segments": [[ln.start, ln.end] for ln in lines],
                     "out_dir": str(work / "voices"), "threshold": 0.75},
        "tts": dict(voice_job, out_dir=str(work / "lines"), lines=tts_lines(lines)),
    }, work, ctx=ctx, on_stage=on_stage)
    worker_seconds = time.monotonic() - t_worker
    labels = result.get("speakers", {}).get("labels", [])
    refs = result.get("speakers", {}).get("refs", {})
    for ln, lb in zip(lines, labels):
        ln.speaker = lb
    ctx.log(f"{len(refs)} speaker(s); separation {result.get('separate_seconds', 0):.0f}s, "
            f"voices {result.get('tts_seconds', 0):.0f}s")
    said = {d["id"]: d for d in result["lines"]}

    # learn: how fast the voice talks in this language, and how fast this PC is
    chars = sum(len(ln.text) for ln in lines)
    seconds = sum(said[ln.id]["seconds"] for ln in lines)
    if seconds > 3:
        estimate.learn(f"dub/cps/{req.voice}/{req.target}", chars / seconds)
        rate = chars / seconds
    if result.get("tts_seconds"):
        estimate.learn(f"dub/tts/{req.voice}/{dev}", seconds / result["tts_seconds"])
    if result.get("separate_seconds") and dur > 10:
        estimate.learn(f"dub/separate/{dev}", dur / result["separate_seconds"])
    ctx.log(f"voice helper took {worker_seconds:.0f}s")

    # 4. lines still too long: shorter wording, said again (one round)
    long = [ln for ln in lines if audio.needs_shorter(said[ln.id]["seconds"], ln.slot)]
    if long:
        ctx.log(f"{len(long)} lines are still too long; asking for shorter ones")
        spans.report("fit", 0.05, "making long lines shorter")
        limits = {}
        for ln in long:
            line_rate = len(ln.text) / max(0.1, said[ln.id]["seconds"])
            limits[ln.id] = max(6, int(ln.slot * line_rate * audio.MAX_TEMPO * 0.95))
        before = {ln.id: ln.text for ln in long}
        with llm.LlamaServer(model, cancel=ctx.cancel, on_log=ctx.log, gpu=dev == "gpu") as server:
            tl.shorten_lines(long, server.chat, tgt=req.target, limits=limits)
        changed = [ln for ln in long if ln.text != before[ln.id]]
        if changed:
            spans.report("fit", 0.3, "saying the shorter lines")
            again = run_worker({
                "device": device,
                "tts": dict(voice_job, out_dir=str(work / "lines2"), refs=refs,
                            lines=[dict(x, speaker=ln.speaker) for x, ln in zip(tts_lines(changed), changed)]),
            }, work, ctx=ctx, on_stage=lambda s, f: spans.report("fit", 0.3 + 0.4 * f, "saying the shorter lines"))
            for d in again["lines"]:
                if d["seconds"] < said[d["id"]]["seconds"]:
                    said[d["id"]] = d
                else:  # the new take is no shorter: keep the old wording too
                    ln = next(x for x in changed if x.id == d["id"])
                    ln.text = before[ln.id]

    # 5. fit and mix
    spans.report("fit", 0.75, "fitting the voices")
    vocals, vsr = audio.read_wav(work / "vocals.wav")
    vocals_mono = vocals.mean(axis=1)
    background, bsr = audio.read_wav(work / "background.wav")
    overall = audio.active_rms(vocals_mono, vsr)
    (work / "fit").mkdir(exist_ok=True)
    clips = []
    for k, ln in enumerate(lines):
        d = said[ln.id]
        tempo = audio.tempo_for(d["seconds"], ln.slot,
                                audio.HARD_TEMPO if audio.needs_shorter(d["seconds"], ln.slot) else audio.MAX_TEMPO)
        fitted = audio.stretch(Path(d["path"]), work / "fit" / f"line{ln.id:04d}.wav", tempo, cancel=ctx.cancel)
        clip, csr = audio.read_wav(fitted)
        clip = clip[:, 0]
        orig = vocals_mono[int(ln.start * vsr):int(ln.end * vsr)]
        clip = clip * audio.gain_to_match(clip, orig, csr, overall)
        clips.append((ln.start, clip))
        if tempo > 1.01:
            ctx.log(f"line {ln.id}: {d['seconds']:.1f}s in a {ln.slot:.1f}s slot, sped up x{tempo:.2f}")
        spans.report("fit", 0.75 + 0.2 * (k + 1) / len(lines), "fitting the voices")
    voice = audio.place(clips, len(background), bsr)
    dub_wav = audio.write_wav(work / "dub.wav", audio.mix(background, voice), bsr)

    # 6. subtitles that match the words now spoken, and the video
    srt.write(tl.spread(lines, cues, req.target), srt_path)
    partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
    try:
        b = burn.make(info, srt_path, req.target, gpu) if req.burn_in else None
        t0 = time.monotonic()
        mux_dub(src, info, dub_wav, srt_path, partial, target=req.target, keep_original=req.keep_original,
                cancel=ctx.cancel, burn_in=b, work=work / "burn",
                on_progress=lambda f, s: spans.report(
                    "save", f, "writing subtitles onto the picture" if b else "saving the video"))
        partial.replace(dst)
        if b:
            burn.learn(info, b, time.monotonic() - t0)
    except BaseException:
        remove_quietly(partial)
        raise
    return dst
