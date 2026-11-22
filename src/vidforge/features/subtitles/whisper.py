"""Speech to timed words with whisper.cpp (`whisper-cli`).

Audio is first turned into 16 kHz mono WAV (what Whisper listens to). whisper-cli then writes
its detailed JSON with every token. Word times come from DTW alignment (`-dtw`, which needs
flash attention off): on test speech they were within about 0.3 s of the real word starts,
where the default token times were up to 1.6 s early. Tokens are joined back into words (a
token that starts with a space starts a new word), and words outside detected speech are
dropped (see vad.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...core import tools
from ...core.ffmpeg import run_ffmpeg
from ...core.gpu import GpuInfo
from ...core.media import MediaInfo
from ...core.process import CancelToken, run_process
from . import vad
from .segment import Word


@dataclass
class Transcript:
    language: str  # ISO 639-1 code whisper detected or was told, e.g. "en", "ar"
    words: list[Word]
    text: str

    def to_json(self) -> dict:
        return {"language": self.language, "text": self.text,
                "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in self.words]}

    @classmethod
    def from_json(cls, data: dict) -> "Transcript":
        return cls(data["language"], [Word(t, s, e) for t, s, e in data["words"]], data.get("text", ""))


def whisper_tool(gpu: GpuInfo | None) -> str:
    """CUDA build on NVIDIA cards, CPU build elsewhere (or when the CUDA one is missing)."""
    if gpu and gpu.vendor == "nvidia" and tools.locate("whisper")[0] is not None:
        return "whisper"
    if tools.locate("whisper-cpu")[0] is not None:
        return "whisper-cpu"
    return "whisper" if gpu and gpu.vendor == "nvidia" else "whisper-cpu"


def extract_audio(src: Path, info: MediaInfo, track: int, dst: Path, *, cancel: CancelToken,
                  on_progress: Callable[[float | None, str], None]) -> Path:
    """Audio track number *track* (0 = first) as 16 kHz mono 16-bit WAV."""
    if not info.audio:
        from ...core.errors import FriendlyError
        raise FriendlyError("no_audio", "this video has no sound")
    track = track if 0 <= track < len(info.audio) else 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", src, "-map", f"0:a:{track}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
               duration=info.duration, on_progress=lambda f, s: on_progress(f, "reading the sound"),
               cancel=cancel)
    return dst


_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")


def ascii_path(path: Path) -> Path:
    """The same file under a name made only of plain characters (Windows 8.3 short path).
    Raises a friendly error if the drive has short names turned off."""
    s = str(path)
    if s.isascii():
        return path
    import sys

    if sys.platform == "win32":
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf)) and buf.value.isascii():
            return Path(buf.value)
    from ...core.errors import FriendlyError

    raise FriendlyError("path_chars", f"the speech tool cannot open files under {s}", path=s)


def build_command(exe: Path, model: Path, wav: Path, out_base: Path, *, language: str | None,
                  translate: bool, dtw: str | None, threads: int) -> list[str]:
    # No -sns ("suppress non-speech tokens"): with it, large-v3-turbo dropped all punctuation
    # and capitals on our test speech. Sound notes it may write instead are removed afterwards.
    cmd = [str(exe), "-m", str(model), "-f", str(wav), "-of", str(out_base), "-ojf", "-pp",
           "-l", language or "auto", "-t", str(threads)]
    if translate:
        cmd.append("-tr")
    if dtw:
        cmd += ["-nfa", "-dtw", dtw]
    return cmd


_SPECIAL = re.compile(r"^\[_[A-Z]+_?\d*\]$|^\[_TT_\d+\]$|^<\|.*\|>$")
# Whole segments that describe sounds rather than speech: [Music], (laughs), ♪ ... ♪
_SOUND_NOTE = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\)|[♪♫].*[♪♫]?|\*[^*]*\*)\s*$")
_QUOTES = str.maketrans("", "", "\"“”„")
_PUNCT_ONLY = re.compile(r"^[^\w\s]+$")


def words_from_json(data: dict) -> tuple[str, list[Word]]:
    """whisper-cli -ojf output -> (language, words). Token offsets are milliseconds."""
    language = (data.get("result") or {}).get("language") or "und"
    words: list[Word] = []
    space_pending = False
    for seg in data.get("transcription", []):
        if _SOUND_NOTE.match(seg.get("text", "")):
            continue
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if not text or _SPECIAL.match(text.strip()):
                continue
            text = text.translate(_QUOTES)  # Whisper sometimes wraps sentences in quotes
            if not text.strip():
                space_pending = space_pending or bool(text)  # a lone space: the next piece starts a word
                continue
            if space_pending and not text.startswith(" ") and not _PUNCT_ONLY.match(text):
                text = " " + text
            space_pending = False
            if words and _PUNCT_ONLY.match(text):
                words[-1].text += text.strip()  # punctuation always belongs to the word before it
                continue
            off = tok.get("offsets") or {}
            start, end = off.get("from", 0) / 1000, off.get("to", 0) / 1000
            dtw = tok.get("t_dtw", -1)
            if dtw is not None and dtw >= 0:  # centiseconds; much closer to the real word start
                start = dtw / 100
                end = max(end, start)
            if words and not text.startswith(" ") and words[-1].end >= start - 0.05:
                words[-1].text += text  # continuation of the previous word
                words[-1].end = max(words[-1].end, end)
            else:
                if not words and not text.startswith(" "):
                    text = " " + text
                words.append(Word(text, start, max(end, start)))
    # Some tokens carry no duration; give each word a sensible end and keep times increasing.
    for i, w in enumerate(words):
        if i + 1 < len(words) and w.end > words[i + 1].start:
            w.end = words[i + 1].start
        if w.end <= w.start:
            w.end = min(w.start + 0.3, words[i + 1].start if i + 1 < len(words) else w.start + 0.3)
    return language, words


def transcribe(wav: Path, out_base: Path, *, gpu: GpuInfo | None, model: Path, vad_model: Path | None,
               dtw: str | None,
               language: str | None, translate: bool, cancel: CancelToken,
               on_progress: Callable[[float | None, str], None], on_log: Callable[[str], None]) -> Transcript:
    import os

    exe = tools.exe_path(whisper_tool(gpu))
    threads = max(2, min(8, (os.cpu_count() or 4) - 1))
    speech = None
    if vad_model is not None:
        on_progress(None, "finding speech")
        speech = vad.detect(wav, vad_model)
        on_log(f"speech regions: {len(speech)}, {sum(e - s for s, e in speech):.0f}s of talking")
    # whisper-cli reads its command line in the old Windows code page, so non-English characters
    # in paths turn into "?". Run it inside the work folder with plain relative names, and give
    # it the model's short (8.3) path when the real path has such characters.
    cmd = build_command(exe, ascii_path(model), Path(wav.name), Path(out_base.name), language=language,
                        translate=translate, dtw=dtw, threads=threads)
    on_log("$ " + " ".join(cmd))

    def handle(line: str) -> None:
        m = _PROGRESS.search(line)
        if m:
            on_progress(int(m.group(1)) / 100, "listening")
        elif line.strip():
            on_log(line)

    run_process(cmd, on_line=handle, cancel=cancel, cwd=wav.parent)  # DLLs still load from exe's folder
    data = json.loads(Path(f"{out_base}.json").read_text(encoding="utf-8", errors="replace"))
    language_code, words = words_from_json(data)
    if speech is not None:
        kept = [w for w in words if vad.inside(w.start, w.end, speech)]
        if len(kept) < len(words):
            on_log(f"dropped {len(words) - len(kept)} words heard outside speech")
        words = kept
    text = "".join(w.text for w in words).strip()
    return Transcript(language_code, words, text)
