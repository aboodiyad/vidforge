"""Translating subtitles with a local language model.

Subtitles are first joined into *lines*: one sentence (or a few short ones in a row), never
longer than about 12 seconds. A line is what gets translated and later spoken, because word
order changes between languages and half a sentence often cannot be translated on its own.

Lines go to the model in batches, numbered, with the lines just before (already translated)
and just after as context. The answer must match a JSON schema with exactly those numbers as
keys, so a line can never go missing or be merged with its neighbour. Each line also gets a
length hint: about as many characters as fit in the time it is spoken, so a dub can keep up.

The translation is then spread back over the original subtitles of that line, keeping their
times exactly, and wrapped by the subtitle rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from ..subtitles.segment import _SENTENCE_END, wrap_text
from ..subtitles.srt import Cue

LANGUAGE_NAMES = {
    "en": "English", "ar": "Arabic", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)",
    "tr": "Turkish", "hi": "Hindi", "ur": "Urdu", "fa": "Persian", "nl": "Dutch", "sv": "Swedish",
    "pl": "Polish", "id": "Indonesian", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "th": "Thai", "vi": "Vietnamese", "ms": "Malay", "bn": "Bengali",
    "ta": "Tamil", "da": "Danish", "fi": "Finnish", "no": "Norwegian", "sw": "Swahili",
}
# Each language's name in its own language, for messages shown to people.
NATIVE_NAMES = {
    "en": "English", "ar": "العربية", "fr": "Français", "es": "Español", "de": "Deutsch", "it": "Italiano",
    "pt": "Português", "tr": "Türkçe", "ru": "Русский", "uk": "Українська", "fa": "فارسی", "ur": "اردو",
    "hi": "हिन्दी", "bn": "বাংলা", "id": "Bahasa Indonesia", "ms": "Bahasa Melayu", "ja": "日本語", "ko": "한국어",
    "zh": "中文", "nl": "Nederlands", "pl": "Polski", "sv": "Svenska", "el": "Ελληνικά", "he": "עברית",
}
# How the target language should sound; Arabic speakers expect Modern Standard Arabic in dubs.
STYLE = {"ar": "Use clear Modern Standard Arabic (فصحى) without diacritics."}
NO_SPACES = {"zh", "ja", "th"}

# Characters a voice says per second, used to give the model a length to aim for. Measured
# with Chatterbox on test speech (Arabic ~10.5); others are rough and are refined by real runs.
DEFAULT_CPS = {"ar": 11.0, "zh": 4.5, "ja": 6.0, "ko": 6.5, "hi": 12.0, "he": 12.0, "ru": 13.0, "el": 13.0}
FALLBACK_CPS = 14.0

Chat = Callable[..., str]  # LlamaServer.chat-like: chat(messages, schema=..., ...) -> str


@dataclass
class Line:
    id: int
    start: float
    end: float
    source: str
    cues: list[int]  # indexes of the original subtitles this line covers
    slot: float = 0.0  # seconds the dub may take (until the next line starts)
    text: str = ""  # the translation
    speaker: int = 0
    tries: int = 0  # how many times it was made shorter

    @property
    def spoken(self) -> float:
        return self.end - self.start


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def cps(lang: str, learned: dict | None = None) -> float:
    if learned and learned.get(lang):
        return float(learned[lang])
    return DEFAULT_CPS.get(lang, FALLBACK_CPS)


def build_lines(cues: list[Cue], *, duration: float | None = None, max_seconds: float = 12.0,
                max_gap: float = 0.5, gap: float = 0.15) -> list[Line]:
    """Join subtitles into lines: the next subtitle joins when the sentence is not finished, the
    pause is short, and the line stays under *max_seconds*."""
    lines: list[Line] = []
    for i, c in enumerate(cues):
        text = " ".join(c.text.split())
        if lines:
            cur = lines[-1]
            unfinished = not cur.source.rstrip().endswith(_SENTENCE_END)
            if unfinished and c.start - cur.end <= max_gap and c.end - cur.start <= max_seconds:
                cur.source += " " + text
                cur.end = c.end
                cur.cues.append(i)
                continue
        lines.append(Line(len(lines), c.start, c.end, text, [i]))
    for k, ln in enumerate(lines):
        nxt = lines[k + 1].start if k + 1 < len(lines) else None
        if nxt is not None:
            ln.slot = max(ln.spoken, nxt - gap - ln.start)
        else:
            end = duration if duration else ln.end + 1.5
            ln.slot = max(ln.spoken, end - ln.start)
    return lines


def char_limit(line: Line, lang: str, rate: float, *, stretch: float = 1.1) -> int:
    """How many characters can be said in the line's slot (allowing a slightly faster voice)."""
    return max(8, int(line.slot * rate * stretch))


# --- prompts -------------------------------------------------------------------------------

def schema(ids: list[int]) -> dict:
    keys = [str(i) for i in ids]
    return {"type": "object", "properties": {k: {"type": "string"} for k in keys},
            "required": keys, "additionalProperties": False}


def _system(src: str, tgt: str, dub: bool) -> str:
    rules = [
        f"You translate the spoken lines of a video from {language_name(src)} into {language_name(tgt)}.",
        "Translate the meaning naturally, the way people really talk, keeping names, numbers and tone.",
        "Translate every numbered line on its own: never move words between lines, never skip one.",
        "Each line has a length limit in characters. Stay within it; say it more briefly if needed.",
    ]
    if dub:
        rules.append("The translation will be spoken aloud by a voice actor in the same time as the original.")
    if tgt in STYLE:
        rules.append(STYLE[tgt])
    rules.append("Answer only with a JSON object: the line numbers are the keys, the translations the values.")
    return "\n".join(rules)


def translate_messages(batch: list[Line], before: list[Line], after: list[Line], *, src: str, tgt: str,
                       limits: dict[int, int], dub: bool) -> list[dict]:
    parts = []
    if before:
        parts.append("Earlier lines, already translated (context only):")
        parts += [f"[{ln.id}] {ln.source}\n    => {ln.text}" for ln in before]
        parts.append("")
    parts.append("Translate these lines:")
    parts += [f"[{ln.id}] (max {limits[ln.id]} characters) {ln.source}" for ln in batch]
    if after:
        parts.append("")
        parts.append("The lines that follow (context only, do not translate):")
        parts += [f"[{ln.id}] {ln.source}" for ln in after]
    return [{"role": "system", "content": _system(src, tgt, dub)}, {"role": "user", "content": "\n".join(parts)}]


def shorten_messages(batch: list[Line], *, tgt: str, limits: dict[int, int]) -> list[dict]:
    sys_text = (
        f"You make {language_name(tgt)} dubbing lines shorter so they can be spoken in less time.\n"
        "Keep the meaning and the natural tone; drop filler words, use shorter words and phrasing.\n"
        "Each line must fit within its character limit.\n"
        + (STYLE.get(tgt, "") + "\n" if tgt in STYLE else "")
        + "Answer only with a JSON object: the line numbers are the keys, the shorter lines the values."
    )
    body = ["Make these lines shorter:"]
    body += [f"[{ln.id}] (max {limits[ln.id]} characters, now {len(ln.text)}) {ln.text}\n    original: {ln.source}"
             for ln in batch]
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": "\n".join(body)}]


def parse_answer(text: str, ids: list[int]) -> dict[int, str]:
    """The model's JSON answer -> {line id: text}. Raises ValueError if a line is missing or empty."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    data = json.loads(m.group(0))
    out: dict[int, str] = {}
    for i in ids:
        v = data.get(str(i))
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"line {i} is missing")
        out[i] = clean(v)
    return out


_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _SPACES.sub(" ", text).strip()
    return text.strip("\"“”«»").strip()


def _ask(chat: Chat, messages: list[dict], ids: list[int], tries: int = 3) -> dict[int, str]:
    last: Exception | None = None
    for attempt in range(tries):
        answer = chat(messages, schema=schema(ids), max_tokens=256 + 160 * len(ids),
                      temperature=0.3 + 0.2 * attempt)
        try:
            return parse_answer(answer, ids)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"the translator gave an unusable answer: {last}")


def translate_lines(lines: list[Line], chat: Chat, *, src: str, tgt: str, rate: float, dub: bool,
                    batch_size: int = 10, on_progress: Callable[[float], None] | None = None) -> None:
    """Fill in `text` for every line."""
    for b in range(0, len(lines), batch_size):
        batch = lines[b:b + batch_size]
        before = lines[max(0, b - 3):b]
        after = lines[b + batch_size:b + batch_size + 2]
        limits = {ln.id: char_limit(ln, tgt, rate) for ln in batch}
        got = _ask(chat, translate_messages(batch, before, after, src=src, tgt=tgt, limits=limits, dub=dub),
                   [ln.id for ln in batch])
        for ln in batch:
            ln.text = got[ln.id]
        if on_progress:
            on_progress(min(1.0, (b + len(batch)) / len(lines)))


def shorten_lines(lines: list[Line], chat: Chat, *, tgt: str, limits: dict[int, int], batch_size: int = 10) -> None:
    """Ask for shorter versions of *lines*; a line only changes if the new one is shorter."""
    for b in range(0, len(lines), batch_size):
        batch = lines[b:b + batch_size]
        got = _ask(chat, shorten_messages(batch, tgt=tgt, limits=limits), [ln.id for ln in batch])
        for ln in batch:
            ln.tries += 1
            if len(got[ln.id]) < len(ln.text):
                ln.text = got[ln.id]


def too_long(lines: list[Line], rate: float, *, allowance: float = 1.15) -> list[Line]:
    """Lines whose translation will probably take longer to say than their slot allows."""
    return [ln for ln in lines if len(ln.text) / rate > ln.slot * allowance]


# --- back onto the subtitles --------------------------------------------------------------

def _units(text: str, lang: str) -> tuple[list[str], str]:
    if lang in NO_SPACES and " " not in text.strip():
        return list(text), ""
    return text.split(), " "


def spread(lines: list[Line], cues: list[Cue], lang: str, max_line: int = 42) -> list[Cue]:
    """Translated subtitles with the original times. A line's words are shared among its
    subtitles in proportion to how much of the original each one held."""
    out: list[Cue] = []
    for ln in lines:
        members = [cues[i] for i in ln.cues]
        units, sep = _units(ln.text, lang)
        if len(members) == 1 or len(units) < len(members):
            out.append(Cue(members[0].start, members[-1].end, wrap_text(ln.text, max_line)))
            continue
        weights = [max(1, len(c.text)) for c in members]
        total = sum(weights)
        bounds, acc = [], 0
        for w in weights[:-1]:
            acc += w
            bounds.append(round(acc / total * len(units)))
        pieces, prev = [], 0
        for k, bnd in enumerate(bounds + [len(units)]):
            # every subtitle keeps at least one word, and leaves at least one for each after it
            remaining = len(members) - k - 1
            bnd = max(prev + 1, min(bnd, len(units) - remaining))
            pieces.append(sep.join(units[prev:bnd]))
            prev = bnd
        for c, text in zip(members, pieces):
            out.append(Cue(c.start, c.end, wrap_text(text, max_line) if sep else text))
    return out

