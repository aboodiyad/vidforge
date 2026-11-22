"""Turn timed words into readable subtitles.

Speech recognisers return long segments or single words; neither reads well on screen. These
rules follow common subtitling practice: short lines, at most two of them, each subtitle on
screen long enough to read but not so long that it lags behind the speech, and breaks at
natural points (end of a sentence, then a comma, then a pause in the speech).
"""

from __future__ import annotations

from dataclasses import dataclass

from .srt import Cue


@dataclass
class Word:
    text: str  # includes its leading space when it starts a new word
    start: float
    end: float


@dataclass
class Rules:
    max_line: int = 42  # characters per line
    max_lines: int = 2
    min_duration: float = 1.0
    max_duration: float = 6.0
    pause_split: float = 0.6  # a silence this long ends a subtitle once it has enough to read
    long_pause: float = 1.0  # a silence this long always ends a subtitle
    gap: float = 0.08  # keep subtitles from touching (about two frames)
    linger: float = 0.4  # how long a subtitle may stay after the last word, if there is room


_SENTENCE_END = (".", "?", "!", "…", "。", "؟", "！", "？")
_CLAUSE_END = (",", ";", ":", "،", "؛", "，", "、")


def _text(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


def _wrap(words: list[Word], max_line: int) -> str:
    """Two lines, split at the word boundary that makes them most even, with a bonus for
    breaking after punctuation."""
    text = _text(words)
    if len(text) <= max_line:
        return text
    best, best_score = text, None
    acc = ""
    for i in range(1, len(words)):
        acc += words[i - 1].text
        top, bottom = acc.strip(), _text(words[i:])
        if len(top) > max_line or len(bottom) > max_line:
            continue
        score = abs(len(top) - len(bottom)) - (8 if top.endswith(_CLAUSE_END + _SENTENCE_END) else 0)
        if best_score is None or score < best_score:
            best, best_score = f"{top}\n{bottom}", score
    return best


def _fits(words: list[Word], rules: Rules) -> bool:
    if words[-1].end - words[0].start > rules.max_duration:
        return False
    return len(_text(words)) <= rules.max_line * rules.max_lines - 2


def _split_point(words: list[Word], rules: Rules) -> int:
    """Where to end the current subtitle when the next word would not fit.

    Every word boundary after the first third is scored: a sentence end is best, then a comma
    or similar, then a pause (longer is better); later boundaries win ties so subtitles stay
    full. The part before the break must still fit on screen."""
    n = len(words)
    best, best_score = n - 1, None
    for i in range(max(1, n // 3), n):
        if not _fits(words[:i], rules):
            break
        prev = words[i - 1].text.strip()
        score = 0.0
        if prev.endswith(_SENTENCE_END):
            score += 100
        elif prev.endswith(_CLAUSE_END):
            score += 50
        score += min(words[i].start - words[i - 1].end, 1.0) * 40
        score += i / n  # tie-break: prefer later
        if best_score is None or score >= best_score:
            best, best_score = i, score
    return best


def group(words: list[Word], rules: Rules | None = None) -> list[list[Word]]:
    rules = rules or Rules()
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if not w.text.strip():
            continue
        if cur:
            pause = w.start - cur[-1].end
            # After a full sentence, start a new subtitle once this one holds a little speech,
            # even without a pause: real speakers often run sentences together.
            sentence_done = cur[-1].text.strip().endswith(_SENTENCE_END) and (
                len(cur) >= 2 and cur[-1].end - cur[0].start >= 0.7)
            # A long silence always ends a subtitle; a short one only once there is enough to
            # read, so a pause after "cucumbers," doesn't leave one word alone on screen.
            enough = cur[-1].end - cur[0].start >= 1.5 or len(_text(cur)) >= 30
            if pause >= rules.long_pause or (pause >= rules.pause_split and enough) or sentence_done:
                groups.append(cur)
                cur = []
            elif not _fits(cur + [w], rules):
                cut = _split_point(cur + [w], rules)
                groups.append((cur + [w])[:cut])
                cur = (cur + [w])[cut:]
                continue
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def cues_from_words(words: list[Word], rules: Rules | None = None) -> list[Cue]:
    rules = rules or Rules()
    groups = group(words, rules)
    cues: list[Cue] = []
    for gi, g in enumerate(groups):
        start, end = g[0].start, g[-1].end
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else None
        limit = (next_start - rules.gap) if next_start is not None else end + rules.linger + rules.min_duration
        end = min(max(end + rules.linger, start + rules.min_duration), limit)
        end = max(end, g[-1].end if next_start is None else min(g[-1].end, limit))
        cues.append(Cue(start, max(end, start + 0.2), _wrap(g, rules.max_line)))
    return cues
