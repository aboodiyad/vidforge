"""Subtitle splitting, SRT reading/writing and whisper output parsing (no models needed)."""

import pytest

from vidforge.features.subtitles import segment, srt, whisper
from vidforge.features.subtitles.segment import Rules, Word


def words_from(text: str, start: float = 0.0, per_word: float = 0.35, gaps: dict | None = None) -> list[Word]:
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(Word(" " + w, t, t + per_word - 0.05))
        t += per_word
    return out


def test_timestamp_and_roundtrip(tmp_path):
    assert srt.timestamp(3725.5) == "01:02:05,500"
    cues = [srt.Cue(1.0, 2.5, "Hello"), srt.Cue(3.0, 4.2, "مرحبا\nبالعالم")]
    p = srt.write(cues, tmp_path / "a.srt")
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw  # BOM + Windows line ends
    back = srt.read(p)
    assert [(c.start, c.end, c.text) for c in back] == [(1.0, 2.5, "Hello"), (3.0, 4.2, "مرحبا\nبالعالم")]


def test_lines_are_short_and_at_most_two():
    text = ("Look at the garden. She grows tomatoes, cucumbers, peppers and even a few watermelons "
            "and grandma says the secret is good soil and lots of patience every single morning.")
    cues = segment.cues_from_words(words_from(text))
    for c in cues:
        lines = c.text.split("\n")
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines), c.text
        assert c.duration <= Rules().max_duration + Rules().linger + 1e-6
    assert " ".join(c.text.replace("\n", " ") for c in cues) == text  # nothing lost or reordered


def test_breaks_at_sentence_end_and_long_pause():
    w = words_from("Welcome to our video. Today we visit grandma.", gaps={4: 0.3})
    cues = segment.cues_from_words(w)
    assert cues[0].text == "Welcome to our video."
    w = words_from("one two three four five six", gaps={3: 1.0})
    assert [c.text for c in segment.cues_from_words(w)] == ["one two three", "four five six"]


def test_cues_do_not_overlap_and_last_long_enough():
    w = words_from("We went to the beach today. Then we drove home.", per_word=0.3, gaps={6: 0.3})
    cues = segment.cues_from_words(w)
    assert len(cues) == 2
    assert cues[0].end <= cues[1].start - Rules().gap + 1e-9
    assert all(c.duration >= 0.2 for c in cues)
    assert cues[1].duration >= Rules().min_duration - 1e-9  # last one gets its minimum time


def test_two_line_wrap_is_balanced():
    w = words_from("The kids are already running ahead, slow down and wait for us")
    text = segment._wrap(w, 42)
    top, bottom = text.split("\n")
    assert abs(len(top) - len(bottom)) <= 12 and top.endswith(",")


def test_words_from_whisper_json_joins_tokens():
    data = {
        "result": {"language": "en"},
        "transcription": [{"tokens": [
            {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
            {"text": " Wel", "offsets": {"from": 100, "to": 300}},
            {"text": "come", "offsets": {"from": 300, "to": 500}},
            {"text": " home", "offsets": {"from": 520, "to": 900}},
            {"text": ".", "offsets": {"from": 900, "to": 950}},
            {"text": "[_TT_47]", "offsets": {"from": 950, "to": 950}},
        ]}],
    }
    lang, words = whisper.words_from_json(data)
    assert lang == "en"
    assert [(w.text, w.start, w.end) for w in words] == [(" Welcome", 0.1, 0.5), (" home.", 0.52, 0.9)]


def test_words_from_json_quotes_punctuation_and_sound_notes():
    seg = lambda text, toks: {"text": text, "tokens": [  # noqa: E731
        {"text": t, "offsets": {"from": a, "to": b}, "t_dtw": d} for t, a, b, d in toks]}
    data = {"result": {"language": "en"}, "transcription": [
        seg(' "Welcome home."', [(' "', 0, 0, 10), ("Welcome", 0, 400, 20), (" home", 400, 800, 60),
                                 ('."', 800, 900, 300), (" ", 900, 900, 300)]),
        seg("[Music]", [(" [", 1000, 1100, 110), ("Music", 1100, 1500, 120), ("]", 1500, 1600, 150)]),
        seg(' "It rains."', [('"It', 2000, 2300, 205), (" rains", 2300, 2600, 230), ('."', 2600, 2700, 260)]),
    ]}
    _, words = whisper.words_from_json(data)
    text = "".join(w.text for w in words).strip()
    assert text == "Welcome home. It rains."  # quotes gone, full stops kept, [Music] dropped
    assert words[0].start == 0.2  # DTW time (centiseconds) used, not the coarse offset


def test_command_has_vad_translate_and_language():
    from pathlib import Path

    cmd = whisper.build_command(Path("w.exe"), Path("m.bin"), Path("a.wav"), Path("out"), language=None,
                                translate=True, dtw="large.v3.turbo", threads=4)
    s = " ".join(cmd)
    assert "-l auto" in s and "-tr" in cmd and "-nfa -dtw large.v3.turbo" in s and "-ojf" in cmd
    assert "--vad" not in cmd  # whisper.cpp's own VAD shifts word times; we filter afterwards


def test_transcript_json_roundtrip():
    t = whisper.Transcript("ar", [Word(" مرحبا", 0.0, 0.5)], "مرحبا")
    assert whisper.Transcript.from_json(t.to_json()).words[0].text == " مرحبا"


def test_subtitle_estimate_scales_with_audio_length(isolated_home):
    from vidforge.core import media
    from vidforge.core.gpu import GpuInfo
    from vidforge.features.subtitles import estimate

    def info(d):
        return media.parse_ffprobe({"format": {"filename": "x.mp4", "duration": str(d)},
                                    "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}]})
    gpu = GpuInfo(0, "NVIDIA GeForce RTX 2060 SUPER", "nvidia", "1", 8192, (7, 5))
    short = estimate.estimate(info(60), level="balanced", gpu=gpu, add_to_video=False, stats={})
    long = estimate.estimate(info(3600), level="balanced", gpu=gpu, add_to_video=False, stats={})
    cpu = estimate.estimate(info(3600), level="balanced", gpu=None, add_to_video=False, stats={})
    assert short.seconds < long.seconds < cpu.seconds
    assert long.download_mb > 500
