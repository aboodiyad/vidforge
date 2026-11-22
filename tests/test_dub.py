"""Translate & dub: lines, prompts and answers, spreading onto subtitles, fitting and mixing,
estimates, muxing arguments and the friendly errors (no models or PyTorch needed)."""

import json

import numpy as np
import pytest

from vidforge.core import models as model_registry
from vidforge.core.errors import FriendlyError
from vidforge.core.media import MediaInfo, StreamInfo
from vidforge.features.dub import addon, audio, estimate, llm, service, voice_worker
from vidforge.features.dub import translate as tl
from vidforge.features.subtitles.segment import wrap_text
from vidforge.features.subtitles.srt import Cue


def cues():
    return [
        Cue(1.0, 4.0, "Good morning everyone, and welcome"),
        Cue(4.1, 6.0, "back to our little show."),
        Cue(7.0, 9.0, "First, we need flour,"),
        Cue(9.2, 11.0, "sugar and eggs."),
        Cue(13.0, 14.0, "Thanks!"),
    ]


def fake_chat(answers):
    """A chat function that records requests and answers by line number."""
    calls = []

    def chat(messages, schema=None, **kw):
        calls.append((messages, schema))
        ids = schema["required"]
        return json.dumps({i: answers(int(i)) for i in ids})

    chat.calls = calls
    return chat


def test_lines_join_unfinished_sentences_and_get_slots():
    lines = tl.build_lines(cues(), duration=15.0)
    assert [ln.cues for ln in lines] == [[0, 1], [2, 3], [4]]
    assert lines[0].source == "Good morning everyone, and welcome back to our little show."
    assert lines[0].slot == pytest.approx(7.0 - 0.15 - 1.0)  # until the next line starts
    assert lines[-1].slot == pytest.approx(2.0)  # until the video ends


def test_long_unfinished_run_is_still_split():
    many = [Cue(i * 3.0, i * 3.0 + 2.9, "and then more words") for i in range(8)]
    lines = tl.build_lines(many, max_seconds=12.0)
    assert all(ln.end - ln.start <= 12.0 for ln in lines) and len(lines) > 1


def test_schema_forces_every_line():
    s = tl.schema([3, 4])
    assert s["required"] == ["3", "4"] and s["additionalProperties"] is False
    assert tl.parse_answer('noise {"3": " مرحبا  بكم ", "4": "«نعم»"} noise', [3, 4]) == {3: "مرحبا بكم", 4: "نعم"}
    with pytest.raises(ValueError):
        tl.parse_answer('{"3": "x"}', [3, 4])
    with pytest.raises(ValueError):
        tl.parse_answer('{"3": "x", "4": "  "}', [3, 4])


def test_translate_in_batches_with_context_and_limits():
    lines = tl.build_lines([Cue(i * 2.0, i * 2.0 + 1.5, f"Line {i}.") for i in range(25)])
    chat = fake_chat(lambda i: f"سطر {i}")
    tl.translate_lines(lines, chat, src="en", tgt="ar", rate=11.0, dub=True, batch_size=10)
    assert [ln.text for ln in lines] == [f"سطر {i}" for i in range(25)]
    assert len(chat.calls) == 3
    second = chat.calls[1][0][1]["content"]
    assert "[9] Line 9." in second and "=> سطر 9" in second  # earlier lines as context
    assert "[20] Line 20." in second.split("context only, do not translate")[1]  # next lines
    assert "(max " in second and "Arabic" in chat.calls[0][0][0]["content"]


def test_bad_answers_are_retried_then_fail():
    replies = iter(['{"0": ""}', '{"0": "ok"}'])
    lines = tl.build_lines([Cue(0, 1, "Hi.")])
    tl.translate_lines(lines, lambda m, schema=None, **kw: next(replies), src="en", tgt="fr", rate=14, dub=False)
    assert lines[0].text == "ok"
    with pytest.raises(RuntimeError):
        tl.translate_lines(lines, lambda m, schema=None, **kw: "{}", src="en", tgt="fr", rate=14, dub=False)


def test_shorten_keeps_only_shorter_and_too_long_uses_rate():
    lines = tl.build_lines([Cue(0, 1.0, "Hello there."), Cue(1.2, 2.2, "Bye.")])
    lines[0].text, lines[1].text = "x" * 40, "short"
    assert tl.too_long(lines, rate=10.0) == [lines[0]]  # 4 s of speech in a 1.05 s slot
    chat = fake_chat(lambda i: "y" * 10 if i == 0 else "much longer than before")
    tl.shorten_lines(lines, chat, tgt="ar", limits={0: 12, 1: 12})
    assert lines[0].text == "y" * 10 and lines[1].text == "short"
    assert "max 12 characters" in chat.calls[0][0][1]["content"]


def test_spread_keeps_times_and_shares_words():
    cs = cues()
    lines = tl.build_lines(cs)
    for ln, t in zip(lines, ["صباح الخير جميعًا ومرحبًا بكم من جديد في برنامجنا الصغير",
                             "أولًا نحتاج إلى الدقيق والسكر والبيض", "شكرًا"]):
        ln.text = t
    out = tl.spread(lines, cs, "ar")
    assert [(c.start, c.end) for c in out] == [(c.start, c.end) for c in cs]
    assert all(c.text.strip() for c in out)
    assert " ".join(c.text.replace("\n", " ") for c in out[:2]) == lines[0].text


def test_spread_with_fewer_words_than_cues_merges_them():
    cs = cues()[:2]
    lines = tl.build_lines(cs)
    lines[0].text = "مرحبًا"
    out = tl.spread(lines, cs, "ar")
    assert len(out) == 1 and (out[0].start, out[0].end) == (1.0, 6.0)


def test_wrap_text_balances_two_lines():
    w = wrap_text("اليوم سنصنع كعكة ليمون بسيطة يمكن لأي شخص خبزها في المنزل")
    top, bottom = w.split("\n")
    assert len(top) <= 42 and len(bottom) <= 42
    assert wrap_text("short") == "short"
    assert "\n" in wrap_text("word " * 30)  # too long for two lines: still split evenly


def test_wav_roundtrip_float_and_pcm16(tmp_path):
    x = (np.sin(np.linspace(0, 20, 1000)) * 0.5).astype(np.float32)
    p = audio.write_wav(tmp_path / "a.wav", np.stack([x, -x], 1), 44100)
    back, sr = audio.read_wav(p)
    assert sr == 44100 and back.shape == (1000, 2) and np.allclose(back[:, 0], x)
    import wave

    with wave.open(str(tmp_path / "b.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    mono, sr = audio.read_wav(tmp_path / "b.wav")
    assert sr == 24000 and mono.shape == (1000, 1) and np.allclose(mono[:, 0], x, atol=1e-4)


def test_tempo_limits():
    assert audio.tempo_for(2.0, 3.0) == 1.0
    assert audio.tempo_for(3.3, 3.0) == pytest.approx(1.1)
    assert audio.tempo_for(9.0, 3.0) == audio.MAX_TEMPO
    assert audio.needs_shorter(9.0, 3.0) and not audio.needs_shorter(3.9, 3.0)
    assert audio.tempo_filter(1.0) == "anull"


def test_place_gain_and_mix():
    sr = 1000
    track = audio.place([(0.5, np.ones(100, np.float32)), (0.55, np.ones(100, np.float32))], 2000, sr)
    assert track[499] == 0 and track[520] == 1 and track[560] == 2 and track[620] == 1 and track[700] == 0
    loud, quiet = np.full(500, 0.5, np.float32), np.full(500, 0.05, np.float32)
    assert audio.gain_to_match(loud, quiet, sr, 0.1) == pytest.approx(0.25)  # clamped
    assert audio.gain_to_match(quiet, loud, sr, 0.1) == pytest.approx(4.0)
    assert audio.gain_to_match(loud, np.zeros(500, np.float32), sr, 0.25) == pytest.approx(0.5)  # fallback level
    m = audio.mix(np.zeros((10, 2), np.float32), np.ones(12, np.float32))
    assert m.shape == (12, 2) and m[11, 1] == 1


def test_speakers_are_grouped_by_voice():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=16), rng.normal(size=16)

    def near(v):
        e = v + rng.normal(scale=0.15, size=16)
        return e / np.linalg.norm(e)

    embeds = np.stack([near(a), near(b), near(a), near(b), near(a), near(b)])
    labels = voice_worker.cluster(embeds, [3, 3, 2, 2, 1, 0.5], threshold=0.75)
    assert labels == [0, 1, 0, 1, 0, 1]
    # a very short line never starts a new speaker
    assert voice_worker.cluster(np.stack([near(a), near(b)]), [3, 0.8], threshold=0.75) == [0, 0]


def test_trim_cuts_silence():
    sr = 1000
    wav = np.concatenate([np.zeros(500), np.full(300, 0.3), np.zeros(500)]).astype(np.float32)
    out = voice_worker._trim(wav, sr)
    assert 300 <= len(out) <= 400


def info(path="", audio_codecs=("aac",), subs=()):
    streams = [StreamInfo(index=0, kind="video", codec="h264", width=640, height=360)]
    for c in audio_codecs:
        streams.append(StreamInfo(index=len(streams), kind="audio", codec=c))
    for c in subs:
        streams.append(StreamInfo(index=len(streams), kind="subtitle", codec=c))
    return MediaInfo(path=path, format_name="mp4", duration=60.0, bit_rate=None, streams=streams)


def test_estimate_parts_and_learning(tmp_path):
    from vidforge.core.gpu import GpuInfo
    from vidforge.features.subtitles.segment import Word
    from vidforge.features.subtitles.whisper import Transcript

    gpu = GpuInfo(index=0, name="RTX", vendor="nvidia")
    tr = Transcript("en", [Word(" hi", i, i + 0.5) for i in range(60)], "hi " * 60)
    subs = estimate.estimate(info(), level="best", dub=False, add_to_video=False, target="ar", gpu=gpu, transcript=tr)
    dub = estimate.estimate(info(), level="best", dub=True, add_to_video=False, target="ar", gpu=gpu, transcript=tr)
    quick = estimate.estimate(info(), level="quick", dub=False, add_to_video=False, target="ar", gpu=gpu, transcript=tr)
    fresh = estimate.estimate(info(), level="best", dub=False, add_to_video=False, target="ar", gpu=gpu, transcript=None)
    assert dub.seconds > subs.seconds > quick.seconds
    assert fresh.seconds > subs.seconds  # must write down the speech first
    assert dub.download_mb > subs.download_mb > 0  # translator, voice models, llama.cpp
    assert not dub.learned
    estimate.learn("dub/tts/gpu", 3.0)
    again = estimate.estimate(info(), level="best", dub=True, add_to_video=False, target="ar", gpu=gpu, transcript=tr)
    assert again.learned and again.seconds < dub.seconds
    estimate.learn("dub/cps/ar", 9.0)
    assert estimate.learned_cps(estimate.up.load_stats()) == {"ar": 9.0}


def test_models_are_pinned_and_share_a_folder():
    from vidforge.features.dub import models

    for key in list(models.TRANSLATORS.values()) + [models.DEMUCS_KEY]:
        assert len(model_registry.MODELS[key].sha256) == 64
    paths = {model_registry.model_path(f"chatterbox-{f}").parent for f, _, _ in models.VOICE_FILES}
    assert len(paths) == 1 and paths.pop().name == models.VOICE_FOLDER
    assert {"ar", "en", "fr"} <= models.VOICE_LANGUAGES


def test_friendly_errors(monkeypatch, tmp_path):
    from vidforge.gui import i18n

    monkeypatch.setattr(addon, "is_installed", lambda: False)
    with pytest.raises(FriendlyError) as e:
        service._check(service.DubRequest(tmp_path / "a.mp4", target="ur", dub=True))
    assert e.value.code == "voice_language" and e.value.params["lang"] == "اردو"
    with pytest.raises(FriendlyError) as e:
        service._check(service.DubRequest(tmp_path / "a.mp4", target="ar", dub=True))
    assert e.value.code == "voice_addon"
    service._check(service.DubRequest(tmp_path / "a.mp4", target="ur", dub=False))  # subtitles only: fine
    with pytest.raises(FriendlyError) as e:
        service._check(service.DubRequest(tmp_path / "a.mp4", target="xx", dub=False))
    assert e.value.code == "bad_language"
    for code in ("voice_language", "voice_addon", "same_language", "bad_language", "no_audio", "no_speech",
                 "file_missing"):
        assert f"err.{code}" in i18n.S


def test_mux_maps_dub_first_and_keeps_original(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(service, "run_ffmpeg", lambda args, **kw: seen.setdefault("args", [str(a) for a in args]))
    service.mux_dub(tmp_path / "in.mp4", info(audio_codecs=("aac", "pcm_s16le"), subs=("subrip",)), tmp_path / "d.wav",
                    tmp_path / "s.srt", tmp_path / "out.mp4", target="ar", keep_original=True,
                    cancel=None, on_progress=None)
    a = " ".join(seen["args"])
    assert "-map 0:v -map 1:a -map 0:1 -map 0:2 -map 0:3 -map 2:0" in a
    assert "-c:a:0 aac" in a and "-metadata:s:a:0 language=ara" in a and "-disposition:a:0 default" in a
    assert "-c:a:2 aac" in a  # PCM can't go into MP4
    assert "-disposition:a:1 0" in a and "-c:s:1 mov_text" in a and "-metadata:s:s:1 language=ara" in a
    seen.clear()
    service.mux_dub(tmp_path / "in.mkv", info(), tmp_path / "d.wav", tmp_path / "s.srt", tmp_path / "out.mkv",
                    target="fr", keep_original=False, cancel=None, on_progress=None)
    a = " ".join(seen["args"])
    assert "-map 0:1" not in a and "-c:s:0 srt" in a and "language=fra" in a and "-map 0:t?" in a


def test_llama_command_uses_key_and_no_web_ui(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.tools, "exe_path", lambda key, exe=None: tmp_path / "llama-server.exe")
    srv = llm.LlamaServer(tmp_path / "m.gguf", cancel=None, on_log=print, gpu=False)
    srv.port = 1234
    cmd = srv.command()
    assert "--api-key" in cmd and srv.key in cmd and "--no-webui" in cmd and cmd[cmd.index("-ngl") + 1] == "0"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_spans_share_progress():
    reports = []

    class Ctx:
        def report(self, p, m, phase=None):
            reports.append(p)

    s = service.Spans(Ctx(), {"a": 1.0, "b": 0.0, "c": 3.0})
    s.report("a", 1.0, "x")
    s.report("c", 0.5, "y")
    s.report("b", None, "z")
    assert reports == [0.25, pytest.approx(0.625), None]


def test_addon_paths_and_status(tmp_path):
    assert addon.status() is None and not addon.is_installed()
    addon.python().parent.mkdir(parents=True)
    addon.python().write_text("")
    (addon.root() / "vidforge-addon.json").write_text(json.dumps({"version": addon.VERSION, "device": "cuda"}))
    assert addon.is_installed()
    (addon.root() / "vidforge-addon.json").write_text(json.dumps({"version": 0, "device": "cuda"}))
    assert not addon.is_installed()  # an old install is redone
    assert addon.requirements_file().is_file()


def test_cli_parses_dub():
    from vidforge.cli import build_parser

    a = build_parser().parse_args(["dub", "x.mp4", "--to", "fr", "--subtitles-only", "--level", "quick"])
    assert a.to == "fr" and a.subtitles_only and a.level == "quick" and a.func.__name__ == "cmd_dub"
