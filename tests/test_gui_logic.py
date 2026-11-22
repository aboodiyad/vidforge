"""Language, formatting, time-remaining and estimate logic (no windows needed)."""

import string
import time

import pytest

from vidforge.core import jobs, media
from vidforge.features.upscale import estimate as est
from vidforge.gui import i18n


@pytest.fixture(autouse=True)
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_every_text_has_both_languages_with_same_placeholders():
    for key, (en, ar) in i18n.S.items():
        assert en.strip() and ar.strip(), key
        assert _fields(en) == _fields(ar), key


def test_language_resolution():
    assert i18n.resolve("system", "ar_JO") == "ar"
    assert i18n.resolve("system", "en_US") == "en"
    assert i18n.resolve("en", "ar_EG") == "en"
    i18n.set_language("ar")
    assert i18n.is_rtl() and i18n.tr("up.start") == "ابدأ"


def test_arabic_plurals():
    i18n.set_language("ar")
    assert i18n.plural("min", 1) == "دقيقة واحدة"
    assert i18n.plural("min", 2) == "دقيقتان"
    assert i18n.plural("min", 5) == "5 دقائق"
    assert i18n.plural("min", 25) == "25 دقيقة"
    assert i18n.videos(3) == "3 فيديوهات"


def test_durations_and_sizes():
    assert i18n.duration(20) == "less than a minute"
    assert i18n.about_time(20) == "less than a minute"
    assert i18n.about_time(190) == "about 3 min"
    assert i18n.duration(3 * 3600 + 120) == "3 hours 2 min"
    assert i18n.duration(130, precise=True) == "2 min 10 sec"
    assert i18n.duration(9, precise=True) == "9 sec"
    assert i18n.size(27_400_000) == "27 MB"
    assert i18n.size(1_900_000) == "1.9 MB"
    assert i18n.size(2_000_000_000) == "2 GB"
    assert est.round_nice(13_612_345) == 14_000_000


def test_job_eta_from_progress_rate():
    job = jobs.Job(1, "x", fn=lambda c: None, state=jobs.JobState.RUNNING)
    now = time.monotonic()
    assert job.eta(now) is None  # nothing measured yet
    for i in range(11):  # 1% per second for 10 s
        job._samples.append((now - 10 + i, 0.20 + i * 0.01))
    eta = job.eta(now)
    assert eta == pytest.approx(70, abs=1)  # 70% left at 1%/s


def test_report_phase_change_resets_rate():
    q = jobs.JobQueue()
    seen = {}

    def work(ctx):
        ctx.report(0.5, "x", phase="downloading")
        ctx.report(0.1, "y", phase="working")
        seen["samples"] = list(ctx._job._samples)
        seen["phase"] = ctx._job.phase

    j = q.submit("t", work)
    q.wait(j, 5)
    q.shutdown()
    assert seen["phase"] == "working" and len(seen["samples"]) == 1
    assert j.started is not None and j.elapsed() >= 0


def _info(bitrate=2_000_000, w=1280, h=720, dur=60.0, codec="h264"):
    return media.parse_ffprobe({
        "format": {"filename": "x.mp4", "format_name": "mp4", "duration": str(dur)},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": codec, "width": w, "height": h,
             "avg_frame_rate": "30/1", "pix_fmt": "yuv420p", "bit_rate": str(bitrate)},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "bit_rate": "128000"},
        ],
    })


def test_estimate_scales_sensibly():
    info = _info()
    base = est.estimate(info, 3840, 2160, mode="fast", model="general", codec="hevc", quality="balanced",
                        container="mp4", stats={})
    assert base.bytes_low < base.bytes < base.bytes_high
    assert base.seconds_low < base.seconds < base.seconds_high
    small = est.estimate(info, 3840, 2160, mode="fast", model="general", codec="hevc", quality="small",
                         container="mp4", stats={})
    h264 = est.estimate(info, 3840, 2160, mode="fast", model="general", codec="h264", quality="balanced",
                        container="mp4", stats={})
    assert small.bytes < base.bytes < h264.bytes
    q = est.estimate(info, 3840, 2160, mode="quality", model="general", codec="hevc", quality="balanced",
                     container="mp4", stats={})
    assert q.seconds > base.seconds * 5  # Best quality is much slower
    audio = 128_000 * 60 / 8
    assert base.bytes > audio


def test_estimate_learns_from_finished_jobs(isolated_home):
    info = _info()
    before = est.estimate(info, 3840, 2160, mode="fast", model="general", codec="hevc", quality="balanced",
                          container="mp4", stats={})
    est.learn(mode="fast", model="general", codec="hevc", quality="balanced", frames=1800,
              in_pixels=1280 * 720, out_pixels=3840 * 2160, work_seconds=1800 / 20,  # 20 fps here
              predicted_video_bytes=100, actual_video_bytes=200)
    after = est.estimate(info, 3840, 2160, mode="fast", model="general", codec="hevc", quality="balanced",
                         container="mp4")
    assert after.learned
    assert after.seconds > before.seconds  # this PC is slower than the default
    assert after.bytes > before.bytes * 1.5  # files came out twice as big
    assert (after.bytes_high - after.bytes_low) / after.bytes < (before.bytes_high - before.bytes_low) / before.bytes
