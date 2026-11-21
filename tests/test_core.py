import hashlib
import sys
import time
import zipfile

import pytest

from vidforge.core import archive, download, ffmpeg, gpu, jobs, media, models, process, settings, tools


# --- process --------------------------------------------------------------------------

def test_run_process_splits_carriage_returns():
    lines = []
    code = "import sys; sys.stdout.write('a\\rb\\nc'); sys.stdout.flush()"
    process.run_process([sys.executable, "-c", code], on_line=lines.append)
    assert lines == ["a", "b", "c"]


def test_run_process_raises_with_tail():
    code = "import sys; print('boom'); sys.exit(3)"
    with pytest.raises(process.ProcessError) as ei:
        process.run_process([sys.executable, "-c", code])
    assert ei.value.returncode == 3
    assert "boom" in ei.value.tail


def test_run_process_cancel_kills_child():
    token = process.CancelToken()
    start = time.monotonic()
    import threading

    threading.Timer(0.3, token.cancel).start()
    with pytest.raises(process.Cancelled):
        process.run_process([sys.executable, "-c", "import time; time.sleep(30)"], cancel=token)
    assert time.monotonic() - start < 10


# --- jobs -----------------------------------------------------------------------------

def test_job_queue_runs_and_reports():
    q = jobs.JobQueue()
    seen = []
    q.add_listener(lambda j: seen.append(j.state))

    def work(ctx):
        ctx.report(0.5, "half")
        return 42

    job = q.submit("t", work)
    assert q.wait(job, 5)
    assert job.state == jobs.JobState.DONE and job.result == 42 and job.progress == 1.0
    assert jobs.JobState.RUNNING in seen
    q.shutdown()


def test_job_queue_failure_and_cancel():
    q = jobs.JobQueue()
    failed = q.submit("fail", lambda ctx: 1 / 0)

    def slow(ctx):
        while True:
            ctx.cancel.raise_if_cancelled()
            time.sleep(0.01)

    running = q.submit("slow", slow)
    queued = q.submit("never", lambda ctx: None)
    assert q.wait(failed, 5) and failed.state == jobs.JobState.FAILED
    while running.state != jobs.JobState.RUNNING:
        time.sleep(0.01)
    q.cancel(queued.id)
    q.cancel(running.id)
    assert q.wait(running, 5) and running.state == jobs.JobState.CANCELLED
    assert queued.state == jobs.JobState.CANCELLED
    q.shutdown()


# --- settings -------------------------------------------------------------------------

def test_settings_roundtrip_and_unknown_keys(tmp_path):
    p = tmp_path / "s.json"
    s = settings.Settings(upscale_mode="quality", gpu_index=1)
    s.save(p)
    p.write_text(p.read_text().replace("{", '{"obsolete": 1,', 1))
    loaded = settings.Settings.load(p)
    assert loaded.upscale_mode == "quality" and loaded.gpu_index == 1
    assert settings.Settings.load(tmp_path / "missing.json") == settings.Settings()


# --- download / archive ---------------------------------------------------------------

def test_download_verifies_sha256(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 3_000_000)
    good = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "out" / "file.bin"
    calls = []
    download.download(src.as_uri(), dest, sha256=good, progress=lambda d, t: calls.append(d))
    assert dest.read_bytes() == src.read_bytes() and calls[-1] == 3_000_000

    bad = tmp_path / "out" / "bad.bin"
    with pytest.raises(download.ChecksumError):
        download.download(src.as_uri(), bad, sha256="0" * 64)
    assert not bad.exists() and not bad.with_name("bad.bin.part").exists()


def test_extract_rejects_path_traversal(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", "no")
    with pytest.raises(archive.UnsafeArchiveError):
        archive.extract(z, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_parse_7z_listing():
    text = ("Path = C:\\dl\\NVEncC.7z\nType = 7z\n\n----------\n"
            "Path = NVEncC\\x64\nFolder = +\n\nPath = NVEncC\\x64\\NVEncC64.exe\nSize = 1\n")
    assert archive.parse_7z_listing(text) == ["NVEncC\\x64", "NVEncC\\x64\\NVEncC64.exe"]
    assert archive.parse_7z_listing("no separator") == []


def test_fetch_installs_bare_exe(monkeypatch, tmp_path):
    exe = tmp_path / "7zr.exe"
    exe.write_bytes(b"MZ")
    digest = hashlib.sha256(b"MZ").hexdigest()
    monkeypatch.setattr(tools, "resolve", lambda k: [tools.ReleaseAsset("26", "7zr.exe", exe.as_uri(), 2, digest)])
    installed = tools.fetch("sevenzip")
    assert installed.name == "7zr.exe" and installed.read_bytes() == b"MZ"


def test_extract_zip(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("pkg/bin/tool.exe", "x")
    archive.extract(z, tmp_path / "out")
    assert (tmp_path / "out" / "pkg" / "bin" / "tool.exe").exists()


# --- tools ----------------------------------------------------------------------------

RELEASE = {
    "tag_name": "latest",
    "assets": [
        {"name": "ffmpeg-n7.1-latest-win64-gpl-7.1.zip", "size": 10, "browser_download_url": "u71",
         "digest": "sha256:" + "a" * 64},
        {"name": "ffmpeg-n8.0-latest-win64-gpl-8.0.zip", "size": 20, "browser_download_url": "u80",
         "digest": "sha256:" + "b" * 64},
        {"name": "ffmpeg-n8.0-latest-win64-lgpl-8.0.zip", "size": 20, "browser_download_url": "x"},
        {"name": "ffmpeg-master-latest-win64-gpl.zip", "size": 30, "browser_download_url": "m"},
    ],
}


def test_pick_asset_matches_pattern_and_reads_digest():
    a = tools.pick_asset(RELEASE, tools.TOOLS["ffmpeg"].source.asset_pattern)
    assert a.name == "ffmpeg-n8.0-latest-win64-gpl-8.0.zip" and a.sha256 == "b" * 64


def test_pick_asset_without_digest_and_no_match():
    rel = {"tag_name": "7.70", "assets": [{"name": "NVEncC_7.70_x64.7z", "size": 1, "browser_download_url": "n"}]}
    a = tools.pick_asset(rel, tools.TOOLS["nvencc"].source.asset_pattern)
    assert a.sha256 is None
    with pytest.raises(LookupError):
        tools.pick_asset(rel, tools.TOOLS["ffmpeg"].source.asset_pattern)


def test_resolve_scans_prereleases_for_assets(monkeypatch):
    releases = [
        {"tag_name": "v1.0.0", "assets": []},  # tagged release without binaries
        {"tag_name": "b12613", "assets": [
            {"name": "llama-b12613-bin-win-cuda-12.4-x64.zip", "size": 5, "browser_download_url": "m",
             "digest": "sha256:" + "c" * 64},
            {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "size": 3, "browser_download_url": "r",
             "digest": "sha256:" + "d" * 64},
        ]},
    ]
    monkeypatch.setattr(tools, "_github_json", lambda url: releases)
    main, runtime = tools.resolve("llama")
    assert main.tag == "b12613" and main.name.startswith("llama-") and runtime.name.startswith("cudart-")


def test_whisper_pattern_accepts_both_namings():
    rx = tools.TOOLS["whisper"].source.asset_pattern
    import re
    assert re.search(rx, "whisper-bin-win-cuda-12.4.0-x64.zip")
    assert re.search(rx, "whisper-cublas-12.4.0-bin-x64.zip")
    assert not re.search(rx, "whisper-bin-win-cuda-13.5-arm64.zip")


def test_fetch_refuses_unverified(monkeypatch):
    monkeypatch.setattr(tools, "resolve", lambda k: [tools.ReleaseAsset("t", "a.7z", "u", 1, None)])
    with pytest.raises(RuntimeError, match="unverified"):
        tools.fetch("nvencc")


def test_fetch_installs_from_archive(monkeypatch, tmp_path):
    z = tmp_path / "pkg.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ffmpeg-x/bin/ffmpeg.exe", "x")
        zf.writestr("ffmpeg-x/bin/ffprobe.exe", "x")
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    monkeypatch.setattr(tools, "resolve", lambda k: [tools.ReleaseAsset("t", "pkg.zip", z.as_uri(), 1, digest)])
    exe = tools.fetch("ffmpeg")
    assert exe.name == "ffmpeg.exe"
    path, origin = tools.locate("ffmpeg")
    assert origin == "managed" and path == exe
    assert tools.exe_path("ffmpeg", "ffprobe.exe").name == "ffprobe.exe"


# --- models ---------------------------------------------------------------------------

def test_model_spec_requires_hash():
    with pytest.raises(ValueError):
        models.ModelSpec("k", "K", "u", "abc", "f", 1, "MIT", "upscale")


# --- ffmpeg progress ------------------------------------------------------------------

def test_progress_parser():
    seen = []
    p = ffmpeg.ProgressParser(10.0, lambda f, s: seen.append((f, s)))
    for line in ["frame=10", "out_time_us=2500000", "speed=2.0x", "progress=continue",
                 "out_time_us=10000000", "progress=end"]:
        assert p.feed(line)
    assert not p.feed("Input #0, mov,mp4 from 'a.mp4':")
    assert seen == [(0.25, "2.0x"), (1.0, "2.0x")]


def test_progress_parser_unknown_duration():
    seen = []
    p = ffmpeg.ProgressParser(None, lambda f, s: seen.append(f))
    p.feed("out_time_us=5")
    p.feed("progress=continue")
    assert seen == [None]


# --- media ----------------------------------------------------------------------------

FFPROBE = {
    "format": {"filename": "a.mkv", "format_name": "matroska,webm", "duration": "4.000", "bit_rate": "5000"},
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 640, "height": 360,
         "avg_frame_rate": "24000/1001", "pix_fmt": "yuv420p", "disposition": {"default": 1}},
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000",
         "tags": {"LANGUAGE": "eng"}},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "spa"}},
        {"index": 3, "codec_type": "video", "codec_name": "mjpeg", "avg_frame_rate": "0/0",
         "disposition": {"attached_pic": 1}},
    ],
}


def test_parse_ffprobe():
    info = media.parse_ffprobe(FFPROBE)
    assert info.duration == 4.0 and info.path == "a.mkv"
    v = info.video
    assert v.index == 0 and v.width == 640 and abs(v.fps - 23.976) < 0.001 and v.default
    assert info.audio[0].language == "eng" and info.audio[0].sample_rate == 48000
    assert info.subtitles[0].language == "spa"
    assert info.streams[3].attached_pic and info.streams[3].fps is None


# --- gpu ------------------------------------------------------------------------------

def test_parse_nvidia_smi_and_rtx_rules():
    text = ("0, NVIDIA GeForce RTX 2060 SUPER, 621.14, 8192, 7.5\n"
            "1, NVIDIA GeForce GTX 1660 Ti, 621.14, 6144, 7.5\n"
            "2, NVIDIA GeForce GTX 1080, 621.14, 8192, 6.1\n")
    g = gpu.parse_nvidia_smi(text)
    assert [x.vram_mb for x in g] == [8192, 6144, 8192]
    assert g[0].compute_cap == (7, 5) and g[0].vsr_capable
    assert not g[1].vsr_capable  # Turing without Tensor cores
    assert not g[2].vsr_capable
