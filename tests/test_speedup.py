"""NVIDIA speed-up: partial zip downloads, card detection, fallback to DirectML, update check."""

import hashlib
import http.server
import threading
import zipfile

import numpy as np
import pytest

from vidforge.core import update
from vidforge.core.download import ChecksumError
from vidforge.core.gpu import GpuInfo
from vidforge.core.zipfetch import fetch_members
from vidforge.features.upscale import estimate, quality, speedup


class RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""

    def log_message(self, *a):
        pass

    def _head(self, code, length, extra=None):
        self.send_response(code)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_HEAD(self):
        self._head(200, len(self.data))

    def do_GET(self):
        rng = self.headers.get("Range")
        a, b = (int(x) for x in rng.split("=")[1].split("-"))
        part = self.data[a:b + 1]
        self._head(206, len(part), {"Content-Range": f"bytes {a}-{b}/{len(self.data)}"})
        self.wfile.write(part)


@pytest.fixture
def served_zip(tmp_path):
    files = {"pkg/a.dll": b"A" * 300_000 + bytes(range(256)) * 100, "pkg/b.dll": b"hello", "pkg/big.dll": b"x" * 2_000_000}
    zpath = tmp_path / "w.whl"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items():
            zf.writestr(k, v)
    RangeHandler.data = zpath.read_bytes()
    srv = http.server.HTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/w.whl", files
    srv.shutdown()


def test_fetch_only_some_members(served_zip, tmp_path):
    url, files = served_zip
    want = {k: hashlib.sha256(files[k]).hexdigest() for k in ("pkg/a.dll", "pkg/b.dll")}
    seen = []
    fetch_members(url, want, tmp_path / "out", progress=lambda d, t: seen.append((d, t)))
    for k in want:
        assert (tmp_path / "out" / k).read_bytes() == files[k]
    assert not (tmp_path / "out" / "pkg" / "big.dll").exists()
    assert seen and seen[-1][1] < len(RangeHandler.data)  # far less than the whole archive


def test_fetch_refuses_wrong_checksum(served_zip, tmp_path):
    url, _ = served_zip
    with pytest.raises(ChecksumError):
        fetch_members(url, {"pkg/b.dll": "0" * 64}, tmp_path / "out")
    assert not (tmp_path / "out" / "pkg" / "b.dll").exists()
    with pytest.raises(LookupError):
        fetch_members(url, {"pkg/none.dll": "0" * 64}, tmp_path / "out")


def gpu(cc, vendor="nvidia"):
    return GpuInfo(index=0, name="card", vendor=vendor, compute_cap=cc)


def test_cards_and_downloads():
    assert speedup.supported(gpu((8, 9))) and speedup.supported(gpu((7, 5)))
    assert not speedup.supported(gpu((6, 1))) and not speedup.supported(gpu(None)) and not speedup.supported(None)
    assert not speedup.supported(gpu((8, 9), vendor="amd"))
    assert speedup.resource_for((8, 9)) == "sm89" and speedup.resource_for((8, 7)) == "ptx"
    mb = speedup.download_mb(gpu((8, 9)))  # RTX 4080
    assert 400 < mb < 450 and speedup.download_mb(gpu((6, 1))) == 0
    for sha in list(speedup.LIBS_CORE.values()) + list(speedup.LIBS_RESOURCES.values()):
        assert len(sha) == 64
    assert not speedup.is_installed()


class FakeEngine:
    scale = 4
    provider = "Dml"

    def __init__(self, *a, **kw):
        pass

    def upscale(self, frame):
        return np.repeat(np.repeat(frame, 4, 0), 4, 1)


def test_auto_engine_falls_back_to_directml(monkeypatch, tmp_path):
    class BrokenTrt(FakeEngine):
        provider = "TensorRT"

        def upscale(self, frame):
            raise RuntimeError("engine build failed")

    import vidforge.features.upscale.trt_engine as trt_engine

    monkeypatch.setattr(quality, "EsrganEngine", FakeEngine)
    monkeypatch.setattr(trt_engine, "TrtEsrganEngine", BrokenTrt)
    monkeypatch.setattr(speedup, "is_installed", lambda: True)
    logs = []
    e = quality.AutoEngine("general", tmp_path / "m.onnx", device_id=0, gpu=gpu((8, 9)), on_log=logs.append)
    assert e.provider == "TensorRT"
    out = e.upscale(np.zeros((2, 3, 3), np.uint8))
    assert out.shape == (8, 12, 3) and e.provider == "Dml" and "DirectML" in logs[-1]
    # no add-on, or an AMD card: DirectML straight away
    monkeypatch.setattr(speedup, "is_installed", lambda: False)
    assert quality.AutoEngine("max", tmp_path / "m.onnx", device_id=0, gpu=gpu((8, 9)), on_log=print).provider == "Dml"


def test_speeds_are_kept_apart():
    assert estimate.speed_key("quality", "general", trt=True) == "quality-trt/general"
    assert estimate.speed_key("quality", "general", trt=False) == "quality/general"
    assert estimate.speed_key("fast", "general", trt=True) == "fast"
    assert estimate.DEFAULT_SPEED["quality-trt/general"] > estimate.DEFAULT_SPEED["quality/general"]


def test_version_compare():
    assert update.is_newer("0.2.0", "0.1.0") and update.is_newer("v1.0", "0.9.9") and update.is_newer("0.2.1", "0.2")
    assert not update.is_newer("0.2.0", "0.2.0") and not update.is_newer("0.1.9", "0.2.0") and not update.is_newer("", "0.1")
