import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vidforge.core import encoders, framepipe, media, sizing, streams
from vidforge.core.gpu import GpuInfo
from vidforge.features.upscale import esrgan_convert as ec
from vidforge.features.upscale import nvencc_vsr as nv
from vidforge.features.upscale import service
from vidforge.features.upscale.esrgan_onnx import merge_tiles, plan_tiles, tile_starts

RTX2060S = GpuInfo(0, "NVIDIA GeForce RTX 2060 SUPER", "nvidia", "621.14", 8192, (7, 5))
RTX4070 = GpuInfo(0, "NVIDIA GeForce RTX 4070", "nvidia", "621.14", 12288, (8, 9))


def make_info(**video) -> media.MediaInfo:
    v = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720,
         "avg_frame_rate": "24000/1001", "pix_fmt": "yuv420p"}
    v.update(video)
    return media.parse_ffprobe({
        "format": {"filename": "in.mkv", "format_name": "matroska", "duration": "10"},
        "streams": [
            v,
            {"index": 1, "codec_type": "audio", "codec_name": "flac", "channels": 2},
            {"index": 2, "codec_type": "audio", "codec_name": "pcm_s16le", "channels": 2},
            {"index": 3, "codec_type": "subtitle", "codec_name": "subrip"},
            {"index": 4, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        ],
    })


# --- sizing ---------------------------------------------------------------------------

@pytest.mark.parametrize("disp,preset,out", [
    ((640, 360), "2160p", (3840, 2160)),
    ((720, 480), "1080p", (1620, 1080)),        # 3:2 keeps its aspect
    ((1920, 804), "2160p", (3840, 1608)),       # scope: width-limited
    ((480, 720), "1080p", (1080, 1620)),        # portrait box turns around
    ((655, 480), "2x", (1310, 960)),
])
def test_target_size(disp, preset, out):
    assert sizing.target_size(*disp, preset) == out


def test_target_size_refuses_downscale():
    with pytest.raises(sizing.SizeError):
        sizing.target_size(3840, 2160, "1080p")


def test_custom_size():
    assert sizing.parse_custom("2560x1080") == (2560, 1080)
    assert all(v % 2 == 0 for v in sizing.parse_custom("1921x1083"))
    with pytest.raises(sizing.SizeError):
        sizing.parse_custom("big")


# --- media extras ---------------------------------------------------------------------

def test_media_display_size_rotation_and_sar():
    info = make_info(width=720, height=480, sample_aspect_ratio="10:11", field_order="tt",
                     side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}])
    v = info.video
    assert v.sar == (10, 11) and v.interlaced and v.rotation == -90
    assert v.display_size == (480, 655)
    assert info.frame_count() == 240
    assert v.fps_str == "24000/1001"


def test_media_hdr_and_depth():
    v = make_info(pix_fmt="yuv420p10le", color_transfer="smpte2084").video
    assert v.hdr and v.bit_depth == 10


# --- streams --------------------------------------------------------------------------

def test_stream_plans_mp4_vs_mkv():
    info = make_info()
    a = streams.plan_audio(info, "mp4")
    assert [p.action for p in a] == ["copy", "transcode"] and a[1].codec == "aac"
    s = streams.plan_subtitles(info, "mp4")
    assert [p.action for p in s] == ["transcode", "drop"]
    assert all(p.action == "copy" for p in streams.plan_subtitles(info, "mkv"))


# --- fast mode command ----------------------------------------------------------------

def test_fast_command_mp4():
    info = make_info()
    cmd = nv.build_command(Path("nv.exe"), Path("in.mkv"), Path("out.mp4"), info,
                           nv.FastOptions(3840, 2160), RTX2060S)
    s = " ".join(cmd)
    assert "--avhw" in cmd and "algo=ngx-vsr,vsr-quality=4" in cmd
    assert "--output-res 3840x2160" in s and "--sar 1:1" in s
    assert "--audio-copy 1" in s and "--audio-codec 2?aac" in s
    assert "--sub-copy 1 --sub-codec mov_text" in s
    assert "--perf-monitor" not in s  # it suppresses NVEncC's live progress line


def test_fast_command_rotated_interlaced_and_sw_decode():
    info = make_info(codec_name="h264", pix_fmt="yuv420p10le", field_order="tt",
                     side_data_list=[{"rotation": 90}])
    cmd = nv.build_command(Path("nv.exe"), Path("in.mkv"), Path("out.mkv"), info,
                           nv.FastOptions(1080, 1920), RTX2060S)
    assert "--avsw" in cmd  # NVDEC has no H.264 10-bit
    assert "--vpp-yadif" in cmd
    i = cmd.index("--output-res")
    assert cmd[i + 1] == "1920x1080"  # stored orientation for rotated video
    assert "--sub-copy" in cmd and "1,2" in cmd  # mkv keeps both subtitle tracks


def test_fast_command_av1_rules():
    info = make_info(codec_name="av1")
    with pytest.raises(ValueError):
        nv.build_command(Path("nv"), Path("a"), Path("b.mp4"), info, nv.FastOptions(3840, 2160, codec="av1"), RTX2060S)
    assert "--avsw" in nv.build_command(Path("nv"), Path("a"), Path("b.mp4"), info, nv.FastOptions(3840, 2160), RTX2060S)
    cmd = nv.build_command(Path("nv"), Path("a"), Path("b.mp4"), info, nv.FastOptions(3840, 2160, codec="av1"), RTX4070)
    assert "--avhw" in cmd and "av1" in cmd


def test_parse_nvencc_progress():
    line = "[52.7%] 68 frames: 35.38 fps, 4198 kbps, remain 0:00:01, GPU 53%, VE 100%, VD 2%, est out size 2.2MB"
    assert nv.parse_progress(line) == (68, 35.38)
    assert nv.parse_progress("encoded 600 frames, 38.42 fps, 4137.31 kbps, 9.86 MB") is None


# --- mode selection -------------------------------------------------------------------

def test_resolve_mode(monkeypatch):
    monkeypatch.setattr(service.tools, "locate", lambda k: (Path("nv.exe"), "managed"))
    gtx = GpuInfo(0, "NVIDIA GeForce GTX 1660", "nvidia", "1", 6144, (7, 5))
    assert service.resolve_mode("auto", RTX2060S) == "fast"
    assert service.resolve_mode("auto", gtx) == "quality"
    assert service.resolve_mode("auto", None) == "quality"
    with pytest.raises(ValueError):
        service.resolve_mode("fast", gtx)
    monkeypatch.setattr(service.tools, "locate", lambda k: (None, "missing"))
    assert service.resolve_mode("auto", RTX2060S) == "quality"


def test_unique_path(tmp_path):
    p = tmp_path / "a.mp4"
    assert service.unique_path(p) == p
    p.write_text("x")
    assert service.unique_path(p).name == "a (1).mp4"


# --- encoders -------------------------------------------------------------------------

def test_encoder_falls_back_to_software(monkeypatch):
    monkeypatch.setattr(encoders, "encoder_works", lambda name, fmt: False)
    enc = encoders.choose("hevc", "balanced", "nvidia")
    assert enc.name == "libx265" and not enc.hardware
    monkeypatch.setattr(encoders, "encoder_works", lambda name, fmt: True)
    enc = encoders.choose("hevc", "high", "nvidia")
    assert enc.name == "hevc_nvenc" and "-cq" in enc.args and enc.pix_fmt == "p010le"


def test_source_matrix():
    assert framepipe.source_matrix(make_info().video) == "bt709"
    assert framepipe.source_matrix(make_info(width=720, height=480).video) == "bt601"
    assert framepipe.source_matrix(make_info(color_space="bt709", height=480).video) == "bt709"


# --- tiling ---------------------------------------------------------------------------

def test_tile_starts_cover_and_clamp():
    assert tile_starts(500, 512, 16) == [0]
    s = tile_starts(1920, 512, 16)
    assert s[0] == 0 and s[-1] == 1920 - 512 and all(b - a <= 512 - 16 for a, b in zip(s, s[1:]))


def test_merge_tiles_identity():
    """With an identity 'model' (scale 1), merging overlapping tiles must give back the frame."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)
    plan = plan_tiles(70, 90, 32, 8)
    tiles = np.stack([frame[y:y + plan.tile_h, x:x + plan.tile_w] for y in plan.ys for x in plan.xs])
    out = merge_tiles(plan, tiles, 1)
    assert np.abs(out.astype(int) - frame).max() <= 1


# --- checkpoint loader / converter ----------------------------------------------------

def _fake_checkpoint(path: Path, tensors: dict[str, np.ndarray], monkeypatch) -> None:
    """Write a .pth in PyTorch's zip layout without needing PyTorch: stand-in `torch` modules
    make the pickle reference torch.FloatStorage and torch._utils._rebuild_tensor_v2."""
    import collections
    import sys
    import types

    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*a):
        raise AssertionError("never called while pickling")

    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    torch.FloatStorage, utils._rebuild_tensor_v2 = FloatStorage, _rebuild_tensor_v2
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)

    class Tensor:
        def __init__(self, key, arr):
            self.key, self.arr = key, arr

        def __reduce__(self):
            storage = ("storage", FloatStorage, self.key, "cpu", self.arr.size)
            strides = tuple(s // 4 for s in self.arr.strides)
            return (_rebuild_tensor_v2, (storage, 0, self.arr.shape, strides, False, collections.OrderedDict()))

    class P(pickle.Pickler):
        def persistent_id(self, obj):
            return obj if isinstance(obj, tuple) and obj and obj[0] == "storage" else None

    arrays = {k: np.ascontiguousarray(v, np.float32) for k, v in tensors.items()}
    buf = io.BytesIO()
    P(buf, protocol=2).dump({"params": collections.OrderedDict(
        (k, Tensor(str(i), v)) for i, (k, v) in enumerate(arrays.items()))})
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for i, v in enumerate(arrays.values()):
            zf.writestr(f"archive/data/{i}", v.tobytes())


def test_tiny_srvgg_checkpoint_converts_and_runs(tmp_path, monkeypatch):
    import onnxruntime as ort

    rng = np.random.default_rng(3)
    t = {
        "body.0.weight": rng.normal(0, .1, (8, 3, 3, 3)), "body.0.bias": np.zeros(8),
        "body.1.weight": np.full(8, 0.1),
        "body.2.weight": rng.normal(0, .1, (8, 8, 3, 3)), "body.2.bias": np.zeros(8),
        "body.3.weight": np.full(8, 0.1),
        "body.4.weight": rng.normal(0, .1, (12, 8, 3, 3)), "body.4.bias": np.zeros(12),
    }
    t = {k: np.asarray(v, np.float32) for k, v in t.items()}
    pth = tmp_path / "tiny.pth"
    _fake_checkpoint(pth, t, monkeypatch)
    sd = ec.load_state_dict(pth)
    assert set(sd) == set(t) and np.allclose(sd["body.2.weight"], t["body.2.weight"])
    arch = ec.detect_arch(sd)
    assert (arch.kind, arch.scale, arch.num_blocks) == ("srvgg", 2, 1)
    out = tmp_path / "tiny.onnx"
    ec.convert(pth, out, fp16=False, uint8_io=True)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": np.full((1, 6, 5, 3), 128, np.uint8)})[0]
    assert y.shape == (1, 12, 10, 3) and y.dtype == np.uint8


def test_loader_rejects_foreign_objects(tmp_path):
    payload = pickle.dumps({"params": print})  # builtins.print is not on the allow-list
    pth = tmp_path / "evil.pth"
    with zipfile.ZipFile(pth, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
    with pytest.raises(ec.CheckpointError):
        ec.load_state_dict(pth)
