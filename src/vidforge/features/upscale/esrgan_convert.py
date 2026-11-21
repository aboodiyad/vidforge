"""Convert official Real-ESRGAN PyTorch checkpoints (.pth) to ONNX without needing PyTorch.

A .pth file is a zip holding a pickled state dict plus raw tensor storages. We unpickle it with
a strict allow-list (only the three classes a plain state dict needs), read the storages with
NumPy, and then build the network directly as an ONNX graph.

Supported architectures (both from the public Real-ESRGAN project, BSD-3-Clause):
- RRDBNet, scale 4 (RealESRGAN_x4plus): 23 residual-in-residual dense blocks.
- SRVGGNetCompact (realesr-general-x4v3, realesr-animevideov3): a plain conv/PReLU stack whose
  output is pixel-shuffled and added to a nearest-neighbour upscale of the input.
"""

from __future__ import annotations

import collections
import pickle
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# --- checkpoint loading ----------------------------------------------------------------

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
}


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class _StorageRef:
    key: str
    dtype: type


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


def _rebuild_tensor(storage, offset, shape, stride, *_ignored):
    return _TensorRef(storage, int(offset), tuple(shape), tuple(stride))


class _StateDictUnpickler(pickle.Unpickler):
    """Only allows what a plain tensor state dict contains; anything else is rejected."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(_STORAGE_DTYPES[name])
        raise CheckpointError(f"checkpoint references disallowed object {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise CheckpointError(f"unexpected persistent id {pid!r}")
        stype = pid[1]
        if not isinstance(stype, _StorageType):
            raise CheckpointError("unsupported storage type")
        return _StorageRef(str(pid[2]), stype.dtype)


def load_state_dict(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl = next((n for n in names if n.endswith("/data.pkl") or n == "data.pkl"), None)
        if pkl is None:
            raise CheckpointError(f"{path.name}: not a PyTorch zip checkpoint")
        prefix = pkl[: -len("data.pkl")]
        obj = _StateDictUnpickler(zf.open(pkl)).load()
        if isinstance(obj, dict):
            for key in ("params_ema", "params", "state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        if not isinstance(obj, dict):
            raise CheckpointError(f"{path.name}: no state dict found")
        storages: dict[str, np.ndarray] = {}
        out: dict[str, np.ndarray] = {}
        for name, ref in obj.items():
            if not isinstance(ref, _TensorRef):
                raise CheckpointError(f"{path.name}: entry {name} is not a tensor")
            st = ref.storage
            if st.key not in storages:
                storages[st.key] = np.frombuffer(zf.read(f"{prefix}data/{st.key}"), dtype=st.dtype)
            buf = storages[st.key]
            item = buf.itemsize
            arr = np.lib.stride_tricks.as_strided(
                buf[ref.offset:], shape=ref.shape, strides=tuple(s * item for s in ref.stride)
            )
            out[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


# --- architecture detection ------------------------------------------------------------

@dataclass(frozen=True)
class ArchInfo:
    kind: str  # "rrdb" or "srvgg"
    scale: int
    num_feat: int
    num_blocks: int  # RRDB blocks, or conv layers in the SRVGG body (excluding first/last)


def detect_arch(sd: dict[str, np.ndarray]) -> ArchInfo:
    if "conv_first.weight" in sd:
        blocks = {int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.rdb1\.", k))}
        in_ch = sd["conv_first.weight"].shape[1]
        if in_ch != 3:
            raise CheckpointError("only scale-4 RRDBNet checkpoints are supported")
        return ArchInfo("rrdb", 4, sd["conv_first.weight"].shape[0], len(blocks))
    if "body.0.weight" in sd:
        convs = sorted(int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.bias$", k)))
        last = sd[f"body.{convs[-1]}.weight"]
        scale = int(round((last.shape[0] / 3) ** 0.5))
        if scale * scale * 3 != last.shape[0]:
            raise CheckpointError("unexpected SRVGG output channel count")
        return ArchInfo("srvgg", scale, sd["body.0.weight"].shape[0], len(convs) - 2)
    raise CheckpointError("unrecognised Real-ESRGAN architecture")


# --- ONNX graph building ---------------------------------------------------------------

class _Graph:
    def __init__(self, dtype: np.dtype):
        self.dtype = np.dtype(dtype)
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def name(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, value: np.ndarray, base: str, keep_dtype: bool = False) -> str:
        arr = value if keep_dtype else value.astype(self.dtype)
        n = self.name(base)
        self.inits.append(numpy_helper.from_array(arr, n))
        return n

    def op(self, op_type: str, inputs: list[str], base: str, **attrs) -> str:
        out = self.name(base)
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=out, **attrs))
        return out

    def conv(self, x: str, sd: dict, prefix: str) -> str:
        w = self.const(sd[f"{prefix}.weight"], prefix + ".w")
        b = self.const(sd[f"{prefix}.bias"], prefix + ".b")
        k = sd[f"{prefix}.weight"].shape[-1]
        return self.op("Conv", [x, w, b], prefix, pads=[k // 2] * 4, kernel_shape=[k, k])

    def lrelu(self, x: str) -> str:
        return self.op("LeakyRelu", [x], "lrelu", alpha=0.2)

    def upsample_nearest(self, x: str, factor: int) -> str:
        scales = self.const(np.array([1, 1, factor, factor], np.float32), "scales", keep_dtype=True)
        return self.op(
            "Resize", [x, "", scales], "up", mode="nearest",
            coordinate_transformation_mode="asymmetric", nearest_mode="floor",
        )

    def scaled_add(self, x: str, residual: str, factor: float) -> str:
        f = self.const(np.array(factor, np.float32), "res_scale")
        return self.op("Add", [self.op("Mul", [x, f], "mul"), residual], "add")


def _rdb(g: _Graph, x: str, sd: dict, p: str) -> str:
    feats = [x]
    for i in range(1, 5):
        cat = g.op("Concat", feats, "cat", axis=1) if len(feats) > 1 else x
        feats.append(g.lrelu(g.conv(cat, sd, f"{p}.conv{i}")))
    x5 = g.conv(g.op("Concat", feats, "cat", axis=1), sd, f"{p}.conv5")
    return g.scaled_add(x5, x, 0.2)


def _build_rrdb(g: _Graph, x: str, sd: dict, arch: ArchInfo) -> str:
    feat = g.conv(x, sd, "conv_first")
    body = feat
    for b in range(arch.num_blocks):
        inner = body
        for r in (1, 2, 3):
            inner = _rdb(g, inner, sd, f"body.{b}.rdb{r}")
        body = g.scaled_add(inner, body, 0.2)
    feat = g.op("Add", [feat, g.conv(body, sd, "conv_body")], "add")
    feat = g.lrelu(g.conv(g.upsample_nearest(feat, 2), sd, "conv_up1"))
    feat = g.lrelu(g.conv(g.upsample_nearest(feat, 2), sd, "conv_up2"))
    return g.conv(g.lrelu(g.conv(feat, sd, "conv_hr")), sd, "conv_last")


def _build_srvgg(g: _Graph, x: str, sd: dict, arch: ArchInfo) -> str:
    convs = sorted(int(m.group(1)) for k in sd if (m := re.match(r"body\.(\d+)\.bias$", k)))
    h = x
    for i, idx in enumerate(convs):
        h = g.conv(h, sd, f"body.{idx}")
        if i < len(convs) - 1:
            slope = sd[f"body.{idx + 1}.weight"].reshape(-1, 1, 1)
            h = g.op("PRelu", [h, g.const(slope, f"body.{idx + 1}.slope")], "prelu")
    out = g.op("DepthToSpace", [h], "shuffle", blocksize=arch.scale, mode="CRD")
    return g.op("Add", [out, g.upsample_nearest(x, arch.scale)], "add")


def build_onnx(
    sd: dict[str, np.ndarray], *, fp16: bool = True, uint8_io: bool = True
) -> tuple[onnx.ModelProto, ArchInfo]:
    """Build the network. With *uint8_io* the model takes and returns uint8 RGB images in
    N x H x W x 3 layout and does the scaling, transposes and rounding on the GPU, so the host
    only moves bytes. Without it, IO is N x 3 x H x W floats in [0, 1] (used for validation)."""
    arch = detect_arch(sd)
    dtype = np.float16 if fp16 else np.float32
    elem = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    g = _Graph(dtype)
    builder = _build_rrdb if arch.kind == "rrdb" else _build_srvgg
    if uint8_io:
        x = g.op("Cast", ["input"], "cast_in", to=elem)
        x = g.op("Transpose", [x], "to_nchw", perm=[0, 3, 1, 2])
        x = g.op("Mul", [x, g.const(np.array(1 / 255, np.float32), "inv255")], "norm")
        y = builder(g, x, sd, arch)
        y = g.op("Mul", [y, g.const(np.array(255, np.float32), "k255")], "denorm")
        y = g.op("Clip", [y, g.const(np.array(0, np.float32), "lo"), g.const(np.array(255, np.float32), "hi")], "clip")
        y = g.op("Round", [y], "round")
        y = g.op("Transpose", [y], "to_nhwc", perm=[0, 2, 3, 1])
        g.nodes.append(helper.make_node("Cast", [y], ["output"], name="cast_out", to=TensorProto.UINT8))
        inputs = [helper.make_tensor_value_info("input", TensorProto.UINT8, ["batch", "height", "width", 3])]
        outputs = [helper.make_tensor_value_info("output", TensorProto.UINT8, ["batch", "out_height", "out_width", 3])]
    else:
        out = builder(g, "input", sd, arch)
        g.nodes.append(helper.make_node("Identity", [out], ["output"], name="output_identity"))
        inputs = [helper.make_tensor_value_info("input", elem, ["batch", 3, "height", "width"])]
        outputs = [helper.make_tensor_value_info("output", elem, ["batch", 3, "out_height", "out_width"])]
    graph = helper.make_graph(g.nodes, f"realesrgan_{arch.kind}_x{arch.scale}", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="vidforge")
    model.ir_version = 8
    model.metadata_props.add(key="scale", value=str(arch.scale))
    onnx.checker.check_model(model)
    return model, arch


def convert(pth: Path, out: Path, *, fp16: bool = True, uint8_io: bool = True) -> ArchInfo:
    model, arch = build_onnx(load_state_dict(pth), fp16=fp16, uint8_io=uint8_io)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    onnx.save(model, str(tmp))
    tmp.replace(out)
    return arch
