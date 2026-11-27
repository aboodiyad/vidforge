"""Real-ESRGAN on NVIDIA's TensorRT: the same ONNX model as the DirectML path, compiled for the
exact graphics card. Used when the NVIDIA speed-up add-on is installed; any problem falls back
to DirectML (see quality.make_engine), so the worst case is today's speed.

TensorRT compiles ("builds") an engine for one input size. The engine is cached per model,
tile size, card, and TensorRT version under data/models/tensorrt, so only the first video of a
given size waits for the build (seconds for the small models, a few minutes for "max").

The CUDA runtime is called through ctypes (cudart from the add-on), so no extra Python CUDA
package is needed. Frames go in and out of page-locked ("pinned") host memory with
asynchronous copies on one CUDA stream; two buffer sets alternate, so while one tile is being
upscaled the next is already uploading (see upscale_many).
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

import numpy as np

from ...core import paths
from .esrgan_onnx import TilePlan, merge_tiles, plan_tiles

H2D, D2H = 1, 2


class CudaError(RuntimeError):
    pass


class Cuda:
    """The few CUDA runtime calls we need."""

    def __init__(self, dll: Path):
        self.lib = ctypes.CDLL(str(dll))
        for name in ("cudaSetDevice", "cudaMalloc", "cudaFree", "cudaMallocHost", "cudaFreeHost", "cudaMemcpyAsync",
                     "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize", "cudaGetDeviceProperties",
                     "cudaDeviceGetAttribute", "cudaRuntimeGetVersion", "cudaDriverGetVersion"):
            if not hasattr(self.lib, name) and not hasattr(self.lib, name + "_v2"):
                raise CudaError(f"{name} missing in {dll.name}")

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise CudaError(f"{what} failed (CUDA error {rc})")

    def set_device(self, i: int) -> None:
        self.check(self.lib.cudaSetDevice(ctypes.c_int(i)), "cudaSetDevice")

    def compute_capability(self, i: int) -> tuple[int, int]:
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.cudaDeviceGetAttribute(ctypes.byref(major), 75, i), "attribute major")  # ComputeCapabilityMajor
        self.check(self.lib.cudaDeviceGetAttribute(ctypes.byref(minor), 76, i), "attribute minor")  # ComputeCapabilityMinor
        return major.value, minor.value

    def malloc(self, n: int) -> int:
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
        return p.value

    def malloc_host(self, n: int) -> int:
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMallocHost(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMallocHost")
        return p.value

    def free(self, p: int) -> None:
        self.lib.cudaFree(ctypes.c_void_p(p))

    def free_host(self, p: int) -> None:
        self.lib.cudaFreeHost(ctypes.c_void_p(p))

    def copy_async(self, dst: int, src: int, n: int, kind: int, stream: int) -> None:
        self.check(self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(n),
                                            ctypes.c_int(kind), ctypes.c_void_p(stream)), "cudaMemcpyAsync")

    def stream(self) -> int:
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s.value or 0

    def sync(self, stream: int) -> None:
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")

    def destroy_stream(self, stream: int) -> None:
        self.lib.cudaStreamDestroy(ctypes.c_void_p(stream))


def pinned_array(cuda: Cuda, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    n = int(np.prod(shape))
    ptr = cuda.malloc_host(n)
    buf = (ctypes.c_uint8 * n).from_address(ptr)
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape), ptr


def engine_cache_dir() -> Path:
    return paths.models_dir() / "tensorrt"


def engine_key(model: Path, shape: tuple[int, int], cc: tuple[int, int], trt_version: str) -> str:
    stat = model.stat()
    h = hashlib.sha1(f"{model.name}|{stat.st_size}|{shape}|sm{cc[0]}{cc[1]}|{trt_version}".encode()).hexdigest()[:12]
    return f"{model.stem}-{shape[0]}x{shape[1]}-sm{cc[0]}{cc[1]}-{h}.engine"


class TrtEsrganEngine:
    """Same interface as esrgan_onnx.EsrganEngine: .scale, .provider, .upscale(frame)."""

    def __init__(self, model_path: Path, addon_dir: Path, *, device_id: int = 0, tile: int = 512, overlap: int = 16,
                 on_log=print):
        trt = load_tensorrt(addon_dir)
        self.trt = trt
        self.cuda = Cuda(addon_dir / "nvidia" / "cuda_runtime" / "bin" / "cudart64_12.dll")
        self.cuda.set_device(device_id)
        self.cc = self.cuda.compute_capability(device_id)
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.on_log = on_log
        import onnx

        meta = {p.key: p.value for p in onnx.load(str(model_path), load_external_data=False).metadata_props}
        self.scale = int(meta.get("scale", 4))
        self.tile, self.overlap = tile, overlap
        self.provider = f"TensorRT {trt.__version__} (sm{self.cc[0]}{self.cc[1]})"
        self._plan: TilePlan | None = None
        self._shape: tuple[int, int] | None = None
        self.stream = self.cuda.stream()
        self._bufs: list[dict] = []

    # --- building and loading -------------------------------------------------------------
    def _build(self, th: int, tw: int) -> bytes:
        trt = self.trt
        builder = trt.Builder(self.logger)
        flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)  # fp16 comes from the model
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse(self.model_path.read_bytes()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not read the model: {errs}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        profile = builder.create_optimization_profile()
        name = network.get_input(0).name
        shape = (1, th, tw, 3)
        profile.set_shape(name, shape, shape, shape)
        config.add_optimization_profile(profile)
        data = builder.build_serialized_network(network, config)
        if data is None:
            raise RuntimeError("TensorRT could not build an engine for this model")
        return bytes(data)

    def _load(self, th: int, tw: int) -> None:
        cache = engine_cache_dir() / engine_key(self.model_path, (th, tw), self.cc, self.trt.__version__)
        if cache.is_file():
            data = cache.read_bytes()
        else:
            import time

            self.on_log(f"preparing the AI for this graphics card ({th}x{tw}); only the first time")
            t0 = time.monotonic()
            data = self._build(th, tw)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(cache)
            self.on_log(f"TensorRT engine built in {time.monotonic() - t0:.0f}s")
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("TensorRT could not load its engine")
        self.context = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.context.set_input_shape(self.in_name, (1, th, tw, 3))
        s = self.scale
        self._free_buffers()
        for _ in range(2):  # two sets: one is uploading while the other runs
            host_in, p_in = pinned_array(self.cuda, (1, th, tw, 3))
            host_out, p_out = pinned_array(self.cuda, (1, th * s, tw * s, 3))
            self._bufs.append({
                "host_in": host_in, "host_out": host_out, "p_in": p_in, "p_out": p_out,
                "dev_in": self.cuda.malloc(host_in.nbytes), "dev_out": self.cuda.malloc(host_out.nbytes),
            })
        self._shape = (th, tw)

    # --- running ----------------------------------------------------------------------------
    def _enqueue(self, b: dict, tile: np.ndarray) -> None:
        np.copyto(b["host_in"][0], tile)
        c, s = self.cuda, self.stream
        c.copy_async(b["dev_in"], b["p_in"], b["host_in"].nbytes, H2D, s)
        self.context.set_tensor_address(self.in_name, b["dev_in"])
        self.context.set_tensor_address(self.out_name, b["dev_out"])
        if not self.context.execute_async_v3(s):
            raise RuntimeError("TensorRT failed to run")
        c.copy_async(b["p_out"], b["dev_out"], b["host_out"].nbytes, D2H, s)

    def upscale_many(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale tiles in order; while the card works on one, the next is prepared."""
        out: list[np.ndarray] = []
        pending: list[dict] = []
        for i, t in enumerate(tiles):
            b = self._bufs[i % 2]
            if len(pending) == 2:  # the buffer we're about to reuse must be finished first
                self.cuda.sync(self.stream)
                out.extend(p["host_out"][0].copy() for p in pending)
                pending.clear()
            self._enqueue(b, t)
            pending.append(b)
        self.cuda.sync(self.stream)
        out.extend(p["host_out"][0].copy() for p in pending)
        return out

    def plan_for(self, h: int, w: int) -> TilePlan:
        if self._plan is None or (self._plan.height, self._plan.width) != (h, w):
            self._plan = plan_tiles(h, w, self.tile, self.overlap)
            if self._shape != (self._plan.tile_h, self._plan.tile_w):
                self._load(self._plan.tile_h, self._plan.tile_w)
        return self._plan

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        plan = self.plan_for(h, w)
        tiles = [frame[y:y + plan.tile_h, x:x + plan.tile_w] for y in plan.ys for x in plan.xs]
        outs = self.upscale_many(tiles)
        if plan.count == 1:
            return outs[0]
        return merge_tiles(plan, np.stack(outs), self.scale)

    def _free_buffers(self) -> None:
        for b in self._bufs:
            self.cuda.free(b["dev_in"])
            self.cuda.free(b["dev_out"])
            self.cuda.free_host(b["p_in"])
            self.cuda.free_host(b["p_out"])
        self._bufs = []

    def close(self) -> None:
        try:
            self.cuda.sync(self.stream)
        except CudaError:
            pass
        self._free_buffers()
        self.cuda.destroy_stream(self.stream)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_tensorrt(addon_dir: Path):
    """Import TensorRT from the add-on folder (kept outside VidForge's own Python so updates of
    VidForge don't remove it)."""
    import os

    if str(addon_dir) not in sys.path:
        sys.path.insert(0, str(addon_dir))
    for d in (addon_dir / "tensorrt_libs", addon_dir / "nvidia" / "cuda_runtime" / "bin"):
        if d.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(d))
    import tensorrt_bindings  # loads its libraries from tensorrt_libs next to it

    return tensorrt_bindings
