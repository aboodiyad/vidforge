"""The NVIDIA speed-up add-on: TensorRT for Best quality upscaling on RTX cards.

Kept in the data folder (addons\\nvidia), so updates of VidForge don't remove it. Everything is
pinned by SHA-256:
- tensorrt_cu12_bindings (the Python part, 0.9 MB) and nvidia_cuda_runtime_cu12 (cudart, 3.6 MB)
  are downloaded whole from PyPI.
- tensorrt_cu12_libs is one 2.25 GB wheel holding TensorRT for every card generation. Only the
  core libraries and the builder resources for this PC's card are read out of it (zipfetch),
  about 0.5 GB. Their hashes were taken from the whole wheel after it matched the SHA-256 that
  NVIDIA's index publishes (818221d3...542f).
TensorRT 11 supports Turing (RTX 20) and newer; older cards keep DirectML.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable

from ...core import paths
from ...core.download import download
from ...core.gpu import GpuInfo
from ...core.process import CancelToken
from ...core.zipfetch import fetch_members

VERSION = 1
TRT_VERSION = "11.5.0.63"
_NV = "https://pypi.nvidia.com"
LIBS_URL = f"{_NV}/tensorrt-cu12-libs/tensorrt_cu12_libs-{TRT_VERSION}-py3-none-win_amd64.whl"
WHEELS = {  # name -> (url, sha256)
    "bindings": (f"{_NV}/tensorrt-cu12-bindings/tensorrt_cu12_bindings-{TRT_VERSION}-cp312-none-win_amd64.whl",
                 "663f209cf0f4789015087d8bcd4ed4b7a2dc2d9494177a3e01548392ff2c2261"),
    "cudart": (f"{_NV}/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
               "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891"),
}
# Files read out of the libs wheel: always the core, plus one builder resource for the card.
LIBS_CORE = {
    "tensorrt_libs/__init__.py": "fa7935877053963869234bb9ba27531c6d33835d9c217cf83caf03ea0e39eb42",
    "tensorrt_libs/nvinfer_11.dll": "0fbbbff1a826c12ca94d948173a7695e57c0699d833cabd4d9d59a37eeb96956",
    "tensorrt_libs/nvinfer_plugin_11.dll": "64813da3124228b01a1b96ac82852e88c2fe9909993973ad3e822a7aa5812c53",
    "tensorrt_libs/nvonnxparser_11.dll": "c5f3e2f480ebaae68b08f8ee202a0b86255f34a62148f11004e3a21787f588de",
}
LIBS_RESOURCES = {  # card generation -> sha256 of its builder resource
    "sm75": "184b1e0ba665d1ddfc07d3eed4d9efe4ed51ba12ca3af36211e09ce2505ac036",
    "sm80": "46a112713972e10081164cc5c47b200f57d92f0a69bbc375e7591f3583813cd0",
    "sm86": "4f896eaf30c13926348f19a04a3444a2bd1dcfc715078bdb6aca398e394810e1",
    "sm89": "9158656fd4a34ac9b065e7766705a216ab6e4554c1a3dc456e18732900d0be2b",
    "sm90": "692084721affb4a65fe033d1145391d93df4bc7f1556143c01925189553976b2",
    "sm100": "c1dfc0a242254b7dd8c944b2f3bfc69e80288b968e817d308379286f961533e0",
    "sm120": "0ba6f46216dfbc1e9e100e2035ca3265c2516549abfa3d55a17eb43938476497",
    "ptx": "2ac6fd47c84696cc633ce8402f95097571b41ebe5c04a423c7dd32026e473bcd",
}
LIBS_SIZES_MB = {  # compressed size inside the wheel = what is downloaded
    "tensorrt_libs/nvinfer_11.dll": 210.0, "tensorrt_libs/nvinfer_plugin_11.dll": 25.5,
    "tensorrt_libs/nvonnxparser_11.dll": 0.9,
    **{f"tensorrt_libs/nvinfer_builder_resource_{k}_11.dll": v for k, v in {
        "sm75": 105.5, "sm80": 180.9, "sm86": 168.5, "sm89": 178.3, "sm90": 503.5, "sm100": 304.3,
        "sm120": 276.2, "ptx": 296.9}.items()},
}


def root() -> Path:
    return paths.data_root() / "addons" / "nvidia"


def _marker() -> Path:
    return root() / "vidforge-addon.json"


def supported(gpu: GpuInfo | None) -> bool:
    return bool(gpu and gpu.vendor == "nvidia" and gpu.compute_cap and gpu.compute_cap >= (7, 5))


def resource_for(cc: tuple[int, int]) -> str:
    """The builder resource for a card: its own generation if TensorRT ships one, else the
    generic (PTX) one, which works on any card TensorRT supports."""
    sm = f"sm{cc[0]}{cc[1]}"
    return sm if sm in LIBS_RESOURCES else "ptx"


def _member(res: str) -> str:
    return f"tensorrt_libs/nvinfer_builder_resource_{res}_11.dll"


def status() -> dict | None:
    try:
        data = json.loads(_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("version") == VERSION else None


def is_installed() -> bool:
    return status() is not None


def download_mb(gpu: GpuInfo | None) -> float:
    if not supported(gpu):
        return 0.0
    res = resource_for(gpu.compute_cap)
    core = sum(LIBS_SIZES_MB.get(m, 0.0) for m in LIBS_CORE)
    return core + LIBS_SIZES_MB.get(_member(res), 0.0) + 5.0


def _unzip(wheel: Path, dest: Path, prefixes: tuple[str, ...]) -> None:
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.startswith(prefixes) and not name.endswith("/"):
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def install(gpu: GpuInfo, *, cancel: CancelToken, progress: Callable[[int, int | None], None] | None = None,
            on_log: Callable[[str], None] = print) -> Path:
    if not supported(gpu):
        raise RuntimeError("the NVIDIA speed-up needs an NVIDIA RTX card (RTX 20 series or newer)")
    res = resource_for(gpu.compute_cap)
    staging = root().with_name("nvidia.new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    tmp = paths.temp_dir() / "downloads"
    for name, (url, sha) in WHEELS.items():
        wheel = tmp / url.rsplit("/", 1)[1]
        download(url, wheel, sha256=sha, cancel=cancel)
        _unzip(wheel, staging, ("tensorrt_bindings/", "nvidia/"))
        wheel.unlink(missing_ok=True)
    members = dict(LIBS_CORE)
    members[_member(res)] = LIBS_RESOURCES[res]
    on_log(f"reading {len(members)} files for {res} out of {LIBS_URL}")
    fetch_members(LIBS_URL, members, staging, cancel=cancel, progress=progress)
    (staging / "vidforge-addon.json").write_text(
        json.dumps({"version": VERSION, "tensorrt": TRT_VERSION, "resource": res}), encoding="utf-8")
    shutil.rmtree(root(), ignore_errors=True)
    staging.replace(root())
    return root()


def remove() -> None:
    shutil.rmtree(root(), ignore_errors=True)
