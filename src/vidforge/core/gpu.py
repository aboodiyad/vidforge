"""GPU detection and the RTX Video Super Resolution capability check.

RTX VSR needs an NVIDIA RTX card (Turing or newer, with Tensor cores) and a driver that ships
NGX. GTX 16xx cards are also compute capability 7.5 but have no Tensor cores, so the name must
contain "RTX". `probe_vsr` settles it for real by running a tiny NVEncC job.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from . import paths, tools
from .process import capture


@dataclass
class GpuInfo:
    index: int
    name: str
    vendor: str  # nvidia / amd / intel / other
    driver: str | None = None
    vram_mb: int | None = None
    compute_cap: tuple[int, int] | None = None

    @property
    def is_rtx(self) -> bool:
        return (
            self.vendor == "nvidia"
            and "RTX" in self.name.upper()
            and self.compute_cap is not None
            and self.compute_cap >= (7, 5)
        )

    @property
    def vsr_capable(self) -> bool:
        """Heuristic only; confirm with probe_vsr()."""
        return self.is_rtx


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n or " arc" in n:
        return "intel"
    return "other"


def parse_nvidia_smi(csv_text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap
    --format=csv,noheader,nounits`."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, driver, mem, cc = parts[:5]
        cap = None
        if "." in cc:
            major, minor = cc.split(".", 1)
            if major.isdigit() and minor.isdigit():
                cap = (int(major), int(minor))
        gpus.append(
            GpuInfo(
                index=int(idx) if idx.isdigit() else len(gpus),
                name=name,
                vendor="nvidia",
                driver=driver,
                vram_mb=int(float(mem)) if mem.replace(".", "", 1).isdigit() else None,
                compute_cap=cap,
            )
        )
    return gpus


def _nvidia_gpus() -> list[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = capture(
            [smi, "--query-gpu=index,name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            timeout=15,
        )
    except Exception:
        return []
    return parse_nvidia_smi(out.stdout) if out.returncode == 0 else []


def _windows_adapters() -> list[GpuInfo]:
    if sys.platform != "win32":
        return []
    try:
        out = capture(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json"],
            timeout=20,
        )
        data = json.loads(out.stdout or "[]")
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    return [
        GpuInfo(index=i, name=d.get("Name") or "?", vendor=_vendor(d.get("Name") or ""),
                driver=d.get("DriverVersion"))
        for i, d in enumerate(data)
    ]


def detect_gpus() -> list[GpuInfo]:
    """NVIDIA GPUs from nvidia-smi (accurate VRAM and compute capability), plus any other
    adapters Windows reports (AMD, Intel) for the DirectML path."""
    gpus = _nvidia_gpus()
    for a in _windows_adapters():
        if a.vendor == "nvidia" and gpus:
            continue
        lowered = a.name.lower()
        # Skip software and virtual display adapters (Microsoft Basic Display, Meta/Parsec
        # virtual monitors, remote desktop); they cannot run any of our GPU paths.
        if any(w in lowered for w in ("basic display", "remote", "virtual", "indirect")):
            continue
        a.index = len(gpus)
        gpus.append(a)
    return gpus


def probe_vsr(timeout: float = 120) -> tuple[bool | None, str]:
    """Run a 1-second NVEncC job with NGX VSR. Returns (ok, detail); ok is None when the
    tools needed for the test are not installed."""
    try:
        ffmpeg = tools.exe_path("ffmpeg")
        nvencc = tools.exe_path("nvencc")
    except FileNotFoundError as e:
        return None, str(e)
    work = paths.temp_dir() / "vsr-probe"
    work.mkdir(parents=True, exist_ok=True)
    src, dst = work / "in.mp4", work / "out.mp4"
    gen = capture(
        [ffmpeg, "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
        timeout=60,
    )
    if gen.returncode != 0:
        return False, "could not create the test clip: " + gen.stderr[-400:]
    run = capture(
        [nvencc, "--avhw", "-i", src, "-o", dst, "--codec", "h264",
         "--vpp-resize", "algo=ngx-vsr,vsr-quality=4", "--output-res", "640x360"],
        timeout=timeout,
    )
    text = run.stdout + run.stderr
    ok = run.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    detail = "NGX VSR works" if ok else (text.strip().splitlines() or ["NVEncC failed"])[-1]
    for p in (src, dst):
        Path(p).unlink(missing_ok=True)
    return ok, detail
