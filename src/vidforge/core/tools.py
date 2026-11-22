"""External tools VidForge drives: where to get them, how to find them, which version.

Tools are fetched from GitHub releases and verified against the SHA-256 digest GitHub
publishes for each release asset. Assets without a published digest are refused unless the
caller passes allow_unverified=True.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paths
from .archive import extract
from .download import USER_AGENT, download, sha256_file
from .process import CancelToken, capture


@dataclass(frozen=True)
class GithubSource:
    repo: str
    asset_pattern: str  # regex matched against the asset file name
    # A fixed tag, or None to take the newest release (prereleases included) that has the
    # assets. whisper.cpp and llama.cpp publish Windows builds only on "bNNNN" prereleases.
    tag: str | None = None
    extra_patterns: tuple[str, ...] = ()  # more assets unpacked into the same folder


@dataclass(frozen=True)
class ToolSpec:
    key: str
    display: str
    exe: str  # main executable name
    version_args: tuple[str, ...]
    version_regex: str
    source: GithubSource
    milestone: str  # first milestone that needs it
    extra_exes: tuple[str, ...] = ()


TOOLS: dict[str, ToolSpec] = {
    t.key: t
    for t in [
        ToolSpec(
            key="sevenzip",
            display="7-Zip (7zr)",
            exe="7zr.exe",
            version_args=(),
            version_regex=r"7-Zip \(r\) (\S+)",
            source=GithubSource("ip7z/7zip", r"^7zr\.exe$"),
            milestone="M0",  # helper: unpacks NVEncC's .7z
        ),
        ToolSpec(
            key="ffmpeg",
            display="FFmpeg",
            exe="ffmpeg.exe",
            extra_exes=("ffprobe.exe",),
            version_args=("-version",),
            version_regex=r"ffmpeg version (\S+)",
            source=GithubSource("BtbN/FFmpeg-Builds", r"^ffmpeg-n[\d.]+-latest-win64-gpl-[\d.]+\.zip$", "latest"),
            milestone="M0",
        ),
        ToolSpec(
            key="nvencc",
            display="NVEncC",
            exe="NVEncC64.exe",
            version_args=("--version",),
            version_regex=r"NVEncC \(x64\) (\S+)",
            source=GithubSource("rigaya/NVEnc", r"^NVEncC_[\d.]+_x64\.7z$"),
            milestone="M1",
        ),
        ToolSpec(
            key="whisper",
            display="whisper.cpp (CUDA)",
            exe="whisper-cli.exe",
            version_args=("--help",),
            version_regex=r"(usage)",
            source=GithubSource(
                "ggml-org/whisper.cpp",
                r"^whisper-(bin-win-cuda-12[\d.]*-x64|cublas-12[\d.]*-bin-x64)\.zip$",
            ),
            milestone="M3",
        ),
        ToolSpec(
            key="whisper-cpu",
            display="whisper.cpp (CPU)",
            exe="whisper-cli.exe",
            version_args=("--help",),
            version_regex=r"(usage)",
            source=GithubSource("ggml-org/whisper.cpp", r"^whisper-blas-bin-x64\.zip$"),
            milestone="M3",  # for PCs without an NVIDIA card
        ),
        ToolSpec(
            key="llama",
            display="llama.cpp (CUDA)",
            exe="llama-server.exe",
            version_args=("--version",),
            version_regex=r"version: (\S+)",
            source=GithubSource(
                "ggml-org/llama.cpp",
                r"^llama-b\d+-bin-win-cuda-12[\d.]*-x64\.zip$",
                extra_patterns=(r"^cudart-llama-bin-win-cuda-12[\d.]*-x64\.zip$",),
            ),
            milestone="M4",
        ),
    ]
}


@dataclass
class ToolStatus:
    spec: ToolSpec
    path: Path | None
    version: str | None
    origin: str  # "managed", "PATH" or "missing"

    @property
    def found(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class ReleaseAsset:
    tag: str
    name: str
    url: str
    size: int
    sha256: str | None


# --- locating -------------------------------------------------------------------------

def managed_dir(key: str) -> Path:
    return paths.tools_dir() / key


def _find_in_managed(spec: ToolSpec) -> Path | None:
    root = managed_dir(spec.key)
    if not root.is_dir():
        return None
    hits = sorted(root.rglob(spec.exe), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def locate(key: str) -> tuple[Path | None, str]:
    spec = TOOLS[key]
    p = _find_in_managed(spec)
    if p:
        return p, "managed"
    on_path = shutil.which(spec.exe) or shutil.which(spec.exe.removesuffix(".exe"))
    if on_path:
        return Path(on_path), "PATH"
    return None, "missing"


def exe_path(key: str, exe: str | None = None) -> Path:
    """Path to a tool's executable (or a sibling exe such as ffprobe), or raise."""
    main, _ = locate(key)
    if main is None:
        raise FileNotFoundError(f"{TOOLS[key].display} is not installed (run: vidforge tools fetch {key})")
    if exe is None:
        return main
    sibling = main.with_name(exe)
    if sibling.exists():
        return sibling
    found = shutil.which(exe)
    if found:
        return Path(found)
    raise FileNotFoundError(f"{exe} not found next to {main}")


def read_version(spec: ToolSpec, exe: Path) -> str | None:
    try:
        out = capture([str(exe), *spec.version_args], timeout=15)
    except (OSError, Exception):
        return None
    m = re.search(spec.version_regex, out.stdout + out.stderr)
    return m.group(1) if m else None


def status(key: str) -> ToolStatus:
    spec = TOOLS[key]
    path, origin = locate(key)
    version = read_version(spec, path) if path else None
    return ToolStatus(spec, path, version, origin)


def all_status() -> list[ToolStatus]:
    return [status(k) for k in TOOLS]


# --- fetching -------------------------------------------------------------------------

def _github_json(url: str) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r)


def pick_asset(release: dict, pattern: str) -> ReleaseAsset:
    """Choose the matching asset from a GitHub release JSON (highest name wins on ties)."""
    rx = re.compile(pattern)
    matches = [a for a in release.get("assets", []) if rx.search(a["name"])]
    if not matches:
        names = ", ".join(a["name"] for a in release.get("assets", []))
        raise LookupError(f"no asset matching {pattern!r} in release {release.get('tag_name')}: {names}")
    best = sorted(matches, key=lambda a: a["name"], reverse=True)[0]
    digest = best.get("digest") or ""
    sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else None
    return ReleaseAsset(release["tag_name"], best["name"], best["browser_download_url"], best["size"], sha)


def pick_assets(release: dict, src: GithubSource) -> list[ReleaseAsset]:
    """Main asset first, then one asset per extra pattern; LookupError if any is missing."""
    return [pick_asset(release, p) for p in (src.asset_pattern, *src.extra_patterns)]


def resolve(key: str) -> list[ReleaseAsset]:
    src = TOOLS[key].source
    api = f"https://api.github.com/repos/{src.repo}/releases"
    if src.tag:
        return pick_assets(_github_json(f"{api}/tags/{src.tag}"), src)
    releases = _github_json(f"{api}?per_page=30")
    for release in releases:
        if release.get("draft"):
            continue
        try:
            return pick_assets(release, src)
        except LookupError:
            continue
    raise LookupError(f"no recent {src.repo} release has an asset matching {src.asset_pattern!r}")


def fetch(
    key: str,
    *,
    allow_unverified: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    """Download and unpack a tool into the managed tools folder; returns the exe path."""
    assets = resolve(key)
    unverified = [a.name for a in assets if a.sha256 is None]
    if unverified and not allow_unverified:
        raise RuntimeError(
            f"{', '.join(unverified)} has no published SHA-256 digest; refusing to install unverified "
            "(pass allow_unverified=True / --allow-unverified to override)"
        )
    sevenzip = None
    if any(a.name.lower().endswith(".7z") for a in assets):
        sevenzip, _ = locate("sevenzip")
        if sevenzip is None:
            sevenzip = fetch("sevenzip", cancel=cancel)

    root = managed_dir(key)
    staging = root.with_name(key + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    total = sum(a.size for a in assets) or None
    offset = 0
    for asset in assets:
        archive = paths.temp_dir() / "downloads" / asset.name
        # Reuse a previous download when it still matches the published digest.
        if not (archive.is_file() and asset.sha256 and sha256_file(archive) == asset.sha256):
            cb = (lambda d, t, o=offset: progress(o + d, total)) if progress else None
            download(asset.url, archive, sha256=asset.sha256, progress=cb, cancel=cancel)
        offset += asset.size
        if asset.name.lower().endswith((".zip", ".7z")):
            extract(archive, staging, sevenzip=sevenzip)
            try:
                archive.unlink(missing_ok=True)
            except PermissionError:
                pass  # removed on the next run; not worth failing the install
        else:  # a bare executable such as 7zr.exe
            shutil.move(str(archive), staging / asset.name)
    (staging / "vidforge-tool.json").write_text(
        json.dumps(
            [{"tag": a.tag, "asset": a.name, "sha256": a.sha256, "url": a.url} for a in assets], indent=2
        ),
        encoding="utf-8",
    )
    shutil.rmtree(root, ignore_errors=True)
    staging.replace(root)
    exe = _find_in_managed(TOOLS[key])
    if exe is None:
        raise FileNotFoundError(f"{TOOLS[key].exe} not found in {asset.name}")
    return exe
