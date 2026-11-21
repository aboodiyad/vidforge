"""HTTP(S) downloads with SHA-256 verification, progress and cancellation."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Callable

from .. import __version__
from .process import CancelToken

USER_AGENT = f"VidForge/{__version__}"
CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: CancelToken | None = None,
    timeout: float = 60,
) -> Path:
    """Download *url* to *dest*. The file only appears at *dest* if the hash matches.

    *sha256* may be None only when the caller has explicitly decided to skip verification.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
            length = resp.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            while True:
                if cancel:
                    cancel.raise_if_cancelled()
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                if progress:
                    progress(done, total)
        digest = h.hexdigest()
        if sha256 is not None and digest.lower() != sha256.lower():
            raise ChecksumError(f"SHA-256 mismatch for {url}: expected {sha256}, got {digest}")
        part.replace(dest)
        return dest
    finally:
        if part.exists():
            part.unlink()
