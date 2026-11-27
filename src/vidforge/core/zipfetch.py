"""Download only some files out of a large zip (or wheel) on a web server.

Zip files keep a table of contents at the end, and the server answers "Range" requests, so the
table and then just the wanted files can be read without the rest. Every file taken out is
checked against a pinned SHA-256 before it is kept, so this is as safe as downloading the whole
archive and checking its hash: the pins were made from an archive whose published hash matched.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .download import USER_AGENT, ChecksumError
from .process import CancelToken

CHUNK = 8 << 20  # bytes per request when reading ahead


class RangeFile:
    """A read-only, seekable file over HTTP Range requests, reading ahead in big chunks."""

    def __init__(self, url: str, *, timeout: float = 60, cancel: CancelToken | None = None,
                 on_bytes: Callable[[int], None] | None = None):
        self.url = url
        self.timeout = timeout
        self.cancel = cancel
        self.on_bytes = on_bytes
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise OSError("the server can't send parts of the file")
        self.pos = 0
        self._buf_start = 0
        self._buf = b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def _fetch(self, start: int, length: int) -> bytes:
        if self.cancel:
            self.cancel.raise_if_cancelled()
        end = min(self.size, start + length) - 1
        req = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise OSError(f"expected part of the file, got HTTP {r.status}")
            data = r.read()
        if self.on_bytes:
            self.on_bytes(len(data))
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            off = self.pos - self._buf_start
            if 0 <= off < len(self._buf):
                piece = self._buf[off:off + n]
            else:
                self._buf_start = self.pos
                self._buf = self._fetch(self.pos, max(n, CHUNK) if n > 65536 else CHUNK)
                piece = self._buf[:n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def close(self) -> None:
        self._buf = b""


def fetch_members(url: str, members: dict[str, str], dest: Path, *, cancel: CancelToken | None = None,
                  progress: Callable[[int, int | None], None] | None = None) -> list[Path]:
    """Extract *members* ({path in zip: sha256}) from the zip at *url* into *dest*, keeping the
    zip's folders. A file only appears once its SHA-256 matches."""
    done = [0]
    total: int | None = None  # known once the table of contents is read

    def counted(n: int) -> None:
        done[0] += n
        if progress:
            progress(done[0], total)

    f = RangeFile(url, cancel=cancel, on_bytes=counted)
    out: list[Path] = []
    with zipfile.ZipFile(f) as zf:
        infos = {i.filename: i for i in zf.infolist()}
        missing = [m for m in members if m not in infos]
        if missing:
            raise LookupError(f"not in the archive: {', '.join(missing)}")
        total = sum(infos[m].compress_size for m in members)
        done[0] = 0  # don't count the table of contents
        for name, sha in members.items():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            h = hashlib.sha256()
            with zf.open(infos[name]) as src, open(part, "wb") as dst:
                while True:
                    if cancel:
                        cancel.raise_if_cancelled()
                    block = src.read(1 << 20)
                    if not block:
                        break
                    h.update(block)
                    dst.write(block)
            if h.hexdigest() != sha:
                part.unlink(missing_ok=True)
                raise ChecksumError(f"SHA-256 mismatch for {name} in {url}")
            shutil.move(str(part), target)
            out.append(target)
    return out
