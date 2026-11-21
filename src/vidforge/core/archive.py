"""Safe extraction of downloaded .zip / .7z archives (rejects paths that escape the target).

.7z archives are unpacked with 7-Zip's standalone 7zr.exe: NVEncC's releases use the BCJ2
filter, which pure-Python 7z readers do not support.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .process import capture


class UnsafeArchiveError(RuntimeError):
    pass


def _check_member(target: Path, name: str) -> None:
    resolved = (target / name).resolve()
    if resolved != target and target not in resolved.parents:
        raise UnsafeArchiveError(f"archive member escapes target folder: {name!r}")


def parse_7z_listing(text: str) -> list[str]:
    """Member paths from `7zr l -slt` output. Entries follow the '----------' separator;
    the 'Path = ' line before it names the archive itself."""
    _, sep, body = text.partition("\n----------")
    if not sep:
        return []
    return [line[len("Path = "):].strip() for line in body.splitlines() if line.startswith("Path = ")]


def extract(archive: Path, target: Path, *, sevenzip: Path | None = None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                _check_member(target, name)
            zf.extractall(target)
    elif suffix == ".7z":
        if sevenzip is None:
            raise FileNotFoundError("7zr.exe is required to unpack .7z archives")
        listing = capture([sevenzip, "l", "-slt", archive], timeout=120)
        if listing.returncode != 0:
            raise RuntimeError(f"7zr could not list {archive.name}: {listing.stdout[-400:]}")
        names = parse_7z_listing(listing.stdout)
        if not names:
            raise RuntimeError(f"7zr listed no files in {archive.name}")
        for name in names:
            _check_member(target, name)
        run = capture([sevenzip, "x", "-y", f"-o{target}", archive], timeout=900)
        if run.returncode != 0:
            raise RuntimeError(f"7zr failed to extract {archive.name}: {run.stdout[-400:]}")
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")
