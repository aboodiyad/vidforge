"""Is there a newer VidForge? Asks GitHub for the latest release's version number (nothing about
the user is sent; GitHub sees only that someone asked). The app then offers the download page;
installing the new version over the old keeps every downloaded program and AI file."""

from __future__ import annotations

import json
import re
import urllib.request

from .. import __version__
from .download import USER_AGENT

REPO = "aboodiyad/vidforge"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"


def parse_version(text: str) -> tuple[int, ...]:
    m = re.search(r"(\d+(?:\.\d+)*)", text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def is_newer(latest: str, current: str = __version__) -> bool:
    a, b = parse_version(latest), parse_version(current)
    n = max(len(a), len(b))
    return bool(a) and a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


def latest_version(timeout: float = 6) -> str | None:
    """The newest release's version, or None if it can't be found out (offline and so on)."""
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest",
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tag = json.load(r).get("tag_name", "")
    except Exception:  # noqa: BLE001 - never bother the user about this
        return None
    return tag.lstrip("vV") or None


def newer_version() -> str | None:
    v = latest_version()
    return v if v and is_newer(v) else None
