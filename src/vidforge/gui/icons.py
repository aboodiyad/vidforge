"""VidForge's own icon set: 24x24 line icons, 1.75 stroke, round caps, drawn for this app.
Rendered from SVG at any size and colour."""

from __future__ import annotations

import functools

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from . import theme

PATHS = {
    "upscale": '<rect x="3" y="13" width="8" height="8" rx="1.6"/><path d="M13.5 3H21v7.5M21 3l-8.5 8.5"/>',
    "face": '<circle cx="12" cy="12" r="9"/><path d="M8.5 14.6c1.9 2 5.1 2 7 0"/><path d="M9.2 9.6h.01M14.8 9.6h.01" stroke-width="2.4"/>',
    "subtitles": '<rect x="3" y="5" width="18" height="14" rx="2.6"/><path d="M7 15h4M13.5 15H17M7 11h10"/>',
    "translate": '<path d="M3.5 6h9M8 3.8V6M5.6 6c.8 3.6 3 6.2 6 7.6M10.4 6c-.8 3.6-3 6.2-6 7.6"/><path d="M13 20.5l4-9.5 4 9.5M14.6 17h4.8"/>',
    "settings": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2.2"/><circle cx="10" cy="17" r="2.2"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2.2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9.5 3v3M14.5 3v3M9.5 18v3M14.5 18v3M3 9.5h3M3 14.5h3M18 9.5h3M18 14.5h3"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2.6"/><path d="M7.5 4v16M16.5 4v16M3 9h4.5M3 15h4.5M16.5 9H21M16.5 15H21"/>',
    "folder": '<path d="M3 7.6A2.6 2.6 0 0 1 5.6 5H9l2 2.4h7.4A2.6 2.6 0 0 1 21 10v7.4a2.6 2.6 0 0 1-2.6 2.6H5.6A2.6 2.6 0 0 1 3 17.4z"/>',
    "play": '<path d="M8 5.6v12.8a.6.6 0 0 0 .9.5l10-6.4a.6.6 0 0 0 0-1l-10-6.4a.6.6 0 0 0-.9.5z"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "bolt": '<path d="M13.2 2.8L5.4 13.2h6.1l-1 8 7.8-10.4h-6.1z"/>',
    "sparkle": '<path d="M11 3.5l1.8 5.2 5.2 1.8-5.2 1.8L11 17.5l-1.8-5.2L4 10.5l5.2-1.8z"/><path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
    "chevron": '<path d="M6.5 9.5l5.5 5.5 5.5-5.5"/>',
    "chevron-up": '<path d="M6.5 14.5L12 9l5.5 5.5"/>',
    "check": '<path d="M5 12.6l4.4 4.4L19 7.4"/>',
    "alert": '<circle cx="12" cy="12" r="9"/><path d="M12 7.6v5.4M12 16.4h.01" />',
    "plus": '<path d="M12 5.5v13M5.5 12h13"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7.5V12l3 2"/>',
    "file": '<path d="M14 3H7.6A2.6 2.6 0 0 0 5 5.6v12.8A2.6 2.6 0 0 0 7.6 21h8.8a2.6 2.6 0 0 0 2.6-2.6V8z"/><path d="M14 3v5h5"/>',
}


def _svg(name: str, color: str) -> bytes:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</svg>'
    ).encode()


@functools.lru_cache(maxsize=None)
def pixmap(name: str, size: int = 20, color: str = theme.TEXT_2, ratio: float = 2.0) -> QPixmap:
    px = QPixmap(int(size * ratio), int(size * ratio))
    px.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(_svg(name, color)))
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, px.width(), px.height()))
    p.end()
    px.setDevicePixelRatio(ratio)
    return px


def icon(name: str, color: str = theme.TEXT_2, active: str | None = None, disabled: str = theme.TEXT_3) -> QIcon:
    """An icon with colours for normal, checked/active and disabled states."""
    ic = QIcon()
    ic.addPixmap(pixmap(name, 24, color), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, 24, active or color), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, 24, active or theme.TEXT), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(pixmap(name, 24, disabled), QIcon.Mode.Disabled, QIcon.State.Off)
    return ic


ICON_SIZE = QSize(20, 20)
