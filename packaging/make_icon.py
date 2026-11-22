"""Draw VidForge's icon and save it as src/vidforge/gui/assets/vidforge.ico (+ a 256 px PNG).

An amber play button on a dark rounded square, with a small spark for "made better". The .ico
holds PNG images at the sizes Windows asks for (allowed since Windows Vista).

    .venv\\Scripts\\python packaging\\make_icon.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parents[1] / "src" / "vidforge" / "gui" / "assets"


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0
    # background: dark rounded square with a faint top light
    bg = QLinearGradient(0, 0, 0, size)
    bg.setColorAt(0, QColor("#262A33"))
    bg.setColorAt(1, QColor("#111215"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bg)
    pad = 8 * s
    p.drawRoundedRect(QRectF(pad, pad, size - 2 * pad, size - 2 * pad), 56 * s, 56 * s)
    # play triangle, slightly right of centre so it looks centred
    amber = QLinearGradient(0, 60 * s, 0, 200 * s)
    amber.setColorAt(0, QColor("#FFC766"))
    amber.setColorAt(1, QColor("#E8921E"))
    tri = QPainterPath()
    tri.moveTo(QPointF(94 * s, 66 * s))
    tri.lineTo(QPointF(196 * s, 128 * s))
    tri.lineTo(QPointF(94 * s, 190 * s))
    tri.closeSubpath()
    p.setBrush(amber)
    p.drawPath(tri)
    # spark (a four-pointed star) at the top right, left out when too small to read
    if size >= 32:
        cx, cy, r, w = 190 * s, 62 * s, 30 * s, 8 * s
        star = QPainterPath()
        star.moveTo(cx, cy - r)
        star.quadTo(cx + w * 0.6, cy - w * 0.6, cx + r, cy)
        star.quadTo(cx + w * 0.6, cy + w * 0.6, cx, cy + r)
        star.quadTo(cx - w * 0.6, cy + w * 0.6, cx - r, cy)
        star.quadTo(cx - w * 0.6, cy - w * 0.6, cx, cy - r)
        p.setBrush(QColor("#FFF1D6"))
        p.drawPath(star)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: Path, images: list[tuple[int, bytes]]) -> None:
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, data in images:
        dim = 0 if size >= 256 else size  # 0 means 256
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    path.write_bytes(header + entries + blobs)


def main() -> int:
    _app = QGuiApplication(sys.argv)  # QPainter needs an application object
    OUT.mkdir(parents=True, exist_ok=True)
    images = [(n, png_bytes(draw(n))) for n in SIZES]
    write_ico(OUT / "vidforge.ico", images)
    draw(256).save(str(OUT / "vidforge.png"))
    print(f"wrote {OUT / 'vidforge.ico'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
