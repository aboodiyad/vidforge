"""Shared by the feature pages: the list of chosen videos (with drag and drop, background
reading, and per-video "after" size / file size / time) and the Start bar."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget

from ...core import media
from ...core.media import MediaInfo
from ...features.upscale import estimate as est
from .. import i18n, theme
from ..i18n import tr
from ..tasks import run_in_background
from .common import button, icon_button, icon_label, label, panel

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".ts", ".m2ts", ".mts", ".webm", ".wmv", ".flv",
             ".mpg", ".mpeg", ".vob", ".3gp"}
VIDEO_FILTER = "Videos (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXT)) + ");;All files (*)"


@dataclass
class Video:
    path: Path
    info: MediaInfo | None = None
    error: str | None = None
    estimates: dict = field(default_factory=dict)  # key -> upscale.estimate.Estimate
    size_error: bool = False


def clock(seconds: float | None) -> str:
    if not seconds:
        return ""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


class VideoRow(QFrame):
    def __init__(self, video: Video, on_remove, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Row")
        self.video = video
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 6, 10)
        lay.setSpacing(14)
        lay.addWidget(icon_label("film", 22, theme.TEXT_3))
        names = QVBoxLayout()
        names.setSpacing(2)
        self.name = label(video.path.name, "RowName")
        self.name.setToolTip(str(video.path))
        self.name.setMinimumWidth(120)
        names.addWidget(self.name)
        self.meta = label(tr("row.reading"), "RowMeta")
        names.addWidget(self.meta)
        lay.addLayout(names, 3)
        self.arrow = label(tr("row.arrow"), "Hint")
        lay.addWidget(self.arrow)
        self.after = label("", "RowAfter")
        self.after.setMinimumWidth(110)
        lay.addWidget(self.after, 1)
        self.size = self._stat("file")
        lay.addLayout(self.size[0], 1)
        self.time = self._stat("clock")
        lay.addLayout(self.time[0], 1)
        rm = icon_button("close", tr("up.remove"))
        rm.clicked.connect(lambda: on_remove(self))
        lay.addWidget(rm)

    @staticmethod
    def _stat(icon_name: str):
        box = QHBoxLayout()
        box.setSpacing(6)
        box.addWidget(icon_label(icon_name, 16, theme.TEXT_3))
        lb = label("", "RowNum")
        box.addWidget(lb, 1)
        return box, lb

    def show(self, key: str) -> None:
        v = self.video
        if v.error:
            self.meta.setText(tr("row.cant_read"))
            self.meta.setObjectName("Danger")
            theme.repolish(self.meta)
            self.meta.setToolTip(v.error)
            for w in (self.arrow, self.after, self.size[1], self.time[1]):
                w.setText("")
            return
        if v.info is None:
            return
        vs = v.info.video
        dw, dh = vs.display_size if vs and vs.display_size else (0, 0)
        self.meta.setText(f"{tr('u.res', w=dw, h=dh)}  ·  {i18n.ltr(clock(v.info.duration))}")
        e = v.estimates.get(key)
        if v.size_error or e is None:
            self.arrow.setText("")
            self.after.setText(tr("err.already_big") if v.size_error else "")
            self.after.setObjectName("Hint" if v.size_error else "RowAfter")
            self.after.setWordWrap(True)
            theme.repolish(self.after)
            self.size[1].setText("")
            self.time[1].setText("")
            return
        self.arrow.setText(tr("row.arrow"))
        self.after.setObjectName("RowAfter")
        theme.repolish(self.after)
        self.after.setText(tr("u.res", w=e.width, h=e.height))
        self.size[1].setText(i18n.about(i18n.size(est.round_nice(e.bytes))))
        self.size[1].setToolTip(f"{i18n.size(e.bytes_low)} – {i18n.size(e.bytes_high)}")
        self.time[1].setText(i18n.about_time(e.seconds))
        self.time[1].setToolTip(f"{i18n.duration(e.seconds_low)} – {i18n.duration(e.seconds_high)}")


class DropZone(QFrame):
    def __init__(self, on_choose, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(8)
        lay.addWidget(icon_label("film", 40, theme.TEXT_3), 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(label(tr("drop.title"), "SectionTitle"), 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(label(tr("drop.or"), "Hint"), 0, Qt.AlignmentFlag.AlignHCenter)
        b = button(tr("drop.choose"), "plus")
        b.clicked.connect(on_choose)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)

    def set_hover(self, on: bool) -> None:
        self.setProperty("hover", "true" if on else "false")
        theme.repolish(self)


class VideoList(QWidget):
    """Drop zone while empty, list of rows once videos are added. Emits `changed` whenever a
    video is added, read or removed, so the page can recompute its estimates."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.videos: list[Video] = []
        self.rows: list[VideoRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.drop = DropZone(self.choose_files)
        lay.addWidget(self.drop)
        self.list_frame, self.list_lay = panel(margins=(10, 10, 10, 10), spacing=8)
        tools_row = QHBoxLayout()
        tools_row.setContentsMargins(4, 2, 4, 0)
        add = button(tr("up.add_more"), "plus", "Ghost")
        add.clicked.connect(self.choose_files)
        clear = button(tr("up.clear"), name="Ghost")
        clear.clicked.connect(self.clear)
        tools_row.addWidget(add)
        tools_row.addStretch(1)
        tools_row.addWidget(clear)
        self.list_lay.addLayout(tools_row)
        lay.addWidget(self.list_frame)
        self._refresh()

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, tr("drop.dialog"), "", VIDEO_FILTER)
        self.add_paths(paths)

    def add_paths(self, paths: list[str]) -> None:
        known = {v.path for v in self.videos}
        for p in paths:
            path = Path(p)
            if not path.is_file() or path in known:
                continue
            video = Video(path)
            self.videos.append(video)
            row = VideoRow(video, self.remove_row)
            self.rows.append(row)
            self.list_lay.insertWidget(self.list_lay.count() - 1, row)
            run_in_background(lambda p=path: media.probe(p), lambda r, v=video: self._probed(v, r))
        self._refresh()

    def add_dropped(self, urls) -> None:
        files = []
        for u in urls:
            p = Path(u.toLocalFile())
            if p.is_dir():
                files += [str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in VIDEO_EXT]
            else:
                files.append(str(p))
        self.add_paths(files)

    def _probed(self, video: Video, result) -> None:
        if isinstance(result, Exception):
            video.error = str(result)
        elif result.video is None:
            video.error = tr("err.no_video")
        else:
            video.info = result
        self.changed.emit()

    def remove_row(self, row: VideoRow) -> None:
        self.rows.remove(row)
        self.videos.remove(row.video)
        row.setParent(None)
        row.deleteLater()
        self._refresh()

    def clear(self) -> None:
        for row in list(self.rows):
            self.remove_row(row)

    def _refresh(self) -> None:
        has = bool(self.videos)
        self.drop.setVisible(not has)
        self.list_frame.setVisible(has)
        self.changed.emit()

    def show_estimates(self, key: str) -> None:
        for row in self.rows:
            row.show(key)


class ActionBar(QFrame):
    """Always-visible bottom bar: what you will get, and the Start button."""

    def __init__(self, on_start, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("JobsPanel")
        bl = QHBoxLayout(self)
        bl.setContentsMargins(40, 14, 40, 14)
        bl.setSpacing(16)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self.summary = label(tr("sum.empty"), "Strong")
        self.note = label("", "Hint", wrap=True)
        texts.addWidget(self.summary)
        texts.addWidget(self.note)
        bl.addLayout(texts, 1)
        self.start = button(tr("up.start"), "play", "Primary")
        self.start.clicked.connect(on_start)
        bl.addWidget(self.start)

    def show_totals(self, videos: int, ready: int, seconds: float, size: float, download_mb: float) -> None:
        if not videos:
            self.summary.setText(tr("sum.empty"))
            self.note.setText("")
        elif ready:
            self.summary.setText(tr("sum.line", count=i18n.videos(ready),
                                    size=i18n.about(i18n.size(est.round_nice(size))),
                                    time=i18n.about_time(seconds)))
            notes = [tr("sum.download", mb=f"{download_mb:.0f}")] if download_mb else []
            notes.append(tr("sum.guess"))
            self.note.setText("  ".join(notes))
        else:
            self.summary.setText(i18n.videos(videos))
            self.note.setText("")
        self.start.setEnabled(ready > 0)

    def show_added(self) -> None:
        self.summary.setText(tr("sum.added"))
        self.note.setText("")
