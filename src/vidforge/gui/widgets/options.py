"""The "More options" panel shared by the feature pages: a two-column grid of plainly named
settings, plus the output rows every page has (video format, file size, file type, save to)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...core.settings import Settings
from .. import icons, theme
from ..i18n import tr
from .common import button, label, panel


def combo(items: list[tuple[str, str]], current, on_change) -> QComboBox:
    c = QComboBox()
    for value, text in items:
        c.addItem(text, value)
    c.setCurrentIndex(max(0, c.findData(current)))
    c.currentIndexChanged.connect(lambda *_: on_change())
    return c


class MoreOptions:
    """A toggle button plus a hidden panel; add rows, then place `button` and `panel`."""

    def __init__(self):
        self.button = button(tr("more.show"), "chevron", "Ghost")
        self.button.clicked.connect(self.toggle)
        self.panel, self.grid = panel(QGridLayout, margins=(20, 18, 20, 18), spacing=12)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnMinimumWidth(0, 180)
        self.grid.setHorizontalSpacing(24)
        self.panel.setVisible(False)
        self._row = 0

    def toggle(self) -> None:
        on = not self.panel.isVisible()
        self.panel.setVisible(on)
        self.button.setText(tr("more.hide") if on else tr("more.show"))
        self.button.setIcon(icons.icon("chevron-up" if on else "chevron", theme.TEXT_2, active=theme.TEXT))

    def row(self, title: str, widget, hint: str | None = None) -> None:
        self.grid.addWidget(label(title, "Strong"), self._row, 0,
                            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeading)
        box = QVBoxLayout()
        box.setSpacing(4)
        if isinstance(widget, QWidget):
            box.addWidget(widget)
        else:
            box.addLayout(widget)
        if hint:
            box.addWidget(label(hint, "Hint", wrap=True))
        self.grid.addLayout(box, self._row, 1)
        self._row += 1


class OutputRows:
    """Video format, file size, file type and save folder, stored in the shared settings."""

    def __init__(self, more: MoreOptions, settings: Settings, on_change):
        self.settings = settings
        self.on_change = on_change
        s = settings
        self.quality = combo([(k, tr(f"q.{k}")) for k in ("high", "balanced", "small")], s.upscale_quality, self._changed)
        self.codec = combo([(k, tr(f"codec.{k}")) for k in ("hevc", "h264", "av1")], s.upscale_codec, self._changed)
        self.container = combo([(k, tr(f"container.{k}")) for k in ("mp4", "mkv")], s.upscale_container, self._changed)
        self.folder = label("", "Body")
        pick = button(tr("folder.choose"), "folder")
        pick.clicked.connect(self._pick)
        self.folder_reset = button(tr("folder.reset"), name="Ghost")
        self.folder_reset.clicked.connect(lambda: self.set_folder(""))
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.folder_reset)
        folder_row.addWidget(pick)
        more.row(tr("more.quality"), self.quality)
        more.row(tr("more.codec"), self.codec)
        more.row(tr("more.container"), self.container)
        more.row(tr("more.folder"), folder_row)
        self.set_folder(s.output_dir, save=False)

    def _changed(self) -> None:
        s = self.settings
        s.upscale_quality = self.quality.currentData()
        s.upscale_codec = self.codec.currentData()
        s.upscale_container = self.container.currentData()
        s.save()
        self.on_change()

    def set_folder(self, path: str, save: bool = True) -> None:
        self.settings.output_dir = path
        self.folder.setText(path or tr("folder.same"))
        self.folder.setToolTip(path)
        self.folder_reset.setVisible(bool(path))
        if save:
            self.settings.save()

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self.folder, tr("folder.dialog"), self.settings.output_dir)
        if d:
            self.set_folder(d)

    def allow_av1(self, on: bool) -> None:
        self.codec.model().item(self.codec.findData("av1")).setEnabled(on)
