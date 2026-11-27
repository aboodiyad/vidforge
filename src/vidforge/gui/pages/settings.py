"""Settings: language, where videos are saved, graphics card, estimate memory, about."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QRadioButton

from ... import __version__
from ...core import gpu as gpu_mod
from ...core.settings import Settings
from ...features.upscale import estimate as est
from ..bridge import JobBridge
from ..i18n import tr
from ..tasks import run_in_background
from ..widgets.common import Page, button, label, panel

REPO_URL = "https://github.com/aboodiyad/vidforge"


def restart_app() -> None:
    """Start a fresh VidForge (so a new language applies) and close this one."""
    exe = sys.executable
    windowed = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.path.exists(windowed):
        exe = windowed
    QProcess.startDetached(exe, ["-m", "vidforge.app"])
    QApplication.quit()


class SettingsPage(Page):
    def __init__(self, bridge: JobBridge, settings: Settings, parent=None):
        super().__init__(tr("set.title"), tr("set.subtitle"), parent)
        self.bridge = bridge
        self.settings = settings
        self.loaded_language = settings.language

        # language
        self.section(tr("set.language"))
        box, lay = panel()
        self.lang_group = QButtonGroup(self)
        for value in ("system", "en", "ar"):
            rb = QRadioButton(tr(f"lang.{value}"))
            rb.setChecked(settings.language == value)
            rb.toggled.connect(lambda on, v=value: on and self._language(v))
            self.lang_group.addButton(rb)
            lay.addWidget(rb)
        self.restart_row = QHBoxLayout()
        self.restart_note = label(tr("set.restart_note"), "Body", wrap=True)
        self.restart_btn = button(tr("set.restart"), name="Primary")
        self.restart_btn.clicked.connect(self._restart)
        self.restart_row.addWidget(self.restart_note, 1)
        self.restart_row.addWidget(self.restart_btn)
        lay.addSpacing(6)
        lay.addLayout(self.restart_row)
        self._show_restart(False)
        self.add(box)

        # folder
        self.section(tr("set.folder"))
        box, lay = panel(QHBoxLayout)
        self.folder = label("", "Body")
        lay.addWidget(self.folder, 1)
        reset = button(tr("folder.reset"), name="Ghost")
        reset.clicked.connect(lambda: self._folder(""))
        pick = button(tr("folder.choose"), "folder")
        pick.clicked.connect(self._pick)
        lay.addWidget(reset)
        lay.addWidget(pick)
        self.folder_reset = reset
        self._folder(settings.output_dir, save=False)
        self.add(box)

        # graphics card
        self.section(tr("set.gpu"), tr("set.gpu.help"))
        self.gpu = QComboBox()
        self.gpu.setMinimumWidth(360)
        self.gpu.setEnabled(False)
        self.gpu.currentIndexChanged.connect(self._gpu_changed)
        self.add(self.gpu)
        run_in_background(gpu_mod.detect_gpus, self._gpus)

        # estimates
        self.section(tr("set.estimates"), tr("set.estimates.desc"))
        row = QHBoxLayout()
        reset_est = button(tr("set.reset"))
        self.reset_msg = label("", "Ok")
        reset_est.clicked.connect(self._reset_estimates)
        row.addWidget(reset_est)
        row.addWidget(self.reset_msg)
        row.addStretch(1)
        self.body.addLayout(row)
        self.body.addSpacing(26)

        # updates
        self.section(tr("set.updates"))
        self.updates = QCheckBox(tr("set.updates.check"))
        self.updates.setChecked(settings.check_updates)
        self.updates.toggled.connect(self._updates)
        self.body.addWidget(self.updates)
        self.body.addSpacing(4)
        self.body.addWidget(label(tr("set.updates.help"), "Hint", wrap=True))
        self.body.addSpacing(26)

        # about
        self.section(tr("set.about"))
        self.body.addWidget(label(tr("set.about.text", v=__version__), "Body", wrap=True))
        self.body.addSpacing(8)
        link = button(tr("set.about.link"), name="Ghost")
        link.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(REPO_URL)))
        self.body.addWidget(link, 0, Qt.AlignmentFlag.AlignLeading)
        self.finish()

    def _updates(self, on: bool) -> None:
        self.settings.check_updates = on
        self.settings.save()

    def _language(self, value: str) -> None:
        self.settings.language = value
        self.settings.save()
        self._show_restart(value != self.loaded_language)

    def _show_restart(self, on: bool) -> None:
        self.restart_note.setVisible(on)
        self.restart_btn.setVisible(on)

    def _restart(self) -> None:
        busy = any(not j.state.finished for j in self.bridge.queue.jobs())
        if busy:
            self.restart_note.setText(tr("set.restart_busy"))
            return
        restart_app()

    def _folder(self, path: str, save: bool = True) -> None:
        self.settings.output_dir = path
        self.folder.setText(path or tr("folder.same"))
        self.folder_reset.setVisible(bool(path))
        if save:
            self.settings.save()

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self, tr("folder.dialog"), self.settings.output_dir)
        if d:
            self._folder(d)

    def _gpus(self, gpus) -> None:
        gpus = gpus if isinstance(gpus, list) else []
        self.gpu.blockSignals(True)
        for g in gpus:
            self.gpu.addItem(g.name, g.index)
        self.gpu.setCurrentIndex(max(0, self.gpu.findData(self.settings.gpu_index)))
        self.gpu.blockSignals(False)
        self.gpu.setEnabled(len(gpus) > 1)

    def _gpu_changed(self, _i: int) -> None:
        if self.gpu.currentData() is not None:
            self.settings.gpu_index = int(self.gpu.currentData())
            self.settings.save()

    def _reset_estimates(self) -> None:
        try:
            (est._stats_file()).unlink(missing_ok=True)
        except OSError:
            pass
        self.reset_msg.setText(tr("set.reset_done"))
