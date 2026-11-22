"""The first start of a fresh install: download the helper programs every part needs (FFmpeg, and
NVEncC on RTX cards) before the main window opens, with a progress bar and a way to try again."""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QProgressBar, QVBoxLayout

from ..core import gpu as gpu_mod
from ..core import tools
from ..core.process import CancelToken, Cancelled
from . import i18n
from .i18n import tr
from .widgets.common import button, label


class _Fetcher(QThread):
    progress = Signal(int, int, str)  # done bytes, total bytes, tool name
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, cancel: CancelToken):
        super().__init__()
        self.cancel = cancel

    def run(self) -> None:
        try:
            gpus = gpu_mod.detect_gpus()
            keys = tools.first_run_keys(any(g.vsr_capable for g in gpus))
            for key in keys:
                name = tools.TOOLS[key].display
                tools.fetch(key, progress=lambda d, t, n=name: self.progress.emit(d, t or 0, n), cancel=self.cancel)
            self.finished_ok.emit()
        except Cancelled:
            pass
        except Exception as e:  # shown to the person, with a button to try again
            self.failed.emit(str(e))


class FirstRunDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VidForge")
        self.setMinimumWidth(520)
        self.cancel = CancelToken()
        self.worker: _Fetcher | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 22)
        lay.setSpacing(10)
        lay.addWidget(label(tr("first.title"), "SectionTitle"))
        lay.addWidget(label(tr("first.desc"), "Body", wrap=True))
        lay.addSpacing(6)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addWidget(self.bar)
        self.status = label(tr("first.checking"), "Hint", wrap=True)
        lay.addWidget(self.status)
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        self.retry = button(tr("first.retry"))
        self.retry.clicked.connect(self.start)
        self.retry.setVisible(False)
        row.addWidget(self.retry)
        self.quit = button(tr("first.quit"))
        self.quit.clicked.connect(self.reject)
        row.addWidget(self.quit)
        lay.addLayout(row)

    def start(self) -> None:
        self.retry.setVisible(False)
        self.status.setObjectName("Hint")
        self.status.setText(tr("first.checking"))
        self.bar.setRange(0, 0)
        self.worker = _Fetcher(self.cancel)
        self.worker.progress.connect(self._progress)
        self.worker.finished_ok.connect(self.accept)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, done: int, total: int, name: str) -> None:
        name = i18n.ltr(name)  # a Latin name inside Arabic text keeps its place
        if total:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(done * 1000 / total))
            self.status.setText(tr("first.getting", name=name, done=i18n.size(done), total=i18n.size(total)))
        else:
            self.status.setText(tr("first.getting_plain", name=name))

    def _failed(self, why: str) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.status.setText(tr("first.failed"))
        self.status.setToolTip(why)
        self.retry.setVisible(True)

    def reject(self) -> None:
        self.cancel.cancel()
        if self.worker is not None:
            self.worker.wait(15000)
        super().reject()

    def exec_if_needed(self) -> bool:
        """Show the dialog on a fresh install (no FFmpeg yet); True when VidForge can start. NVEncC
        comes in the same step on RTX cards; later it can be (re)installed on "My PC"."""
        if tools.locate("ffmpeg")[0] is not None:
            return True
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.start()
        return self.exec() == QDialog.DialogCode.Accepted
