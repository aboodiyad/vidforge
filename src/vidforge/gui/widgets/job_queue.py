"""The list of started videos: what each one is doing, how far along, and how long is left."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QVBoxLayout,
    QWidget,
)

from ...core.jobs import Job, JobState
from .. import i18n, theme
from ..bridge import JobBridge
from ..i18n import tr
from .common import button, icon_button, label


def friendly_error(job: Job) -> str:
    if job.error_code and f"err.{job.error_code}" in i18n.S:
        return tr(f"err.{job.error_code}", **job.error_params)
    return tr("err.generic")


def status_text(job: Job) -> tuple[str, str]:
    """(status line, time line) in the current language."""
    if job.state == JobState.QUEUED:
        return tr("st.queued"), ""
    if job.state == JobState.DONE:
        return tr("st.done", t=i18n.duration(job.elapsed(), precise=True)), ""
    if job.state == JobState.CANCELLED:
        return tr("st.cancelled"), ""
    if job.state == JobState.FAILED:
        return tr("st.failed", why=friendly_error(job)), ""
    if job.phase == "downloading":
        return tr("st.downloading"), ""
    if job.phase in ("", "preparing") or job.progress is None:
        return tr("st.preparing"), ""
    status = tr("st.working", pct=i18n.pct(job.progress))
    eta = job.eta()
    return status, (tr("st.left", t=i18n.duration(eta, precise=True)) if eta is not None else tr("st.measuring"))


class DetailsDialog(QDialog):
    def __init__(self, job: Job, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("details.title"))
        self.resize(720, 420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.addWidget(label(friendly_error(job), "Strong", wrap=True))
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setLayoutDirection(Qt.LayoutDirection.LeftToRight)  # logs are English
        text.setPlainText((job.error or "") + "\n\n" + "\n".join(job.log[-80:]))
        lay.addWidget(text, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        copy = button(tr("details.copy"))
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(text.toPlainText()))
        close = button(tr("details.close"))
        close.clicked.connect(self.accept)
        row.addWidget(copy)
        row.addWidget(close)
        lay.addLayout(row)


def show_in_folder(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class JobRow(QFrame):
    def __init__(self, job: Job, bridge: JobBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("JobRow")
        self.job = job
        self.bridge = bridge
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        lay.setSpacing(14)
        text = QVBoxLayout()
        text.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.name = label(job.title, "JobName")
        self.name.setToolTip(job.title)
        top.addWidget(self.name, 1)
        self.time = label("", "JobTime")
        top.addWidget(self.time)
        text.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        text.addWidget(self.bar)
        self.status = label("", "JobStatus")
        text.addWidget(self.status)
        lay.addLayout(text, 1)

        self.stop = button(tr("btn.stop"), "stop")
        self.stop.clicked.connect(lambda: bridge.queue.cancel(job.id))
        self.show_btn = button(tr("btn.show"), "folder")
        self.show_btn.clicked.connect(lambda: show_in_folder(Path(self.job.result)))
        self.play = button(tr("btn.play"), "play")
        self.play.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.job.result))))
        self.details = button(tr("btn.details"))
        self.details.clicked.connect(lambda: DetailsDialog(self.job, self.window()).exec())
        for b in (self.stop, self.show_btn, self.play, self.details):
            lay.addWidget(b)
        self.refresh()

    def refresh(self) -> None:
        job = self.job
        status, time_line = status_text(job)
        self.status.setText(status)
        self.status.setToolTip(job.message)
        self.time.setText(time_line)
        state = job.state.value
        for w in (self.status, self.bar):
            w.setProperty("state", state)
            theme.repolish(w)
        running = job.state == JobState.RUNNING
        if running and job.progress is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int((job.progress or 0) * 1000) if job.state != JobState.QUEUED else 0)
        done_file = job.state == JobState.DONE and isinstance(job.result, (str, Path)) and Path(job.result).is_file()
        self.stop.setVisible(not job.state.finished)
        self.show_btn.setVisible(done_file)
        self.play.setVisible(done_file and Path(job.result).suffix.lower() in (".mp4", ".mkv", ".mov"))
        self.details.setVisible(job.state == JobState.FAILED)


class JobQueueWidget(QFrame):
    """Bottom panel. Hidden while empty so the page keeps its room."""

    def __init__(self, bridge: JobBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("JobsPanel")
        self.bridge = bridge
        self.rows: dict[int, JobRow] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 12, 40, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        self.title = label(tr("jobs.title"), "SectionTitle")
        head.addWidget(self.title)
        self.count = label("", "Hint")
        head.addWidget(self.count)
        head.addStretch(1)
        self.clear = button(tr("jobs.clear"), name="Ghost")
        self.clear.clicked.connect(self.clear_finished)
        head.addWidget(self.clear)
        lay.addLayout(head)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.list = QVBoxLayout(holder)
        self.list.setContentsMargins(0, 0, 0, 0)
        self.list.setSpacing(8)
        self.list.addStretch(1)
        self.scroll.setWidget(holder)
        lay.addWidget(self.scroll)
        bridge.job_updated.connect(self.update_job)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)
        self._update_header()

    def update_job(self, job: Job) -> None:
        row = self.rows.get(job.id)
        if row is None:
            row = JobRow(job, self.bridge)
            self.rows[job.id] = row
            self.list.insertWidget(self.list.count() - 1, row)
        row.refresh()
        self._update_header()

    def tick(self) -> None:
        for row in self.rows.values():
            if row.job.state == JobState.RUNNING:
                row.refresh()

    def clear_finished(self) -> None:
        for jid, row in list(self.rows.items()):
            if row.job.state.finished:
                row.setParent(None)
                row.deleteLater()
                del self.rows[jid]
        self._update_header()

    def _update_header(self) -> None:
        running = sum(1 for r in self.rows.values() if not r.job.state.finished)
        self.count.setText(tr("jobs.running", n=running) if running else "")
        self.clear.setVisible(any(r.job.state.finished for r in self.rows.values()))
        self.setVisible(bool(self.rows))
        visible_rows = min(len(self.rows), 2)
        self.scroll.setFixedHeight(max(1, visible_rows) * 70)
