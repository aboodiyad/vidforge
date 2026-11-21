"""Job list with progress bars, status text and cancel buttons."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...core.jobs import Job, JobState
from ..bridge import JobBridge

COLS = ["Job", "Status", "Progress", ""]


class JobQueueWidget(QWidget):
    def __init__(self, bridge: JobBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self._rows: dict[int, int] = {}
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(2, 180)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        bridge.job_updated.connect(self.update_job)

    def update_job(self, job: Job) -> None:
        row = self._rows.get(job.id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[job.id] = row
            self.table.setItem(row, 0, QTableWidgetItem(job.title))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            self.table.setCellWidget(row, 2, bar)
            btn = QPushButton("Cancel")
            btn.clicked.connect(lambda _=False, jid=job.id: self.bridge.queue.cancel(jid))
            self.table.setCellWidget(row, 3, btn)

        status = job.state.value.capitalize()
        if job.state == JobState.FAILED and job.error:
            status = f"Failed: {job.error.splitlines()[0]}"
        elif job.message:
            status = f"{status}: {job.message}"
        item = QTableWidgetItem(status)
        if job.error:
            item.setToolTip(job.error)
        self.table.setItem(row, 1, item)

        bar: QProgressBar = self.table.cellWidget(row, 2)
        if job.progress is None and job.state == JobState.RUNNING:
            bar.setRange(0, 0)  # busy indicator
        else:
            bar.setRange(0, 1000)
            bar.setValue(int((job.progress or 0) * 1000))
        btn: QPushButton = self.table.cellWidget(row, 3)
        btn.setEnabled(not job.state.finished)
