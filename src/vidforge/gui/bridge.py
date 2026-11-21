"""Moves JobQueue notifications (worker threads) onto the Qt UI thread."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..core.jobs import Job, JobQueue


class JobBridge(QObject):
    job_updated = Signal(object)  # Job

    def __init__(self, queue: JobQueue, parent: QObject | None = None):
        super().__init__(parent)
        self.queue = queue
        # Emitting from a worker thread is safe; Qt queues the call to the receiver's thread.
        queue.add_listener(self._on_job)

    def _on_job(self, job: Job) -> None:
        self.job_updated.emit(job)
