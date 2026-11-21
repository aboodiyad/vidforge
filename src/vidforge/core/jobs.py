"""Background job queue shared by the GUI and the CLI.

GPU work is heavy, so by default one worker runs jobs one at a time. Listeners are called
from the worker thread; the GUI forwards them to the UI thread through a Qt signal.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .process import CancelToken, Cancelled


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


@dataclass
class Job:
    id: int
    title: str
    fn: Callable[["JobContext"], Any] = field(repr=False)
    state: JobState = JobState.QUEUED
    progress: float | None = None  # 0..1, None = indeterminate
    message: str = ""
    error: str | None = None
    result: Any = None
    created: float = field(default_factory=time.time)
    cancel_token: CancelToken = field(default_factory=CancelToken, repr=False)
    log: list[str] = field(default_factory=list, repr=False)


class JobContext:
    """What a job function receives: progress reporting, logging and cancellation."""

    def __init__(self, job: Job, notify: Callable[[Job], None]):
        self._job = job
        self._notify = notify
        self._last_emit = 0.0

    @property
    def cancel(self) -> CancelToken:
        return self._job.cancel_token

    def report(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self._job.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self._job.message = message
        now = time.monotonic()
        # Throttle UI updates; always emit when a message changes.
        if message is not None or now - self._last_emit >= 0.1:
            self._last_emit = now
            self._notify(self._job)

    def log(self, line: str) -> None:
        self._job.log.append(line)
        if len(self._job.log) > 2000:
            del self._job.log[:500]


class JobQueue:
    def __init__(self, workers: int = 1):
        self._jobs: dict[int, Job] = {}
        self._queue: "queue.Queue[Job | None]" = queue.Queue()
        self._ids = itertools.count(1)
        self._listeners: list[Callable[[Job], None]] = []
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._worker, name=f"job-worker-{i}", daemon=True)
            for i in range(workers)
        ]
        for t in self._threads:
            t.start()

    def add_listener(self, fn: Callable[[Job], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, job: Job) -> None:
        for fn in list(self._listeners):
            try:
                fn(job)
            except Exception:  # a broken listener must not kill the worker
                traceback.print_exc()

    def submit(self, title: str, fn: Callable[[JobContext], Any]) -> Job:
        job = Job(id=next(self._ids), title=title, fn=fn)
        with self._lock:
            self._jobs[job.id] = job
        self._notify(job)
        self._queue.put(job)
        return job

    def cancel(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.state.finished:
            return
        job.cancel_token.cancel()
        if job.state == JobState.QUEUED:
            job.state = JobState.CANCELLED
            self._notify(job)

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def wait(self, job: Job, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while not job.state.finished:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    def shutdown(self) -> None:
        for job in self.jobs():
            self.cancel(job.id)
        for _ in self._threads:
            self._queue.put(None)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if job.state != JobState.QUEUED:
                continue  # cancelled while waiting
            job.state = JobState.RUNNING
            self._notify(job)
            ctx = JobContext(job, self._notify)
            try:
                job.result = job.fn(ctx)
                job.state = JobState.DONE
                job.progress = 1.0
            except Cancelled:
                job.state = JobState.CANCELLED
            except Exception as exc:
                job.state = JobState.FAILED
                job.error = str(exc) or exc.__class__.__name__
                job.log.append(traceback.format_exc())
            self._notify(job)
