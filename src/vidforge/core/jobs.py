"""Background job queue shared by the GUI and the CLI.

GPU work is heavy, so by default one worker runs jobs one at a time. Listeners are called
from the worker thread; the GUI forwards them to the UI thread through a Qt signal.
"""

from __future__ import annotations

import collections
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
    message: str = ""  # technical detail (English), for logs and tooltips
    phase: str = ""  # short machine-readable step, e.g. "preparing", "downloading", "working"
    error: str | None = None
    error_code: str | None = None  # set when the error has a known, translatable cause
    error_params: dict = field(default_factory=dict)
    result: Any = None
    created: float = field(default_factory=time.time)
    started: float | None = None  # time.monotonic() when it began running
    finished: float | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken, repr=False)
    log: list[str] = field(default_factory=list, repr=False)
    # (monotonic time, progress) samples while working, for the time-remaining estimate.
    _samples: collections.deque = field(default_factory=lambda: collections.deque(maxlen=400), repr=False)

    def elapsed(self, now: float | None = None) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else (now or time.monotonic())
        return max(0.0, end - self.started)

    def eta(self, now: float | None = None, window: float = 30.0) -> float | None:
        """Seconds left, from the progress rate over the last *window* seconds of work.
        None until there is enough data to say anything honest."""
        if self.state != JobState.RUNNING or not self._samples:
            return None
        now = now or time.monotonic()
        t1, p1 = self._samples[-1]
        if p1 >= 1.0:
            return 0.0
        old = [(t, p) for t, p in self._samples if now - t <= window] or [self._samples[0]]
        t0, p0 = old[0]
        if (t1 - t0) < 2.0 or (p1 - p0) <= 0.002:
            return None
        rate = (p1 - p0) / (t1 - t0)
        return max(0.0, (1.0 - p1) / rate - (now - t1))


class JobContext:
    """What a job function receives: progress reporting, logging and cancellation."""

    def __init__(self, job: Job, notify: Callable[[Job], None]):
        self._job = job
        self._notify = notify
        self._last_emit = 0.0

    @property
    def cancel(self) -> CancelToken:
        return self._job.cancel_token

    def report(self, progress: float | None = None, message: str | None = None, phase: str | None = None) -> None:
        if phase is not None and phase != self._job.phase:
            self._job.phase = phase
            self._job._samples.clear()  # a new step has its own pace
        if progress is not None:
            self._job.progress = max(0.0, min(1.0, progress))
            self._job._samples.append((time.monotonic(), self._job.progress))
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
            job.started = time.monotonic()
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
                job.error_code = getattr(exc, "code", None)
                job.error_params = dict(getattr(exc, "params", None) or {})
                job.log.append(traceback.format_exc())
            job.finished = time.monotonic()
            self._notify(job)
