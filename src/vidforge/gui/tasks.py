"""Run short blocking calls (GPU detection, probing) off the UI thread."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as e:  # handed to the callback instead of crashing the UI
            result = e
        self.signals.done.emit(result)


_alive: set[_Task] = set()


def run_in_background(fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
    """Call fn() on a pool thread; on_done(result_or_exception) runs on the UI thread."""
    task = _Task(fn)
    _alive.add(task)  # keep the Python object (and its signal) alive until it fires
    task.signals.done.connect(on_done)
    task.signals.done.connect(lambda _r, t=task: _alive.discard(t))
    QThreadPool.globalInstance().start(task)
