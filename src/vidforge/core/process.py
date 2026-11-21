"""Run external tools with live output lines and cancellation.

FFmpeg, NVEncC and whisper.cpp redraw progress with carriage returns, so output is split on
both '\\r' and '\\n'. Cancelling kills the whole process tree.
"""

from __future__ import annotations

import collections
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable, Sequence


class Cancelled(Exception):
    """Raised when a job is cancelled."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class ProcessError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, tail: Iterable[str]):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.tail = list(tail)
        name = os.path.basename(self.cmd[0]) if self.cmd else "process"
        last = "\n".join(self.tail[-10:])
        super().__init__(f"{name} exited with code {returncode}\n{last}")


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
    else:
        proc.kill()


def _pump(stream, on_line: Callable[[str], None]) -> None:
    buf = []
    while True:
        ch = stream.read(1)
        if not ch:
            break
        if ch in "\r\n":
            if buf:
                on_line("".join(buf))
                buf.clear()
        else:
            buf.append(ch)
    if buf:
        on_line("".join(buf))


def run_process(
    cmd: Sequence[str],
    *,
    on_line: Callable[[str], None] | None = None,
    cancel: CancelToken | None = None,
    cwd: str | os.PathLike | None = None,
    env: dict[str, str] | None = None,
    ok_codes: Iterable[int] = (0,),
    tail_lines: int = 60,
) -> int:
    """Run *cmd*, streaming merged stdout/stderr lines to *on_line*.

    Raises Cancelled if *cancel* fires, ProcessError on an unexpected exit code.
    """
    tail: collections.deque[str] = collections.deque(maxlen=tail_lines)

    def handle(line: str) -> None:
        tail.append(line)
        if on_line:
            on_line(line)

    if cancel:
        cancel.raise_if_cancelled()
    proc = subprocess.Popen(
        list(map(str, cmd)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    reader = threading.Thread(target=_pump, args=(proc.stdout, handle), daemon=True)
    reader.start()
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel and cancel.cancelled:
                    _kill_tree(proc)
                    proc.wait()
                    raise Cancelled()
    finally:
        reader.join(timeout=2)
        if proc.stdout:
            proc.stdout.close()
    if proc.returncode not in set(ok_codes):
        raise ProcessError(cmd, proc.returncode, tail)
    return proc.returncode


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[=>]")


def pty_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winpty  # noqa: F401
    except ImportError:
        return False
    return True


def run_process_console(
    cmd: Sequence[str],
    *,
    on_line: Callable[[str], None] | None = None,
    cancel: CancelToken | None = None,
    ok_codes: Iterable[int] = (0,),
    tail_lines: int = 60,
) -> int:
    """Like run_process, but the child gets a pseudo-console (Windows ConPTY).

    Some tools (NVEncC) only print live progress when attached to a console. Falls back to
    plain pipes where ConPTY is unavailable."""
    if not pty_available():
        return run_process(cmd, on_line=on_line, cancel=cancel, ok_codes=ok_codes, tail_lines=tail_lines)
    from winpty import PtyProcess

    tail: collections.deque[str] = collections.deque(maxlen=tail_lines)
    if cancel:
        cancel.raise_if_cancelled()
    proc = PtyProcess.spawn([str(c) for c in cmd], dimensions=(50, 500))
    chunks: "queue.Queue[str | None]" = queue.Queue()

    def reader() -> None:
        while True:
            try:
                chunks.put(proc.read(4096))
            except EOFError:
                break
            except Exception:
                break
        chunks.put(None)

    threading.Thread(target=reader, daemon=True).start()
    pending = ""
    try:
        while True:
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                if cancel and cancel.cancelled:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=_NO_WINDOW)
                    raise Cancelled()
                continue
            if chunk is None:
                break
            pending += _ANSI.sub("", chunk)
            *lines, pending = re.split(r"[\r\n]+", pending)
            for line in lines:
                line = line.rstrip()
                if line:
                    tail.append(line)
                    if on_line:
                        on_line(line)
        if pending.strip():
            tail.append(pending.strip())
            if on_line:
                on_line(pending.strip())
        for _ in range(50):  # the exit status can lag the end of output slightly
            if not proc.isalive():
                break
            time.sleep(0.1)
        code = proc.exitstatus if proc.exitstatus is not None else -1
    finally:
        try:
            proc.close(force=True)
        except Exception:
            pass
    if code not in set(ok_codes):
        raise ProcessError(cmd, code, tail)
    return code


def capture(cmd: Sequence[str], timeout: float = 30) -> subprocess.CompletedProcess:
    """Run a short command and capture its output (for version checks and probes)."""
    return subprocess.run(
        list(map(str, cmd)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )
