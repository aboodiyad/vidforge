"""A local translator: llama.cpp's `llama-server` on 127.0.0.1, spoken to over its
OpenAI-style HTTP API.

The server is started for one job and stopped right after, so its graphics memory is free for
the voice step. `--fit on` lets llama.cpp keep as many layers on the graphics card as fit and
run the rest on the processor, so smaller cards still work (just slower).
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable

from ...core import tools
from ...core.process import CancelToken, Cancelled
from ..subtitles.whisper import ascii_path

CTX = 8192


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, model: Path, *, cancel: CancelToken, on_log: Callable[[str], None], gpu: bool = True):
        self.model = model
        self.cancel = cancel
        self.on_log = on_log
        self.gpu = gpu
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.tail: deque[str] = deque(maxlen=40)
        # A one-time key, so web pages open in a browser can't use the server while it runs.
        self.key = secrets.token_hex(16)
        self.last_usage: dict = {}

    # --- lifecycle ---
    def command(self) -> list[str]:
        exe = tools.exe_path("llama")
        # llama.cpp reads its command line in the old Windows code page: give it the short path.
        cmd = [str(exe), "-m", str(ascii_path(self.model)), "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(CTX), "-np", "1", "--no-webui", "--api-key", self.key, "--jinja", "-fa", "auto",
               "--reasoning-budget", "0"]  # Qwen3: answer directly, no hidden "thinking" text
        if not self.gpu:
            cmd += ["-ngl", "0"]
        return cmd

    def start(self, timeout: float = 300) -> "LlamaServer":
        self.port = free_port()
        cmd = self.command()
        self.on_log("$ " + " ".join(cmd).replace(self.key, "***"))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._pump, daemon=True).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel.cancelled:
                self.stop()
                raise Cancelled()
            if self.proc.poll() is not None:
                raise RuntimeError("the translator stopped while starting:\n" + "\n".join(list(self.tail)[-12:]))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("the translator did not start in time")

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                self.tail.append(line)
                # Only the interesting lines: the server is chatty about every request.
                if any(k in line for k in ("error", "offload", "CUDA", "fit", "model size", "failed")):
                    self.on_log(line)

    def stop(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                p.kill()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- requests ---
    def chat(self, messages: list[dict], *, schema: dict | None = None, max_tokens: int = 2048,
             temperature: float = 0.3) -> str:
        """One chat completion; with *schema* the answer is forced to match that JSON schema."""
        body: dict = {
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        result: dict = {}
        error: list[BaseException] = []

        def call() -> None:
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    result.update(json.load(r))
            except BaseException as e:  # handed back to the waiting thread
                error.append(e)

        t = threading.Thread(target=call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.2)
            if self.cancel.cancelled:
                self.stop()  # ends the request too
                raise Cancelled()
        if error:
            raise RuntimeError(f"the translator failed: {error[0]}")
        self.last_usage = result.get("usage", {})
        return result["choices"][0]["message"]["content"]
