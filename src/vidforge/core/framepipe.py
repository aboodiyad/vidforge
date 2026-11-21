"""Raw-frame pipeline: FFmpeg decodes to RGB frames, Python processes them, FFmpeg encodes.

Frames travel as rgb24 over pipes. The reader forces a constant frame rate, so frame N always
sits at N / fps and the copied audio stays in sync. Colour: the source is converted to RGB with
its own matrix (BT.709 for HD, BT.601 for SD when untagged); the output is encoded as BT.709.
"""

from __future__ import annotations

import collections
import subprocess
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from . import tools
from .media import MediaInfo, StreamInfo
from .process import Cancelled, CancelToken, ProcessError, _NO_WINDOW, _kill_tree
from .streams import container_of, plan_audio, plan_subtitles

_MATRIX = {
    "bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601",
    "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc",
}


def source_matrix(v: StreamInfo) -> str:
    if v.color_space in _MATRIX:
        return _MATRIX[v.color_space]
    return "bt709" if (v.height or 0) >= 720 else "bt601"


class _Pipe:
    """A child process with one binary pipe; stderr is drained into a tail buffer."""

    def __init__(self, cmd: list[str], *, stdin: bool):
        self.cmd = cmd
        self.tail: collections.deque[str] = collections.deque(maxlen=40)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL if stdin else subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW,
            bufsize=0,
        )
        self._drain = threading.Thread(target=self._read_err, daemon=True)
        self._drain.start()

    def _read_err(self) -> None:
        for raw in iter(self.proc.stderr.readline, b""):
            self.tail.append(raw.decode("utf-8", "replace").rstrip())

    def kill(self) -> None:
        _kill_tree(self.proc)

    def finish(self, timeout: float | None = None) -> None:
        rc = self.proc.wait(timeout)
        self._drain.join(timeout=2)
        if rc != 0:
            raise ProcessError(self.cmd, rc, self.tail)


class FrameReader:
    def __init__(self, src: Path, info: MediaInfo, *, deinterlace: bool | None = None):
        v = info.video
        if v is None:
            raise ValueError("the source has no video stream")
        ffmpeg = str(tools.exe_path("ffmpeg"))
        rot = abs(v.rotation) % 180 == 90
        self.width, self.height = (v.height, v.width) if rot else (v.width, v.height)
        filters = []
        if deinterlace if deinterlace is not None else v.interlaced:
            filters.append("bwdif=mode=send_frame")
        rng = "pc" if v.color_range == "pc" else "tv"
        filters.append(f"scale=in_color_matrix={source_matrix(v)}:in_range={rng}:flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24")
        self.fps = v.fps_str or "30"
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src),
               "-map", f"0:{v.index}", "-vf", ",".join(filters),
               "-fps_mode", "cfr", "-r", self.fps, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        self.frame_bytes = self.width * self.height * 3
        self._pipe = _Pipe(cmd, stdin=False)

    def __iter__(self) -> Iterator[np.ndarray]:
        out = self._pipe.proc.stdout
        while True:
            buf = bytearray(self.frame_bytes)
            view = memoryview(buf)
            got = 0
            while got < self.frame_bytes:
                n = out.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < self.frame_bytes:
                if got:
                    self._pipe.tail.append(f"truncated final frame ({got} bytes)")
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
        self._pipe.finish()

    def kill(self) -> None:
        self._pipe.kill()


class FrameWriter:
    """Encodes rgb24 frames of size (in_w, in_h), scaled to (out_w, out_h), and muxes the
    source's audio, subtitles, chapters and metadata into *dst*."""

    def __init__(
        self,
        src: Path,
        dst: Path,
        info: MediaInfo,
        *,
        in_size: tuple[int, int],
        out_size: tuple[int, int],
        fps: str,
        encoder_args: list[str],
        pix_fmt: str,
    ):
        ffmpeg = str(tools.exe_path("ffmpeg"))
        container = container_of(str(dst))
        (iw, ih), (ow, oh) = in_size, out_size
        vf = []
        if (iw, ih) != (ow, oh):
            vf.append(f"scale={ow}:{oh}:flags=lanczos")
        vf.append(f"scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format={pix_fmt}")
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{iw}x{ih}", "-framerate", fps, "-i", "pipe:0",
               "-i", str(src), "-map", "0:v:0"]
        tail: list[str] = []
        for out_i, p in enumerate(plan_audio(info, container)):
            cmd += ["-map", f"1:{p.stream.index}"]
            if p.action == "copy":
                tail += [f"-c:a:{out_i}", "copy"]
            else:
                tail += [f"-c:a:{out_i}", p.codec, f"-b:a:{out_i}", "192k"]
        sub_i = 0
        for p in plan_subtitles(info, container):
            if p.action == "drop":
                continue
            cmd += ["-map", f"1:{p.stream.index}"]
            tail += [f"-c:s:{sub_i}", p.codec if p.action == "transcode" else "copy"]
            sub_i += 1
        if container == "mkv":
            cmd += ["-map", "1:t?"]
            tail += ["-c:t", "copy"]
        cmd += ["-vf", ",".join(vf), *encoder_args, *tail,
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-map_metadata", "1", "-map_chapters", "1"]
        if container == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(dst))
        self.command = cmd
        self._pipe = _Pipe(cmd, stdin=True)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._pipe.proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except (BrokenPipeError, OSError):
            # The encoder died; report its own error message instead of the pipe error.
            self._pipe.proc.wait()
            raise ProcessError(self.command, self._pipe.proc.returncode or -1, self._pipe.tail)

    def close(self) -> None:
        self._pipe.proc.stdin.close()
        self._pipe.finish()

    def kill(self) -> None:
        self._pipe.kill()
