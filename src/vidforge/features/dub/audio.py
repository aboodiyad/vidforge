"""Fitting the new voice into the video's time and mixing it over the background sound.

Each spoken line has a *slot*: from the moment the original line starts until the next line
starts. A dubbed line that is too long is sped up (Rubber Band, which keeps the pitch; FFmpeg's
atempo if Rubber Band is missing), but only up to a limit, because faster than about 1.35x
starts to sound rushed. Lines that would need more are first made shorter by the translator.
The voice is matched in loudness to the original speaker, line by line, and added to the
background (music and sound effects with the original voices removed).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core import tools
from ...core.ffmpeg import run_ffmpeg
from ...core.process import CancelToken, capture

MAX_TEMPO = 1.35  # normal speed-up limit
HARD_TEMPO = 1.6  # used only when a line is still too long after being shortened
SR = 44100


# --- WAV files (PCM 16-bit and 32-bit float) ---------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """(samples as float32 shaped [frames, channels], sample rate)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError(f"{path}: missing fmt or data")
    tag, channels, rate, _, _, bits = fmt
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: the real format is in the sub-format GUID
        tag = 3 if bits == 32 and b"\x03\x00\x00\x00\x00\x00\x10\x00" in data[:200] else 1
    if tag == 3 and bits == 32:
        a = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4")
    elif tag == 1 and bits == 16:
        a = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"{path}: unsupported WAV format {tag}/{bits}")
    return a[: len(a) // channels * channels].reshape(-1, channels).astype(np.float32), rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    """32-bit float WAV (keeps headroom; the final encode limits peaks)."""
    s = np.ascontiguousarray(samples, dtype="<f4")
    if s.ndim == 1:
        s = s[:, None]
    ch = s.shape[1]
    raw = s.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, ch, rate, rate * ch * 4, ch * 4, 32)
    header += b"data" + struct.pack("<I", len(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + raw)
    return path


# --- fitting --------------------------------------------------------------------------------

@dataclass
class Fit:
    id: int
    start: float
    slot: float
    seconds: float  # length of the voice before speeding up
    tempo: float = 1.0

    @property
    def fitted(self) -> float:
        return self.seconds / self.tempo


def tempo_for(seconds: float, slot: float, limit: float = MAX_TEMPO) -> float:
    """1.0 when it fits, else the speed-up needed, capped at *limit*."""
    if slot <= 0 or seconds <= slot:
        return 1.0
    return min(seconds / slot, limit)


def needs_shorter(seconds: float, slot: float, limit: float = MAX_TEMPO) -> bool:
    return slot > 0 and seconds / slot > limit


_rubberband: bool | None = None


def has_rubberband() -> bool:
    global _rubberband
    if _rubberband is None:
        try:
            out = capture([str(tools.exe_path("ffmpeg")), "-hide_banner", "-filters"], timeout=20)
            _rubberband = " rubberband " in out.stdout
        except Exception:
            _rubberband = False
    return _rubberband


def tempo_filter(tempo: float) -> str:
    if abs(tempo - 1.0) < 0.01:
        return "anull"
    if has_rubberband():
        return f"rubberband=tempo={tempo:.4f}:pitchq=quality:window=standard"
    # atempo accepts 0.5..2 in one pass, which covers every tempo used here
    return f"atempo={tempo:.4f}"


def stretch(src: Path, dst: Path, tempo: float, *, cancel: CancelToken) -> Path:
    """Speed a clip up by *tempo* and resample it to mono 44.1 kHz float."""
    run_ffmpeg(["-i", src, "-af", f"{tempo_filter(tempo)},aresample={SR}", "-ac", "1", "-c:a", "pcm_f32le", dst],
               cancel=cancel)
    return dst


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def active_rms(x: np.ndarray, rate: int, floor_db: float = -45.0) -> float:
    """Loudness of the parts that actually have sound (pauses would make quiet speech look
    quieter than it is)."""
    win = max(1, int(0.03 * rate))
    n = len(x) // win
    if n == 0:
        return rms(x)
    frames = x[: n * win].reshape(n, win)
    levels = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    loud = levels[20 * np.log10(levels + 1e-12) > floor_db]
    return float(np.sqrt(np.mean(np.square(loud)))) if len(loud) else rms(x)


def place(clips: list[tuple[float, np.ndarray]], total: int, rate: int) -> np.ndarray:
    """Put mono clips into one track at their start times."""
    track = np.zeros(total, dtype=np.float32)
    for start, clip in clips:
        a = max(0, int(round(start * rate)))
        b = min(total, a + len(clip))
        if b > a:
            track[a:b] += clip[: b - a]
    return track


def gain_to_match(clip: np.ndarray, original: np.ndarray, rate: int, fallback: float) -> float:
    """Gain that makes the dubbed line as loud as the original line, within sensible limits."""
    target = active_rms(original, rate)
    if target < 1e-4:
        target = fallback
    have = active_rms(clip, rate)
    if have < 1e-6:
        return 1.0
    return float(np.clip(target / have, 0.25, 4.0))


def mix(background: np.ndarray, voice: np.ndarray) -> np.ndarray:
    """Background (frames, 2) + mono voice on both channels."""
    n = max(len(background), len(voice))
    out = np.zeros((n, 2), dtype=np.float32)
    bg = background if background.shape[1] == 2 else np.repeat(background[:, :1], 2, axis=1)
    out[: len(bg)] += bg
    out[: len(voice)] += voice[:, None]
    return out
