"""Find where people are talking, with the Silero voice detector (MIT licence, ONNX, CPU).

Whisper sometimes "hears" words in music or silence. We run Whisper on the whole recording
(so its timings stay true) and afterwards drop any words that fall outside detected speech.
(whisper.cpp's own --vad cuts the silence out before listening, which shifts its word times.)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512  # samples per step (32 ms)
CONTEXT = 64  # Silero v5 looks at the last 64 samples of the previous chunk too


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected 16 kHz mono 16-bit audio")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def speech_probabilities(audio: np.ndarray, model_path: Path) -> np.ndarray:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros(CONTEXT, np.float32)
    sr = np.array(SR, np.int64)
    n = len(audio) // CHUNK
    probs = np.zeros(n, np.float32)
    for i in range(n):
        chunk = audio[i * CHUNK:(i + 1) * CHUNK]
        x = np.concatenate([context, chunk])[None]
        out, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        probs[i] = out[0, 0]
        context = chunk[-CONTEXT:]
    return probs


def regions(probs: np.ndarray, *, threshold: float = 0.5, min_speech: float = 0.25,
            min_silence: float = 0.3, pad: float = 0.15) -> list[tuple[float, float]]:
    """Speech regions in seconds from per-chunk probabilities."""
    step = CHUNK / SR
    raw, start = [], None
    for i, p in enumerate(probs):
        if p >= threshold and start is None:
            start = i
        elif p < threshold - 0.15 and start is not None:  # a little hysteresis
            raw.append((start * step, i * step))
            start = None
    if start is not None:
        raw.append((start * step, len(probs) * step))
    merged: list[tuple[float, float]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    total = len(probs) * step
    return [(max(0.0, s - pad), min(total, e + pad)) for s, e in merged if e - s >= min_speech]


def detect(wav: Path, model_path: Path) -> list[tuple[float, float]]:
    return regions(speech_probabilities(read_wav(wav), model_path))


def inside(t0: float, t1: float, speech: list[tuple[float, float]], slack: float = 0.3) -> bool:
    """True if [t0, t1] overlaps any speech region (widened by *slack* seconds)."""
    return any(t0 < e + slack and t1 > s - slack for s, e in speech)
