"""Follow faces from frame to frame and steady their landmarks.

Restoring each frame on its own makes faces shimmer: the detector's landmarks wobble by a
pixel or two, so every frame's face crop is aligned a little differently. Here each face keeps
an identity across frames (matched by box overlap) and its landmarks go through a One-Euro
filter (Casiez et al., 2012): strong smoothing while the face is still, little smoothing while
it moves fast, so there is neither jitter nor lag.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .detect import Face


class OneEuro:
    """One-Euro low-pass filter over an array of values."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    face: Face
    landmarks: np.ndarray  # smoothed, frame pixels
    filter: OneEuro
    unit: float = 1.0  # face size when first seen; fixed so size jitter can't move points
    missed: int = 0
    age: int = 0
    memory: dict = field(default_factory=dict)  # per-face state for later stages (e.g. last output)


class FaceTracker:
    def __init__(self, *, max_missed: int = 4, min_iou: float = 0.25, smooth: bool = True):
        self.smooth = smooth
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def reset(self) -> None:
        self.tracks = []

    def update(self, faces: list[Face], dt: float) -> list[Track]:
        """Match this frame's faces to tracks; returns the tracks seen in this frame."""
        pairs = sorted(
            ((iou(t.face.box, f.box), ti, fi) for ti, t in enumerate(self.tracks) for fi, f in enumerate(faces)),
            reverse=True,
        )
        used_t, used_f = set(), set()
        for score, ti, fi in pairs:
            if score < self.min_iou or ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            t, f = self.tracks[ti], faces[fi]
            # Smooth in units of the face's size so the filter reacts the same for small and big
            # faces. The unit is fixed per track: rescaling by each frame's (wobbly) size estimate
            # would itself shake the points.
            t.landmarks = t.filter(f.landmarks / t.unit, dt) * t.unit if self.smooth else f.landmarks.copy()
            t.face, t.missed = f, 0
            t.age += 1
        for fi, f in enumerate(faces):
            if fi not in used_f:
                tr = Track(next(self._ids), f, f.landmarks.copy(), OneEuro(), unit=max(f.size, 1.0))
                tr.filter(f.landmarks / tr.unit, dt)
                self.tracks.append(tr)
                used_t.add(len(self.tracks) - 1)
        seen = []
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.missed += 1
            elif t.missed == 0:
                seen.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return seen


def scene_changed(prev_small: np.ndarray | None, small: np.ndarray, threshold: float = 40.0) -> bool:
    """True on a hard cut: the mean difference between tiny greyscale frames jumps."""
    if prev_small is None:
        return False
    return float(np.mean(np.abs(prev_small.astype(np.int16) - small.astype(np.int16)))) > threshold
