"""Per-frame face enhancement: detect, track, align, restore, steady over time, blend back."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .align import crop_face, face_transform, match_colors, paste_face, soft_mask
from .detect import FaceDetector
from .restore import FaceRestorer
from .track import FaceTracker, scene_changed

CROP = 512


@dataclass
class EnhanceSettings:
    strength: float = 1.0  # 0 = untouched, 1 = fully restored face
    fidelity: float = 0.7  # CodeFormer only: higher keeps more of the original face
    temporal: float = 0.45  # how much of the previous frame's face may carry over (still faces)
    still_threshold: float = 4.0  # crop change (grey levels) at which the carry-over reaches zero
    min_face_px: float = 14.0  # faces smaller than this (in the source) are left alone
    color_match: bool = True
    smooth_landmarks: bool = True  # off only for comparisons


@dataclass
class FrameStats:
    faces: int = 0
    frames: int = 0

    @property
    def per_frame(self) -> float:
        return self.faces / self.frames if self.frames else 0.0


class FaceEnhancer:
    def __init__(self, detector: FaceDetector, restorer: FaceRestorer, settings: EnhanceSettings, fps: float):
        self.detector = detector
        self.restorer = restorer
        self.s = settings
        self.dt = 1.0 / (fps or 30.0)
        self.tracker = FaceTracker(smooth=settings.smooth_landmarks)
        self.mask = soft_mask(CROP)
        self._prev_small: np.ndarray | None = None
        self.stats = FrameStats()

    def process(self, source: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Find faces in *source*, restore them inside *out* (which may be larger) and return it."""
        self.stats.frames += 1
        small = cv2.resize(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        if scene_changed(self._prev_small, small):
            self.tracker.reset()  # a new shot: don't carry faces (or their history) across the cut
        self._prev_small = small

        tracks = self.tracker.update(self.detector.detect(source), self.dt)
        # Faces are cut from the *source* frame, never from an AI-upscaled one: restoration
        # models are trained on real, degraded photos, and an upscaler's smooth output makes
        # them invent texture. The restored face is then mapped onto the (bigger) output.
        to_source = np.diag([source.shape[1] / out.shape[1], source.shape[0] / out.shape[0], 1.0])
        jobs = []
        for t in tracks:
            if t.face.size < self.s.min_face_px:
                continue
            m = face_transform(t.landmarks, CROP)  # source frame -> crop
            if m is None:
                continue
            m_out = m @ to_source  # output frame -> crop
            jobs.append((t, m_out, crop_face(source, m, CROP)))
        if not jobs:
            return out
        restored = self.restorer.restore([c for _, _, c in jobs], self.s.fidelity)
        for (t, m_out, crop), face in zip(jobs, restored):
            face = self._steady(t.memory, crop, face)
            if self.s.color_match:
                face = match_colors(face, crop, self.mask)
            paste_face(out, face, m_out, self.mask, self.s.strength)
        self.stats.faces += len(jobs)
        return out

    def _steady(self, memory: dict, crop: np.ndarray, face: np.ndarray) -> np.ndarray:
        """Blend with this face's previous result when the input barely changed.

        Crops are aligned to the same template every frame, so a still face lands on the same
        pixels and its restorations can be averaged; when the face moves or talks, the input
        crop changes and the carry-over fades out, so there is no ghosting."""
        tiny = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        prev_tiny, prev_face = memory.get("tiny"), memory.get("face")
        result = face.astype(np.float32)
        if prev_tiny is not None and self.s.temporal > 0:
            change = float(np.mean(np.abs(tiny - prev_tiny)))
            k = self.s.temporal * max(0.0, 1.0 - change / self.s.still_threshold)
            if k > 0:
                result = result * (1.0 - k) + prev_face * k
        memory["tiny"], memory["face"] = tiny, result
        return (result + 0.5).astype(np.uint8)
