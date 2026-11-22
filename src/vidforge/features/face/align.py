"""Cut a face out to the standard 512x512 pose the restoration models expect, and put the
restored face back where it came from with a soft edge.

The template is the widely used 5-point face layout of the FFHQ dataset at 512 pixels (eyes,
nose tip, mouth corners), which GFPGAN and CodeFormer were trained on.
"""

from __future__ import annotations

import cv2
import numpy as np

FFHQ_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float64)


def face_transform(landmarks: np.ndarray, size: int = 512) -> np.ndarray:
    """2x3 similarity transform (rotation, uniform scale, shift) from frame to face crop."""
    template = FFHQ_512 * (size / 512.0)
    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), template.astype(np.float32),
                                       method=cv2.LMEDS)
    if m is None:  # degenerate landmarks: fall back to a least-squares fit
        m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), template.astype(np.float32))
    return m


def crop_face(frame: np.ndarray, m: np.ndarray, size: int = 512) -> np.ndarray:
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


_MASKS: dict[tuple[int, float], np.ndarray] = {}


def soft_mask(size: int = 512, feather: float = 0.07) -> np.ndarray:
    """Soft oval over the face itself (forehead to chin, cheek to cheek) in crop space.

    Only the face is replaced; hair and background stay from the original frame. Restored
    hair and background never quite match their surroundings, and a square paste shows them
    as a visible box."""
    key = (size, feather)
    if key not in _MASKS:
        s = size / 512.0
        m = np.zeros((size, size), np.float32)
        # Oval in FFHQ coordinates: centred a little below the eyes, wide enough for the cheeks.
        cv2.ellipse(m, (int(256 * s), int(300 * s)), (int(158 * s), int(196 * s)), 0, 0, 360, 1.0, -1)
        f = max(2, int(size * feather))
        m = cv2.GaussianBlur(m, (f * 4 + 1, f * 4 + 1), f)
        _MASKS[key] = np.clip(m * 1.15, 0, 1)  # keep the middle fully opaque after blurring
    return _MASKS[key]


def match_colors(restored: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift the restored face's mean colour (in Lab) to the original's, so skin tone and
    lighting stay as they were. Only the means move; the restored detail stays."""
    w = mask > 0.5
    if not np.any(w):
        return restored
    r = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)
    o = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    shift = o[w].mean(axis=0) - r[w].mean(axis=0)
    r += shift
    return cv2.cvtColor(np.clip(r, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def paste_face(frame: np.ndarray, face: np.ndarray, m: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> None:
    """Blend *face* (crop space) back into *frame* (in place), touching only the face region."""
    h, w = frame.shape[:2]
    size = face.shape[0]
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float64) @ inv.T
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    shifted = inv.copy()
    shifted[:, 2] -= (x0, y0)
    rw, rh = x1 - x0, y1 - y0
    face_roi = cv2.warpAffine(face, shifted, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(mask, shifted, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)[..., None] * strength
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi + (face_roi.astype(np.float32) - roi) * alpha + 0.5).astype(np.uint8)
