"""Find faces with YuNet (a small face detector from the OpenCV project, MIT licence), run
through ONNX Runtime.

The model looks at a 640x640 picture. Its output is a grid at three scales (every 8, 16 and
32 pixels); each grid cell has a face score, a box and five landmarks, all relative to the
cell. We letterbox the frame into 640x640, decode the cells, and drop overlapping boxes.

Landmark order: the eye on the image's left, the eye on the image's right, the nose tip, the
mouth corner on the image's left, the mouth corner on the image's right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

INPUT = 640
STRIDES = (8, 16, 32)


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h (float, frame pixels)
    landmarks: np.ndarray  # 5 x 2 (float, frame pixels)
    score: float

    @property
    def size(self) -> float:
        """Face scale: twice the distance from the eye centre to the mouth centre."""
        eyes = self.landmarks[:2].mean(axis=0)
        mouth = self.landmarks[3:].mean(axis=0)
        return float(np.linalg.norm(eyes - mouth) * 2.0) or float(max(self.box[2], self.box[3]))


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression on x, y, w, h boxes; returns kept indices."""
    order = np.argsort(-scores)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        overlap = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[overlap <= threshold]
    return keep


def decode(outputs: list[np.ndarray], score_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw YuNet outputs (cls x3, obj x3, bbox x3, kps x3) -> boxes, landmarks, scores in 640 space."""
    boxes, lms, scores = [], [], []
    for k, stride in enumerate(STRIDES):
        cls, obj = outputs[k][0, :, 0], outputs[3 + k][0, :, 0]
        bbox, kps = outputs[6 + k][0], outputs[9 + k][0]
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        idx = np.nonzero(score >= score_threshold)[0]
        if idx.size == 0:
            continue
        cols = INPUT // stride
        r, c = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        b = bbox[idx]
        cx, cy = (c + b[:, 0]) * stride, (r + b[:, 1]) * stride
        w, h = np.exp(b[:, 2]) * stride, np.exp(b[:, 3]) * stride
        boxes.append(np.stack([cx - w / 2, cy - h / 2, w, h], axis=1))
        p = kps[idx].reshape(-1, 5, 2)
        lms.append(np.stack([(p[..., 0] + c[:, None]) * stride, (p[..., 1] + r[:, None]) * stride], axis=-1))
        scores.append(score[idx])
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0, 5, 2)), np.zeros(0)
    return np.concatenate(boxes), np.concatenate(lms), np.concatenate(scores)


class FaceDetector:
    def __init__(self, model_path: Path, *, score: float = 0.6, nms_threshold: float = 0.3):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Tiny network: the CPU is fast enough and keeps the GPU free for restoration.
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.score = score
        self.nms_threshold = nms_threshold

    def detect(self, frame_rgb: np.ndarray) -> list[Face]:
        h, w = frame_rgb.shape[:2]
        scale = INPUT / max(h, w)
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = np.zeros((INPUT, INPUT, 3), np.uint8)
        canvas[:sh, :sw] = cv2.resize(frame_rgb, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # YuNet was trained on BGR pixel values 0-255.
        blob = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: blob})
        boxes, lms, scores = decode(outputs, self.score)
        if not len(scores):
            return []
        keep = nms(boxes, scores, self.nms_threshold)
        return [Face(box=boxes[i] / scale, landmarks=(lms[i] / scale).astype(np.float64), score=float(scores[i]))
                for i in keep]
