"""Run a face-restoration network (GFPGAN or CodeFormer, ONNX) on aligned 512x512 crops.

Both models take RGB in [-1, 1], N x 3 x 512 x 512, and return the same. CodeFormer has one
more input: the fidelity weight w (0 = strongest repair, 1 = closest to the input face).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ONNX_TYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


class FaceRestorer:
    def __init__(self, model_path: Path, *, provider: str = "dml", device_id: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        if provider == "dml":
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        inputs = self.session.get_inputs()
        image = next(i for i in inputs if len(i.shape) == 4)
        self.image_name = image.name
        self.image_type = _ONNX_TYPES.get(image.type, np.float32)
        self.size = image.shape[2] if isinstance(image.shape[2], int) else 512
        others = [i for i in inputs if i is not image]
        self.weight = (others[0].name, _ONNX_TYPES.get(others[0].type, np.float64), others[0].shape) if others else None
        self.batch_fixed = isinstance(image.shape[0], int)

    @property
    def has_fidelity(self) -> bool:
        return self.weight is not None

    def restore(self, crops: list[np.ndarray], fidelity: float = 0.7) -> list[np.ndarray]:
        """crops: list of size x size x 3 uint8 RGB -> restored, same format."""
        if not crops:
            return []
        if self.batch_fixed or len(crops) == 1:
            return [self._run(np.stack([c]), fidelity)[0] for c in crops]
        return list(self._run(np.stack(crops), fidelity))

    def _run(self, batch: np.ndarray, fidelity: float) -> np.ndarray:
        x = batch.transpose(0, 3, 1, 2).astype(np.float32) * (2.0 / 255.0) - 1.0
        feed = {self.image_name: x.astype(self.image_type)}
        if self.weight:
            name, dtype, shape = self.weight
            feed[name] = np.array(fidelity if not shape else [fidelity], dtype=dtype)
        y = self.session.run(None, feed)[0].astype(np.float32)
        y = (np.clip(y, -1.0, 1.0) + 1.0) * 127.5
        return (y + 0.5).astype(np.uint8).transpose(0, 2, 3, 1)
