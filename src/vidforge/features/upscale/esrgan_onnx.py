"""Run a Real-ESRGAN ONNX model (uint8 NHWC in/out, see esrgan_convert) over video frames.

Frames larger than the tile size are split into equally sized overlapping tiles, so every
inference call has the same input shape (DirectML compiles once per shape). Tiles are placed
in raster order; where a tile overlaps what is already on the canvas, it is cross-faded in with
a linear ramp. Only those thin strips are blended, so the CPU cost stays small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of equally sized tiles covering [0, length); the last one is pulled back
    to end exactly at the edge (so its overlap may be wider than *overlap*)."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def fade_in(n: int) -> np.ndarray:
    """Weights for the incoming tile across an overlap of n samples, rising from ~0 to ~1."""
    return ((np.arange(n, dtype=np.float32) + 0.5) / n) if n > 0 else np.zeros(0, np.float32)


@dataclass
class TilePlan:
    height: int
    width: int
    tile_h: int
    tile_w: int
    ys: list[int]
    xs: list[int]

    @property
    def count(self) -> int:
        return len(self.ys) * len(self.xs)


def plan_tiles(height: int, width: int, tile: int, overlap: int) -> TilePlan:
    th, tw = min(tile, height), min(tile, width)
    return TilePlan(height, width, th, tw, tile_starts(height, th, overlap), tile_starts(width, tw, overlap))


def merge_tiles(plan: TilePlan, tiles: np.ndarray, scale: int) -> np.ndarray:
    """Assemble upscaled tiles (N x sTh x sTw x 3 uint8, raster order) into one frame."""
    if plan.count == 1:
        return tiles[0]
    s = scale
    th, tw = plan.tile_h * s, plan.tile_w * s
    canvas = np.empty((plan.height * s, plan.width * s, 3), np.uint8)
    k = 0
    for iy, y in enumerate(plan.ys):
        oy = (plan.ys[iy - 1] + plan.tile_h - y) * s if iy else 0  # overlap with the row above
        for ix, x in enumerate(plan.xs):
            ox = (plan.xs[ix - 1] + plan.tile_w - x) * s if ix else 0  # overlap with the left tile
            t = tiles[k]
            k += 1
            Y, X = y * s, x * s
            # Fresh area: copy straight in.
            canvas[Y + oy:Y + th, X + ox:X + tw] = t[oy:, ox:]
            # Left strip (full tile height below the top overlap): cross-fade horizontally.
            if ox:
                a = fade_in(ox)[None, :, None]
                dst = canvas[Y + oy:Y + th, X:X + ox]
                dst[:] = (dst * (1 - a) + t[oy:, :ox] * a + 0.5).astype(np.uint8)
            # Top strip (full tile width): cross-fade vertically.
            if oy:
                a = fade_in(oy)[:, None, None]
                dst = canvas[Y:Y + oy, X:X + tw]
                dst[:] = (dst * (1 - a) + t[:oy, :] * a + 0.5).astype(np.uint8)
    return canvas


class EsrganEngine:
    def __init__(
        self,
        model_path: Path,
        *,
        provider: str = "dml",
        device_id: int = 0,
        tile: int = 512,
        overlap: int = 16,
        batch: int = 1,
    ):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        if provider == "dml":
            # DirectML requires these two settings.
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers = [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        if "uint8" not in inp.type:
            raise ValueError(f"{model_path.name}: expected a uint8-IO model (see esrgan_convert)")
        self.input_name = inp.name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.scale = int(meta.get("scale", 4))
        self.tile = tile
        self.overlap = overlap
        self.batch = max(1, batch)
        self._plan: TilePlan | None = None

    def plan_for(self, h: int, w: int) -> TilePlan:
        if self._plan is None or (self._plan.height, self._plan.width) != (h, w):
            self._plan = plan_tiles(h, w, self.tile, self.overlap)
        return self._plan

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """frame: H x W x 3 uint8 RGB -> (scale*H) x (scale*W) x 3 uint8 RGB."""
        h, w, _ = frame.shape
        plan = self.plan_for(h, w)
        if plan.count == 1:
            return self.session.run(None, {self.input_name: frame[None]})[0][0]
        tiles = np.stack([frame[y:y + plan.tile_h, x:x + plan.tile_w] for y in plan.ys for x in plan.xs])
        outs = [
            self.session.run(None, {self.input_name: tiles[i:i + self.batch]})[0]
            for i in range(0, len(tiles), self.batch)
        ]
        return merge_tiles(plan, np.concatenate(outs), self.scale)
