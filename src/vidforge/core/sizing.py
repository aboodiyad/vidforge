"""Output-size presets shared by the upscaling and face features.

Sizes are computed from the *display* size (pixel aspect and rotation applied), and outputs
always use square pixels with even dimensions.
"""

from __future__ import annotations

# Landscape bounding boxes; for portrait video the box is turned around.
BOXES = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}
FACTORS = {"2x": 2.0, "3x": 3.0, "4x": 4.0}
PRESETS = ("2x", "3x", "4x", *BOXES)
LABELS = {
    "2x": "2× size", "3x": "3× size", "4x": "4× size",
    "1080p": "1080p (Full HD)", "1440p": "1440p (QHD)", "2160p": "2160p (4K UHD)",
}


class SizeError(ValueError):
    pass


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def target_size(display_w: int, display_h: int, preset: str) -> tuple[int, int]:
    if display_w <= 0 or display_h <= 0:
        raise SizeError("source has no valid size")
    if preset in FACTORS:
        f = FACTORS[preset]
    elif preset in BOXES:
        bw, bh = BOXES[preset]
        if display_h > display_w:
            bw, bh = bh, bw
        f = min(bw / display_w, bh / display_h)
    else:
        raise SizeError(f"unknown size preset {preset!r}; choose from {', '.join(PRESETS)}")
    w, h = _even(display_w * f), _even(display_h * f)
    if w <= display_w and h <= display_h:
        raise SizeError(f"the source ({display_w}x{display_h}) is already at or above {LABELS[preset]}")
    return w, h


def parse_custom(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as e:
        raise SizeError(f"custom size must look like 1920x1080, got {text!r}") from e
    if w < 16 or h < 16:
        raise SizeError("custom size is too small")
    return _even(w), _even(h)
