"""User settings stored as JSON. Unknown keys are ignored so old files keep loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import paths


@dataclass
class Settings:
    # "auto" picks fast (RTX VSR) when available, else quality; or force "fast" / "quality".
    language: str = "system"  # system / en / ar
    upscale_mode: str = "auto"
    upscale_size: str = "2160p"
    upscale_model: str = "general"
    upscale_vsr_quality: int = 4
    upscale_codec: str = "hevc"
    upscale_quality: str = "balanced"
    upscale_container: str = "mp4"
    face_model: str = "gfpgan"
    face_strength: float = 1.0
    face_fidelity: float = 0.7
    face_size: str = "keep"
    face_background: str = "auto"
    gpu_index: int = 0
    output_dir: str = ""
    keep_temp_files: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or paths.settings_file()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        path = path or paths.settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)
