"""Models for translating and dubbing, each pinned by SHA-256.

- Translation, run by llama.cpp:
  "best": Gemma 3 12B instruct (Gemma Terms of Use), 4-bit GGUF quantized by Unsloth (Google
  only publishes an 8 GB QAT file). 7.3 GB: on an 8 GB card a few layers run on the processor.
  In tests its Arabic was clearly better than Qwen's (no made-up words), at about 5x the time.
  "quick": Qwen3-8B (Apache-2.0), 4-bit GGUF from Qwen's own repository. 5 GB, fits 8 GB cards.
- Voice: Chatterbox Multilingual (MIT) from ResembleAI/chatterbox, pinned to one commit. Hashes
  of the large files are the LFS SHA-256 values Hugging Face publishes; the small tokenizer file
  was checked against its git blob id and then pinned.
- Voice, option: XTTS-v2 (Coqui Public Model License: non-commercial, so personal use only), from
  coqui/XTTS-v2 pinned to one commit; LFS SHA-256 values, the two small JSON files checked against
  their git blob ids and then pinned.
- Separating voices from music: Demucs "htdemucs" (MIT). Facebook's file name carries the first
  8 hex digits of its SHA-256 (8726e21a), which matches the full hash pinned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...core import models
from ...core.process import CancelToken

TRANSLATORS = {"best": "gemma-3-12b-it-q4_k_m", "quick": "qwen3-8b-q4_k_m"}
models.register(models.ModelSpec(
    key=TRANSLATORS["quick"], display="Qwen3-8B (Q4_K_M)",
    url="https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    filename="Qwen3-8B-Q4_K_M.gguf", size_mb=5027.8, license="Apache-2.0", feature="dub",
))
models.register(models.ModelSpec(
    key=TRANSLATORS["best"], display="Gemma 3 12B (Q4_K_M)",
    url="https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
    sha256="15b8fd9d8672cd4240c178c217ca781409291f34e353d2e913b29c7602ceb3ff",
    filename="gemma-3-12b-it-Q4_K_M.gguf", size_mb=7300.8, license="Gemma Terms of Use", feature="dub",
))

VOICE_FOLDER = "chatterbox-mtl"
_CB = "https://huggingface.co/ResembleAI/chatterbox/resolve/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/{}"
VOICE_FILES = [
    ("ve.pt", "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1", 5.7),
    ("t3_mtl23ls_v2.safetensors", "b1237586127ce98e7800a68e49938eb5092846862aabcb6e17b2fda7889a6c75", 2144.0),
    ("s3gen.pt", "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3", 1057.2),
    ("grapheme_mtl_merged_expanded_v1.json", "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf", 0.1),
    ("conds.pt", "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e", 0.1),
]
for _file, _sha, _mb in VOICE_FILES:
    models.register(models.ModelSpec(
        key=f"chatterbox-{_file}", display=f"Chatterbox Multilingual ({_file})", url=_CB.format(_file),
        sha256=_sha, filename=_file, size_mb=_mb, license="MIT", feature="dub", folder=VOICE_FOLDER,
    ))

DEMUCS_KEY = "htdemucs"
DEMUCS_FILE = "955717e8-8726e21a.th"
models.register(models.ModelSpec(
    key=DEMUCS_KEY, display="Demucs htdemucs",
    url=f"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/{DEMUCS_FILE}",
    sha256="8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4",
    filename=DEMUCS_FILE, size_mb=84.1, license="MIT", feature="dub",
))

XTTS_FOLDER = "xtts-v2"
_XT = "https://huggingface.co/coqui/XTTS-v2/resolve/6c2b0d75eae4b7047358e3b6bd9325f857d43f77/{}"
XTTS_FILES = [
    ("config.json", "ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8", 0.1),
    ("vocab.json", "928260878a59da8a72a2a5b7687fea29d5106137669d90945430fe17e415304a", 0.4),
    ("model.pth", "c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187", 1867.9),
    ("speakers_xtts.pth", "f0f6137c19a4eab0cbbe4c99b5babacf68b1746e50da90807708c10e645b943b", 7.8),
]
for _file, _sha, _mb in XTTS_FILES:
    models.register(models.ModelSpec(
        key=f"xtts-{_file}", display=f"XTTS-v2 ({_file})", url=_XT.format(_file), sha256=_sha, filename=_file,
        size_mb=_mb, license="Coqui Public Model License (non-commercial)", feature="dub",
        commercial_ok=False, folder=XTTS_FOLDER,
    ))

# The voices, and the languages each can speak (ISO 639-1).
ENGINES = ("chatterbox", "xtts")
VOICE_LANGUAGES = {
    "chatterbox": {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
                   "pl", "pt", "ru", "sv", "sw", "tr", "zh"},
    "xtts": {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh"},
}
# XTTS's own name for a language when it differs from ISO 639-1.
XTTS_CODES = {"zh": "zh-cn"}

Progress = Callable[[int, int | None], None]


def llm_installed(level: str) -> bool:
    return models.is_installed(TRANSLATORS[level])


def voice_keys(engine: str) -> list[str]:
    """Model files a voice needs. Telling speakers apart always uses Chatterbox's small voice
    encoder (ve.pt), and the background always comes from Demucs."""
    if engine == "xtts":
        own = [f"xtts-{f}" for f, _, _ in XTTS_FILES] + ["chatterbox-ve.pt"]
    else:
        own = [f"chatterbox-{f}" for f, _, _ in VOICE_FILES]
    return own + [DEMUCS_KEY]


def voice_installed(engine: str = "chatterbox") -> bool:
    return all(models.is_installed(k) for k in voice_keys(engine))


def ensure_llm(level: str, *, progress: Progress | None = None, cancel: CancelToken | None = None) -> Path:
    return models.ensure(TRANSLATORS[level], progress=progress, cancel=cancel)


def ensure_voice(engine: str = "chatterbox", *, progress: Progress | None = None,
                 cancel: CancelToken | None = None) -> dict[str, Path]:
    """Download what the voice needs; returns the folders/files by name: "voice" (the chosen
    voice's folder), "encoder" (folder with ve.pt) and "demucs" (the separation model file)."""
    keys = voice_keys(engine)
    total = int(sum(models.MODELS[k].size_mb for k in keys) * 1e6)
    done_before = 0
    for k in keys:
        if not models.is_installed(k):
            cb = (lambda d, t, o=done_before: progress(o + d, total)) if progress else None
            models.ensure(k, progress=cb, cancel=cancel)
        done_before += int(models.MODELS[k].size_mb * 1e6)
    return {"voice": models.model_path(keys[0]).parent, "encoder": models.model_path("chatterbox-ve.pt").parent,
            "demucs": models.model_path(DEMUCS_KEY)}


def download_mb(level: str, *, voice: str | None) -> float:
    """MB still to download; *voice* is the voice engine, or None for subtitles only."""
    mb = 0.0 if llm_installed(level) else models.MODELS[TRANSLATORS[level]].size_mb
    if voice:
        for k in voice_keys(voice):
            if not models.is_installed(k):
                mb += models.MODELS[k].size_mb
    return mb
