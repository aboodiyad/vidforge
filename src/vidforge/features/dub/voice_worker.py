"""Voice work that needs PyTorch. Runs inside the voice add-on's own Python (see addon.py), never
inside VidForge itself, so it imports nothing from VidForge.

    python voice_worker.py job.json

The job file lists the steps to do, any of:
  separate  - split the sound into the voices and everything else (Demucs htdemucs)
  speakers  - tell apart who speaks each line, and cut a short sample of each voice
  tts       - say every translated line in the matching voice (Chatterbox Multilingual)
Progress goes to stdout as lines starting with "@@" followed by JSON; everything else is log
text. The results are written to the "result" path in the job file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every model file comes from VidForge, verified
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

# PyTorch and soundfile load in main(), so VidForge's tests can import the plain helpers here.
sf = None
torch = None


def emit(**kw) -> None:
    print("@@" + json.dumps(kw, ensure_ascii=True), flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_device(wanted: str) -> str:
    return "cuda" if wanted == "cuda" and torch.cuda.is_available() else "cpu"


# --- separate ----------------------------------------------------------------------------

def separate(job: dict, device: str) -> dict:
    from demucs.apply import apply_model
    from demucs.states import load_model

    emit(stage="separate", progress=0.0)
    model = load_model(job["model"])
    model.to(device).eval()
    wav, sr = sf.read(job["input"], dtype="float32", always_2d=True)  # (frames, channels)
    if sr != model.samplerate:
        raise SystemExit(f"expected {model.samplerate} Hz audio, got {sr}")
    mix = torch.from_numpy(wav.T.copy())
    if mix.shape[0] == 1:
        mix = mix.repeat(2, 1)
    mix = mix[:2]
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std().clamp_min(1e-8)
    # Work in pieces so progress can be shown; each piece carries a second of overlap on both
    # sides that is thrown away, so the joins are clean.
    piece, pad = 30 * sr, 1 * sr
    n = mix.shape[1]
    vocals = torch.zeros_like(mix)
    voc_i = model.sources.index("vocals")
    starts = list(range(0, n, piece))
    with torch.inference_mode():
        for k, s in enumerate(starts):
            a, b = max(0, s - pad), min(n, s + piece + pad)
            chunk = ((mix[:, a:b] - mean) / std)[None].to(device)
            out = apply_model(model, chunk, device=device, split=True, overlap=0.25, progress=False)[0]
            v = out[voc_i].cpu() * std + mean
            e = min(n, s + piece)
            vocals[:, s:e] = v[:, s - a:e - a]
            emit(stage="separate", progress=(k + 1) / len(starts))
    background = mix - vocals
    sf.write(job["vocals"], vocals.numpy().T, sr, subtype="FLOAT")
    sf.write(job["background"], background.numpy().T, sr, subtype="FLOAT")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"vocals": job["vocals"], "background": job["background"]}


# --- speakers ----------------------------------------------------------------------------

def _load_mono(path: str, target_sr: int) -> np.ndarray:
    import librosa

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    return librosa.resample(mono, orig_sr=sr, target_sr=target_sr) if sr != target_sr else mono


def cluster(embeds: np.ndarray, durations: list[float], threshold: float, min_found: float = 1.5) -> list[int]:
    """Greedy grouping by voice: the longest lines found the groups (they give the most reliable
    voice print), shorter ones join the closest group."""
    order = sorted(range(len(embeds)), key=lambda i: -durations[i])
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels = [0] * len(embeds)
    for i in order:
        e = embeds[i]
        sims = [float(e @ (c / np.linalg.norm(c))) for c in centroids]
        best = int(np.argmax(sims)) if sims else -1
        if best >= 0 and (sims[best] >= threshold or durations[i] < min_found):
            labels[i] = best
            centroids[best] = centroids[best] + e
            counts[best] += 1
        else:
            labels[i] = len(centroids)
            centroids.append(e.copy())
            counts.append(1)
    # Number speakers in order of first appearance.
    remap: dict[int, int] = {}
    for lb in labels:
        remap.setdefault(lb, len(remap))
    return [remap[lb] for lb in labels]


def speakers(job: dict, device: str, ve=None) -> dict:
    emit(stage="speakers", progress=0.0)
    if ve is None:
        from chatterbox.models.voice_encoder import VoiceEncoder

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(Path(job["model_dir"]) / "ve.pt", map_location="cpu", weights_only=True))
        ve.to(device).eval()
    sr = 16000
    wav16 = _load_mono(job["vocals"], sr)
    segs = job["segments"]
    clips, durs = [], []
    for s, e in segs:
        c = wav16[int(s * sr):int(e * sr)]
        durs.append(len(c) / sr)
        if len(c) < int(1.6 * sr):  # the voice encoder looks at 1.6 s windows
            c = np.tile(c if len(c) else np.zeros(sr // 10, dtype=np.float32), int(1.6 * sr / max(1, len(c))) + 1)[: int(1.6 * sr)]
        clips.append(c.astype(np.float32))
    with torch.inference_mode():
        embeds = np.asarray(ve.embeds_from_wavs(clips, sample_rate=sr), dtype=np.float32)
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    labels = cluster(embeds, durs, job.get("threshold", 0.75)) if len(segs) else []
    emit(stage="speakers", progress=0.6)

    # A voice sample per speaker: their longest lines joined, up to 10 seconds.
    ref_sr = 24000
    wav24 = _load_mono(job["vocals"], ref_sr)
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    gap = np.zeros(int(0.15 * ref_sr), dtype=np.float32)
    for spk in sorted(set(labels)):
        idx = sorted((i for i, lb in enumerate(labels) if lb == spk), key=lambda i: -durs[i])
        parts, total = [], 0.0
        for i in idx:
            s, e = segs[i]
            c = wav24[int(s * ref_sr):int(e * ref_sr)]
            parts += [c, gap]
            total += len(c) / ref_sr
            if total >= 10:
                break
        ref = np.concatenate(parts)[: 10 * ref_sr]
        peak = float(np.abs(ref).max()) or 1.0
        ref = ref / peak * 0.9
        p = out_dir / f"speaker{spk}.wav"
        sf.write(str(p), ref, ref_sr, subtype="PCM_16")
        refs[str(spk)] = str(p)
    emit(stage="speakers", progress=1.0)
    return {"labels": labels, "refs": refs, "embeddings": len(embeds)}


# --- tts ---------------------------------------------------------------------------------

def _trim(wav: np.ndarray, sr: int, floor_db: float = -42.0) -> np.ndarray:
    """Cut silence at both ends (keeps 40 ms around the speech)."""
    if not len(wav):
        return wav
    win = int(0.02 * sr)
    n = len(wav) // win
    if n == 0:
        return wav
    frames = wav[: n * win].reshape(n, win)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    loud = np.where(db > max(floor_db, db.max() - 40))[0]
    if not len(loud):
        return wav[:0]
    keep = int(0.04 * sr)
    a = max(0, loud[0] * win - keep)
    b = min(len(wav), (loud[-1] + 1) * win + keep)
    return wav[a:b]


def tts(job: dict, device: str) -> dict:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    emit(stage="tts", progress=0.0, msg="loading voice")
    t0 = time.monotonic()
    model = ChatterboxMultilingualTTS.from_local(job["model_dir"], device)
    log(f"voice model loaded in {time.monotonic() - t0:.1f}s on {device}")
    results: dict = {}
    if job.get("speakers"):
        sp = dict(job["speakers"], model_dir=job["model_dir"])
        results["speakers"] = speakers(sp, device, ve=model.ve)
        refs = results["speakers"]["refs"]
        labels = results["speakers"]["labels"]
        for line, lb in zip(job["lines"], labels):
            line.setdefault("speaker", lb)
    else:
        refs = job.get("refs", {})
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = job["lines"]
    conds: dict[str, object] = {}
    done: list[dict] = []
    total_chars = sum(len(x["text"]) for x in lines) or 1
    chars = 0
    lang = job["language"]
    order = sorted(range(len(lines)), key=lambda i: (str(lines[i].get("speaker", 0)), i))
    t_start = time.monotonic()
    for k in order:
        line = lines[k]
        spk = str(line.get("speaker", 0))
        if spk not in conds:
            ref = refs.get(spk) or next(iter(refs.values()), None)
            if ref:
                model.prepare_conditionals(ref, exaggeration=job.get("exaggeration", 0.5))
            conds[spk] = model.conds
        model.conds = conds[spk]
        expect = line.get("expect", 0) or len(line["text"]) / 12.0
        wav = None
        for attempt in range(3):
            torch.manual_seed(1234 + int(line["id"]) * 7 + attempt)
            with torch.inference_mode():
                out = model.generate(line["text"], language_id=lang, exaggeration=job.get("exaggeration", 0.5),
                                     cfg_weight=job.get("cfg_weight", 0.0), temperature=0.8)
            take = _trim(out.squeeze(0).cpu().numpy().astype(np.float32), model.sr)
            if len(take) > 0.2 * model.sr and (wav is None or len(take) < len(wav)):
                wav = take  # of odd takes, the shortest one babbles least
            # A run-on take (the model talking on past the end) is redone, at most twice.
            if wav is not None and len(wav) / model.sr <= max(2.0 * expect, expect + 3):
                break
            log(f"line {line['id']}: odd length {len(take) / model.sr:.1f}s for {expect:.1f}s expected, redoing")
        if wav is None:
            wav = np.zeros(int(0.3 * model.sr), dtype=np.float32)
        p = out_dir / f"line{int(line['id']):04d}.wav"
        sf.write(str(p), wav, model.sr, subtype="PCM_16")
        done.append({"id": line["id"], "path": str(p), "seconds": len(wav) / model.sr, "speaker": spk})
        chars += len(line["text"])
        emit(stage="tts", progress=chars / total_chars, line=line["id"])
    results["lines"] = sorted(done, key=lambda d: d["id"])
    results["tts_seconds"] = time.monotonic() - t_start
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def main(argv: list[str]) -> int:
    global sf, torch
    import soundfile as sf
    import torch

    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    device = pick_device(job.get("device", "cuda"))
    log(f"torch {torch.__version__}, device {device}")
    result: dict = {"device": device}
    t = time.monotonic()
    if job.get("separate"):
        result["separate"] = separate(job["separate"], device)
        result["separate_seconds"] = time.monotonic() - t
    if job.get("speakers") and not job.get("tts"):
        result["speakers"] = speakers(job["speakers"], device)
    if job.get("tts"):
        if job.get("speakers"):
            job["tts"]["speakers"] = job["speakers"]
        result.update(tts(job["tts"], device))
    Path(job["result"]).write_text(json.dumps(result), encoding="utf-8")
    emit(stage="done", progress=1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
