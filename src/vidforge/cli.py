"""Headless entry point: `vidforge <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from dataclasses import asdict

from . import __version__
from .core import gpu, media, paths, tools


def _mb(n: int | None) -> str:
    return f"{n / 1e6:.1f} MB" if n else "?"


def _progress_printer(label: str):
    last = [-1]

    def cb(done: int, total: int | None) -> None:
        pct = int(done * 100 / total) if total else -1
        if pct != last[0]:
            last[0] = pct
            shown = f"{pct}%" if pct >= 0 else _mb(done)
            print(f"\r{label}: {shown}   ", end="", flush=True)

    return cb


def cmd_info(args) -> int:
    print(f"VidForge {__version__}")
    print(f"Data folder: {paths.data_root()}")
    print("\nGPUs:")
    gpus = gpu.detect_gpus()
    if not gpus:
        print("  none detected")
    for g in gpus:
        cc = ".".join(map(str, g.compute_cap)) if g.compute_cap else "-"
        vram = f"{g.vram_mb} MiB" if g.vram_mb else "?"
        print(f"  [{g.index}] {g.name} ({g.vendor}) driver {g.driver or '?'}, VRAM {vram}, "
              f"CC {cc}, RTX VSR: {'likely' if g.vsr_capable else 'no'}")
    print("\nTools:")
    for st in tools.all_status():
        where = f"{st.path} [{st.origin}]" if st.found else "not installed"
        ver = f" {st.version}" if st.version else ""
        print(f"  {st.spec.display:<20} ({st.spec.milestone}){ver}: {where}")
    return 0


def cmd_probe(args) -> int:
    info = media.probe(args.file)
    if args.json:
        print(json.dumps(asdict(info), indent=2))
        return 0
    dur = f"{info.duration:.2f}s" if info.duration else "?"
    print(f"{info.path}\n  format {info.format_name}, duration {dur}")
    for s in info.streams:
        if s.kind == "video":
            extra = f"{s.width}x{s.height} {s.fps and round(s.fps, 3)} fps {s.pix_fmt}"
            if s.attached_pic:
                extra += " (cover)"
        elif s.kind == "audio":
            extra = f"{s.channels} ch {s.sample_rate} Hz"
        else:
            extra = ""
        lang = f" [{s.language}]" if s.language else ""
        print(f"  #{s.index} {s.kind:<8} {s.codec:<10} {extra}{lang}{' default' if s.default else ''}")
    return 0


def cmd_tools(args) -> int:
    if args.action == "list":
        for key, spec in tools.TOOLS.items():
            print(f"{key:<8} {spec.display:<20} {spec.source.repo}  (needed from {spec.milestone})")
        return 0
    keys = list(tools.TOOLS) if args.key == "all" else [args.key]
    for key in keys:
        if key not in tools.TOOLS:
            print(f"unknown tool {key!r}; choose from {', '.join(tools.TOOLS)}", file=sys.stderr)
            return 2
        if args.action == "resolve":
            for a in tools.resolve(key):
                print(f"{key}: {a.tag} {a.name} ({_mb(a.size)})\n  {a.url}\n  sha256: {a.sha256 or 'NOT PUBLISHED'}")
        else:  # fetch
            exe = tools.fetch(key, allow_unverified=args.allow_unverified,
                              progress=_progress_printer(f"Downloading {key}"))
            print(f"\nInstalled {key}: {exe}")
    return 0


def _run_jobs(args, make_job) -> int:
    """Run one job per input file in order, with a live progress line and time left."""
    from .core.jobs import JobQueue, JobState

    queue = JobQueue()
    width = 0

    def show(job) -> None:
        nonlocal width
        pct = f"{job.progress * 100:5.1f}%" if job.progress is not None else "  ...  "
        eta = job.eta()
        left = f", {int(eta) // 60}:{int(eta) % 60:02d} left" if eta is not None else ""
        line = f"\r[{pct}] {job.title}: {job.message}{left}"
        print(line.ljust(width), end="", flush=True)
        width = len(line)

    queue.add_listener(show)
    failed = False
    for f in args.files:
        job = queue.submit(Path(f).name, make_job(Path(f)))
        try:
            queue.wait(job)
        except KeyboardInterrupt:
            queue.cancel(job.id)
            queue.wait(job, 30)
        print()
        if job.state == JobState.DONE:
            print(f"  -> {job.result}  ({job.elapsed():.0f}s)")
        else:
            failed = True
            print(f"  {job.state.value}: {job.error or ''}")
            if args.verbose:
                print("\n".join(job.log[-30:]))
    queue.shutdown()
    return 1 if failed else 0


def _out(args):
    single = args.output and len(args.files) == 1
    return (Path(args.output) if single else None), (Path(args.output_dir) if args.output_dir else None)


def cmd_upscale(args) -> int:
    from .features.upscale.service import UpscaleRequest, run_upscale

    output, output_dir = _out(args)

    def make(path: Path):
        req = UpscaleRequest(
            input=path, output=output, output_dir=output_dir, size=args.size, mode=args.mode,
            codec=args.codec, quality=args.quality, container=args.container,
            vsr_quality=args.vsr_quality, model=args.model, gpu_index=args.gpu, overwrite=args.overwrite,
        )
        return lambda ctx: run_upscale(req, ctx)

    return _run_jobs(args, make)


def cmd_faces(args) -> int:
    from .features.face.service import FaceRequest, run_faces

    output, output_dir = _out(args)

    def make(path: Path):
        req = FaceRequest(
            input=path, output=output, output_dir=output_dir, size=args.size, background=args.background,
            model=args.model, strength=args.strength, fidelity=args.fidelity, esrgan_model=args.ai_style,
            codec=args.codec, quality=args.quality, container=args.container, gpu_index=args.gpu,
            overwrite=args.overwrite,
        )
        return lambda ctx: run_faces(req, ctx)

    return _run_jobs(args, make)


def cmd_subs(args) -> int:
    from .features.subtitles.service import SubtitleRequest, run_subtitles

    _, output_dir = _out(args)

    def make(path: Path):
        req = SubtitleRequest(
            input=path, output_dir=output_dir, audio_track=args.track - 1, language=args.language,
            translate=args.english, level=args.level, add_to_video=args.add_to_video, burn_in=args.burn_in,
            gpu_index=args.gpu, overwrite=args.overwrite,
        )
        return lambda ctx: run_subtitles(req, ctx)

    return _run_jobs(args, make)


def cmd_dub(args) -> int:
    from .features.dub.service import DubRequest, run_dub

    _, output_dir = _out(args)

    def make(path: Path):
        req = DubRequest(
            input=path, target=args.to, dub=not args.subtitles_only, add_to_video=args.add_to_video,
            level=args.level, keep_original=not args.no_original, language=args.language,
            audio_track=args.track - 1, output_dir=output_dir, gpu_index=args.gpu, overwrite=args.overwrite,
            keep_temp=args.keep_temp, voice=args.voice, burn_in=args.burn_in,
        )
        return lambda ctx: run_dub(req, ctx)

    return _run_jobs(args, make)


def cmd_addon(args) -> int:
    from .features.dub import addon

    if args.action == "status":
        st = addon.status()
        print(f"voice add-on: {'installed (' + st['device'] + ')' if st else 'not installed'} at {addon.root()}")
        return 0
    if args.action == "remove":
        addon.remove()
        print("voice add-on removed")
        return 0
    from .core.process import CancelToken

    g = gpu.detect_gpus()
    dev = addon.device_for(g[0] if g else None)
    print(f"Installing the voice add-on ({dev}, about {addon.INSTALL_MB[dev] / 1000:.1f} GB) into {addon.root()}")
    addon.install(g[0] if g else None, cancel=CancelToken(),
                  on_log=(print if args.verbose else (lambda line: None)),
                  on_step=lambda i, n, name: print(f"[{i}/{n}] {name}"))
    print("done")
    return 0


def cmd_vsr_probe(args) -> int:
    ok, detail = gpu.probe_vsr()
    state = {True: "OK", False: "FAILED", None: "SKIPPED"}[ok]
    print(f"RTX VSR test: {state} - {detail}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vidforge", description="VidForge command line")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show GPUs and installed tools").set_defaults(func=cmd_info)

    pr = sub.add_parser("probe", help="show streams of a media file")
    pr.add_argument("file")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_probe)

    t = sub.add_parser("tools", help="list, resolve or fetch external tools")
    t.add_argument("action", choices=["list", "resolve", "fetch"])
    t.add_argument("key", nargs="?", default="all")
    t.add_argument("--allow-unverified", action="store_true",
                   help="install even if the release publishes no SHA-256 digest")
    t.set_defaults(func=cmd_tools)

    sub.add_parser("vsr-probe", help="run a 1-second RTX VSR test encode").set_defaults(func=cmd_vsr_probe)

    from .core import sizing
    from .features.upscale.models import QUALITY_MODELS
    from .features.upscale.service import CODECS, CONTAINERS, MODES

    u = sub.add_parser("upscale", help="upscale videos (fast: RTX VSR, quality: Real-ESRGAN)")
    u.add_argument("files", nargs="+")
    u.add_argument("-s", "--size", default="2160p", help=f"{', '.join(sizing.PRESETS)} or WxH (default 2160p)")
    u.add_argument("-m", "--mode", choices=MODES, default="auto")
    u.add_argument("--model", choices=list(QUALITY_MODELS), default="general", help="Quality-mode model")
    u.add_argument("--vsr-quality", type=int, choices=[1, 2, 3, 4], default=4, help="Fast-mode VSR level")
    u.add_argument("-c", "--codec", choices=CODECS, default="hevc")
    u.add_argument("-q", "--quality", choices=["high", "balanced", "small"], default="balanced")
    u.add_argument("--container", choices=CONTAINERS, default="mp4")
    u.add_argument("-o", "--output", help="output file (single input only)")
    u.add_argument("--output-dir")
    u.add_argument("--gpu", type=int, default=0)
    u.add_argument("--overwrite", action="store_true")
    u.add_argument("-v", "--verbose", action="store_true")
    u.set_defaults(func=cmd_upscale)

    fx = sub.add_parser("faces", help="fix blurry faces (GFPGAN or CodeFormer), optionally upscaling too")
    fx.add_argument("files", nargs="+")
    fx.add_argument("--model", choices=["gfpgan", "codeformer"], default="gfpgan",
                    help="gfpgan = natural (default), codeformer = strong repair (non-commercial licence)")
    fx.add_argument("--strength", type=float, default=1.0, help="0.3-1.0: how much faces change")
    fx.add_argument("--fidelity", type=float, default=0.7, help="CodeFormer only: 0 = repair more, 1 = keep face")
    fx.add_argument("-s", "--size", default="keep", help=f"keep (default), {', '.join(sizing.PRESETS)} or WxH")
    fx.add_argument("-b", "--background", choices=["auto", "fast", "quality"], default="auto",
                    help="how to upscale when --size is not keep")
    fx.add_argument("--ai-style", choices=list(QUALITY_MODELS), default="general", help="Best-quality background model")
    fx.add_argument("-c", "--codec", choices=CODECS, default="hevc")
    fx.add_argument("-q", "--quality", choices=["high", "balanced", "small"], default="balanced")
    fx.add_argument("--container", choices=CONTAINERS, default="mp4")
    fx.add_argument("-o", "--output", help="output file (single input only)")
    fx.add_argument("--output-dir")
    fx.add_argument("--gpu", type=int, default=0)
    fx.add_argument("--overwrite", action="store_true")
    fx.add_argument("-v", "--verbose", action="store_true")
    fx.set_defaults(func=cmd_faces)

    sb = sub.add_parser("subs", help="write subtitles (.srt) from the speech in videos")
    sb.add_argument("files", nargs="+")
    sb.add_argument("-l", "--language", default="auto", help="spoken language: auto (default) or a code like en, ar, fr")
    sb.add_argument("--english", action="store_true", help="write English subtitles whatever the spoken language")
    sb.add_argument("--level", choices=["quick", "balanced", "best"], default="balanced")
    sb.add_argument("--add-to-video", action="store_true", help="also save a copy of the video with the subtitles inside")
    sb.add_argument("--burn-in", action="store_true",
                    help="save a copy of the video with the subtitles written onto the picture (always shown; encodes the video again)")
    sb.add_argument("--track", type=int, default=1, help="sound track number (1 = first)")
    sb.add_argument("--output-dir")
    sb.add_argument("-o", "--output", help=argparse.SUPPRESS)
    sb.add_argument("--gpu", type=int, default=0)
    sb.add_argument("--overwrite", action="store_true")
    sb.add_argument("-v", "--verbose", action="store_true")
    sb.set_defaults(func=cmd_subs)

    from .features.dub.service import LEVELS as DUB_LEVELS

    d = sub.add_parser("dub", help="translate the speech in videos: subtitles and a new voice")
    d.add_argument("files", nargs="+")
    d.add_argument("-t", "--to", default="ar", help="language to translate into, e.g. ar, en, fr (default ar)")
    d.add_argument("--subtitles-only", action="store_true", help="only make translated subtitles, no new voice")
    d.add_argument("--add-to-video", action="store_true",
                   help="with --subtitles-only: also save a copy of the video with the subtitles inside")
    d.add_argument("--burn-in", action="store_true",
                   help="save a copy of the video with the subtitles written onto the picture (always shown; encodes the video again)")
    d.add_argument("--level", choices=DUB_LEVELS, default="best",
                   help="translator: best (Gemma 3 12B, default) or quick (Qwen3 8B)")
    d.add_argument("--voice", choices=["chatterbox", "xtts"], default="chatterbox",
                   help="chatterbox (default, MIT) or xtts (XTTS-v2: non-commercial licence, personal use only)")
    d.add_argument("--no-original", action="store_true", help="don't keep the original sound as a second track")
    d.add_argument("-l", "--language", default="auto", help="spoken language if the video was never transcribed")
    d.add_argument("--track", type=int, default=1, help="sound track number (1 = first)")
    d.add_argument("--output-dir")
    d.add_argument("-o", "--output", help=argparse.SUPPRESS)
    d.add_argument("--gpu", type=int, default=0)
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--keep-temp", action="store_true", help="keep the work files (voices, lines) for checking")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_dub)

    a = sub.add_parser("addon", help="install or remove the voice add-on (PyTorch, for dubbing)")
    a.add_argument("action", choices=["status", "install", "remove"])
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_addon)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths.ensure_dirs()
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, LookupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
