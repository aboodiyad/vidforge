"""Headless entry point: `vidforge <command>`."""

from __future__ import annotations

import argparse
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


def cmd_upscale(args) -> int:
    from pathlib import Path

    from .core.jobs import JobQueue, JobState
    from .features.upscale.service import UpscaleRequest, run_upscale

    queue = JobQueue()
    width = 0

    def show(job) -> None:
        nonlocal width
        pct = f"{job.progress * 100:5.1f}%" if job.progress is not None else "  ...  "
        line = f"\r[{pct}] {job.title}: {job.message}"
        print(line.ljust(width), end="", flush=True)
        width = len(line)

    queue.add_listener(show)
    failed = False
    for f in args.files:
        req = UpscaleRequest(
            input=Path(f),
            output=Path(args.output) if args.output and len(args.files) == 1 else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            size=args.size, mode=args.mode, codec=args.codec, quality=args.quality,
            container=args.container, vsr_quality=args.vsr_quality, model=args.model,
            gpu_index=args.gpu, overwrite=args.overwrite,
        )
        job = queue.submit(Path(f).name, lambda ctx, r=req: run_upscale(r, ctx))
        try:
            queue.wait(job)
        except KeyboardInterrupt:
            queue.cancel(job.id)
            queue.wait(job, 30)
        print()
        if job.state == JobState.DONE:
            print(f"  -> {job.result}")
        else:
            failed = True
            print(f"  {job.state.value}: {job.error or ''}")
            if args.verbose:
                print("\n".join(job.log[-30:]))
    queue.shutdown()
    return 1 if failed else 0


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
