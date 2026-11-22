"""Build the Windows installer: dist\\VidForge-Setup-<version>.exe

    .venv\\Scripts\\python packaging\\build.py

Steps:
1. Download a portable CPython (python-build-standalone, pinned, SHA-256 checked) into
   build\\app\\python, so people don't need Python installed.
2. Install VidForge and its packages into it (pip, from this checkout).
3. Trim parts never used (tests, Tk, IDLE), pre-compile the .py files, write the icon and a
   list of the third-party parts and their licences.
4. Compile packaging\\vidforge.iss with Inno Setup (ISCC.exe).

The installer puts VidForge in %LOCALAPPDATA%\\Programs\\VidForge (no administrator rights).
Programs and AI models are not included: VidForge downloads them on first use, each checked
against its SHA-256, into vidforge-data next to the app.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
APP = BUILD / "app"
DIST = ROOT / "dist"
CACHE = BUILD / "cache"

PYTHON_TAG = "20261110"
PYTHON_FILE = "cpython-3.12.16+20261110-x86_64-pc-windows-msvc-install_only.tar.gz"
PYTHON_URL = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PYTHON_TAG}/" \
             + PYTHON_FILE.replace("+", "%2B")
PYTHON_SHA256 = "e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b"  # GitHub release digest

TRIM = ["Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "tcl", "DLLs/_tkinter.pyd",
        "DLLs/tcl86t.dll", "DLLs/tk86t.dll", "include"]
# Parts of packages VidForge never loads: Qt's developer tools (Designer, Linguist, QML tools),
# QML modules, Qt's own translations, software OpenGL (Widgets draw without it), headers, and
# OpenCV's built-in video reader (VidForge reads video through FFmpeg).
SP = "Lib/site-packages/"
TRIM += [SP + "PySide6/" + p for p in ("qml", "translations", "opengl32sw.dll", "metatypes", "include",
                                       "typesystems", "glue", "doc", "scripts")]
TRIM += [SP + "cv2/opencv_videoio_ffmpeg510_64.dll"]
TRIM_GLOBS = [SP + "PySide6/*.exe"]


def log(msg: str) -> None:
    print(f"== {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def version() -> str:
    ns: dict = {}
    exec((ROOT / "src" / "vidforge" / "__init__.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def get_python() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / PYTHON_FILE
    if not archive.is_file() or sha256(archive) != PYTHON_SHA256:
        log(f"downloading {PYTHON_FILE}")
        part = archive.with_suffix(".part")
        with urllib.request.urlopen(PYTHON_URL, timeout=120) as r, open(part, "wb") as out:
            shutil.copyfileobj(r, out)
        if sha256(part) != PYTHON_SHA256:
            part.unlink()
            raise SystemExit("Python download failed its SHA-256 check")
        part.replace(archive)
    return archive


def unpack_python(archive: Path) -> Path:
    log("unpacking Python")
    with tarfile.open(archive) as tf:
        tf.extractall(APP, filter="data")  # the archive's top folder is "python"
    return APP / "python" / "python.exe"


def pip_install(python: Path) -> None:
    log("installing VidForge and its packages")
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
    subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-cache-dir", "--no-warn-script-location",
                    str(ROOT)], check=True, env=env)


def trim_and_compile(python: Path) -> None:
    log("trimming and pre-compiling")
    base = python.parent
    for rel in TRIM:
        p = base / rel
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    for pattern in TRIM_GLOBS:
        for p in base.glob(pattern):
            p.unlink()
    # Command shortcuts packages add (Qt tools, test runners); only VidForge's own are kept.
    for p in (base / "Scripts").glob("*.exe"):
        if not p.name.lower().startswith("vidforge"):
            p.unlink()
    for cache in base.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    # Compiled now, so the files exist at install time and the uninstaller removes them.
    subprocess.run([str(python), "-I", "-m", "compileall", "-q", "-j", "0", str(base / "Lib")], check=False)


def vc_runtime(python: Path) -> None:
    """Put Microsoft's C++ runtime next to python.exe. A fresh Windows may not have it, and
    onnxruntime and OpenCV need it; Windows looks in the program's own folder first. PySide6 ships
    these files (redistributable under Microsoft's licence); only missing ones are copied."""
    qt = python.parent / "Lib" / "site-packages" / "PySide6"
    for pattern in ("msvcp140*.dll", "vcruntime140*.dll", "concrt140.dll"):
        for dll in qt.glob(pattern):
            target = python.parent / dll.name
            if not target.exists():
                shutil.copyfile(dll, target)


def notices(python: Path) -> None:
    """THIRD-PARTY-NOTICES.txt: every bundled package with its licence (the full licence texts are
    inside each package's .dist-info folder)."""
    out = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.metadata as m\n"
         "for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()):\n"
         "    md = d.metadata\n"
         "    lic = md.get('License-Expression') or md.get('License') or ''\n"
         "    if not lic or len(lic) > 80:\n"
         "        lic = '; '.join(c.split('::')[-1].strip() for c in md.get_all('Classifier') or [] if c.startswith('License')) or 'see its .dist-info folder'\n"
         "    print(f\"{md['Name']} {md['Version']}: {lic}\")\n"],
        capture_output=True, text=True, check=True).stdout
    text = ("VidForge bundles these parts. Each keeps its own licence; full texts are in\n"
            "python\\Lib\\site-packages\\<package>.dist-info. Python itself: PSF License (python\\LICENSE.txt).\n"
            "Qt for Python (PySide6) is LGPLv3: its libraries are the unmodified PyPI ones and can be replaced.\n"
            "Programs and AI models that VidForge downloads later are listed in README.md.\n\n" + out)
    (APP / "THIRD-PARTY-NOTICES.txt").write_text(text, encoding="utf-8")


def find_iscc() -> Path:
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles(x86)", ""),
                 os.environ.get("ProgramFiles", "")):
        p = Path(base) / ("Programs" if base == os.environ.get("LOCALAPPDATA") else "") / "Inno Setup 6" / "ISCC.exe"
        if p.is_file():
            return p
    found = shutil.which("iscc")
    if found:
        return Path(found)
    raise SystemExit("Inno Setup 6 not found (winget install JRSoftware.InnoSetup)")


def compile_installer(ver: str) -> Path:
    log("building the installer")
    DIST.mkdir(exist_ok=True)
    iss = ROOT / "packaging" / "vidforge.iss"
    subprocess.run([str(find_iscc()), "/Q", f"/DAppVersion={ver}", f"/DSourceDir={APP}", f"/DOutputDir={DIST}",
                    f"/DIconFile={ROOT / 'src' / 'vidforge' / 'gui' / 'assets' / 'vidforge.ico'}", str(iss)],
                   check=True)
    return DIST / f"VidForge-Setup-{ver}.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-installer", action="store_true", help="only build the app folder")
    args = ap.parse_args()
    ver = version()
    if APP.exists():
        shutil.rmtree(APP)
    APP.mkdir(parents=True)
    python = unpack_python(get_python())
    pip_install(python)
    trim_and_compile(python)
    vc_runtime(python)
    shutil.copyfile(ROOT / "src" / "vidforge" / "gui" / "assets" / "vidforge.ico", APP / "vidforge.ico")
    shutil.copyfile(ROOT / "README.md", APP / "README.md")
    notices(python)
    size = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file())
    log(f"app folder: {size / 1e6:.0f} MB")
    if args.skip_installer:
        return 0
    setup = compile_installer(ver)
    log(f"done: {setup} ({setup.stat().st_size / 1e6:.0f} MB, sha256 {sha256(setup)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
