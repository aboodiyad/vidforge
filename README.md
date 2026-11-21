# VidForge

A Windows app for making videos look better, built from open-source parts.

**Works now**
- **Upscale** videos to 1080p, 1440p, 4K or 2×/3×/4×:
  - **Fast mode**: NVIDIA RTX Video Super Resolution (needs an RTX 20-series or newer card).
  - **Quality mode**: Real-ESRGAN AI upscaling on any GPU (NVIDIA, AMD or Intel). Slower, more detail.
  - Keeps audio tracks, subtitles and chapters.

**Coming next**: face enhancement, subtitle generation, translation and dubbing.

## Try it

You need **Windows 10 or 11** and an internet connection for the first start.

1. On this page, click the green **Code** button, then **Download ZIP**.
2. Right-click the downloaded ZIP, choose **Extract All…**, and extract it to a normal folder
   (for example your Desktop). Don't run it from inside the ZIP.
3. Open the extracted folder and double-click **`Start VidForge.bat`**.
   - If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**.
   - The first start installs everything into that folder (Python packages about 1 GB,
     FFmpeg and NVEncC about 500 MB). It takes a few minutes. If Python 3.12 is missing,
     it offers to install it.
4. When the VidForge window opens: go to **Upscale**, add a video, pick a size, click **Upscale**.
   Progress shows in the **Jobs** panel at the bottom. The result is saved next to the video.

Later, just double-click **`Start VidForge.bat`** again.

### Tips
- **Auto** mode picks Fast on RTX cards and Quality on everything else.
- Quality mode downloads its AI model (5–67 MB) the first time you use it.
- Quality mode is much slower than Fast: start with a short clip.
- The **System** page shows your GPU and can test whether RTX Video Super Resolution works.

## Command line

```powershell
.venv\Scripts\vidforge upscale my_video.mp4 --size 2160p --mode auto
.venv\Scripts\vidforge upscale --help
```

## Credits
- [NVEncC](https://github.com/rigaya/NVEnc) by rigaya (MIT), using NVIDIA RTX Video Super Resolution
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) models by Xintao Wang et al. (BSD-3-Clause)
- [FFmpeg](https://ffmpeg.org) (GPL build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds))
- [ONNX Runtime](https://onnxruntime.ai) with DirectML, [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/)

Tools and models are downloaded from their official releases and checked against SHA-256 checksums.
