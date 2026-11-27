# VidForge

A Windows app for making videos look better, built from open-source parts.

**Works now**
- **Upscale** videos to 1080p, 1440p, 4K or 2×/3×/4×:
  - **Fast mode**: NVIDIA RTX Video Super Resolution (needs an RTX 20-series or newer card).
  - **Quality mode**: Real-ESRGAN AI upscaling on any GPU (NVIDIA, AMD or Intel). Slower, more detail.
  - Keeps audio tracks, subtitles and chapters.
  - Shows the new size, the file size and how long it will take **before** you start, and the
    time left while it works. The guesses get better with every video you finish.
- **Fix faces** in videos: finds every face, follows it from frame to frame, and makes it clear and
  sharp with GFPGAN (natural) or CodeFormer (strong repair, free for personal use only). It can
  make the video bigger at the same time.
- **Write the words**: listens to the video and writes subtitles (.srt) in the spoken language or
  in English, with each line at the right moment. Can also add them inside a copy of the video,
  as a track you turn on and off, or **written onto the picture** so they show everywhere (phones,
  WhatsApp, TV); that copy is encoded again, so it takes longer.
- **Translate & dub**: turns the speech into another language. Translated subtitles always; and if
  you want, a new voice: every person speaks the new language in a voice like their own, while the
  music and sounds stay. The result is a copy of the video with the new voice as the main sound,
  the original sound as a second track, and the translated subtitles as a track you can turn on
  (or written onto the picture, if you tick that).
  Translation runs on your PC with a local AI (Gemma 3 12B for "Best", Qwen3 8B for "Quick").
  Two voice makers: **Natural** (Chatterbox, the default) and **XTTS** (XTTS-v2: about twice as
  fast; its licence allows personal use only, not selling).
  The voice needs a one-time **voice add-on** (about 5.5 GB with an NVIDIA card), offered on that page.
- **English and Arabic** (right-to-left). Change it in **Settings**.

## Download and install

You need **Windows 10 or 11 (64-bit)** and an internet connection for the first start.

1. Go to the [**Releases** page](https://github.com/aboodiyad/vidforge/releases/latest) and download
   **`VidForge-Setup-<version>.exe`** (about 140 MB).
2. Double-click it. If Windows shows "Windows protected your PC", click **More info**, then
   **Run anyway** (VidForge is free and not paid-signed, so Windows doesn't know it yet).
   No administrator rights are needed; it installs for you only.
3. Start **VidForge** from the Start menu (or the desktop icon). The first time, it downloads the
   free programs it needs (FFmpeg, and NVEncC on RTX cards: about 500 MB) and then opens.
4. Pick a page on the left (Make bigger, Fix faces, Write the words, Translate & dub), add a video,
   and press **Start**. Progress shows in **Your list** at the bottom. Results are saved next to
   the video; your original is never changed.

AI models download the first time each part needs them, and are kept in the `vidforge-data`
folder inside the install folder. To remove VidForge, use **Settings > Apps** in Windows; it asks
whether to delete those downloads too.

### Run from the source code (for developers)

Download the code (green **Code** button > **Download ZIP**, then extract it) and double-click
**`Start VidForge.bat`**. The first start creates a Python environment in that folder (needs
Python 3.12; it offers to install it). To build the installer yourself:

```powershell
winget install JRSoftware.InnoSetup
.venv\Scripts\python packaging\build.py
```

### Tips
- **Auto** mode picks Fast on RTX cards and Quality on everything else.
- Quality mode downloads its AI model (5–67 MB) the first time you use it.
- Quality mode is much slower than Fast: start with a short clip.
- **NVIDIA speed-up** (RTX 20 series and newer): on the Make bigger page, Best quality offers a
  one-time download (about 350–450 MB, only the parts your card needs) that runs the AI on
  NVIDIA's TensorRT. On an RTX 2060 SUPER it made Best quality about 2× faster with the same
  pictures; Fix faces uses it too when it makes the background bigger with Best quality.
- **Updates:** VidForge shows a "New version" button when one is out. Download it and install it
  over the old one: programs, AI files and settings you already have are kept.
- The **System** page shows your GPU and can test whether RTX Video Super Resolution works.
- Translate & dub downloads its AI files the first time: the translator (7.3 GB for Best, 5 GB for
  Quick) and the voice models (3.3 GB). A new voice takes about twice as long as the video on an
  RTX 2060 SUPER; subtitles only take a minute or two. It needs a card with 8 GB of video memory
  to run at full speed; with less it still works, more slowly.
- The Natural voice speaks 23 languages (Arabic, English, French, Spanish, German, Turkish, Russian,
  Hindi, Chinese, Japanese and more); XTTS speaks 17, including Arabic, Czech and Hungarian.
  Subtitles can be translated into any language in the list.
- XTTS downloads its own voice files (1.9 GB) the first time you pick it.

## Command line

In an installed copy, use VidForge's own Python, for example
`& "$env:LOCALAPPDATA\Programs\VidForge\python\python.exe" -m vidforge upscale my_video.mp4`.
From the source code:

```powershell
.venv\Scripts\vidforge upscale my_video.mp4 --size 2160p --mode auto
.venv\Scripts\vidforge upscale --help
.venv\Scripts\vidforge faces my_video.mp4 --model gfpgan --size keep
.venv\Scripts\vidforge faces --help
.venv\Scripts\vidforge subs my_video.mp4 --level balanced
.venv\Scripts\vidforge subs my_video.mp4 --burn-in      # subtitles written onto the picture
.venv\Scripts\vidforge subs --help
.venv\Scripts\vidforge addon install          # the voice add-on, once
.venv\Scripts\vidforge dub my_video.mp4 --to ar
.venv\Scripts\vidforge dub my_video.mp4 --to fr --subtitles-only --add-to-video
.venv\Scripts\vidforge dub --help
```

## Credits
- [NVEncC](https://github.com/rigaya/NVEnc) by rigaya (MIT), using NVIDIA RTX Video Super Resolution
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) models by Xintao Wang et al. (BSD-3-Clause)
- [GFPGAN](https://github.com/TencentARC/GFPGAN) (Apache-2.0) and [CodeFormer](https://github.com/sczhou/CodeFormer)
  (S-Lab License 1.0, non-commercial) face models, [YuNet](https://github.com/opencv/opencv_zoo) face detector (MIT),
  as ONNX files from [FaceFusion's model releases](https://github.com/facefusion/facefusion-assets)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with OpenAI Whisper models (MIT) and the
  [Silero](https://github.com/snakers4/silero-vad) voice detector (MIT)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) with [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)
  (Gemma Terms of Use; GGUF by [Unsloth](https://huggingface.co/unsloth/gemma-3-12b-it-GGUF)) and
  [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF) (Apache-2.0)
- [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) voice by Resemble AI (MIT; its
  output carries Resemble's inaudible Perth watermark), [Demucs](https://github.com/facebookresearch/demucs)
  voice separation (MIT), [PyTorch](https://pytorch.org) (BSD)
- [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) by Coqui (Coqui Public Model License, non-commercial),
  run with the [coqui-tts](https://github.com/idiap/coqui-ai-TTS) library (MPL-2.0)
- [OpenCV](https://opencv.org) (Apache-2.0); interface font IBM Plex Sans Arabic (SIL Open Font License)
- [FFmpeg](https://ffmpeg.org) (GPL build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds))
- [ONNX Runtime](https://onnxruntime.ai) with DirectML, [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/)

Tools and models are downloaded from their official releases and checked against SHA-256 checksums.
