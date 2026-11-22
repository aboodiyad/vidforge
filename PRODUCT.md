# Product

<!-- impeccable:product-schema 1 -->

## Platform

Windows desktop app (Python + PySide6 / Qt Widgets). Not web; the nearest Impeccable value is `web`.

## Users
The owner, plus friends and family who are not technical, including children. They have a video
(old, small or blurry) and want a bigger, sharper copy without learning video jargon. English and
Arabic speakers.

## Product Purpose
Make videos look better on the user's own PC with free, open-source AI: upscaling now; face
enhancement, subtitles, and translation/dubbing later. Success: a person picks a video, understands
what will happen (how big the result is, how long it takes), presses one button, and gets the file.

## Positioning
Runs fully on the user's PC with open-source parts, no account or cloud. Two clear choices for
upscaling: Fast (NVIDIA RTX Video Super Resolution) and Best quality (Real-ESRGAN on any GPU).

## Operating Context
Used occasionally, a few videos at a time. Jobs take seconds to many minutes and run in a queue,
so the user often leaves and comes back. Output is saved next to the source video by default.

## Capabilities and Constraints
- Upscale presets: 2x/3x/4x, 1080p, 1440p, 4K; Fast needs an NVIDIA RTX card, Quality works on any
  DirectX 12 GPU.
- Size and time are estimates: shown before starting, refined from the user's own finished jobs.
- Languages: English and Arabic (right-to-left). First launch follows the Windows language.
- Personal use: non-commercial model licences are allowed as options.

## Product Principles
1. Plain words first: every option says what it does for the person, in a sentence a child gets.
2. Show the outcome before the work: new size, file size and time are visible before pressing start.
3. One obvious path; technical controls live under "More options".
4. Honest estimates: say "about", give ranges, and learn from real runs.
5. Never lose the user's original video.

## Accessibility & Inclusion
Full right-to-left layout for Arabic. Readable text sizes, high contrast on the dark theme, keyboard
reachable controls.
