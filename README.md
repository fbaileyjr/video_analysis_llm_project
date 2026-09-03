# Local Video Analyzer

A fully local pipeline for analyzing videos with a small LLM/VLM stack — built
for a 16GB Apple Silicon Mac (e.g. Mac Mini M4). Nothing leaves your machine.

## How it works

```
video.mp4
   │
   ├── ffmpeg → audio.wav → whisper.cpp → transcript.txt
   │
   └── ffmpeg → frames/*.jpg → local vision model (Ollama) → frame_captions.json
                                                                    │
                              transcript + captions ────────────────┘
                                        │
                              local text model (Ollama)
                                        │
                                  summary.md
```

1. **Audio** is pulled out with `ffmpeg` and transcribed with `whisper.cpp`
   (a fast, low-memory local Whisper implementation).
2. **Frames** are sampled every N seconds with `ffmpeg` and each one is
   captioned by a small local vision-language model running in Ollama.
3. A small local text model reads the transcript + all the frame captions
   together and writes a final analysis.

Models run **sequentially**, not simultaneously, to keep peak memory use low
on a 16GB machine.

## Requirements

- macOS on Apple Silicon (tested for M4, works on any M-series chip)
- [Homebrew](https://brew.sh)
- ~10GB free disk space for models

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

This installs `ffmpeg`, `whisper-cpp`, and `ollama`, downloads a Whisper
model, and pulls two small local models:

| Purpose       | Default model | Approx. size |
|---------------|---------------|--------------|
| Vision (frames) | `moondream`   | ~1.7GB       |
| Text (summary)  | `qwen2.5:3b`  | ~2GB         |

These were picked specifically to leave headroom on 16GB of unified memory.
You can swap in larger models (see **Customizing models** below) if a given
video needs better quality and you're willing to trade off speed.

## Usage

```bash
python3 analyze_video.py path/to/video.mp4
```

Optional flags:

```bash
python3 analyze_video.py video.mp4 \
  --interval 5 \                       # seconds between sampled frames (default: 4)
  --vision-model moondream \           # any vision-capable Ollama model
  --text-model qwen2.5:3b \            # any Ollama text model
  --question "Does anyone use a laptop on screen?"
```

Output is written to `<video_name>_analysis/`:

```
video_analysis/
├── audio.wav
├── transcript.txt
├── frames/
│   ├── frame_00001.jpg
│   └── ...
├── frame_captions.json     # timestamped captions for every sampled frame
└── summary.md               # final written analysis + full transcript
```

## Analyzing a whole collection

To run it over every video in a folder:

```bash
for f in /path/to/videos/*.mp4; do
  python3 analyze_video.py "$f"
done
```

Each video gets its own `<name>_analysis/` folder, so results don't overwrite
each other.

## Customizing models

Swap models by name (must already be pulled with `ollama pull <model>`):

- **Better frame captioning, more RAM/time:** `qwen2.5vl:3b` or `minicpm-v`
- **Better summarization, more RAM/time:** `qwen2.5:7b` (~5GB, still fits on
  16GB but leaves less headroom — close other apps first)

Check what's currently pulled with `ollama list`.

## RAM notes for 16GB machines

- The vision model and text model are called separately in this script, so
  they're never both loaded at full weight at the same time — this is the
  main reason it stays comfortable on 16GB.
- If you go up to 7B+ models, watch `Activity Monitor` → Memory Pressure the
  first time you run a new model. If it goes into the red, drop back to a
  smaller model or increase `--interval` so fewer frames are captioned per run.
- Longer transcripts and more frame captions increase the context length fed
  to the final text model — for very long videos, consider raising
  `--interval` (e.g. 10–15s) to keep the synthesis prompt manageable.

## Motion analysis: tracking objects and measuring speed

`analyze_video.py` describes what a vision model *sees* in isolated frames —
it can't reliably tell you how fast something is moving. For that, use
`track_objects.py`, which takes a different approach: it actually detects
and tracks objects frame-by-frame with YOLOv8 + ByteTrack, computes real
pixel-displacement-based speeds, and only then hands the resulting numbers
(not images) to a local LLM to interpret.

```
video.mp4
   │
   ├── YOLOv8 detection (every sampled frame)
   │        │
   │        └── ByteTrack — links the same object across frames
   │                 │
   │                 └── position(t) for each tracked object
   │                          │
   │                          └── speed = pixel distance ÷ time between frames
   │                                   │
   │                                   └── local text model interprets the numbers
   │                                            │
   │                                            └── motion_analysis.md
```

### Usage

```bash
python3 track_objects.py video.mp4
```

Useful flags:

```bash
python3 track_objects.py video.mp4 \
  --fps 10 \                     # frames/sec sampled for tracking (higher = more accurate, slower)
  --classes car,person \         # only track these object types (default: everything YOLO knows)
  --pixels-per-unit 40 \         # calibration: how many pixels = 1 real-world unit
  --unit meters \                # unit name for the calibration
  --question "Which object accelerated the most?"
```

Output goes to `<video_name>_motion_analysis/`:

```
video_motion_analysis/
├── trajectories.json     # per-object positions + computed speeds
└── motion_analysis.md    # LLM's written interpretation of the tracking data
```

### About calibration (getting real speeds, not just pixels/sec)

Without calibration, you still get valid **relative** comparisons ("object 3
moved about twice as fast as object 1") — the numbers are just in
pixels-per-second, not real units.

To get real-world speed (m/s, mph, etc.), measure something of known size in
the footage and compute pixels-per-unit:

```
pixels_per_unit = (pixel length of a known object) / (its real-world length)
```

Example: a car is 180px long on screen and is actually ~4.5m long:

```
180 / 4.5 = 40 pixels per meter → --pixels-per-unit 40 --unit meters
```

This assumes the camera and the moving objects are roughly the same distance
from the lens throughout the clip — if the camera pans/zooms or objects move
significantly toward/away from the camera, a single calibration number will
be increasingly approximate.

### Notes and limits

- **Detection quality drives everything downstream.** If YOLO misses an
  object in some frames, its trajectory will have gaps, which can distort
  speed estimates. `trajectories.json` includes `num_detections` per object
  so you can sanity-check how much data a given speed estimate rests on.
- **`--fps` is the main accuracy/speed trade-off.** 10fps is a reasonable
  default; raise it for fast-moving objects, lower it (e.g. 5) for slower
  scenes to save time on longer videos.
- **The LLM step never looks at frames for this script** — it only reasons
  over the computed trajectory/speed numbers, which avoids the
  hallucinated-motion problem you get from asking a vision model to eyeball
  speed from still images.

## Troubleshooting

- **"Could not reach Ollama"** — run `ollama serve` in a separate terminal
  (or re-run `setup.sh`, which starts it in the background).
- **No transcript produced** — the audio track may be silent, unsupported,
  or below Whisper's detection threshold; check `audio.wav` plays correctly.
- **`whisper-cli` not found** — some Homebrew versions install it as `whisper`
  instead of `whisper-cli`; run `whisper-cli --help` or `whisper --help` to
  check which one is on your `PATH` and edit `analyze_video.py` accordingly.
- **`track_objects.py` fails on import** — run
  `python3 -m pip install --break-system-packages ultralytics opencv-python-headless`
  (also done automatically by `setup.sh`).
- **First run of `track_objects.py` is slow** — YOLOv8n downloads its
  weights (~6MB) the first time it runs; after that it's cached locally.
