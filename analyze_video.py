#!/usr/bin/env python3
"""
analyze_video.py — local video analysis pipeline for a 16GB Apple Silicon Mac.

Pipeline:
  1. Extract audio from the video (ffmpeg) and transcribe it (whisper.cpp)
  2. Extract frames at a fixed interval (ffmpeg)
  3. Caption each frame with a local vision model (via Ollama)
  4. Feed the transcript + frame captions to a local text model (via Ollama)
     to produce a final written analysis/summary

Everything runs locally. No data leaves your machine.

Usage:
  python3 analyze_video.py video.mp4
  python3 analyze_video.py video.mp4 --interval 5 --question "Does anyone use a laptop on screen?"

Run setup.sh first to install dependencies and pull the default models.
"""

import argparse
import base64
import json
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_WHISPER_MODEL = str(Path.home() / ".whisper-models" / "ggml-base.en.bin")


def run(cmd, **kwargs):
    """Run a subprocess command, raising a clear error if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    audio_path = out_dir / "audio.wav"
    print("→ Extracting audio...")
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(audio_path),
    ])
    return audio_path


def transcribe_audio(audio_path: Path, out_dir: Path, whisper_model: str) -> str:
    print("→ Transcribing audio with whisper.cpp...")
    out_prefix = out_dir / "transcript"
    run([
        "whisper-cli",
        "-m", whisper_model,
        "-f", str(audio_path),
        "-otxt",
        "-of", str(out_prefix),
    ])
    transcript_file = out_prefix.with_suffix(".txt")
    if not transcript_file.exists():
        print("Warning: no transcript produced (silent audio track?).", file=sys.stderr)
        return ""
    return transcript_file.read_text().strip()


def extract_frames(video_path: Path, out_dir: Path, interval_seconds: int) -> list[Path]:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    print(f"→ Extracting one frame every {interval_seconds}s...")
    fps = f"1/{interval_seconds}"
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "frame_%05d.jpg"),
    ])
    return sorted(frames_dir.glob("frame_*.jpg"))


def ollama_generate(model: str, prompt: str, image_path: Path | None = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if image_path is not None:
        payload["images"] = [base64.b64encode(image_path.read_bytes()).decode()]

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"Could not reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        print("Is `ollama serve` running? Run setup.sh or `ollama serve` in another terminal.", file=sys.stderr)
        sys.exit(1)


def caption_frames(frames: list[Path], vision_model: str, interval_seconds: int) -> list[dict]:
    captions = []
    print(f"→ Captioning {len(frames)} frames with {vision_model}...")
    for i, frame in enumerate(frames):
        timestamp = i * interval_seconds
        caption = ollama_generate(
            vision_model,
            "Describe what is visible in this video frame in one or two concise sentences. "
            "Mention people, objects, text on screen, and setting if relevant.",
            image_path=frame,
        )
        captions.append({"timestamp_sec": timestamp, "frame": frame.name, "caption": caption})
        print(f"   [{timestamp:>5}s] {caption}")
    return captions


def synthesize(transcript: str, captions: list[dict], text_model: str, question: str | None) -> str:
    print(f"→ Synthesizing final analysis with {text_model}...")
    caption_lines = "\n".join(
        f"- [{c['timestamp_sec']}s] {c['caption']}" for c in captions
    )
    prompt = f"""You are analyzing a video using its audio transcript and a series of
timestamped visual descriptions sampled from the footage.

TRANSCRIPT:
{transcript or "(no speech detected)"}

VISUAL DESCRIPTIONS (timestamped):
{caption_lines}

Write a clear, well-organized analysis of the video covering: what happens overall,
the setting, notable visual and spoken content, and how the visuals and speech relate
to each other over time. Note any places where the visual and audio content seem to
contradict or don't match, since one of them may be mistaken.
"""
    if question:
        prompt += f"\nAlso specifically answer this question: {question}\n"

    return ollama_generate(text_model, prompt)


def main():
    parser = argparse.ArgumentParser(description="Analyze a video locally using local LLMs.")
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument("--interval", type=int, default=4, help="Seconds between sampled frames (default: 4)")
    parser.add_argument("--vision-model", default="moondream", help="Ollama vision model (default: moondream)")
    parser.add_argument("--text-model", default="qwen2.5:3b", help="Ollama text model (default: qwen2.5:3b)")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Path to whisper.cpp .bin model")
    parser.add_argument("--question", default=None, help="Optional specific question to answer about the video")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <video_name>_analysis)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or Path(f"{args.video.stem}_analysis")
    out_dir.mkdir(exist_ok=True)

    audio_path = extract_audio(args.video, out_dir)
    transcript = transcribe_audio(audio_path, out_dir, args.whisper_model)

    frames = extract_frames(args.video, out_dir, args.interval)
    captions = caption_frames(frames, args.vision_model, args.interval)
    (out_dir / "frame_captions.json").write_text(json.dumps(captions, indent=2))

    summary = synthesize(transcript, captions, args.text_model, args.question)

    summary_path = out_dir / "summary.md"
    summary_path.write_text(
        f"# Analysis: {args.video.name}\n\n{summary}\n\n"
        f"---\n\n## Raw transcript\n\n{transcript or '(none)'}\n"
    )

    print(f"\n✅ Done. Results saved in: {out_dir}/")
    print(f"   - {summary_path}")
    print(f"   - {out_dir / 'frame_captions.json'}")
    print(f"   - {out_dir / 'transcript.txt'}")


if __name__ == "__main__":
    main()
