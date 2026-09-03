#!/usr/bin/env python3
"""
track_objects.py — local object motion analysis for a 16GB Apple Silicon Mac.

Unlike analyze_video.py (which asks a vision-language model to *describe*
frames), this script actually *measures* motion:

  1. Run YOLOv8 object detection + ByteTrack tracking on every frame
  2. Follow each object's pixel position across frames to build a trajectory
  3. Compute pixel speed per object (and real-world speed, if you provide
     a calibration)
  4. Hand the resulting NUMBERS (not images) to a local LLM to interpret
     and summarize — the LLM never has to guess motion from pictures

Usage:
  python3 track_objects.py video.mp4
  python3 track_objects.py video.mp4 --fps 10 --classes car,person
  python3 track_objects.py video.mp4 --pixels-per-unit 12.5 --unit meters

Calibration:
  YOLO + tracking only gives you PIXEL displacement per second. To convert
  that to real-world speed (m/s, mph, etc.) you need to tell the script how
  many pixels correspond to one real-world unit in your footage. Easiest way:
  measure something of known size in the frame (a car is ~4.5m long, a door
  is ~2m tall) and divide its pixel length by its real length.

  Example: a car that's 180px long in frame and is actually 4.5m long:
    pixels_per_unit = 180 / 4.5 = 40
    --pixels-per-unit 40 --unit meters

  Without calibration, you still get valid RELATIVE speed comparisons
  ("object 3 moved about twice as fast as object 1") — just not real units.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"Could not reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        print("Run `ollama serve` in another terminal, then try again.", file=sys.stderr)
        sys.exit(1)


def run_tracking(video_path: Path, sample_fps: float, classes: list[str] | None):
    """Run YOLO detection + ByteTrack tracking over the video, frame by frame."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run setup.sh first, or:", file=sys.stderr)
        print("  python3 -m pip install --break-system-packages ultralytics opencv-python-headless", file=sys.stderr)
        sys.exit(1)

    print("→ Loading YOLOv8n (downloads automatically on first run)...")
    model = YOLO("yolov8n.pt")

    class_filter = None
    if classes:
        # Map requested class names -> YOLO class indices
        name_to_id = {v: k for k, v in model.names.items()}
        class_filter = [name_to_id[c] for c in classes if c in name_to_id]
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            print(f"Warning: unknown class names ignored: {missing}", file=sys.stderr)
            print(f"Available classes: {sorted(name_to_id)}", file=sys.stderr)

    print(f"→ Running detection + tracking (sampling ~{sample_fps} fps)...")
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=class_filter,
        vid_stride=max(1, round(30 / sample_fps)),  # assumes ~30fps source; approximate
        persist=True,
        verbose=False,
        stream=True,
    )

    # trajectories[track_id] = list of {frame_idx, x, y, w, h, cls}
    trajectories: dict[int, list[dict]] = defaultdict(list)
    class_names: dict[int, str] = {}

    for frame_idx, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.xywh.cpu().tolist()      # center x, center y, w, h
        ids = result.boxes.id.int().cpu().tolist()
        clss = result.boxes.cls.int().cpu().tolist()
        for (cx, cy, w, h), track_id, cls_id in zip(boxes, ids, clss):
            class_names[track_id] = model.names[cls_id]
            trajectories[track_id].append({
                "frame_idx": frame_idx,
                "x": round(cx, 1),
                "y": round(cy, 1),
                "w": round(w, 1),
                "h": round(h, 1),
            })

    return trajectories, class_names


def compute_speeds(trajectories, class_names, sample_fps, pixels_per_unit, unit):
    """Turn raw positions into per-object speed statistics."""
    dt = 1.0 / sample_fps  # seconds between sampled frames
    objects = []

    for track_id, points in trajectories.items():
        if len(points) < 2:
            continue
        speeds_px_s = []
        for a, b in zip(points, points[1:]):
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            frame_gap = max(1, b["frame_idx"] - a["frame_idx"])
            dist_px = (dx ** 2 + dy ** 2) ** 0.5
            speeds_px_s.append(dist_px / (dt * frame_gap))

        avg_px_s = sum(speeds_px_s) / len(speeds_px_s)
        max_px_s = max(speeds_px_s)

        entry = {
            "id": track_id,
            "class": class_names.get(track_id, "unknown"),
            "num_detections": len(points),
            "avg_speed_px_per_sec": round(avg_px_s, 1),
            "max_speed_px_per_sec": round(max_px_s, 1),
            "start_pos": {"x": points[0]["x"], "y": points[0]["y"]},
            "end_pos": {"x": points[-1]["x"], "y": points[-1]["y"]},
        }
        if pixels_per_unit:
            entry["avg_speed"] = round(avg_px_s / pixels_per_unit, 2)
            entry["max_speed"] = round(max_px_s / pixels_per_unit, 2)
            entry["unit"] = f"{unit}/sec"

        objects.append(entry)

    objects.sort(key=lambda o: o["avg_speed_px_per_sec"], reverse=True)
    return objects


def synthesize(objects, text_model: str, calibrated: bool, question: str | None) -> str:
    print(f"→ Interpreting results with {text_model}...")
    lines = []
    for o in objects:
        if calibrated:
            lines.append(
                f"- Object #{o['id']} ({o['class']}): avg speed {o['avg_speed']} {o['unit']}, "
                f"max speed {o['max_speed']} {o['unit']}, tracked over {o['num_detections']} frames, "
                f"moved from {o['start_pos']} to {o['end_pos']}"
            )
        else:
            lines.append(
                f"- Object #{o['id']} ({o['class']}): avg speed {o['avg_speed_px_per_sec']} px/sec, "
                f"max speed {o['max_speed_px_per_sec']} px/sec, tracked over {o['num_detections']} frames, "
                f"moved from {o['start_pos']} to {o['end_pos']}"
            )
    data_block = "\n".join(lines) if lines else "(no objects were tracked for long enough to measure speed)"

    unit_note = (
        "Speeds are in real-world units based on the calibration provided."
        if calibrated else
        "Speeds are in PIXELS PER SECOND, not real-world units, since no calibration "
        "was provided — only compare objects relative to each other, don't state these "
        "as real-world speeds."
    )

    prompt = f"""You are analyzing measured motion data extracted from a video by an
object detection and tracking system (not a description of what the video
looks like — these are computed numbers).

{unit_note}

TRACKED OBJECTS:
{data_block}

Write a clear analysis: which objects were moving fastest and slowest,
any notable patterns (e.g. objects that sped up, slowed down, or stayed
mostly still), and anything worth flagging (e.g. very few detections for
an object, meaning its speed estimate is less reliable).
"""
    if question:
        prompt += f"\nAlso specifically answer this question: {question}\n"

    return ollama_generate(text_model, prompt)


def main():
    parser = argparse.ArgumentParser(description="Track objects and measure their speed locally.")
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument("--fps", type=float, default=10.0,
                         help="Frames per second to sample for tracking (default: 10). "
                              "Higher = more accurate motion, slower to run.")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated list of object classes to track (e.g. 'car,person'). "
                              "Default: track everything YOLO recognizes.")
    parser.add_argument("--pixels-per-unit", type=float, default=None,
                         help="Calibration: how many pixels equal one real-world unit. "
                              "See the docstring at the top of this file for how to measure this.")
    parser.add_argument("--unit", type=str, default="meters",
                         help="Real-world unit name for --pixels-per-unit (default: meters)")
    parser.add_argument("--text-model", default="qwen2.5:3b", help="Ollama text model for the summary")
    parser.add_argument("--question", default=None, help="Optional specific question to answer")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out or Path(f"{args.video.stem}_motion_analysis")
    out_dir.mkdir(exist_ok=True)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

    trajectories, class_names = run_tracking(args.video, args.fps, classes)
    objects = compute_speeds(trajectories, class_names, args.fps, args.pixels_per_unit, args.unit)

    (out_dir / "trajectories.json").write_text(json.dumps(objects, indent=2))

    calibrated = args.pixels_per_unit is not None
    summary = synthesize(objects, args.text_model, calibrated, args.question)

    summary_path = out_dir / "motion_analysis.md"
    cal_note = (
        f"Calibration: {args.pixels_per_unit} px = 1 {args.unit}"
        if calibrated else
        "No calibration provided — speeds below are in pixels/sec (relative comparison only)."
    )
    summary_path.write_text(
        f"# Motion analysis: {args.video.name}\n\n{cal_note}\n\n{summary}\n"
    )

    print(f"\n✅ Done. {len(objects)} objects tracked. Results saved in: {out_dir}/")
    print(f"   - {summary_path}")
    print(f"   - {out_dir / 'trajectories.json'}")
    if not calibrated:
        print("\n   Tip: add --pixels-per-unit and --unit to get real-world speed estimates.")


if __name__ == "__main__":
    main()
