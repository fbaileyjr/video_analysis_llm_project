#!/usr/bin/env bash
#
# setup.sh — installs everything needed to run analyze_video.py on a
# Mac Mini M4 (16GB) or similar Apple Silicon machine.
#
# What this installs:
#   - ffmpeg          (audio/frame extraction)
#   - whisper-cpp      (local speech-to-text)
#   - ollama           (local LLM/VLM runtime)
#   - a small vision model (moondream) + a small text model (qwen2.5:3b)
#
# Safe to re-run — every step checks if it's already done.

set -euo pipefail

echo "== 1/6: Checking Homebrew =="
if ! command -v brew &>/dev/null; then
  echo "Homebrew not found. Install it from https://brew.sh first, then re-run this script."
  exit 1
fi

echo "== 2/6: Installing ffmpeg =="
brew list ffmpeg &>/dev/null || brew install ffmpeg

echo "== 3/6: Installing whisper-cpp =="
brew list whisper-cpp &>/dev/null || brew install whisper-cpp

echo "== 4/6: Downloading a whisper model (base.en) =="
WHISPER_MODEL_DIR="$HOME/.whisper-models"
mkdir -p "$WHISPER_MODEL_DIR"
WHISPER_MODEL_PATH="$WHISPER_MODEL_DIR/ggml-base.en.bin"
if [ ! -f "$WHISPER_MODEL_PATH" ]; then
  curl -L -o "$WHISPER_MODEL_PATH" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
else
  echo "Whisper model already present at $WHISPER_MODEL_PATH"
fi

echo "== 5/6: Installing Python deps for object tracking (YOLO) =="
if ! command -v python3 &>/dev/null; then
  echo "python3 not found. Install Python 3.10+ first, then re-run this script."
  exit 1
fi
python3 -m pip install --quiet --break-system-packages ultralytics opencv-python-headless

echo "== 6/6: Installing Ollama + pulling models =="
if ! command -v ollama &>/dev/null; then
  brew install ollama
fi

# Start the Ollama server in the background if it isn't already running.
if ! pgrep -x "ollama" &>/dev/null; then
  echo "Starting ollama serve in the background..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 3
fi

echo "Pulling vision model (moondream — small, good for frame captioning)..."
ollama pull moondream

echo "Pulling text model (qwen2.5:3b — small, good for summarizing on 16GB)..."
ollama pull qwen2.5:3b

cat <<EOF

✅ Setup complete.

  Whisper model:  $WHISPER_MODEL_PATH
  Vision model:   moondream (via ollama)
  Text model:     qwen2.5:3b (via ollama)

Next step:
  python3 analyze_video.py /path/to/your/video.mp4

See README.md for options and RAM notes.
EOF
