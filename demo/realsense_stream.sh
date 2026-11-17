#!/usr/bin/env bash
set -euo pipefail

# Demo: live RGB + depth view of a plugged-in Intel RealSense camera (not part of training).
# Usage: demo/realsense_stream.sh [--sudo] [--width 640 --height 480 --fps 30 --max-depth 6]
#   --sudo  run the viewer as root; macOS often needs it for USB access to the camera
# The RealSense SDK comes from the optional `camera` extra in pyproject.toml (macOS arm64, Linux x86_64, Windows).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/demo/realsense_stream.py"

if [[ "${1:-}" == "--sudo" ]]; then
  shift
  # Install as the normal user, so the .venv does not end up owned by root; only the viewer runs as root
  uv sync --project "$ROOT_DIR" --extra camera --inexact
  exec sudo "$ROOT_DIR/.venv/bin/python" "$SCRIPT" "$@"
fi

exec uv run --project "$ROOT_DIR" --extra camera python "$SCRIPT" "$@"
