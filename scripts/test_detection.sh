#!/usr/bin/env bash
set -euo pipefail

# Evaluates a trained 2D detector (YOLO26) on SUN RGB-D.
# Usage: scripts/test_detection.sh <config.yaml> <checkpoint.pt> [split]
CONFIG="${1:-configs/detection.yaml}"
CHECKPOINT="${2:?usage: scripts/test_detection.sh <config.yaml> <checkpoint.pt> [val|test]}"
SPLIT="${3:-val}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.evaluation.test_detection \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --split "$SPLIT"
