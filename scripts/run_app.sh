#!/usr/bin/env bash
set -euo pipefail

# Runs the full detection + segmentation app end-to-end.
# Usage: scripts/run_app.sh <config.yaml> <detection_checkpoint.pt> <segmentation_checkpoint.pt>
CONFIG="${1:?usage: scripts/run_app.sh <config.yaml> <detection_checkpoint.pt> <segmentation_checkpoint.pt>}"
DETECTION_CHECKPOINT="${2:?usage: scripts/run_app.sh <config.yaml> <detection_checkpoint.pt> <segmentation_checkpoint.pt>}"
SEGMENTATION_CHECKPOINT="${3:?usage: scripts/run_app.sh <config.yaml> <detection_checkpoint.pt> <segmentation_checkpoint.pt>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.app \
  --config "$CONFIG" \
  --detection-checkpoint "$DETECTION_CHECKPOINT" \
  --segmentation-checkpoint "$SEGMENTATION_CHECKPOINT"
