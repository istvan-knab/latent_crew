#!/usr/bin/env bash
set -euo pipefail

# Evaluates a trained segmentation checkpoint. Usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>
CONFIG="${1:?usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>}"
CHECKPOINT="${2:?usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.evaluation.test_segmentation --config "$CONFIG" --checkpoint "$CHECKPOINT"
