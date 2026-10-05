#!/usr/bin/env bash
set -euo pipefail

# Evaluates a trained detection checkpoint. Usage: scripts/test_detection.sh <config.yaml> <checkpoint.pt>
CONFIG="${1:?usage: scripts/test_detection.sh <config.yaml> <checkpoint.pt>}"
CHECKPOINT="${2:?usage: scripts/test_detection.sh <config.yaml> <checkpoint.pt>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.evaluation.test_detection --config "$CONFIG" --checkpoint "$CHECKPOINT"
