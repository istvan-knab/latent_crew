#!/usr/bin/env bash
set -euo pipefail

# Evaluates a trained segmentation checkpoint. Usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>
CONFIG="${1:?usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>}"
CHECKPOINT="${2:?usage: scripts/test_segmentation.sh <config.yaml> <checkpoint.pt>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# W&B key from .env if it exists (template: .env.example); in Docker, Compose provides it instead
if [[ -f "$ROOT_DIR/.env" ]]; then export UV_ENV_FILE="$ROOT_DIR/.env"; fi

uv run --project "$ROOT_DIR" python -m latent_crew.evaluation.test_segmentation --config "$CONFIG" --checkpoint "$CHECKPOINT"
