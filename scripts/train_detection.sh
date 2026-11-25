#!/usr/bin/env bash
set -euo pipefail

# Trains the 2D object detector (YOLO26) on SUN RGB-D. Usage: scripts/train_detection.sh <config.yaml>
# The dataset from the config is checked first and only downloaded/prepared if it is missing or incomplete,
# then a YOLO-format dataset is built from the SUN RGB-D 2D box metadata.
CONFIG="${1:-configs/detection.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# W&B key from .env if it exists (template: .env.example); in Docker, Compose provides it instead
if [[ -f "$ROOT_DIR/.env" ]]; then export UV_ENV_FILE="$ROOT_DIR/.env"; fi

uv run --project "$ROOT_DIR" python -m latent_crew.training.train_detection --config "$CONFIG"
