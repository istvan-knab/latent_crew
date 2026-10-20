#!/usr/bin/env bash
set -euo pipefail

# Trains the segmentation model. Usage: scripts/train_segmentation.sh <config.yaml>
# The dataset from the config is checked first and only downloaded/prepared if it is missing or incomplete.
CONFIG="${1:?usage: scripts/train_segmentation.sh <config.yaml>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.training.train_segmentation --config "$CONFIG"
