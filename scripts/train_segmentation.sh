#!/usr/bin/env bash
set -euo pipefail

# Trains the segmentation model. Usage: scripts/train_segmentation.sh <config.yaml>
CONFIG="${1:?usage: scripts/train_segmentation.sh <config.yaml>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Download the datasets into ./data if they are not there yet
DATA_DIR="$ROOT_DIR/data"
for DATASET in sunrgbd nyu_depth_v2; do
  if [ -z "$(ls -A "$DATA_DIR/$DATASET" 2>/dev/null)" ]; then
    echo "Dataset $DATASET not found in $DATA_DIR/$DATASET, downloading..."
    echo "TODO: download $DATASET into $DATA_DIR/$DATASET"
  fi
done

uv run --project "$ROOT_DIR" python -m latent_crew.training.train_segmentation --config "$CONFIG"
