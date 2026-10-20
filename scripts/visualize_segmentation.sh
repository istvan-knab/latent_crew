#!/usr/bin/env bash
set -euo pipefail

# Saves one sample (RGB, depth, mask, overlay) as PNG. Usage: scripts/visualize_segmentation.sh <config.yaml> [extra args]
# Extra args: --split train|test, --index N or --id sun_000123, --augment, --out path.png
CONFIG="${1:?usage: scripts/visualize_segmentation.sh <config.yaml> [--index N | --id ID] [--augment]}"
shift
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run --project "$ROOT_DIR" python -m latent_crew.utils.visualization --config "$CONFIG" "$@"
