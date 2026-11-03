#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter optimisation of the segmentation training.
# Usage: scripts/hpo_segmentation.sh <config.yaml> <hpo_config.yaml> [--optimizer ifbo|priorband]   (default: ifbo)
CONFIG="${1:?usage: scripts/hpo_segmentation.sh <config.yaml> <hpo_config.yaml> [--optimizer ifbo|priorband]}"
HPO_CONFIG="${2:?usage: scripts/hpo_segmentation.sh <config.yaml> <hpo_config.yaml> [--optimizer ifbo|priorband]}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# W&B key from .env if it exists (template: .env.example); in Docker, Compose provides it instead
if [[ -f "$ROOT_DIR/.env" ]]; then export UV_ENV_FILE="$ROOT_DIR/.env"; fi

uv run --project "$ROOT_DIR" python -m latent_crew.training.hpo_segmentation --config "$CONFIG" --hpo-config "$HPO_CONFIG" "${@:3}"
