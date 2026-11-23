#!/usr/bin/env bash
set -euo pipefail

# Runs a W&B hyperparameter sweep of the segmentation model (one sweep in W&B, many trials), then retrains the
# best trial at full length (trials are shorter and may be stopped early by Hyperband).
# Usage: scripts/sweep_segmentation.sh [--no-final] <sweep.yaml | entity/project/sweep_id> [trials]
#   sweep.yaml   registers a new sweep, then starts an agent on it
#   sweep path   starts another agent on an existing sweep (continue it later, or in parallel on a second GPU)
#   trials       how many trials this agent runs before it exits (default 25)
#   --no-final   stop after the trials: no best-config export and no full-length run (use it for extra
#                parallel agents, so only one of them starts the final training)
FINAL=1
if [[ "${1:-}" == "--no-final" ]]; then FINAL=0; shift; fi
SWEEP="${1:?usage: scripts/sweep_segmentation.sh [--no-final] <sweep.yaml | entity/project/sweep_id> [trials]}"
TRIALS="${2:-25}"
BEST_CONFIG="configs/experiments/sweep_best.yaml"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" # the trial command in the sweep YAML uses paths relative to the repository root
# W&B key from .env if it exists (template: .env.example); in Docker, Compose provides it instead
if [[ -f "$ROOT_DIR/.env" ]]; then export UV_ENV_FILE="$ROOT_DIR/.env"; fi

# 1. Sweep: create it if a YAML was given, then run the trials
if [[ -f "$SWEEP" ]]; then
  SWEEP="$(uv run --project "$ROOT_DIR" python -m latent_crew.training.sweep create "$SWEEP" | tail -n 1)"
  echo "Sweep: $SWEEP (add agents with: scripts/sweep_segmentation.sh --no-final $SWEEP [trials])"
fi
uv run --project "$ROOT_DIR" wandb agent --count "$TRIALS" "$SWEEP"
if [[ "$FINAL" == 0 ]]; then exit 0; fi

# 2. Best trial -> experiment config on top of the 100-epoch baseline
uv run --project "$ROOT_DIR" python -m latent_crew.training.sweep best "$SWEEP" --out "$BEST_CONFIG"

# 3. Final full-length training with it (a normal run with its own W&B run and model artifact)
"$ROOT_DIR/scripts/train_segmentation.sh" "$BEST_CONFIG"
