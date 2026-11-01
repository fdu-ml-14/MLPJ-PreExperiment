#!/usr/bin/env bash
# One-episode pi0-base smoke test: T1, action_step=4, episode 0.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"

RUN_DIR="${RUN_DIR:-$PI0_ROOT/runs/pi0_base_20261023}"
rm -rf "$RUN_DIR/smoke"
mkdir -p "$RUN_DIR/smoke"

"$PI0_ROOT/envs/train/bin/python" "$REPO_DIR/scripts/extended_eval.py" \
  --model pi0_base \
  --tasks widowx_spoon_on_towel \
  --action-steps 4 \
  --episode-ids 0 \
  --seed 42 \
  --max-steps 60 \
  --output-dir "$RUN_DIR/smoke" \
  --record
