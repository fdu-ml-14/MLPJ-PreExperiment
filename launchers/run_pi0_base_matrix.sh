#!/usr/bin/env bash
# pi0-base main matrix: T1-T4 x action_step 1/4 x 10 episodes = 80 episodes.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"

RUN_DIR="${RUN_DIR:-$PI0_ROOT/runs/pi0_base_20261023}"
rm -f "$RUN_DIR/episodes.csv" "$RUN_DIR/results.csv" "$RUN_DIR/representatives.json" "$RUN_DIR/manifest.json"
rm -rf "$RUN_DIR/videos" "$RUN_DIR/final_frames"
mkdir -p "$RUN_DIR"

"$PI0_ROOT/envs/train/bin/python" "$REPO_DIR/scripts/extended_eval.py" \
  --model pi0_base \
  --tasks widowx_spoon_on_towel,widowx_carrot_on_plate,widowx_stack_cube,widowx_put_eggplant_in_basket \
  --action-steps 1,4 \
  --episode-ids 0-9 \
  --seed 42 \
  --max-steps 60 \
  --output-dir "$RUN_DIR" \
  --record
