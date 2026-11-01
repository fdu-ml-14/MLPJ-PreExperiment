#!/usr/bin/env bash
# Generic single-model SimplerEnv evaluation entry.
# Usage: eval_simpler.sh <model> [tasks] [action_steps] [episode_ids] [output_dir]
#   model: pi0_bridge | pi0_base | octo_base | spatialvla_bridge
#   tasks: comma separated, empty = model default
#   action_steps: e.g. 1,4
#   episode_ids: e.g. 0-9
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"

MODEL="${1:?usage: eval_simpler.sh <model> [tasks] [action_steps] [episode_ids] [output_dir]}"
TASKS="${2:-}"
ACTION_STEPS="${3:-1,4}"
EPISODE_IDS="${4:-0-9}"
OUTPUT_DIR="${5:-$PI0_ROOT/runs/${MODEL}_$(date +%Y%m%d)}"
PYTHON="${PYTHON:-$PI0_ROOT/envs/train/bin/python}"

mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$REPO_DIR/scripts/extended_eval.py" \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --action-steps "$ACTION_STEPS" \
  --episode-ids "$EPISODE_IDS" \
  --seed 42 \
  --max-steps 60 \
  --output-dir "$OUTPUT_DIR" \
  --record
