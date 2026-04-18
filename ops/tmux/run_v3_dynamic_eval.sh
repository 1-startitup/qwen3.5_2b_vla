#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$RUN_DIR" "$(dirname "$DIFF_REPORT")"

export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Waiting for checkpoint: $CHECKPOINT_DIR"
until [[ -f "$CHECKPOINT_DIR/action_head.pt" && -f "$CHECKPOINT_DIR/config.yaml" ]]; do
  sleep 30
done

echo "Checkpoint detected. Starting official eval."

"$PYTHON" "$ROOT/eval_libero.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --suites libero_spatial \
  --eval_level standard \
  --mode official \
  --n_episodes 10 \
  --max_steps 300 \
  --deterministic_seed 0 \
  --record_tasks 0 \
  --record_episodes 0 \
  --no_traj_plot \
  --output "$OFFICIAL_EVAL_JSON" \
  2>&1 | tee "$OFFICIAL_EVAL_LOG"
