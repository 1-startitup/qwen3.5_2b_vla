#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"

cd "$ROOT"
mkdir -p "$(dirname "$TRAIN_LOG")"

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "[train] root=$ROOT"
echo "[train] config=$TRAIN_CONFIG"
echo "[train] log=$TRAIN_LOG"
echo "[train] run_name=$TRAIN_WANDB_RUN_NAME"
echo "[train] batch_size=$TRAIN_BATCH_SIZE grad_accum=$TRAIN_GRAD_ACCUM_STEPS max_steps=$TRAIN_MAX_STEPS"

"$ACCELERATE_BIN" launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes 1 \
  train.py \
  --config "$TRAIN_CONFIG" \
  --dataset.batch_size="$TRAIN_BATCH_SIZE" \
  --training.gradient_accumulation_steps="$TRAIN_GRAD_ACCUM_STEPS" \
  --training.max_steps="$TRAIN_MAX_STEPS" \
  --training.wandb_run_name="$TRAIN_WANDB_RUN_NAME" \
  --training.output_dir="$TRAIN_OUTPUT_DIR" \
  "$@" |& tee -a "$TRAIN_LOG"
