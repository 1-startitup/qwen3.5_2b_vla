#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla"
ACCELERATE_BIN="/home/frankkkz/miniconda3/envs/lerobot_pi0/bin/accelerate"
CONFIG="$ROOT/config/libero_train_qwen_3_5_2b_pi0_5_lora_r16_bs37_30k.yaml"
ACCEL_CONFIG="$ROOT/config/accelerate_single_gpu_bf16.yaml"

cd "$ROOT"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

"$ACCELERATE_BIN" launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes 1 \
  train.py \
  --config "$CONFIG" \
  "$@"
