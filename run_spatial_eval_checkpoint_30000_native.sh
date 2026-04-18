#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/frankkkz/qwen35_2b_vla"
PYTHON_BIN="/home/frankkkz/miniconda3/envs/lerobot_pi0/bin/python"
CHECKPOINT_DIR="$ROOT/checkpoints/qwen_3.5_2b_pi0.5_lora_r16_bs32_ga2_30k_native/checkpoint-30000"
OUTPUT_JSON="$ROOT/eval_results_libero_spatial_checkpoint_30000_native.json"
ROLLOUT_DIR="$ROOT/rollouts/libero_spatial_checkpoint_30000_native"

cd "$ROOT"
mkdir -p "$ROLLOUT_DIR"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" eval_libero.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --suites libero_spatial \
  --eval_level standard \
  --n_episodes 10 \
  --max_steps 300 \
  --output "$OUTPUT_JSON" \
  --chunk_size 50 \
  --deterministic_seed 0 \
  --num_inference_steps 10 \
  --rollout_dir "$ROLLOUT_DIR" \
  --record_tasks 10 \
  --record_episodes 1
