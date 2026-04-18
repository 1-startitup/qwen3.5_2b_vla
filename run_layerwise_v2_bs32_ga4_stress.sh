#!/usr/bin/env bash
set -euo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0
cd /home/frankkkz/qwen35_2b_vla

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs32_ga4_timing10_nodeepspeed}"
POWER_LOG="${POWER_LOG:-/tmp/qwen35_v2_bs32_ga4_power.csv}"

rm -f "$POWER_LOG"

nvidia-smi \
  --query-gpu=timestamp,power.draw,utilization.gpu,memory.used \
  --format=csv,noheader \
  -l 1 > "$POWER_LOG" &
MON_PID=$!

cleanup() {
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

/usr/bin/time -f 'REAL_SECONDS %e' \
  python train.py \
    --config config/libero_train_2b_layerwise_v2_smoke1000.yaml \
    --dataset.batch_size "$BATCH_SIZE" \
    --training.gradient_accumulation_steps "$GRAD_ACCUM" \
    --training.max_steps "$MAX_STEPS" \
    --training.eval_every 1000 \
    --training.save_every "$MAX_STEPS" \
    --training.log_every 1 \
    --training.output_dir "$OUTPUT_DIR"

echo "=== POWER_TAIL ==="
tail -n 20 "$POWER_LOG"
