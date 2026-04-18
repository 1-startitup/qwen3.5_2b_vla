#!/usr/bin/env bash
set -euo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0

cd /home/frankkkz/qwen35_2b_vla

CHECKPOINT_DIR="/home/frankkkz/qwen35_2b_vla/checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh/checkpoint-30000"
ROLLOUT_DIR="/home/frankkkz/qwen35_2b_vla/rollouts/libero_spatial_checkpoint_30000_alltasks"

for i in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== task_${i} ==="
  python eval_libero.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --suites libero_spatial \
    --eval_level spatial \
    --task_ids "${i}" \
    --n_episodes 1 \
    --max_steps 300 \
    --output "/home/frankkkz/qwen35_2b_vla/eval_results_frames_task${i}.json" \
    --chunk_size 1 \
    --deterministic_seed 0 \
    --num_inference_steps 10 \
    --rollout_dir "${ROLLOUT_DIR}" \
    --record_tasks 10 \
    --record_episodes 1 \
    --save_frames
done
