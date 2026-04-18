#!/usr/bin/env bash
set -euo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0

cd /home/frankkkz/qwen35_2b_vla

python eval_libero.py \
  --checkpoint_dir /home/frankkkz/qwen35_2b_vla/checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh/checkpoint-30000 \
  --suites libero_spatial \
  --eval_level standard \
  --n_episodes 10 \
  --max_steps 300 \
  --output /home/frankkkz/qwen35_2b_vla/eval_results_libero_standard_checkpoint_30000_fixedload.json \
  --chunk_size 1 \
  --deterministic_seed 0 \
  --num_inference_steps 10 \
  --rollout_dir /home/frankkkz/qwen35_2b_vla/rollouts/libero_standard_checkpoint_30000_fixedload \
  --record_tasks 10 \
  --record_episodes 1
