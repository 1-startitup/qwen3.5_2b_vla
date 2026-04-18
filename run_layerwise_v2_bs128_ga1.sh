#!/usr/bin/env bash
set -euo pipefail

# Single-GPU native Linux launcher for the validated v2 LoRA setup.
# This uses `accelerate launch` without DeepSpeed so it still works in the
# current environment while keeping an Accelerate-based entrypoint.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0
cd /home/frankkkz/qwen35_2b_vla

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

accelerate launch \
  --config_file config/accelerate_single_gpu_bf16.yaml \
  --num_processes 1 \
  train.py \
  --config config/libero_train_2b_layerwise_v2_bs128_ga1.yaml
