#!/usr/bin/env bash
set -eo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0
set -u
cd /home/frankkkz/qwen35_2b_vla

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"

# Make CUDA runtime and importable wheel libraries visible to the DeepSpeed
# JIT compiler and linker.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

python - <<'PY'
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

print("=== DeepSpeed Preflight ===", flush=True)
print(f"flash_linear_attention_available={is_flash_linear_attention_available()}", flush=True)
print(f"causal_conv1d_available={is_causal_conv1d_available()}", flush=True)
PY

accelerate launch \
  --config_file config/deepspeed_zero2_single_fast.yaml \
  --num_processes 1 \
  train.py \
  --config config/libero_train_2b_layerwise_v2_bs128_ga1_deepspeed_train.yaml \
  "$@"
