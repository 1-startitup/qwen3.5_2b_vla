#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/frankkkz/qwen35_2b_vla"
CONFIG_PATH="config/libero_train_2b_layerwise_action_only_v1.yaml"
TRAIN_LOG="train_lora_r32_bs32_sdpa_actiononly_v1_40k.log"
EVAL_LOG="eval_libero_actiononly_v1_checkpoint_40000.log"
FINAL_CKPT="checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1/checkpoint-40000"
EVAL_OUTPUT="checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1/eval_results_checkpoint_40000_v1.json"
ROLLOUT_DIR="rollouts/qwen35_2b_vla_action_only_v1_checkpoint_40000"

source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
export WANDB_MODE="${WANDB_MODE-offline}"
conda activate lerobot_pi0
cd "$ROOT_DIR"

echo "[$(date)] Starting action-only v1 training run" | tee -a "$TRAIN_LOG"
echo "Config: $CONFIG_PATH" | tee -a "$TRAIN_LOG"

python train.py \
  --config "$CONFIG_PATH" \
  2>&1 | tee -a "$TRAIN_LOG"

train_status=${PIPESTATUS[0]}
if [ "$train_status" -ne 0 ]; then
  echo "[$(date)] Training failed with status $train_status" | tee -a "$TRAIN_LOG"
  exit "$train_status"
fi

if [ ! -d "$FINAL_CKPT" ]; then
  echo "[$(date)] Expected final checkpoint missing: $FINAL_CKPT" | tee -a "$EVAL_LOG"
  exit 1
fi

echo "[$(date)] Starting LIBERO eval from $FINAL_CKPT" | tee -a "$EVAL_LOG"
python eval_libero.py \
  --checkpoint_dir "$FINAL_CKPT" \
  --suites libero_spatial \
  --n_episodes 10 \
  --max_steps 300 \
  --output "$EVAL_OUTPUT" \
  --rollout_dir "$ROLLOUT_DIR" \
  --record_tasks 1 \
  --record_episodes 1 \
  2>&1 | tee -a "$EVAL_LOG"

eval_status=${PIPESTATUS[0]}
if [ "$eval_status" -ne 0 ]; then
  echo "[$(date)] Eval failed with status $eval_status" | tee -a "$EVAL_LOG"
  exit "$eval_status"
fi

echo "[$(date)] Action-only v1 training + eval complete" | tee -a "$EVAL_LOG"
