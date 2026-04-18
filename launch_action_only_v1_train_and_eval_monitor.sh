#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/frankkkz/qwen35_2b_vla"
TRAIN_LOG="train_action_only_v1_40k_direct.log"
EVAL_LOG="eval_action_only_v1_40k_monitor.log"
TRAIN_PID_FILE="train_action_only_v1_40k.pid"
EVAL_PID_FILE="eval_action_only_v1_40k_monitor.pid"
OUTPUT_DIR="checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1"
ROLLOUT_DIR="rollouts/qwen35_2b_vla_action_only_v1_checkpoint_40000"

cd "$ROOT_DIR"

pkill -f "config/libero_train_2b_layerwise_action_only_v1.yaml" || true
sleep 2

rm -f "$TRAIN_LOG" "$EVAL_LOG" "$TRAIN_PID_FILE" "$EVAL_PID_FILE"
rm -rf "$OUTPUT_DIR" "$ROLLOUT_DIR"

nohup bash -lc '
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
  export WANDB_MODE=offline
  source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
  conda activate lerobot_pi0
  cd /home/frankkkz/qwen35_2b_vla
  python train.py \
    --config config/libero_train_2b_layerwise_action_only_v1.yaml \
    --training.wandb_project= \
    > train_action_only_v1_40k_direct.log 2>&1
' >/dev/null 2>&1 &
echo $! > "$TRAIN_PID_FILE"

nohup bash -lc '
  cd /home/frankkkz/qwen35_2b_vla
  while [ ! -d checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1/checkpoint-40000 ]; do
    sleep 60
  done
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
  export WANDB_MODE=offline
  source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
  conda activate lerobot_pi0
  python eval_libero.py \
    --checkpoint_dir checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1/checkpoint-40000 \
    --suites libero_spatial \
    --n_episodes 10 \
    --max_steps 300 \
    --output checkpoints/qwen35_2b_vla_libero_layerwise_action_only_v1/eval_results_checkpoint_40000_v1.json \
    --rollout_dir rollouts/qwen35_2b_vla_action_only_v1_checkpoint_40000 \
    --record_tasks 1 \
    --record_episodes 1 \
    > eval_action_only_v1_40k_monitor.log 2>&1
' >/dev/null 2>&1 &
echo $! > "$EVAL_PID_FILE"

echo "TRAIN_PID=$(cat "$TRAIN_PID_FILE")"
echo "EVAL_MONITOR_PID=$(cat "$EVAL_PID_FILE")"
