#!/usr/bin/env bash
set -eo pipefail

export TZ="America/New_York"

ROOT="/home/frankkkz/qwen35_2b_vla"
TRAIN_LOG="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.log"
CHECKPOINT_DIR="$ROOT/checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh/checkpoint-30000"
RESULT_JSON="$ROOT/eval_results_libero_spatial_checkpoint_30000.json"
ROLLOUT_DIR="$ROOT/rollouts/libero_spatial_checkpoint_30000"
WATCH_LOG="$ROOT/run_libero_spatial_eval_when_30000_done.log"
WATCH_PID="$ROOT/run_libero_spatial_eval_when_30000_done.pid"
REPORT_SCRIPT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla/generate_desktop_reports.py"
POLL_SECONDS=60

source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0
cd "$ROOT"

echo "$$" > "$WATCH_PID"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$WATCH_LOG"
}

log "Watcher started. Waiting for 30000-step training to finish."
log "Train log: $TRAIN_LOG"
log "Target checkpoint: $CHECKPOINT_DIR"

python "$REPORT_SCRIPT" --write-current 2>&1 | tee -a "$WATCH_LOG"

while true; do
  if grep -q "Training complete!" "$TRAIN_LOG" 2>/dev/null; then
    log "Training completion detected in log."
    break
  fi
  if [ -f "$CHECKPOINT_DIR/action_head.pt" ] && [ -f "$CHECKPOINT_DIR/config.yaml" ]; then
    log "Checkpoint-30000 detected."
    break
  fi
  sleep "$POLL_SECONDS"
done

sleep 10
mkdir -p "$ROLLOUT_DIR"

log "Starting LIBERO spatial eval from $CHECKPOINT_DIR."

python eval_libero.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --suites libero_spatial \
  --eval_level spatial \
  --n_episodes 10 \
  --max_steps 300 \
  --output "$RESULT_JSON" \
  --chunk_size 1 \
  --deterministic_seed 0 \
  --num_inference_steps 10 \
  --rollout_dir "$ROLLOUT_DIR" \
  --record_tasks 3 \
  --record_episodes 1 \
  2>&1 | tee -a "$WATCH_LOG"

python "$REPORT_SCRIPT" --write-current 2>&1 | tee -a "$WATCH_LOG"

log "LIBERO spatial eval complete. Results saved to $RESULT_JSON"
