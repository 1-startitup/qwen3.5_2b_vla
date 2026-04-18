#!/usr/bin/env bash
set -eo pipefail

export TZ="America/New_York"

ROOT="/home/frankkkz/qwen35_2b_vla"
BASE_OUT="$ROOT/checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh"
WATCH_LOG="$ROOT/continue_to_40k_if_done_before_2026_04_06_1900.log"
WATCH_PID="$ROOT/continue_to_40k_if_done_before_2026_04_06_1900.pid"
CONT_LOG="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_continue_40k_from30k.log"
CONT_PID="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_continue_40k_from30k.pid"
CONT_RUN_NAME="layerwise-fm-v2-bs128-ga1-deepspeed-continue-40k-from30k"
CUTOFF_HUMAN="2026-04-06 19:00:00"
CUTOFF_EPOCH="$(date -d "$CUTOFF_HUMAN" +%s)"

cd "$ROOT"
echo "$$" > "$WATCH_PID"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$WATCH_LOG"
}

is_training_active() {
  pgrep -f "train.py --config config/libero_train_2b_layerwise_v2_bs128_ga1_deepspeed_train.yaml --training.max_steps 30000 --training.output_dir $BASE_OUT" >/dev/null 2>&1
}

log "Watcher started. Cutoff is $CUTOFF_HUMAN America/New_York."
log "Waiting for the 30000-step run at $BASE_OUT to finish."

while is_training_active; do
  sleep 60
done

NOW_EPOCH="$(date +%s)"
CKPT30000="$BASE_OUT/checkpoint-30000"

if [ ! -f "$CKPT30000/action_head.pt" ]; then
  log "30000-step checkpoint not found at $CKPT30000. Not starting continuation."
  exit 0
fi

if [ "$NOW_EPOCH" -ge "$CUTOFF_EPOCH" ]; then
  log "Training finished after cutoff. Current time is $(date '+%F %T %Z'). Not starting continuation."
  exit 0
fi

log "30000-step training finished before cutoff. Starting continuation to 40000 from $CKPT30000."

nohup bash ./run_layerwise_v2_bs128_ga1_deepspeed_fast.sh \
  --training.max_steps 40000 \
  --training.resume_from_checkpoint "$CKPT30000" \
  --training.output_dir "$BASE_OUT" \
  --training.wandb_run_name "$CONT_RUN_NAME" \
  > "$CONT_LOG" 2>&1 < /dev/null &

CONT_RUN_PID=$!
echo "$CONT_RUN_PID" > "$CONT_PID"
log "Continuation launched with PID $CONT_RUN_PID. Log: $CONT_LOG"
