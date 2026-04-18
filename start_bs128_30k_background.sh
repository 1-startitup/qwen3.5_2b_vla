#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/frankkkz/qwen35_2b_vla"
LOG="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.log"
OUT="$ROOT/checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh"
RUN_NAME="layerwise-fm-v2-bs128-ga1-deepspeed-train-30k-bs128fresh"
PIDFILE="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.pid"
CHILDPIDFILE="$ROOT/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.child.pid"

cd "$ROOT"

pkill -TERM -f "run_deepspeed_probe.sh" >/dev/null 2>&1 || true
pkill -TERM -f "train.py --config config/libero_train_2b_layerwise_v2_bs128_ga1_deepspeed_train.yaml --training.max_steps 30000" >/dev/null 2>&1 || true
sleep 3

rm -f "$PIDFILE" "$CHILDPIDFILE" "$LOG"

nohup bash ./run_layerwise_v2_bs128_ga1_deepspeed_fast.sh \
  --training.max_steps 30000 \
  --training.output_dir "$OUT" \
  --training.wandb_run_name "$RUN_NAME" \
  > "$LOG" 2>&1 < /dev/null &

RUN_PID=$!
echo "$RUN_PID" > "$PIDFILE"

sleep 5
TRAIN_PID="$(pgrep -f "train.py --config config/libero_train_2b_layerwise_v2_bs128_ga1_deepspeed_train.yaml --training.max_steps 30000 --training.output_dir $OUT" | head -n 1 || true)"
if [ -n "$TRAIN_PID" ]; then
  echo "$TRAIN_PID" > "$CHILDPIDFILE"
fi

echo "RUN_PID=$RUN_PID"
echo "TRAIN_PID=${TRAIN_PID:-}"
echo "LOG=$LOG"
