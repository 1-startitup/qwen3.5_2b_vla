#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla"
LOG="$ROOT/train_qwen35_pi05_lora_r16_bs37_30k.log"
PIDFILE="$ROOT/train_qwen35_pi05_lora_r16_bs37_30k.pid"

cd "$ROOT"

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "Training already running: PID=$OLD_PID"
    exit 0
  fi
fi

rm -f "$PIDFILE"

nohup bash "$ROOT/run_qwen35_pi05_lora_r16_bs37_30k.sh" > "$LOG" 2>&1 < /dev/null &
RUN_PID=$!
echo "$RUN_PID" > "$PIDFILE"

sleep 5
if kill -0 "$RUN_PID" >/dev/null 2>&1; then
  echo "STARTED PID=$RUN_PID"
  echo "LOG=$LOG"
else
  echo "FAILED_TO_START"
  exit 1
fi
