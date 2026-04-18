#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla"
LOG="$ROOT/eval_libero_standard_checkpoint_4000_chunk50_task0.log"
PIDFILE="$ROOT/eval_libero_standard_checkpoint_4000_chunk50_task0.pid"

cd "$ROOT"
rm -f "$PIDFILE"

nohup bash "$ROOT/run_closed_loop_standard_eval_checkpoint_4000_chunk50_task0.sh" > "$LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PIDFILE"

sleep 5
if kill -0 "$pid" >/dev/null 2>&1; then
  echo "STARTED PID=$pid"
  echo "LOG=$LOG"
else
  echo "FAILED_TO_START"
  exit 1
fi
