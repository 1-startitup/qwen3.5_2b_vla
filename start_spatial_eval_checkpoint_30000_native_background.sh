#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/frankkkz/qwen35_2b_vla"
LOG="$ROOT/eval_libero_spatial_checkpoint_30000_native.log"
PIDFILE="$ROOT/eval_libero_spatial_checkpoint_30000_native.pid"

cd "$ROOT"

if [ -f "$PIDFILE" ]; then
  OLD_PID="$(cat "$PIDFILE" || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "Eval already running: PID=$OLD_PID"
    exit 0
  fi
fi

rm -f "$PIDFILE"

nohup bash "$ROOT/run_spatial_eval_checkpoint_30000_native.sh" > "$LOG" 2>&1 < /dev/null &
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
