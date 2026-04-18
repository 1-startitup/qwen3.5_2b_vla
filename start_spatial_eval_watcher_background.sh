#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/frankkkz/qwen35_2b_vla"
WATCHER="$ROOT/run_libero_spatial_eval_when_30000_done.sh"
PIDFILE="$ROOT/run_libero_spatial_eval_when_30000_done.pid"
LOGFILE="$ROOT/run_libero_spatial_eval_when_30000_done.log"

cd "$ROOT"
pkill -TERM -f "run_libero_spatial_eval_when_30000_done.sh" >/dev/null 2>&1 || true
sleep 2

nohup bash "$WATCHER" > /dev/null 2>&1 < /dev/null &
WATCH_PID=$!
echo "$WATCH_PID" > "$PIDFILE"

sleep 2
echo "WATCH_PID=$WATCH_PID"
echo "LOGFILE=$LOGFILE"
