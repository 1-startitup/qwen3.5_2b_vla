#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla"
PIDFILE="$ROOT/train_qwen35_pi05_lora_r16_bs37_30k.paused_pids"

if [ ! -f "$PIDFILE" ]; then
  echo "NO_PIDFILE"
  exit 1
fi

mapfile -t pids < "$PIDFILE"
if [ "${#pids[@]}" -eq 0 ]; then
  echo "EMPTY_PIDFILE"
  exit 1
fi

kill -CONT "${pids[@]}"
echo "RESUMED_PIDS=${pids[*]}"
