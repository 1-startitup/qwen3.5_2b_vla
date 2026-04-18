#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla"
CFG="$ROOT/config/libero_train_qwen_3_5_2b_pi0_5_lora_r16_bs37_30k.yaml"
PIDFILE="$ROOT/train_qwen35_pi05_lora_r16_bs37_30k.paused_pids"

mapfile -t pids < <(pgrep -f "train.py --config $CFG" || true)
if [ "${#pids[@]}" -eq 0 ]; then
  echo "NO_TRAIN_PIDS"
  exit 0
fi

printf "%s\n" "${pids[@]}" > "$PIDFILE"
kill -STOP "${pids[@]}"

echo "PAUSED_PIDS=${pids[*]}"
echo "PIDFILE=$PIDFILE"
