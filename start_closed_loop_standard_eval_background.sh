#!/usr/bin/env bash
set -euo pipefail

cd /home/frankkkz/qwen35_2b_vla

nohup bash ./run_closed_loop_standard_eval_30k.sh \
  > eval_libero_standard_checkpoint_30000_fixedload.log \
  2>&1 < /dev/null &

pid=$!
echo "${pid}" > eval_libero_standard_checkpoint_30000_fixedload.pid
echo "${pid}"
