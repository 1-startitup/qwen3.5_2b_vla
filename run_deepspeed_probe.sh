#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/frankkkz/qwen35_2b_vla"
CONFIG="${CONFIG:-config/libero_train_2b_layerwise_v2_bs128_ga1_fast.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-config/deepspeed_zero2_single_fast.yaml}"
MAX_STEPS="${MAX_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/ds_probe}"
POWER_LOG="${POWER_LOG:-/tmp/ds_probe_power.csv}"
RUN_LOG="${RUN_LOG:-./checkpoints/ds_probe.log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /home/frankkkz/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_pi0
set -u
cd "$ROOT"

mkdir -p "$(dirname "$RUN_LOG")" "$OUTPUT_DIR"
rm -f "$POWER_LOG" "$RUN_LOG"

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$POWER_LOG" &
MON_PID=$!
cleanup() {
  kill "$MON_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

/usr/bin/time -f 'ELAPSED_SECONDS=%e' \
  accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes 1 \
  train.py \
  --config "$CONFIG" \
  --training.max_steps "$MAX_STEPS" \
  --training.save_every 1000 \
  --training.eval_every 1000 \
  --training.save_optimizer_state false \
  --training.output_dir "$OUTPUT_DIR" \
  $EXTRA_ARGS \
  2>&1 | tee "$RUN_LOG"

cleanup
trap - EXIT

echo "---POWER-STATS---"
awk -F', *' '
BEGIN { n=0; sum=0; max=0; maxu=0; maxm=0 }
{
  p=$2+0; u=$3+0; m=$4+0;
  if (p > 0) {
    n++;
    sum += p;
    if (p > max) max = p;
    if (u > maxu) maxu = u;
    if (m > maxm) maxm = m;
  }
}
END {
  if (n > 0) {
    printf("samples=%d avg_power=%.2fW max_power=%.2fW max_util=%d%% max_mem=%dMiB\n", n, sum / n, max, maxu, maxm);
  } else {
    print "no power samples";
  }
}' "$POWER_LOG"
