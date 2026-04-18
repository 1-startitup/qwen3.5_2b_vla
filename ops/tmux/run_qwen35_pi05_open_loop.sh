#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

RESOLVED_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir)"
RESOLVED_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$RESOLVED_CHECKPOINT_DIR")"
RESOLVED_CONFIG="$(qwen35_resolve_eval_config "$RESOLVED_CHECKPOINT_DIR")"
RESOLVED_OPEN_LOOP_JSON="$(qwen35_pick_path "${OPEN_LOOP_JSON:-}" "$(qwen35_default_open_loop_json "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_OPEN_LOOP_LOG="$(qwen35_pick_path "${OPEN_LOOP_LOG:-}" "$(qwen35_default_open_loop_log "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_DESKTOP_ANALYSIS_DIR="$(qwen35_pick_path "${DESKTOP_ANALYSIS_DIR:-}" "$(qwen35_default_desktop_analysis_dir "$RESOLVED_CHECKPOINT_STEP")")"

cd "$ROOT"
mkdir -p "$(dirname "$RESOLVED_OPEN_LOOP_JSON")"
mkdir -p "$RESOLVED_DESKTOP_ANALYSIS_DIR"

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "[open-loop] config=$RESOLVED_CONFIG"
echo "[open-loop] checkpoint=$RESOLVED_CHECKPOINT_DIR"
echo "[open-loop] output=$RESOLVED_OPEN_LOOP_JSON"

"$PYTHON_BIN" open_loop_train_eval.py \
  --config "$RESOLVED_CONFIG" \
  --checkpoint_dir "$RESOLVED_CHECKPOINT_DIR" \
  --output "$RESOLVED_OPEN_LOOP_JSON" \
  --batch_size "$OPEN_LOOP_BATCH_SIZE" \
  --num_workers "$OPEN_LOOP_WORKERS" \
  --max_batches "$OPEN_LOOP_MAX_BATCHES" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --deterministic_seed "$DETERMINISTIC_SEED" \
  --device cuda |& tee -a "$RESOLVED_OPEN_LOOP_LOG"

cp "$RESOLVED_OPEN_LOOP_JSON" "$RESOLVED_DESKTOP_ANALYSIS_DIR/"
cp "$RESOLVED_OPEN_LOOP_LOG" "$RESOLVED_DESKTOP_ANALYSIS_DIR/"
echo "[open-loop] exported analysis bundle to $RESOLVED_DESKTOP_ANALYSIS_DIR"
