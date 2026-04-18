#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

RESOLVED_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir)"
RESOLVED_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$RESOLVED_CHECKPOINT_DIR")"
RESOLVED_FIXED_STANDARD_JSON="$(qwen35_pick_path "${FIXED_STANDARD_JSON:-}" "$(qwen35_default_fixed_standard_json "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_FIXED_STANDARD_LOG="$(qwen35_pick_path "${FIXED_STANDARD_LOG:-}" "$(qwen35_default_fixed_standard_log "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_DESKTOP_ANALYSIS_DIR="$(qwen35_pick_path "${DESKTOP_ANALYSIS_DIR:-}" "$(qwen35_default_desktop_analysis_dir "$RESOLVED_CHECKPOINT_STEP")")"

cd "$ROOT"
mkdir -p "$(dirname "$RESOLVED_FIXED_STANDARD_JSON")"
mkdir -p "$RESOLVED_DESKTOP_ANALYSIS_DIR"

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "[fixed-standard-eval] checkpoint=$RESOLVED_CHECKPOINT_DIR"
echo "[fixed-standard-eval] output=$RESOLVED_FIXED_STANDARD_JSON"

"$PYTHON_BIN" eval_libero.py \
  --checkpoint_dir "$RESOLVED_CHECKPOINT_DIR" \
  --suites libero_spatial \
  --eval_level standard \
  --n_episodes 10 \
  --max_steps 300 \
  --output "$RESOLVED_FIXED_STANDARD_JSON" \
  --chunk_size "$CHUNK_SIZE" \
  --deterministic_seed "$DETERMINISTIC_SEED" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --record_tasks 0 \
  --record_episodes 0 \
  --no_traj_plot |& tee -a "$RESOLVED_FIXED_STANDARD_LOG"

cp "$RESOLVED_FIXED_STANDARD_JSON" "$RESOLVED_DESKTOP_ANALYSIS_DIR/"
cp "$RESOLVED_FIXED_STANDARD_LOG" "$RESOLVED_DESKTOP_ANALYSIS_DIR/"
echo "[fixed-standard-eval] exported analysis bundle to $RESOLVED_DESKTOP_ANALYSIS_DIR"
