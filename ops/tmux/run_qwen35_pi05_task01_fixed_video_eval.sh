#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

TASK_ID="${TASK_ID:-1}"
TASK_EPISODES="${TASK_EPISODES:-10}"
TASK_RECORD_EPISODES="${TASK_RECORD_EPISODES:-$TASK_EPISODES}"

RESOLVED_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir)"
RESOLVED_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$RESOLVED_CHECKPOINT_DIR")"
TASK_TAG="$(printf 'task%02d' "$TASK_ID")"
TASK_RECORD_TASKS="$((TASK_ID + 1))"
TASK_VIDEO_JSON="${TASK_VIDEO_JSON:-$ROOT/tmp/${RUN_NAME}_${TASK_TAG}_eval_results_libero_spatial_checkpoint_${RESOLVED_CHECKPOINT_STEP}_native_ep${TASK_EPISODES}_frames_fixed.json}"
TASK_VIDEO_LOG="${TASK_VIDEO_LOG:-$ROOT/tmp/${RUN_NAME}_${TASK_TAG}_eval_results_libero_spatial_checkpoint_${RESOLVED_CHECKPOINT_STEP}_native_ep${TASK_EPISODES}_frames_fixed.log}"
TASK_VIDEO_ROLLOUT_DIR="${TASK_VIDEO_ROLLOUT_DIR:-$ROOT/rollouts/${RUN_NAME}_${TASK_TAG}_libero_spatial_checkpoint_${RESOLVED_CHECKPOINT_STEP}_native_ep${TASK_EPISODES}_frames_fixed}"

cd "$ROOT"
mkdir -p "$TASK_VIDEO_ROLLOUT_DIR"
mkdir -p "$(dirname "$TASK_VIDEO_JSON")"

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

echo "[task01-fixed-video-eval] checkpoint=$RESOLVED_CHECKPOINT_DIR"
echo "[task01-fixed-video-eval] task_id=$TASK_ID episodes=$TASK_EPISODES"
echo "[task01-fixed-video-eval] output=$TASK_VIDEO_JSON"
echo "[task01-fixed-video-eval] rollout_dir=$TASK_VIDEO_ROLLOUT_DIR"

"$PYTHON_BIN" eval_libero.py \
  --checkpoint_dir "$RESOLVED_CHECKPOINT_DIR" \
  --suites libero_spatial \
  --eval_level standard \
  --n_episodes "$TASK_EPISODES" \
  --max_steps 300 \
  --output "$TASK_VIDEO_JSON" \
  --chunk_size "$CHUNK_SIZE" \
  --deterministic_seed "$DETERMINISTIC_SEED" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --rollout_dir "$TASK_VIDEO_ROLLOUT_DIR" \
  --record_tasks "$TASK_RECORD_TASKS" \
  --record_episodes "$TASK_RECORD_EPISODES" \
  --task_ids "$TASK_ID" \
  --save_frames \
  --no_traj_plot |& tee -a "$TASK_VIDEO_LOG"
