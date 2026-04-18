#!/usr/bin/env bash

qwen35_pick_path() {
  local preferred="$1"
  local fallback="$2"
  if [[ -n "$preferred" ]]; then
    printf '%s\n' "$preferred"
  else
    printf '%s\n' "$fallback"
  fi
}

resolve_qwen35_checkpoint_dir() {
  local requested="${CHECKPOINT_DIR:-}"
  if [[ -n "$requested" && -d "$requested" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  local run_dir="$ROOT/checkpoints/$RUN_NAME"
  if [[ ! -d "$run_dir" ]]; then
    echo "Missing checkpoint run directory: $run_dir" >&2
    return 1
  fi

  if [[ -n "${CHECKPOINT_STEP:-}" && "${CHECKPOINT_STEP}" != "latest" ]]; then
    local candidate="$run_dir/checkpoint-$CHECKPOINT_STEP"
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  local latest
  latest="$(find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -z "$latest" ]]; then
    echo "No checkpoints found under $run_dir" >&2
    return 1
  fi

  printf '%s\n' "$latest"
}

qwen35_checkpoint_step_from_dir() {
  local checkpoint_dir="$1"
  basename "$checkpoint_dir" | sed 's/^checkpoint-//'
}

qwen35_resolve_eval_config() {
  local checkpoint_dir="$1"
  if [[ -f "$checkpoint_dir/config.yaml" ]]; then
    printf '%s\n' "$checkpoint_dir/config.yaml"
  else
    printf '%s\n' "$TRAIN_CONFIG"
  fi
}

qwen35_default_open_loop_json() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_open_loop_analysis_checkpoint_${step}.json"
}

qwen35_default_open_loop_log() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_open_loop_analysis_checkpoint_${step}.log"
}

qwen35_default_fixed_standard_json() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_eval_results_libero_spatial_checkpoint_${step}_native_fixed_full.json"
}

qwen35_default_fixed_standard_log() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_eval_results_libero_spatial_checkpoint_${step}_native_fixed_full.log"
}

qwen35_default_fixed_video_json() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_eval_results_libero_spatial_checkpoint_${step}_native_ep1_frames_fixed.json"
}

qwen35_default_fixed_video_log() {
  local step="$1"
  printf '%s\n' "$ROOT/tmp/${RUN_NAME}_eval_results_libero_spatial_checkpoint_${step}_native_ep1_frames_fixed.log"
}

qwen35_default_fixed_video_rollout_dir() {
  local step="$1"
  printf '%s\n' "$ROOT/rollouts/${RUN_NAME}_libero_spatial_checkpoint_${step}_native_ep1_frames_fixed"
}

qwen35_default_desktop_video_dir() {
  local step="$1"
  printf '/mnt/c/Users/frank/OneDrive/Desktop/%s_spatial_agentview_videos_checkpoint_%s_fixed\n' "$RUN_NAME" "$step"
}

qwen35_default_desktop_analysis_dir() {
  local step="$1"
  printf '/mnt/c/Users/frank/OneDrive/Desktop/%s_open_closed_loop_analysis_checkpoint_%s_fixed\n' "$RUN_NAME" "$step"
}
