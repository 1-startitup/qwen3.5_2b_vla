#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

TASK_ID="${TASK_ID:-1}"
TASK_EPISODES="${TASK_EPISODES:-10}"

RESOLVED_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir)"
RESOLVED_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$RESOLVED_CHECKPOINT_DIR")"
TASK_TAG="$(printf 'task%02d' "$TASK_ID")"
TASK_VIDEO_JSON="${TASK_VIDEO_JSON:-$ROOT/tmp/${RUN_NAME}_${TASK_TAG}_eval_results_libero_spatial_checkpoint_${RESOLVED_CHECKPOINT_STEP}_native_ep${TASK_EPISODES}_frames_fixed.json}"
TASK_VIDEO_ROLLOUT_DIR="${TASK_VIDEO_ROLLOUT_DIR:-$ROOT/rollouts/${RUN_NAME}_${TASK_TAG}_libero_spatial_checkpoint_${RESOLVED_CHECKPOINT_STEP}_native_ep${TASK_EPISODES}_frames_fixed}"
OUT_DIR="${TASK_VIDEO_OUT_DIR:-/mnt/c/Users/frank/OneDrive/Desktop/${RUN_NAME}_${TASK_TAG}_agentview_videos_checkpoint_${RESOLVED_CHECKPOINT_STEP}_fixed}"
FLAT_VIDEO_DIR="$OUT_DIR/agentview_mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed."
  exit 1
fi

BASE_DIR="$TASK_VIDEO_ROLLOUT_DIR/libero_spatial/standard"
if [[ ! -d "$BASE_DIR" ]]; then
  echo "Missing rollout directory: $BASE_DIR"
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$FLAT_VIDEO_DIR"

shopt -s nullglob
for taskdir in "$BASE_DIR"/task_*; do
  task_name="$(basename "$taskdir")"
  mkdir -p "$OUT_DIR/$task_name"

  for ep_dir in "$taskdir"/episode_*; do
    [[ -d "$ep_dir" ]] || continue
    frame_dir="$ep_dir/agentview_frames"
    [[ -d "$frame_dir" ]] || continue

    episode_name="$(basename "$ep_dir")"
    out_ep_dir="$OUT_DIR/$task_name/$episode_name"
    mkdir -p "$out_ep_dir"

    out_video="$out_ep_dir/agentview.mp4"
    ffmpeg -y -loglevel error -framerate 20 \
      -i "$frame_dir/frame_%04d.png" \
      -c:v libx264 \
      -pix_fmt yuv420p \
      "$out_video"

    cp "$out_video" "$FLAT_VIDEO_DIR/${task_name}_${episode_name}.mp4"
  done
done
shopt -u nullglob

if [[ -f "$TASK_VIDEO_JSON" ]]; then
  cp "$TASK_VIDEO_JSON" "$OUT_DIR/$(basename "$TASK_VIDEO_JSON")"
fi

cat > "$OUT_DIR/README.txt" <<EOF
Qwen3.5 pi0.5 task-specific fixed-checker agentview bundle
==========================================================

Checkpoint: $RESOLVED_CHECKPOINT_DIR
Task id: $TASK_ID
Rollout base: $BASE_DIR
Eval JSON: $TASK_VIDEO_JSON

This bundle contains one agentview mp4 per recorded episode.
EOF

echo "bundle ready at $OUT_DIR"
