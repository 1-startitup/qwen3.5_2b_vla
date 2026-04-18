#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

RESOLVED_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir)"
RESOLVED_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$RESOLVED_CHECKPOINT_DIR")"
RESOLVED_FIXED_VIDEO_JSON="$(qwen35_pick_path "${FIXED_VIDEO_JSON:-}" "$(qwen35_default_fixed_video_json "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_FIXED_VIDEO_ROLLOUT_DIR="$(qwen35_pick_path "${FIXED_VIDEO_ROLLOUT_DIR:-}" "$(qwen35_default_fixed_video_rollout_dir "$RESOLVED_CHECKPOINT_STEP")")"
RESOLVED_DESKTOP_VIDEO_DIR="$(qwen35_pick_path "${DESKTOP_VIDEO_DIR:-}" "$(qwen35_default_desktop_video_dir "$RESOLVED_CHECKPOINT_STEP")")"

BASE_DIR="$RESOLVED_FIXED_VIDEO_ROLLOUT_DIR/libero_spatial/standard"
OUT_DIR="$RESOLVED_DESKTOP_VIDEO_DIR"
FLAT_VIDEO_DIR="$OUT_DIR/agentview_mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed."
  exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Missing rollout directory: $BASE_DIR"
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$FLAT_VIDEO_DIR"

shopt -s nullglob
for taskdir in "$BASE_DIR"/task_*; do
  task_name="$(basename "$taskdir")"
  ep_dir="$taskdir/episode_00"
  frame_dir="$ep_dir/agentview_frames"
  if [[ ! -d "$frame_dir" ]]; then
    continue
  fi

  mkdir -p "$OUT_DIR/$task_name"
  cp -R "$taskdir" "$OUT_DIR/" 2>/dev/null || true

  out_video="$OUT_DIR/$task_name/episode_00/agentview.mp4"
  ffmpeg -y -loglevel error -framerate 20 \
    -i "$frame_dir/frame_%04d.png" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    "$out_video"

  cp "$out_video" "$FLAT_VIDEO_DIR/${task_name}.mp4"
  echo "rendered $task_name"
done
shopt -u nullglob

if [[ -f "$RESOLVED_FIXED_VIDEO_JSON" ]]; then
  cp "$RESOLVED_FIXED_VIDEO_JSON" "$OUT_DIR/$(basename "$RESOLVED_FIXED_VIDEO_JSON")"
fi

cat > "$OUT_DIR/README.txt" <<EOF
Qwen3.5 pi0.5 fixed-checker agentview bundle
============================================

Checkpoint: $RESOLVED_CHECKPOINT_DIR
Rollout base: $BASE_DIR
Eval JSON: $RESOLVED_FIXED_VIDEO_JSON

This bundle mirrors the earlier task-wise agentview export layout:
  - one folder per task
  - episode_00/agentview.mp4
  - a flat agentview_mp4 directory for quick browsing
EOF

echo "bundle ready at $OUT_DIR"
