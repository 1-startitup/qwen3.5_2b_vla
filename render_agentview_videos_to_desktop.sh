#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/mnt/c/Users/frank/OneDrive/Desktop/Qwen35_2B_VLA_v2_Spatial_Eval_Report/assets}"
BASE_DIR="${2:-/home/frankkkz/qwen35_2b_vla/rollouts/libero_spatial_checkpoint_30000_alltasks/libero_spatial/spatial}"

mkdir -p "${OUT_DIR}"

for taskdir in "${BASE_DIR}"/*; do
  name="$(basename "${taskdir}")"
  frames="${taskdir}/episode_00/agentview_frames"
  if [[ -d "${frames}" ]]; then
    ffmpeg -y -loglevel error -framerate 20 \
      -i "${frames}/frame_%04d.png" \
      -c:v libx264 \
      -pix_fmt yuv420p \
      "${OUT_DIR}/${name}_agentview.mp4"
    echo "made ${name}_agentview.mp4"
  fi
done
