#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_runtime.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed."
  exit 1
fi

ATTACH=1
KILL_EXISTING=0
declare -A AUTOSTART=(
  [train]=0
  [open-loop]=0
  [fixed-standard]=0
  [fixed-video]=0
  [render]=0
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      [[ $# -ge 2 ]] || { echo "missing value for --start"; exit 1; }
      case "$2" in
        train|open-loop|fixed-standard|fixed-video|render) AUTOSTART["$2"]=1 ;;
        *) echo "unknown --start target: $2"; exit 1 ;;
      esac
      shift 2
      ;;
    --session-name)
      [[ $# -ge 2 ]] || { echo "missing value for --session-name"; exit 1; }
      SESSION_NAME="$2"
      shift 2
      ;;
    --kill-existing)
      KILL_EXISTING=1
      shift
      ;;
    --no-attach)
      ATTACH=0
      shift
      ;;
    *)
      echo "unknown argument: $1"
      exit 1
      ;;
  esac
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [[ "$KILL_EXISTING" -eq 1 ]]; then
    tmux kill-session -t "$SESSION_NAME"
  else
    if [[ "$ATTACH" -eq 1 ]]; then
      exec tmux attach -t "$SESSION_NAME"
    fi
    echo "tmux session already exists: $SESSION_NAME"
    exit 0
  fi
fi

ROOT_CMD="cd \"$ROOT\""
CONTEXT_CMD="bash \"$SCRIPT_DIR/show_project_context.sh\""
TRAIN_CMD="bash \"$SCRIPT_DIR/run_qwen35_pi05_train.sh\""
OPEN_LOOP_CMD="bash \"$SCRIPT_DIR/run_qwen35_pi05_open_loop.sh\""
FIXED_STANDARD_CMD="bash \"$SCRIPT_DIR/run_qwen35_pi05_fixed_standard_eval.sh\""
FIXED_VIDEO_CMD="bash \"$SCRIPT_DIR/run_qwen35_pi05_fixed_spatial_video_eval.sh\""
RENDER_CMD="bash \"$SCRIPT_DIR/render_qwen35_pi05_fixed_agentview_bundle.sh\""
MONITOR_CMD="while true; do clear; date; echo; echo 'GPU'; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null || true; echo; echo 'Processes'; ps -eo pid,etime,cmd | grep -E 'train.py|eval_libero.py|open_loop_train_eval.py' | grep -v grep || true; echo; echo 'Train log tail'; tail -n 25 \"$TRAIN_LOG\" 2>/dev/null || true; sleep 5; done"

DISPLAY_CHECKPOINT_STEP="$CHECKPOINT_STEP"
if DISPLAY_CHECKPOINT_DIR="$(resolve_qwen35_checkpoint_dir 2>/dev/null)"; then
  DISPLAY_CHECKPOINT_STEP="$(qwen35_checkpoint_step_from_dir "$DISPLAY_CHECKPOINT_DIR")"
fi
DISPLAY_VIDEO_DIR="$(qwen35_pick_path "${DESKTOP_VIDEO_DIR:-}" "$(qwen35_default_desktop_video_dir "$DISPLAY_CHECKPOINT_STEP")")"
DISPLAY_ANALYSIS_DIR="$(qwen35_pick_path "${DESKTOP_ANALYSIS_DIR:-}" "$(qwen35_default_desktop_analysis_dir "$DISPLAY_CHECKPOINT_STEP")")"

tmux new-session -d -s "$SESSION_NAME" -n control "bash --login"
tmux set-option -t "$SESSION_NAME" remain-on-exit on
tmux set-option -t "$SESSION_NAME" mouse on

tmux send-keys -t "$SESSION_NAME:control" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:control" "clear" C-m
tmux send-keys -t "$SESSION_NAME:control" "printf '%s\n' 'Qwen35 pi0.5 native tmux project' '--------------------------------' 'run name        : $RUN_NAME' 'checkpoint step : $DISPLAY_CHECKPOINT_STEP' 'context         : $CONTEXT_CMD' 'train           : $TRAIN_CMD' 'open-loop       : $OPEN_LOOP_CMD' 'fixed-standard  : $FIXED_STANDARD_CMD' 'fixed-video     : $FIXED_VIDEO_CMD' 'render-videos   : $RENDER_CMD' 'desktop videos  : $DISPLAY_VIDEO_DIR' 'desktop analysis: $DISPLAY_ANALYSIS_DIR' 'context doc     : $PROJECT_CONTEXT_DOC' 'context json    : $PROJECT_CONTEXT_JSON' '' 'tuning rule     : maximize sample/s near OOM; probe micro-batch first, then grad-accum'" C-m

tmux new-window -t "$SESSION_NAME" -n context "bash --login"
tmux send-keys -t "$SESSION_NAME:context" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:context" "$CONTEXT_CMD" C-m

tmux new-window -t "$SESSION_NAME" -n train "bash --login"
tmux send-keys -t "$SESSION_NAME:train" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:train" "$TRAIN_CMD"
[[ "${AUTOSTART[train]}" -eq 1 ]] && tmux send-keys -t "$SESSION_NAME:train" C-m

tmux new-window -t "$SESSION_NAME" -n open-loop "bash --login"
tmux send-keys -t "$SESSION_NAME:open-loop" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:open-loop" "$OPEN_LOOP_CMD"
[[ "${AUTOSTART[open-loop]}" -eq 1 ]] && tmux send-keys -t "$SESSION_NAME:open-loop" C-m

tmux new-window -t "$SESSION_NAME" -n fixed-eval "bash --login"
tmux send-keys -t "$SESSION_NAME:fixed-eval" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:fixed-eval" "$FIXED_STANDARD_CMD"
[[ "${AUTOSTART[fixed-standard]}" -eq 1 ]] && tmux send-keys -t "$SESSION_NAME:fixed-eval" C-m

tmux new-window -t "$SESSION_NAME" -n video-eval "bash --login"
tmux send-keys -t "$SESSION_NAME:video-eval" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:video-eval" "$FIXED_VIDEO_CMD"
[[ "${AUTOSTART[fixed-video]}" -eq 1 ]] && tmux send-keys -t "$SESSION_NAME:video-eval" C-m

tmux new-window -t "$SESSION_NAME" -n render "bash --login"
tmux send-keys -t "$SESSION_NAME:render" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:render" "$RENDER_CMD"
[[ "${AUTOSTART[render]}" -eq 1 ]] && tmux send-keys -t "$SESSION_NAME:render" C-m

tmux new-window -t "$SESSION_NAME" -n monitor "bash --login"
tmux send-keys -t "$SESSION_NAME:monitor" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:monitor" "$MONITOR_CMD" C-m

tmux select-window -t "$SESSION_NAME:control"

if [[ "$ATTACH" -eq 1 ]]; then
  exec tmux attach -t "$SESSION_NAME"
fi

echo "created tmux session: $SESSION_NAME"
