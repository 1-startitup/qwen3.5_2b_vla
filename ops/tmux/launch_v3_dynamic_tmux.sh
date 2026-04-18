#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  exec tmux attach -t "$SESSION_NAME"
fi

ROOT_CMD="cd \"$ROOT\""
TRAIN_CMD="$PYTHON train.py --config \"$TRAIN_CONFIG\""
EVAL_CMD="bash \"$SCRIPT_DIR/run_v3_dynamic_eval.sh\""
ANALYZE_CMD="bash \"$SCRIPT_DIR/run_v3_dynamic_analyze.sh\""
CLAUDE_CMD="bash \"$SCRIPT_DIR/run_v3_claude_review.sh\""
CODEX_PING_CMD="bash \"$SCRIPT_DIR/run_v3_codex_ping.sh\""
CODEX_REVIEW_CMD="bash \"$SCRIPT_DIR/run_v3_codex_review.sh\""
MERGE_CMD="bash \"$SCRIPT_DIR/run_v3_merge_consensus.sh\""
MONITOR_CMD="while true; do clear; date; echo; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true; echo; ps -eo pid,etime,cmd | grep -E \"train.py|eval_libero.py|claude|codex\" | grep -v grep || true; echo; echo \"iter dir: $ITER_DIR_LINUX\"; sleep 5; done"

tmux new-session -d -s "$SESSION_NAME" -n control "bash --login"
tmux set-option -t "$SESSION_NAME" remain-on-exit on
tmux set-option -t "$SESSION_NAME" mouse on

tmux send-keys -t "$SESSION_NAME:control" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:control" "printf '%s\n' 'V3 dynamic pipeline' '-------------------' 'current version : $CURR_VERSION' 'previous version: $PREV_VERSION' 'train config    : $TRAIN_CONFIG' 'shared iter dir : $ITER_DIR_LINUX' 'version manifest: $VERSION_MANIFEST_LINUX' 'official eval   : $OFFICIAL_EVAL_JSON' 'diff report     : $DIFF_REPORT' 'claude review   : $CLAUDE_REVIEW_JSON' 'codex status    : $CODEX_PING_STATUS' 'codex review    : $CODEX_REVIEW_JSON' 'consensus       : $CONSENSUS_JSON' '' 'Run order:' '1. train' '2. eval' '3. analyze' '4. claude review' '5. codex ping/review' '6. consensus merge' '7. user approval -> touch APPROVED' '' 'Fallbacks:' '- codex usage_limited -> Claude-only consensus for this iteration' '- codex timeout/error  -> Claude-only consensus for this iteration'" C-m

tmux new-window -t "$SESSION_NAME" -n train "bash --login"
tmux send-keys -t "$SESSION_NAME:train" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:train" "$TRAIN_CMD"

tmux new-window -t "$SESSION_NAME" -n eval "bash --login"
tmux send-keys -t "$SESSION_NAME:eval" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:eval" "$EVAL_CMD"

tmux new-window -t "$SESSION_NAME" -n analyze "bash --login"
tmux send-keys -t "$SESSION_NAME:analyze" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:analyze" "$ANALYZE_CMD"

tmux new-window -t "$SESSION_NAME" -n claude-review "bash --login"
tmux send-keys -t "$SESSION_NAME:claude-review" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:claude-review" "$CLAUDE_CMD"

tmux new-window -t "$SESSION_NAME" -n codex-ping "bash --login"
tmux send-keys -t "$SESSION_NAME:codex-ping" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:codex-ping" "$CODEX_PING_CMD"

tmux new-window -t "$SESSION_NAME" -n codex-review "bash --login"
tmux send-keys -t "$SESSION_NAME:codex-review" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:codex-review" "$CODEX_REVIEW_CMD"

tmux new-window -t "$SESSION_NAME" -n merge "bash --login"
tmux send-keys -t "$SESSION_NAME:merge" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:merge" "$MERGE_CMD"

tmux new-window -t "$SESSION_NAME" -n monitor "bash --login"
tmux send-keys -t "$SESSION_NAME:monitor" "$ROOT_CMD" C-m
tmux send-keys -t "$SESSION_NAME:monitor" "$MONITOR_CMD" C-m

tmux select-window -t "$SESSION_NAME:control"
exec tmux attach -t "$SESSION_NAME"
