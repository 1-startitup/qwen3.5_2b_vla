#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$ITER_DIR_LINUX/reviews"

if [[ ! -x "$CODEX_BIN_LINUX" ]]; then
  echo '{"decision":"ambiguous","dominant_bucket":"mixed","evidence":["codex binary unavailable"],"next_version":"manual_review","notes":["codex binary missing"]}' > "$CODEX_REVIEW_JSON"
  exit 0
fi

echo "Waiting for diff report: $DIFF_REPORT"
until [[ -f "$DIFF_REPORT" ]]; do
  sleep 15
done

if [[ ! -f "$CODEX_PING_STATUS" ]]; then
  bash "$SCRIPT_DIR/run_v3_codex_ping.sh"
fi

PING_STATUS="$(cat "$CODEX_PING_STATUS" 2>/dev/null || echo error)"
if [[ "$PING_STATUS" != "ok" ]]; then
  echo "{\"decision\":\"ambiguous\",\"dominant_bucket\":\"mixed\",\"evidence\":[\"codex ping status: $PING_STATUS\"],\"next_version\":\"manual_review\",\"notes\":[\"fallback to Claude-only consensus unless Codex headless recovers\"]}" > "$CODEX_REVIEW_JSON"
  exit 0
fi

SCHEMA_WIN="$(wslpath -w "$AGENT_REVIEW_SCHEMA_LINUX")"
OUT_WIN="$(wslpath -w "$CODEX_REVIEW_JSON")"
ITER_WIN="$(wslpath -w "$ITER_DIR_LINUX")"
MANIFEST_TEXT="$(cat "$VERSION_MANIFEST_LINUX")"

PROMPT=$(cat <<EOF
You are reviewing a robotics training iteration report.

Context:
- Current version: $CURR_VERSION
- Previous version: $PREV_VERSION
- Only one module should have changed.
- The version manifest is the canonical roadmap; follow it rather than inventing new modules.

Return structured JSON only with:
- decision
- dominant_bucket
- evidence
- next_version
- notes

Version manifest:
$MANIFEST_TEXT

Report:
$(cat "$DIFF_REPORT")
EOF
)

set +e
timeout 120 "$CODEX_BIN_LINUX" exec \
  --skip-git-repo-check \
  -C "$ITER_WIN" \
  --sandbox read-only \
  --output-schema "$SCHEMA_WIN" \
  -o "$OUT_WIN" \
  "$PROMPT" \
  2>"$CODEX_REVIEW_STDERR"
STATUS=$?
set -e

if [[ $STATUS -ne 0 ]]; then
  NOTE="fallback to Claude-only or desktop handoff"
  if grep -qi 'usage limit' "$CODEX_REVIEW_STDERR"; then
    NOTE="codex usage limited during review; keep Claude-only consensus for this iteration"
  elif [[ $STATUS -eq 124 ]]; then
    NOTE="codex review timed out; keep Claude-only consensus for this iteration"
  fi
  echo "{\"decision\":\"ambiguous\",\"dominant_bucket\":\"mixed\",\"evidence\":[\"codex review failed\"],\"next_version\":\"manual_review\",\"notes\":[\"$NOTE\"]}" > "$CODEX_REVIEW_JSON"
fi

echo "$CODEX_REVIEW_JSON"
