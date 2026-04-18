#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$ITER_DIR_LINUX/reviews"

if ! command -v claude >/dev/null 2>&1; then
  echo '{"decision":"ambiguous","dominant_bucket":"mixed","evidence":["claude CLI unavailable"],"next_version":"manual_review","notes":["no automated Claude review available"]}' > "$CLAUDE_REVIEW_JSON"
  exit 0
fi

echo "Waiting for diff report: $DIFF_REPORT"
until [[ -f "$DIFF_REPORT" ]]; do
  sleep 15
done

MANIFEST_TEXT="$(cat "$VERSION_MANIFEST_LINUX")"

PROMPT=$(cat <<EOF
You are reviewing a robotics training iteration report.

Context:
- Current version: $CURR_VERSION
- Previous version: $PREV_VERSION
- Only one module should have changed.
- The version manifest is the canonical roadmap; follow it rather than inventing new modules.

Return structured JSON only.
Infer:
1. decision: accept / rollback / ambiguous
2. dominant_bucket
3. evidence: 2 to 5 short bullet-like strings
4. next_version
5. notes

Version manifest:
$MANIFEST_TEXT

Report:
$(cat "$DIFF_REPORT")
EOF
)

set +e
claude -p \
  --bare \
  --permission-mode default \
  --tools "" \
  --output-format json \
  --json-schema "$(cat "$AGENT_REVIEW_SCHEMA_LINUX")" \
  "$PROMPT" > "$CLAUDE_REVIEW_RAW" 2>&1
STATUS=$?
set -e

if [[ $STATUS -eq 0 ]]; then
  cp "$CLAUDE_REVIEW_RAW" "$CLAUDE_REVIEW_JSON"
else
  NOTE="fall back to manual inspection of diff report"
  if grep -qi 'Not logged in' "$CLAUDE_REVIEW_RAW"; then
    NOTE="claude CLI is not logged in; run /login before enabling automated Claude review"
  fi
  echo "{\"decision\":\"ambiguous\",\"dominant_bucket\":\"mixed\",\"evidence\":[\"claude review failed\"],\"next_version\":\"manual_review\",\"notes\":[\"$NOTE\"]}" > "$CLAUDE_REVIEW_JSON"
fi

echo "$CLAUDE_REVIEW_JSON"
