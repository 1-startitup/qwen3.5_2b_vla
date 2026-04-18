#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$ITER_DIR_LINUX/reviews"

if ! command -v claude >/dev/null 2>&1; then
  echo '{"final_decision":"ambiguous","agreed_bucket":"mixed","recommended_next_version":"manual_review","shared_rationale":["claude CLI unavailable"],"disagreements":["no merger available"]}' > "$CONSENSUS_JSON"
  exit 0
fi

echo "Waiting for agent reviews."
until [[ -f "$CLAUDE_REVIEW_JSON" && -f "$CODEX_REVIEW_JSON" ]]; do
  sleep 15
done

PROMPT=$(cat <<EOF
Merge these two agent reviews into one consensus JSON.

Current version: $CURR_VERSION
Previous version: $PREV_VERSION

Follow the version manifest and prefer conservative routing when one review is missing or degraded.

Version manifest:
$(cat "$VERSION_MANIFEST_LINUX")

Claude review:
$(cat "$CLAUDE_REVIEW_JSON")

Codex review:
$(cat "$CODEX_REVIEW_JSON")
EOF
)

set +e
claude -p \
  --bare \
  --permission-mode default \
  --tools "" \
  --output-format json \
  --json-schema "$(cat "$CONSENSUS_SCHEMA_LINUX")" \
  "$PROMPT" > "$CONSENSUS_RAW" 2>&1
STATUS=$?
set -e

if [[ $STATUS -eq 0 ]]; then
  cp "$CONSENSUS_RAW" "$CONSENSUS_JSON"
else
  REASON="fall back to reading Claude review and Codex status directly"
  if grep -qi 'Not logged in' "$CONSENSUS_RAW"; then
    REASON="claude CLI is not logged in, so automated consensus merge is unavailable"
  fi
  echo "{\"final_decision\":\"ambiguous\",\"agreed_bucket\":\"mixed\",\"recommended_next_version\":\"manual_review\",\"shared_rationale\":[\"consensus merge failed\"],\"disagreements\":[\"$REASON\"]}" > "$CONSENSUS_JSON"
fi

echo "$CONSENSUS_JSON"
