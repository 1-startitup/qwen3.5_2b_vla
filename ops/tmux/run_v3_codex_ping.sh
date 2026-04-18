#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$ITER_DIR_LINUX/reviews"

if [[ ! -x "$CODEX_BIN_LINUX" ]]; then
  echo "binary_missing" > "$CODEX_PING_STATUS"
  exit 0
fi

CODEX_PING_SCHEMA_LINUX="$ITER_DIR_LINUX/reviews/codex_ping.schema.json"
cat > "$CODEX_PING_SCHEMA_LINUX" <<EOF
{"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"],"additionalProperties":false}
EOF

SCHEMA_WIN="$(wslpath -w "$CODEX_PING_SCHEMA_LINUX")"
OUT_WIN="$(wslpath -w "$CODEX_PING_JSON")"
ITER_WIN="$(wslpath -w "$ITER_DIR_LINUX")"

set +e
timeout 15 "$CODEX_BIN_LINUX" exec \
  --skip-git-repo-check \
  -C "$ITER_WIN" \
  --sandbox read-only \
  --output-schema "$SCHEMA_WIN" \
  -o "$OUT_WIN" \
  "Return exactly the JSON object {\"reply\":\"PONG\"}." \
  >"$CODEX_PING_STDOUT" 2>"$CODEX_PING_STDERR"
STATUS=$?
set -e

if [[ $STATUS -eq 0 && -f "$CODEX_PING_JSON" ]] && grep -qi 'PONG' "$CODEX_PING_JSON"; then
  echo "ok" > "$CODEX_PING_STATUS"
elif grep -qi 'usage limit' "$CODEX_PING_STDERR"; then
  echo "usage_limited" > "$CODEX_PING_STATUS"
elif [[ $STATUS -eq 124 ]]; then
  echo "timeout" > "$CODEX_PING_STATUS"
else
  echo "error" > "$CODEX_PING_STATUS"
fi

echo "$CODEX_PING_STATUS"
