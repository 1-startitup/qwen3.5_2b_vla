#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/v3_dynamic.env"

mkdir -p "$(dirname "$DIFF_REPORT")"
mkdir -p "$ITER_DIR_LINUX/reviews"

cd "$ROOT"

echo "Waiting for eval json: $OFFICIAL_EVAL_JSON"
until [[ -f "$OFFICIAL_EVAL_JSON" ]]; do
  sleep 15
done

echo "Eval json detected. Building diff report."

"$PYTHON" "$ROOT/tools/diff_vs_prev.py" \
  --curr "$OFFICIAL_EVAL_JSON" \
  --prev "$PREV_EVAL_JSON" \
  --label "$CURR_VERSION" \
  --prev-label "$PREV_VERSION" \
  --out "$DIFF_REPORT"

cp "$OFFICIAL_EVAL_JSON" "$ITER_DIR_LINUX/official_full_ep10.json"
cp "$DIFF_REPORT" "$ITER_DIR_LINUX/diff_report.md"
cp "$VERSION_MANIFEST_LINUX" "$ITER_DIR_LINUX/version_manifest.yaml"

echo "diff report: $DIFF_REPORT"
echo "shared iter dir: $ITER_DIR_LINUX"
