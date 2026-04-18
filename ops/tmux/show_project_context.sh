#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen35_pi05_native.env"

clear
printf '%s\n' \
  'Qwen35 pi0.5 Project Context' \
  '============================' \
  '' \
  "Context doc : $PROJECT_CONTEXT_DOC" \
  "Context json: $PROJECT_CONTEXT_JSON" \
  '' \
  'Training priority:' \
  '  1. maximize sample/s under the time budget' \
  '  2. push micro-batch close to OOM gradually' \
  '  3. only then increase grad accumulation' \
  '' \
  'Do not tune by power.draw alone. Use s/step, sample/s, and crash boundary.' \
  ''

sed -n '1,220p' "$PROJECT_CONTEXT_DOC"
