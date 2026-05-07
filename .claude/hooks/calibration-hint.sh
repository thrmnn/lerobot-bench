#!/usr/bin/env bash
# PostToolUse hook on Edit|Write of results/calibration-*.json: runs
# scripts/auto_downscope.py --dry-run and appends any non-empty diff as a hint.
# Informational only: always exits 0 so it never blocks tool calls.
#
# auto_downscope.py may not exist yet; the call is wrapped in `set +e` so a
# missing script just produces no hint.
set -uo pipefail  # NOTE: no -e by design

LOG="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/.calibration-hints.log"
mkdir -p "$(dirname "$LOG")"

file_path="$(jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"

# Only react to results/calibration-*.json paths.
if [[ -z "$file_path" ]] || ! printf '%s' "$file_path" \
    | grep -qE 'results/calibration-.*\.json$'; then
  exit 0
fi

# Resolve project root for the script path. Fall back to cwd.
root="${CLAUDE_PROJECT_DIR:-$PWD}"
script="$root/scripts/auto_downscope.py"

set +e
out=""
if [[ -f "$script" ]]; then
  out="$(python "$script" "$file_path" --dry-run 2>&1)"
  rc=$?
else
  out=""
  rc=0
fi
set -uo pipefail

if [[ -n "$out" && "$rc" -eq 0 ]]; then
  {
    printf '\n[%s] calibration hint for %s\n' "$(date -Is)" "$file_path"
    printf '%s\n' "$out"
  } >>"$LOG"
fi

exit 0
