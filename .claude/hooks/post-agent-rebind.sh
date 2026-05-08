#!/usr/bin/env bash
# PostToolUse hook on Agent: cleans worktree pycache leak and rebinds parent's
# editable install. Exists because `pip install -e .` inside an agent worktree
# silently retargets the parent project's __editable__.<pkg>.pth at the
# worktree's stale src/.
#
# See user memory: feedback_editable_install_drift, reference_worktree_pycache_leak.
set -euo pipefail

LOG="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/hooks/.last-postagent.log"
mkdir -p "$(dirname "$LOG")"

# Climb from the current cwd until we find the parent project root. The parent
# is defined as: the nearest ancestor that contains pyproject.toml AND is NOT
# inside a .claude/worktrees/ path.
find_parent_root() {
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pyproject.toml" && "$d" != *"/.claude/worktrees/"* ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

PARENT="$(find_parent_root || true)"
if [[ -z "${PARENT:-}" ]]; then
  printf '[%s] no parent project root found from cwd=%s; skipping\n' \
    "$(date -Is)" "$PWD" >>"$LOG"
  exit 0
fi

{
  printf '\n[%s] post-agent rebind starting (parent=%s)\n' "$(date -Is)" "$PARENT"

  # Narrow pycache cleanup: only inside tests/ subtrees of the parent.
  cleaned=$(find "$PARENT" -type d -name __pycache__ -path '*/tests/*' \
    -prune -print -exec rm -rf {} + 2>/dev/null | wc -l)
  printf '  cleaned %s tests/__pycache__ dirs\n' "$cleaned"

  # Rebind editable install on the parent. --no-deps is critical: we never want
  # an agent hook to touch the lerobot==0.5.1 pin or any other dep.
  if (cd "$PARENT" && python -m pip install -e . --no-deps -q) 2>&1; then
    printf '  rebound editable install at %s\n' "$PARENT"
  else
    printf '  WARN: pip install -e . --no-deps failed (non-fatal)\n'
  fi
} >>"$LOG" 2>&1

exit 0
