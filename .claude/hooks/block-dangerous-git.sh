#!/usr/bin/env bash
# UserPromptSubmit hook: refuses prompts that ask for known-dangerous git ops
# (--no-verify, force push to main, hard reset to origin/main). Non-zero exit
# blocks the prompt with a stderr message visible to the user.
#
# Override: rephrase the prompt to remove the trigger token, or invoke the
# operation manually outside the agent.
set -euo pipefail

# Hook stdin payload: JSON with a "prompt" field per Claude Code hook spec.
prompt="$(jq -r '.prompt // empty' 2>/dev/null || true)"

if [[ -z "$prompt" ]]; then
  exit 0
fi

# Case-insensitive match on the danger patterns. Anchors loose on purpose so
# variations ("git push -f main", "push --force origin main", etc.) all trip.
if printf '%s' "$prompt" | grep -qiE -- '(--no-verify|push.*--force.*main|push.*-f.*main|reset.*--hard.*origin/main)'; then
  printf 'Refusing dangerous git operation. Override with explicit confirmation if intentional.\n' >&2
  exit 2
fi

exit 0
