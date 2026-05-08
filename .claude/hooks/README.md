# Claude Code hooks for lerobot-bench

Project-local hooks wired in `.claude/settings.local.json`. They exist to
prevent the four worktree gotchas memorialized in user memory:

- editable install drifts to last-installed worktree's `src/`
- worktree pytest leaks `.pyc` into parent `tests/__pycache__`
- subagents need worktree isolation
- editable install in worktrees points at main `src/` tree

Plus a guardrail against destructive `git` operations.

## What ships

| Script                              | Trigger                                                                    | Effect |
|-------------------------------------|----------------------------------------------------------------------------|--------|
| `post-agent-rebind.sh`              | `PostToolUse`, matcher `Agent`                                             | Cleans `tests/__pycache__` in parent project root, then re-runs `pip install -e . --no-deps -q` to rebind the parent's editable install. Always exits 0. |
| `block-dangerous-git.sh`            | `UserPromptSubmit`, matcher `*`                                            | Refuses prompts mentioning `--no-verify`, force-push to `main`, or `reset --hard origin/main`. Exits 2 to block. |
| `calibration-hint.sh`               | `PostToolUse`, matcher `Edit\|Write` on `results/calibration-*.json`        | Runs `scripts/auto_downscope.py <file> --dry-run`; appends non-empty output to the hints log. Informational, always exits 0. Tolerates a missing script. |

## Log paths

- `.claude/hooks/.last-postagent.log` — appended every Agent completion.
- `.claude/hooks/.calibration-hints.log` — appended only when the dry-run
  produces a non-empty diff. Likely empty until `scripts/auto_downscope.py`
  exists.

Both logs are gitignored; see `.claude/hooks/.gitignore`.

## How to disable

Comment or delete the relevant entry in `.claude/settings.local.json`. To
disable everything project-locally without touching the file, set
`CLAUDE_DISABLE_HOOKS=1` and re-launch Claude Code (Claude Code respects this
env var; project hooks remain on disk untouched).

## How to test manually

### `post-agent-rebind.sh`

```bash
# From inside an agent worktree:
bash .claude/hooks/post-agent-rebind.sh
tail -n 20 "$(git rev-parse --show-toplevel)/.claude/hooks/.last-postagent.log"
```

Expected log lines: `post-agent rebind starting (parent=...)`,
`cleaned N tests/__pycache__ dirs`, `rebound editable install at ...`.

### `block-dangerous-git.sh`

```bash
# Simulate a UserPromptSubmit payload:
echo '{"prompt": "please git push --force main"}' \
  | bash .claude/hooks/block-dangerous-git.sh; echo "exit=$?"

# Benign prompt:
echo '{"prompt": "please run the tests"}' \
  | bash .claude/hooks/block-dangerous-git.sh; echo "exit=$?"
```

The first should print the refusal to stderr and exit 2; the second exits 0.

### `calibration-hint.sh`

```bash
echo '{"tool_input": {"file_path": "results/calibration-libero.json"}}' \
  | bash .claude/hooks/calibration-hint.sh; echo "exit=$?"
```

Always exits 0 even if `scripts/auto_downscope.py` is missing.

## Constraints honored

- Never bumps `lerobot==0.5.1`. The reinstall passes `--no-deps`.
- Never runs `--no-verify` or skips git hooks.
- Never modifies the user-global `~/.claude/settings.json`.
- Never writes to `results/` (the existing `PreToolUse` block in
  `settings.json` already prevents that anyway).
