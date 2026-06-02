# Orchestration

How parallel-agent authoring waves run against this repo without stepping
on each other or on `main`'s branch protection. This is the *meta* doc —
`docs/RUNBOOK.md` covers running the sweep; this covers running the
agents that build the repo.

## Why this exists

Most non-trivial changes to this repo are dispatched as a **wave**: several
authoring agents run *concurrently*, each in its own git worktree, each
owning a **non-overlapping set of files**. Concurrency only works if no two
agents can ever touch the same path, and if their PRs merge into a
strict-protected `main` in a disciplined serial order. This doc codifies
both invariants so a wave is mechanical, not improvised.

## The two invariants

1. **Disjoint file ownership.** Every agent in a wave owns exactly one
   `.github/CODEOWNERS` glob (or a strict subset of one). No file is owned
   by two agents. An agent that discovers it needs a file outside its set
   **stops and reports it** rather than editing across the boundary.
2. **Serial drain into `main`.** `main` is strict-protected (linear
   history, squash-merge, required up-to-date branch). PRs therefore merge
   **one at a time**, each rebased/updated onto the latest `main` before it
   goes in.

## File ownership mirrors CODEOWNERS

The disjoint file sets are exactly the path globs in
[`.github/CODEOWNERS`](../.github/CODEOWNERS). Today every glob resolves to
the same maintainer (`@thrmnn`), but the globs are the **partition keys**
for a wave regardless of GitHub owner. The current partition:

| Glob | Surface | Typical agent |
|---|---|---|
| `/src/`, `/tests/` | library + unit tests | eval / stats engineer |
| `/scripts/`, `/configs/` | run/sweep/calibration CLI + pinned configs | sweep SRE |
| `/dashboard/`, `/space/` | the two public read surfaces | frontend engineer |
| `/site/` | static results site | site engineer |
| `/design/` | brand / visual identity | design |
| `/paper/` | arXiv writeup, deck, figures | researcher-writeup |
| `/.github/`, root meta (`Makefile`, `CONTRIBUTING.md`, this doc) | CI / release / process | meta (Claude Code) |
| `/research/` | future world-model / JEPA-planner track | WM track (not yet active) |

`/dashboard/` and `/space/` are split across agents only when the change is
surface-local; anything touching the shared v1-policy filter lives in
`src/lerobot_bench/leaderboard_filter.py` (the `/src/` owner) so the two
read surfaces cannot drift apart.

## Dispatching a wave

1. **Read the CODEOWNERS globs.** They are the menu of assignable file sets.
2. **Assign disjoint sets.** Give each agent one glob (or a named subset,
   e.g. "only `scripts/probes/`"). Write the owned set into the agent's
   brief verbatim: *"You OWN ONLY these files. Do not touch anything else."*
3. **Isolate with `isolation: "worktree"`.** Each agent gets its own
   working copy under `.claude/worktrees/`, branched from `main`. This is
   mandatory — concurrent agents sharing one working tree race the shared
   `HEAD`.
4. **One PR per agent.** Conventional-commit branch name, conventional PR
   title, tight body (what + why + test evidence).
5. **Drain serially** (next section).

## Test runs in a worktree

A worktree's editable install (`__editable__.<pkg>.pth`) is **static** — it
points at whichever `src/` was last `pip install -e`'d, which is the
**parent** tree, not the worktree. So:

- **Never** run `pip install -e .` inside a worktree. It silently rebinds
  the *parent's* editable install to the worktree's (soon-stale) `src/`,
  poisoning every other agent and the parent.
- Always run tests with the worktree's `src/` on `PYTHONPATH`:

  ```bash
  PYTHONPATH=$(pwd)/src /home/theo/miniforge3/envs/lerobot/bin/python -m pytest <targets> -q
  ```

  (`MUJOCO_GL=egl` is set by `tests/conftest.py`.) Without the explicit
  `PYTHONPATH` the worktree silently tests the parent's source and you get
  a false pass.

## Serial drain discipline

`main` is `strict: true` + squash-merge. Merges **serialize** — you cannot
land two PRs that were both branched off the same older `main` without
re-updating the second. Drive the drain as a loop:

1. Pick the next PR. Update its branch onto current `main`
   (`gh pr merge --auto` + "Update branch", or rebase locally and push).
2. Wait for required checks to go green on the *updated* branch.
3. Squash-merge it.
4. **Re-detect what actually landed before touching the next PR.** Squash
   merges rewrite history, so `git branch --merged` will *not* list a
   squash-merged branch. Use the PR state instead:

   ```bash
   gh pr list --state merged --limit 20
   ```

5. Repeat for the next PR.

### CHANGELOG is the recurring conflict

`CHANGELOG.md`'s `[Unreleased]` section is the one file almost every PR in a
wave edits, so it is the predictable merge conflict during the drain. When
PR *N* lands, PR *N+1*'s `[Unreleased]` block goes stale. Resolve by keeping
**both** entries (this is an append, never a replace) and re-running checks.
Two ways to cut the conflict down:

- Have each agent add its CHANGELOG line as the **last** bullet of its
  subsection, so conflicts are append-at-end rather than interleaved.
- If a wave is large, let one agent own CHANGELOG entirely and have the
  others note their user-visible change in the PR body for that agent to
  collate — but the default is each PR carries its own line.

## When an agent is blocked

If an agent needs to edit a file outside its owned set, it does **not** edit
it. It reports the wanted path in its result `notes` and finishes with the
changes it *could* make. The orchestrator then either reassigns that path to
the right owner in a follow-up wave or hands it to the owning agent next
round. Crossing the boundary "just this once" is what produces the
two-agents-one-file race the worktree isolation exists to prevent.

## Hard rules (every agent, every wave)

- Own only your file set. Outside it: stop and report, don't touch.
- Branch from `main`; conventional-commit branch + PR title.
- Never `pip install -e .` in a worktree; test via `PYTHONPATH=$(pwd)/src`.
- Never `--no-verify`, never force-push. If a hook fails, fix the cause.
- One PR per agent; drain serially; detect merges with
  `gh pr list --state merged`, not `git branch --merged`.
