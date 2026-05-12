# Dashboard -- local-first operator UI

A local-only Gradio app that watches an in-flight (or completed)
lerobot-bench sweep. No Hub fetches; reads `results/` and `logs/` from
disk. Intended audience: the operator running `make sweep` who wants
to know "what is this thing actually doing right now?" without
tailing four terminals.

This is **not** the public-facing leaderboard -- that lives under
`space/` and reads from the HF Hub dataset. See "Public Space vs
local dashboard" below for the split.

## Launch

```bash
make dashboard
# opens http://127.0.0.1:7860
```

The dashboard has no GPU dependency; the heaviest module it imports
is pandas. A fresh `pip install -r dashboard/requirements.txt` boots
without the project's `[all]` extras (which pull torch + lerobot +
sim envs).

## Configure

| Env var | Default | Purpose |
|---|---|---|
| `DASHBOARD_RESULTS_DIR` | `<repo>/results` | where to find `sweep_manifest.json` and `calibration-*.json` |
| `DASHBOARD_LOGS_DIR` | `<repo>/logs` | where to tail `sweep-*.log` |

Both values are echoed in the **resolved-paths banner** directly
under the project-context header on every tab. If the dashboard
looks empty, the banner is the first place to check -- the previous
operator footgun was a dashboard pointing at an empty worktree
`results/` while the real sweep was running against the parent
checkout's results.

## Tabs

| Tab | When to use it |
|---|---|
| **Sweep progress** | Live per-`(policy, env)` grid: status, seeds-done, ETA. The default landing tab during an overnight sweep. |
| **Calibration inspector** | Static view of the latest `results/calibration-*.json` -- per-cell ms/step + VRAM + auto-downscope reason. Pre-sweep sanity check that the matrix is sane on this hardware. |
| **Rollout preview** | Four-dropdown cascade (policy / env / seed / episode) into the MP4 archive. Use this to skim a few qualitative rollouts after a cell completes. |
| **Live event log** | Tail of the active `logs/sweep-*.log` with colour-coded dispatch / success / error / breach lines, refreshing every 2 s. Replaces flipping back to the terminal. |
| **About** | Project pitch, 60-second methodology brief, sweep-flow diagram, v1 scope and limits, per-tab reading guide. First-time reviewer should start here. |

The persistent header above the tab strip keeps the project pitch,
sweep status (`M/N seeds done`), and links to the long-form docs
(paper Limitations, MDE table, failure taxonomy) on screen at all
times so a context-switch back into the dashboard doesn't require
re-reading from scratch.

## Public Space vs local dashboard

| | local dashboard (`dashboard/`) | public Space (`space/`) |
|---|---|---|
| Audience | the operator running the sweep | the reader / reviewer |
| Data source | local disk (`results/`, `logs/`) | HF Hub dataset by direct URL |
| Built for | live sweep visibility, taxonomy labelling | leaderboard browsing |
| Runtime | a laptop running `make dashboard` | HF Spaces free CPU tier |
| Refresh | live, polling | one-shot per page load |
| Gradio version | `gradio>=5,<6` | `gradio>=5,<6` (same pin) |

The two apps share the `_helpers` split-module pattern (no Gradio
import in the helpers module so tests can exercise them without
the heavy dep) but are otherwise independent.

## Empty-state checklist

If the dashboard is empty, check (in order):

1. **Resolved-paths banner** -- is `DASHBOARD_RESULTS_DIR` pointing
   at the right tree? Inside a worktree, `results/` is usually
   empty; set the env var to the parent checkout's results dir.
2. **Sweep progress tab** -- the "Sweep run" dropdown lists every
   `sweep_manifest.json` it found under `DASHBOARD_RESULTS_DIR`.
   No entries means no sweep has written its manifest yet (a brand
   new run takes ~30 s to flush the first cell).
3. **Calibration inspector tab** -- the status line at the top of
   the tab says which calibration JSON was read. Empty means no
   `results/calibration-*.json` exists yet -- run `make calibrate`
   first.
4. **Rollout preview tab** -- the status line reports how many MP4s
   were indexed and which roots were scanned. Zero usually means
   the Windows-mounted `~/Robotics-Data` drive is unmounted; the
   local `results/<run>/videos/` still works if `record_video:
   true` was set at sweep time.
5. **Live event log tab** -- the "Sweep log file" dropdown lists
   every `sweep-*.log` under `DASHBOARD_LOGS_DIR`. Empty means the
   sweep driver hasn't written its first log line yet.

If all five tabs are empty and the banner paths are correct, the
sweep is either not running or has not yet written anything. Tail
`logs/sweep-*.log` from a separate terminal to confirm.

## Architecture (one paragraph)

Two-file split: `app.py` (Gradio wiring, ~600 lines) imports from
`_helpers.py` (pure Python, ~1000 lines). The helpers module has no
Gradio import so `tests/test_dashboard.py` exercises its full surface
without the heavy dep (Gradio is in `[space]` extras, not `[dev]`).
The persistent header, per-tab intros, column glossaries, and About
tab body all render from helpers (`persistent_header_markdown`,
`per_tab_intro_markdown`, `column_glossary_markdown`,
`methodology_markdown`) so prose is testable too. Thresholds in
those strings interpolate the same constants
(`SLOW_MS_PER_STEP_THRESHOLD` et al.) that `downscope_reason` uses
-- the test suite pins this so the dashboard prose can't silently
drift from the calibration rule.
