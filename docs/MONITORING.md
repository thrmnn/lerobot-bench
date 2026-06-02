# Experiment monitoring layer

Status: **proposed schemas pending user confirmation.** The dashboard
code is live; the two on-disk contracts described below (the run
registry and the training JSONL) are PROPOSALS the user should confirm
before downstream tooling (the world-model repo) hard-codes them.

Owner: dashboard. Source of truth for the operator-facing monitoring
story. The public HF Space (`space/`) is a separate, read-only artifact
and is out of scope here.

## Why this exists

`dashboard/app.py` started as a single-question tool: *is my overnight
sweep healthy?* That question is answered well by the **Status** tab.
But research infra needs three more things the operator kept doing by
hand:

1. **Look at more than the live run.** Compare this sweep against last
   week's; review a finished run while a new one is queued.
2. **Compare numbers that should agree.** Paper-reported vs our re-run;
   and (forward-looking) a world-model planner vs a VLA baseline.
3. **Drill into failures**, not just see the success rate.
4. **Watch slow-lane training** that has no wandb (offline laptop).

The monitoring layer adds those without disturbing the single-sweep
view: the Status tab still defaults to the newest run and still passes
the 5-second test on first paint.

## The tabs

| Tab | Question | Data source |
|---|---|---|
| **Status** | Is the *selected* run healthy? | manifest + parquet of the selected run |
| **Pre-flight** | Did calibration downscope anything? | `results/calibration-*.json` |
| **Rollouts** | What does a policy actually do? | MP4 archive (flat naming) |
| **Compare** | Do paper / re-run (and WM / VLA) agree? | parquet + `MODEL_CARDS.md` paper rates |
| **Failures** | *How* does a cell fail? | parquet (label-free) + failed-episode MP4s |
| **Training** | Is the WM training run progressing? | `results/wm-runs/<id>/progress.jsonl` |

### Status — cross-run selector

The Status tab gains a **run selector** dropdown. It defaults to the
newest run (the live sweep), so the landing experience is unchanged. The
selector value is the run *name* (the directory basename under
`results/`); a `gr.State` holds it and every status handler
(`refresh_status`, the progress grid, the live leaderboard, the
results-df loader) resolves the selected run via
`resolve_selected_run`, falling back to newest when the selection is
stale (the run dir was moved, or `DASHBOARD_RESULTS_DIR` changed
mid-session). The raw-log accordion still tails the newest `sweep-*.log`
— logs are not yet keyed per-run on disk.

### Compare — two views on shared axes

Both comparison views are "two success rates per `(policy, env)` cell
plus a colour-chipped delta", so they share one table renderer and the
existing `delta_chip` thresholds (green |Δ| < 0.05 · yellow ≤ 0.15 ·
red > 0.15):

- **Paper-reported vs measured.** Left is the policy's published rate,
  read from `docs/MODEL_CARDS.md` via the `PolicySpec.paper_reported_success`
  map (the same source the policy cards use — `MODEL_CARDS.md` is
  read-only to this layer). Right is our pooled re-run rate from the
  selected run's parquet. Cells with no parquet rows show `(pending)`.
- **World-model vs VLA (forward-looking).** Left is a world-model
  planner run *as a policy* (`act(obs) -> action`); right is a VLA
  baseline, on the envs both ran. **In v1 this is an empty state**: the
  parquet is filtered to v1 leaderboard policies on load, so no
  world-model rows are present. The view activates the moment a non-v1
  policy appears in the parquet. The "is this a WM policy?" test is
  currently the heuristic "policy name not in `V1_POLICIES`"
  (`wm_policy_names`) — PROVISIONAL until a real `kind` column lands
  (see *Out of scope*, below).

### Failures — graceful, label-free drill-down

**The parquet has no `failure_mode` column yet.** A real one is a future
schema bump (see below and `docs/FAILURE_TAXONOMY.md`). Until then the
Failures tab degrades gracefully and shows only the signals already on
disk for a selected cell:

- success / episode-length (`n_steps`) distributions;
- the **cap-hit rate** — the fraction of *failed* episodes whose
  `n_steps` reached the env's `max_steps`. This is the closest
  label-free proxy to the taxonomy's **Timeout** mode (a high cap-hit
  rate means failures are timeouts, not early aborts);
- direct MP4 links to failed episodes via the flat video naming
  `{policy}__{env}__seed{seed}__ep{NNN}.mp4`.

When the real `failure_mode` column lands, this tab grows a per-mode bar
chart; the prose framing already lives in `docs/FAILURE_TAXONOMY.md`.

### Training — slow-lane WM-run visibility

Tails the newest `results/wm-runs/<run_id>/progress.jsonl` if present;
absent => a friendly empty state (the v1 default — no WM training has
run). Refreshes every 5 s. The writer is `scripts/wm_run_log.py`.

## Proposed schema 1 — run registry

**Status: PROPOSED, pending user confirmation.**

There is no new on-disk file for the run registry. A "run" is exactly
what `discover_sweep_runs` already finds: a directory under `results/`
containing a `sweep_manifest.json`. The registry is the in-memory list
of those, and the proposal is only about *what the selector stores*:

- The selector's value is the run **name** (directory basename), never a
  `Path`. This keeps the selection stable across a re-discovery: a run
  that finishes only changes its label (`[running]` -> `[done]`), not
  its name.
- Resolution is always "selected name, else newest" (`resolve_selected_run`).
  A stale name (dir gone) silently falls back to newest rather than
  erroring.

If the user wants per-run *logs* (so the raw-log accordion follows the
selected run rather than the newest log), that is a follow-up: it needs
the sweep driver to name logs per run, which touches `scripts/run_sweep.py`
(out of scope for this wave).

## Proposed schema 2 — training-progress JSONL

**Status: PROPOSED, pending user confirmation.** The world-model repo
will import `scripts/wm_run_log.py` later; confirm the shape before that
happens.

- **Path:** `results/wm-runs/<run_id>/progress.jsonl`, one per training
  run. `<run_id>` is the dashboard's display name (the subdir basename).
- **Record (one JSON object per line):**

  ```json
  {"ts": "2026-06-02T18:00:00+00:00", "run_id": "jepa-pusht-001",
   "step": 1200, "metric": "loss", "value": 0.0431}
  ```

  | key | type | meaning |
  |---|---|---|
  | `ts` | str (ISO-8601, `+00:00`) | when the record was written |
  | `run_id` | str | run identifier (matches the subdir name) |
  | `step` | int | training step the metric was measured at |
  | `metric` | str | metric name (`loss`, `val_loss`, `lr`, …) |
  | `value` | float | the measured value |

- **Append-only, flushed per record** so a `kill -9` leaves a valid
  prefix; the dashboard reader (`read_wm_progress`) skips a half-written
  trailing line and tolerates both missing and extra keys, so the WM
  repo can extend the record later without breaking the Training tab.
- **Offline-first.** `scripts/wm_run_log.py` is stdlib-only (no wandb,
  no network, no third-party deps) and imports nothing from
  `dashboard/` or `src/lerobot_bench/`. The `WM_RUNS_SUBDIR` /
  `WM_PROGRESS_FILENAME` constants are duplicated between the writer and
  `dashboard/_helpers.py` on purpose to keep that import boundary clean;
  if you rename one, rename both.

Usage:

```bash
# from a training loop
python -c "from scripts.wm_run_log import log_progress; \
  log_progress('jepa-pusht-001', step=1200, metric='loss', value=0.0431)"

# from the shell (one record)
python scripts/wm_run_log.py --run-id jepa-pusht-001 \
  --step 1200 --metric loss --value 0.0431

# smoke test (a few synthetic records under a temp dir)
python scripts/wm_run_log.py --run-id smoke --results-dir /tmp/wm --demo
```

## Out of scope for this wave (deliberately untouched)

- **`failure_mode` parquet column.** A real failure-mode label is a
  future per-episode parquet schema bump (owned by `src/lerobot_bench/`
  + the eval writer). This layer only *reads* the parquet; it does not
  change the schema. See `docs/FAILURE_TAXONOMY.md`.
- **World-model dispatch.** Evaluating a WM/JEPA planner as a policy is
  a future `load_policy` dispatch branch in `src/lerobot_bench/eval.py`
  (a `kind` field), not a dashboard change. The Compare tab's
  `wm_policy_names` heuristic is the placeholder until that lands.
- **The sweep parquet, manifest, and `scripts/run_sweep.py`** are not
  modified by the monitoring layer.

## Tests

The data layer is gradio-free and exercised directly by
`tests/test_dashboard.py` (the test job has no Gradio install). The
Gradio wiring in `dashboard/app.py` is smoke-tested by building the app
and invoking each tab's handler in its empty-state and populated paths.
