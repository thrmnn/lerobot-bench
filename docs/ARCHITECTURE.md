# Architecture

> Source of truth: [`docs/DESIGN.md`](DESIGN.md) (technical design) and
> [`docs/CEO-PLAN.md`](CEO-PLAN.md) (strategic framing).
> This file is the short index — links and diagrams only.

## High-level dataflow

```
                 ┌────────────────────────┐
                 │  configs/sweep_*.yaml  │
                 └────────────┬───────────┘
                              │
              ┌───────────────▼───────────────┐
              │  scripts/run_sweep.py         │
              │  (orchestrator + checkpoint)  │
              └─┬──────────┬───────────────┬──┘
                │          │               │
                ▼          ▼               ▼
       ┌──────────┐  ┌──────────┐   ┌─────────────┐
       │ envs.py  │  │policies  │   │  eval.py    │
       │ registry │  │ registry │   │ (runs cell) │
       └──────────┘  └──────────┘   └──────┬──────┘
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                      ┌───────────────┐         ┌──────────────┐
                      │ render.py     │         │ stats.py     │
                      │ MP4 + thumbs  │         │ bootstrap CI │
                      └───────┬───────┘         └──────┬───────┘
                              │                        │
                              └───────────┬────────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ results/<sweep>/     │
                              │   ├ results.parquet  │
                              │   ├ videos/*.mp4     │
                              │   └ manifest.json    │
                              └──────────┬───────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
                  ┌──────────────────┐          ┌────────────────────┐
                  │ HF Hub dataset   │          │ space/app.py       │
                  │ thrmnn/          │ <─reads─ │ Gradio UI          │
                  │ lerobot-bench-   │          │ (leaderboard +     │
                  │ results-v1       │          │  browse-rollouts)  │
                  └──────────────────┘          └────────────────────┘
```

## Module layout

| Module | Purpose |
| --- | --- |
| `lerobot_bench.envs` | Sim env registry: gym IDs, `max_steps`, success thresholds |
| `lerobot_bench.policies` | Policy registry: HF Hub repo IDs + revision SHAs + env compat |
| `lerobot_bench.eval` | Core eval loop: `(policy, env, seed, n_episodes) -> CellResult` |
| `lerobot_bench.stats` | Bootstrap CIs, paired Wilcoxon, Cohen's h, effect sizes |
| `lerobot_bench.render` | Episode → MP4 (256px / 10fps / ≤2MB), thumbnail strips |
| `lerobot_bench.checkpointing` | Per-cell skip logic on resume |
| `lerobot_bench.cli` | `lerobot-bench` entrypoint |

## Data contracts

See `docs/DESIGN.md` § Architecture sketch for the full `results.parquet`
schema and `manifest.json` field list. Headlines:

- **Granularity**: one row per episode (5 seeds × ≤50 episodes per cell).
- **Join key**: `sweep_timestamp` joins parquet rows to `manifest.json`.
- **Reproducibility key**: `(policy_revision, sweep_timestamp, seed_idx, episode_idx)`.

## Reproducibility & seeding contract

Mid-cell resume is **not** bit-reproducible because the torch generator advances
across episodes within a cell. `checkpointing.py` only resumes at cell boundaries.
Full seeding contract in `docs/DESIGN.md` § Methodology.

## Deploy

- **GitHub repo**: `thrmnn/lerobot-bench` — code, this repo.
- **HF Hub dataset**: `Theozinh0/lerobot-bench-results-v1` — parquet + videos.
- **HF Space**: `huggingface.co/spaces/thrmnn/lerobot-bench` — its own git remote.
  `space/` ships via `make space-deploy` which runs `git push hf-space main`.

No GitHub Actions deploy workflow in v1 — the bench itself runs on the dev box,
not in CI. CI is for lint + typecheck + fast tests only.
