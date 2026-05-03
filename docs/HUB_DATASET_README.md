---
license: mit
language:
- en
tags:
- robotics
- benchmark
- lerobot
- evaluation
- pretrained-policies
- pusht
- aloha
- libero
size_categories:
- 1K<n<10K
task_categories:
- robotics
pretty_name: LeRobot Multi-Policy Benchmark Results (v1)
configs:
- config_name: results
  data_files: results.parquet
---

# LeRobot Multi-Policy Benchmark — Results (v1)

Public, reproducible evaluation of pretrained [LeRobot](https://github.com/huggingface/lerobot) policies on standard simulation environments. One row per episode; videos for browsable failure analysis.

- **Code**: <https://github.com/thrmnn/lerobot-bench>
- **Live leaderboard (HF Space)**: <https://huggingface.co/spaces/Theozinh0/lerobot-bench>
- **Methodology**: see `docs/DESIGN.md` § Methodology in the GitHub repo.
- **Statistical rigor doc**: see `docs/MDE_TABLE.md` for minimum-detectable-difference analysis at N=250.
- **Failure taxonomy**: see `docs/FAILURE_TAXONOMY.md` for the six-mode rollout labeling rubric.

## What's in this dataset

| File | Contents |
|---|---|
| `results.parquet` | Flat per-episode outcome table (12 columns; one row per episode across all (policy, env, seed) cells) |
| `sweep_manifest.json` | Per-cell run metadata: status (`completed` / `failed` / `skipped`), exit codes, stderr tails, started/finished UTC |
| `_provenance.json` | Audit trail: code SHA, lerobot version, n_cells, n_episodes, total_video_bytes, published_utc |
| `videos/<policy>/<env>/seed<N>/episode<E>.mp4` | Per-episode rollout videos (256×256, 10 fps, H.264, ≤2 MiB; adaptive encoder ladder for long episodes) |

## Schema (`results.parquet`)

| Column | Type | Description |
|---|---|---|
| `policy` | str | Policy name (matches `configs/policies.yaml` in the source repo) |
| `env` | str | Environment name (matches `configs/envs.yaml`) |
| `seed` | int | Seed index (0..4); `numpy.random.seed(seed*1000)` applied at cell start |
| `episode_index` | int | 0..n_episodes-1 within the cell; `env.reset(seed=seed*1000+episode_index)` |
| `success` | bool | `final_reward >= env.success_threshold` |
| `return_` | float | Cumulative reward over the episode |
| `n_steps` | int | Steps before terminated/truncated/max_steps |
| `wallclock_s` | float | Wall-clock seconds for the episode |
| `video_sha256` | str (nullable) | sha256 of the rendered MP4 if recorded |
| `code_sha` | str | git SHA of the lerobot-bench commit that produced the row |
| `lerobot_version` | str | lerobot version (e.g., `0.5.1`) |
| `timestamp_utc` | str | ISO 8601 UTC timestamp of the row write |

## Methodology in one paragraph

5 seeds × 50 episodes per (policy, env) cell, giving N=250 binary success outcomes per cell. Per-cell aggregation uses the Wilson 95% CI; pairwise policy comparisons use the percentile bootstrap (10,000 resamples) and paired Wilcoxon, with Cohen's h for effect size. The "inconclusive at this N" band is `2 × Wilson half-width @ p=0.5,N=250 = 0.123` — any |Δp̂| smaller than this is reported as inconclusive rather than ranked. Per-cell seeding is `numpy.random.seed(seed*1000) + torch.manual_seed(...)` once at cell start; per-episode seeding is `env.reset(seed=seed*1000+episode_index)`. Cell-boundary resume only — mid-cell crashes restart from episode 0 to preserve byte-identical reproducibility.

## Reproducibility

```bash
git clone https://github.com/thrmnn/lerobot-bench
cd lerobot-bench
pip install -e .[dev]
pip install -e /path/to/lerobot                 # until lerobot==0.5.1 is on PyPI
huggingface-cli login                            # for re-publish only

# One cell:
python scripts/run_one.py \
  --policy diffusion_policy \
  --env pusht \
  --seed 0 \
  --n-episodes 50 \
  --out-parquet results/single.parquet

# Full sweep:
python scripts/run_sweep.py --config configs/sweep_full.yaml
```

The sweep is resumable cell-by-cell; killing it mid-cell restarts that cell from episode 0 on the next invocation.

## Citation

```bibtex
@misc{hermann2026lerobotbench,
  title  = {LeRobot Multi-Policy Benchmark: Open, Reproducible Evaluation of Pretrained Policies on PushT, Aloha, and Libero},
  author = {Hermann, Théo},
  year   = {2026},
  url    = {https://github.com/thrmnn/lerobot-bench}
}
```

## License

MIT. Per-policy weights are subject to their upstream licenses (apache-2.0 for the lerobot/* checkpoints in v1; see `docs/MODEL_CARDS.md` for the per-policy table).
