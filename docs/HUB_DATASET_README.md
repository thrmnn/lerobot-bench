---
license: cc-by-4.0
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
- **Live leaderboard (HF Space)**: <https://huggingface.co/spaces/thrmnn/lerobot-bench>
- **Methodology**: see `docs/DESIGN.md` § Methodology in the GitHub repo.
- **Statistical rigor doc**: see `docs/MDE_TABLE.md` for minimum-detectable-difference analysis at N=250.
- **Failure taxonomy**: see `docs/FAILURE_TAXONOMY.md` for the six-mode rollout labeling rubric.

## Policy scope (v1)

The published v1 set is **`act`, `diffusion_policy`, `smolvla_libero`, `no_op`, `random`**. `xvla_libero` is **excluded** from this dataset — its rows and its MP4s are filtered out at publish time and it is deferred to v1.1 (unresolved Hub-JSON processor wiring). Do not expect any `xvla` rows or videos here.

## What's in this dataset

| File | Contents |
|---|---|
| `results.parquet` | Flat per-episode outcome table (one row per episode across all v1 (policy, env, seed) cells) |
| `sweep_manifest.json` | Per-cell run metadata: status (`completed` / `failed` / `skipped`), exit codes, stderr tails, started/finished UTC |
| `_provenance.json` | Audit trail: code SHA, lerobot version, n_cells, n_episodes, total_video_bytes, published_utc |
| `videos/{policy}__{env}__seed{seed}__ep{episode:03d}.mp4` | Per-episode rollout videos, **flat-named** (256×256, 10 fps, H.264, ≤2 MiB; adaptive encoder ladder for long episodes). Example: `diffusion_policy__pusht__seed3__ep042.mp4`. |

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
| `errored` | bool | True if the episode crashed (OOM / env death) rather than failing the task. Back-filled `false` for rows written before the column existed. |
| `eval_run_id` | str | Per-sweep-invocation provenance handle. Empty string for rows written before the column existed. |

## Methodology in one paragraph

5 seeds × 50 episodes per (policy, env) cell, giving N=250 binary success outcomes per cell. Two cells were auto-downscoped to 125 episodes (25 per seed) by the cost-budget rule in `docs/DESIGN.md` § Methodology — `diffusion_policy × pusht` and `xvla_libero × libero_10`; the latter is excluded from this dataset regardless (see Policy scope). Per-cell aggregation uses the Wilson 95% CI; pairwise policy comparisons use the percentile bootstrap (10,000 resamples) and paired Wilcoxon, with Cohen's h for effect size. The "inconclusive at this N" band is `2 × Wilson half-width @ p=0.5,N=250 = 0.123` — any |Δp̂| smaller than this is reported as inconclusive rather than ranked. Per-cell seeding is `numpy.random.seed(seed*1000) + torch.manual_seed(...)` once at cell start; per-episode seeding is `env.reset(seed=seed*1000+episode_index)`. Cell-boundary resume only — mid-cell crashes restart from episode 0 to preserve byte-identical reproducibility.

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

This dataset (the parquet tables, manifests, and rendered MP4 rollouts) is released under **[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** — attribute the LeRobot Multi-Policy Benchmark (see Citation). This covers the *evaluation artifacts produced by this benchmark*, not the upstream policy weights or environments, which keep their own licenses.

Per-policy weights are subject to their upstream licenses (apache-2.0 for the `lerobot/*` checkpoints in v1; see `docs/MODEL_CARDS.md` for the per-policy table).

## Attribution / NOTICE

This dataset is derived from running third-party open-source policies and simulation environments. The benchmark code and these artifacts build on:

- **[LeRobot](https://github.com/huggingface/lerobot)** (Hugging Face) — Apache-2.0. Policy implementations, pretrained `lerobot/*` checkpoints, and evaluation harness primitives.
- **[gym-pusht](https://github.com/huggingface/gym-pusht)** — Apache-2.0. The PushT environment.
- **[gym-aloha](https://github.com/huggingface/gym-aloha)** — Apache-2.0. The Aloha (transfer-cube) environment.
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** — the Libero environment suite (libero_10, etc.), under its upstream license.

See the `NOTICE` file at the root of the [source repository](https://github.com/thrmnn/lerobot-bench) for the full third-party attribution text. Upstream Apache-2.0 components retain their original copyright notices and license terms; nothing here relicenses them.
