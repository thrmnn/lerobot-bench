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

Public, reproducible evaluation of pretrained [LeRobot](https://github.com/huggingface/lerobot) policies on standard simulation environments. One row per episode; videos for browsable failure analysis. This is **v1.0.2**.

The headline cell is **`act` × `aloha_transfer_cube` = 0.824 [0.772, 0.866]** (Wilson 95% CI, N=250, Hub-default inference). This supersedes the earlier v1.0.0 reading of **0.016**, which was a normalization bug *in our own eval harness* — image observations were fed to ACT un-normalized — fixed in PR #51 and confirmed by a controlled 2×2 ablation (the recovery is 100% the normalization fix, 0% temporal ensembling). See `docs/MODEL_CARDS.md` § ACT and `docs/INFERENCE_AUDIT.md` for the full account.

- **Code**: <https://github.com/thrmnn/lerobot-bench>
- **Live leaderboard (HF Space)**: <https://huggingface.co/spaces/thrmnn/embodimetry>
- **Methodology**: see `docs/DESIGN.md` § Methodology in the GitHub repo.
- **Statistical rigor doc**: see `docs/MDE_TABLE.md` for minimum-detectable-difference analysis at N=250.
- **Failure taxonomy**: see `docs/FAILURE_TAXONOMY.md` for the six-mode rollout labeling rubric.

## Policy scope (v1)

The published v1 set is **`act`, `diffusion_policy`, `smolvla_libero`, `no_op`, `random`**. `xvla_libero` is **excluded** from this dataset — its rows and its MP4s are filtered out at publish time and it is deferred to v1.1 (unresolved Hub-JSON processor wiring). Do not expect any `xvla` rows or videos here.

| Policy | Checkpoint (repo_id) | Weights license | Envs in v1 | Source |
|---|---|---|---|---|
| `act` | `lerobot/act_aloha_sim_transfer_cube_human` | apache-2.0 | `aloha_transfer_cube` | Zhao et al. 2023 ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)) |
| `diffusion_policy` | `lerobot/diffusion_pusht` | apache-2.0 | `pusht` | Chi et al. 2023 ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)) |
| `smolvla_libero` | `lerobot/smolvla_libero` | apache-2.0 | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | Shukor et al. 2025 ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) |
| `no_op` | n/a (weights-free) | MIT (this repo) | all six v1 envs | baseline (zero action) |
| `random` | n/a (weights-free) | MIT (this repo) | all six v1 envs | baseline (uniform action) |

Per-policy provenance (revision SHAs, parameter scale, calibrated ms/step, paper-vs-measured deltas) is in `docs/MODEL_CARDS.md`. Upstream weights keep their own licenses; this dataset's CC-BY-4.0 covers only the *evaluation artifacts produced here*.

**Success criterion**: a binary per-episode `success = final_reward >= env.success_threshold`. Each (policy, env) cell is 5 seeds × 50 episodes = **N=250** binary outcomes, except `diffusion_policy × pusht`, auto-downscoped to 25 episodes/seed (N=125) by the cost-budget rule (its iterative-denoising p95 tail crossed the threshold).

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
| `code_sha` | str | git SHA of the embodimetry commit that produced the row |
| `lerobot_version` | str | lerobot version (e.g., `0.5.1`) |
| `timestamp_utc` | str | ISO 8601 UTC timestamp of the row write |
| `errored` | bool | True if the episode crashed (OOM / env death) rather than failing the task. Back-filled `false` for rows written before the column existed. |
| `eval_run_id` | str | Per-sweep-invocation provenance handle. Empty string for rows written before the column existed. |

## Methodology in one paragraph

5 seeds × 50 episodes per (policy, env) cell, giving N=250 binary success outcomes per cell. The full v1 dispatch was 22 cells (18 published) × 5 seeds = 110 cell-seed runs, 0 failures. Two cells were auto-downscoped to 25 episodes/seed (N=125) by the cost-budget rule in `docs/DESIGN.md` § Methodology — `diffusion_policy × pusht` (published) and `xvla_libero × libero_10`; the latter is excluded from this dataset regardless (see Policy scope), and its downscope was dispatch-time only. The on-disk parquet holds 5250 rows = (20 cells × 250) + (2 downscoped × 125); after the xvla strip the *published* set is 18 cells (the published downscope is `diffusion_policy × pusht` only). Per-cell aggregation uses the Wilson 95% CI; pairwise policy comparisons use the percentile bootstrap (10,000 resamples) and paired Wilcoxon, with Cohen's h for effect size. The "inconclusive at this N" band is `2 × Wilson half-width @ p=0.5,N=250 = 0.123` — any |Δp̂| smaller than this is reported as inconclusive rather than ranked. Per-cell seeding is `numpy.random.seed(seed*1000) + torch.manual_seed(...)` once at cell start; per-episode seeding is `env.reset(seed=seed*1000+episode_index)`. Cell-boundary resume only — mid-cell crashes restart from episode 0 to preserve byte-identical reproducibility.

## How to load

```python
from datasets import load_dataset

ds = load_dataset("thrmnn/embodimetry-v1", "results", split="train")
df = ds.to_pandas()  # one row per episode; columns per the schema above

# Per-cell success rate (matches the leaderboard aggregation):
df.groupby(["policy", "env"])["success"].mean()
```

Or read the parquet directly without `datasets`:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/thrmnn/embodimetry-v1/results.parquet")
```

## Reproducibility

```bash
git clone https://github.com/thrmnn/lerobot-bench
cd embodimetry
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
