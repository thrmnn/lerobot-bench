# Example: run one (policy, env, seed) cell end to end

This walks through running a **single benchmark cell** with
`scripts/run_one.py` and inspecting the per-episode rows it produces. A
*cell* is one `(policy, env, seed)` triple — the atomic unit the full
sweep is built from.

> **This example documents a command; it does not run on import.** A
> real cell loads a policy checkpoint and steps a sim env — minutes of
> GPU work. Run the command yourself when you have an env set up. To
> *read* already-produced rows without running anything, see
> [`read_results.py`](read_results.py).

## Prerequisites

A working install (`pip install -e ".[all]"` from the repo root) with
`lerobot==0.5.1`. See [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md)
for the full setup, including the `MUJOCO_GL` rendering backend you may
need on a headless box.

## Step 1 — dry-run first (no GPU, no download)

`--dry-run` resolves the policy and env through the registries and
prints the cell it *would* run. It imports neither torch nor lerobot,
so it is a safe, instant check that your `(policy, env)` pair is valid:

```bash
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --dry-run
```

If the pair is not compatible you get exit code `5` and a message
listing the valid envs for that policy. (`act` runs only on
`aloha_transfer_cube`; `diffusion_policy` only on `pusht`; the LIBERO
VLAs on the four `libero_*` suites — see the README § v1 scope matrix.)

## Step 2 — run the cell

Use a small `--n-episodes` for a quick smoke run (the sweep default is
`50`):

```bash
python scripts/run_one.py \
    --policy act \
    --env aloha_transfer_cube \
    --seed 0 \
    --n-episodes 5
```

Expected final log line (your success count will vary — the policy is
stochastic, but the cell is deterministic given the seed):

```
[run-one] policy=act env=aloha_transfer_cube seed=0 eps=5 success=3/5 rows_appended=5 out=results/results.parquet
```

Exit codes worth knowing:

| Exit | Meaning |
|---|---|
| `0` | Cell ran cleanly, rows appended. |
| `2` | Cell ran but some episodes errored — rows still appended (with `success=False`). |
| `3` | Policy not runnable (missing pinned `revision_sha`). |
| `4` | Missing runtime — `lerobot` or sim extras not installed. |
| `5` | Policy/env not in the registry, or env not in the policy's `env_compat`. |

Useful flags: `--no-record-video` skips the MP4 render entirely;
`--device cpu` forces CPU (slower, not the leaderboard configuration);
`--out-parquet` / `--videos-dir` redirect the outputs. Run
`python scripts/run_one.py --help` for the full list.

## Step 3 — inspect the rows you just produced

The cell appended one row **per episode** to `results/results.parquet`.
That parquet has one row per `(policy, env, seed, episode_index)` with
this schema (`lerobot_bench.checkpointing.RESULT_SCHEMA`):

| Column | Meaning |
|---|---|
| `policy` | policy name, e.g. `act` |
| `env` | env name, e.g. `aloha_transfer_cube` |
| `seed` | seed index (0..4 in the full sweep) |
| `episode_index` | episode within the cell (0-based) |
| `success` | binary outcome — the leaderboard's unit of evidence |
| `return_` | cumulative episode reward |
| `n_steps` | steps taken before termination/truncation |
| `wallclock_s` | episode wall-clock time |
| `video_sha256` | SHA-256 of the rendered MP4 (`""` if no video) |
| `code_sha` | `lerobot-bench` commit SHA |
| `lerobot_version` | pinned `lerobot` version (`0.5.1`) |
| `timestamp_utc` | ISO-8601 run timestamp |

Quick look at what landed:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('results/results.parquet')
cell = df[(df.policy == 'act') & (df.env == 'aloha_transfer_cube') & (df.seed == 0)]
print(cell[['episode_index', 'success', 'return_', 'n_steps']].to_string(index=False))
print(f'success rate: {cell.success.mean():.2f}  (n={len(cell)})')
"
```

A success rate from 5 episodes is **noisy** — do not read a ranking
into it. The next example, [`read_results.py`](read_results.py),
attaches a Wilson confidence interval so you can see how noisy. The
full sweep runs 5 seeds × 50 episodes = 250 episodes per cell precisely
because 5 is not enough.

## Reproducibility note

Re-running the *same* cell on equivalent hardware reproduces the exact
per-episode `success` and `n_steps` sequence — that is the benchmark's
reproducibility contract. To *verify* a published cell rather than
produce a fresh one, use `make reproduce` instead; see
[`docs/REPRODUCE.md`](../docs/REPRODUCE.md).
