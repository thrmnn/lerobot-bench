# Reproduce a leaderboard cell

`lerobot-bench` is a *reproducibility* benchmark. Every number on the
leaderboard is a claim, and this page is how anyone confirms a claim is true —
not "roughly right within error bars", but **bit-for-bit identical**.

If you only read one section, read [The contract](#the-contract).

## The contract

A **cell** is one `(policy, env, seed)` triple. The leaderboard's success rate
for a `(policy, env)` pair is the pooled outcome of its five seed cells, each
50 episodes.

The reproducibility contract (full statement in
[`docs/DESIGN.md`](DESIGN.md) § Methodology → Seeding contract):

> Same `lerobot` version **+** same pinned checkpoint SHA **+** same seed
> → **identical per-episode outcomes.**

Concretely, a cell is deterministic given those three inputs:

- At cell start the script seeds NumPy and Torch from `seed_idx * 1000`.
- Each episode `e` resets the env with `seed = seed_idx * 1000 + e`.
- Policy stochasticity inherits the Torch generator (it is *not* re-seeded per
  episode).

So re-running a cell on equivalent hardware must reproduce the exact same
per-episode `success` boolean sequence and the exact same `n_steps` sequence.
This is **not a statistical tolerance check** — a single flipped episode is a
real signal that something drifted, and `make reproduce` reports it as a
failure.

> One consequence of the seeding contract: the Torch generator advances across
> episodes within a cell, so mid-cell resume is *not* bit-reproducible. A
> reproduce run always re-runs a whole cell from episode 0.

## Prerequisites

You need the same environment the published sweep used:

1. **The conda env** with `lerobot==0.5.1` (the pin is sacred — a different
   `lerobot` release will diverge).

   ```bash
   conda activate lerobot      # or your env name
   python -c "import lerobot; print(lerobot.__version__)"   # must print 0.5.1
   ```

2. **`lerobot-bench` installed with all extras**, from the repo root:

   ```bash
   pip install -e ".[all]"
   ```

3. **The reference results parquet.** `make reproduce` compares against
   `results/sweep-full/results.parquet` by default. If you do not have it
   locally, pull the published dataset:

   ```bash
   huggingface-cli download thrmnn/lerobot-bench-results-v1 \
       --repo-type dataset --local-dir results/sweep-full
   ```

4. **A CUDA GPU.** The published sweep ran on an RTX 4060 Laptop (8 GB). Other
   GPUs should reproduce the contract; if you only have CPU, pass
   `--device cpu` to `scripts/reproduce_cell.py` directly (slower, and not the
   configuration the leaderboard was measured on).

## The command

```bash
make reproduce CELL=policy/env/seed
```

`CELL` is a `policy/env/seed` string — the same identifier the leaderboard
shows for a cell. Examples:

```bash
make reproduce CELL=act/pusht/0
make reproduce CELL=diffusion_policy/aloha_transfer_cube/3
```

To compare against a non-default reference parquet, or to run on CPU, call the
script directly:

```bash
python scripts/reproduce_cell.py --policy act --env pusht --seed 0 \
    --reference path/to/other/results.parquet

python scripts/reproduce_cell.py --policy act --env pusht --seed 0 --dry-run
```

`--dry-run` confirms the cell exists in the reference and reports its episode
count **without** running anything — a cheap sanity check before committing to
a multi-minute re-run.

## Expected wall-clock per cell

A cell is 50 episodes. From the Day 0b calibration (`scripts/calibrate.py`,
RTX 4060 Laptop):

| Policy class                  | Approx. per cell |
|--------------------------------|------------------|
| ACT                            | ~11 min          |
| Diffusion Policy               | ~20 min          |
| VLAs (SmolVLA, Pi0)            | longer — tens of minutes |

`make reproduce` skips video rendering (the contract is over `success` /
`n_steps`), so it is slightly lighter than the original sweep cell. Budget
a single cell under one hour on equivalent hardware.

## Reading the verdict

The script ends with one of two verdicts and a matching exit code.

**Reproduced** — exit `0`:

```
REPRODUCED ✓  act/pusht/seed0  (50/50 episodes identical)
```

Every episode's `success` and `n_steps` matched the reference. The cell's
leaderboard contribution is confirmed.

**Mismatch** — exit `1`:

```
MISMATCH ✗  act/pusht/seed0  (3/50 episodes diverged)
  first divergence at episode 7, column 'success':
    reference   = True
    reproduced  = False

likely causes:
  - lerobot version drift: ...
  - checkpoint SHA drift: ...
  - nondeterminism: ...
```

The re-run disagreed with the reference. The script names the first divergent
episode, both values, and the likely causes.

Other exit codes:

- `2` — the reference parquet is missing, or the requested cell is not in it.
  Check the [prerequisites](#prerequisites) and the `policy/env/seed` spelling.
- `3` — the re-run itself failed (`run_one.py` exited non-zero); see its output
  above the verdict for the cause.

## What a mismatch implies

A mismatch means the contract was broken somewhere. In order of likelihood:

1. **`lerobot` version drift.** The most common cause. Confirm
   `pip show lerobot` reports `0.5.1`. Any other release can change env
   dynamics or policy inference and break determinism.
2. **Checkpoint SHA drift.** The policy's `revision_sha` in
   `configs/policies.yaml` must be the exact commit the published sweep
   pinned. A re-tagged or moved Hub checkpoint produces different actions.
3. **Genuine nondeterminism.** A code path that escaped the seeding contract
   (see [`docs/DESIGN.md`](DESIGN.md) § Methodology). This is the interesting
   case and worth filing as an issue — it is a bug in the benchmark, not in
   your setup.

A reproduced cell is the benchmark working as designed. A mismatch is the
benchmark telling you something — start with the `lerobot` version.
