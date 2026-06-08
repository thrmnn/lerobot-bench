# Getting started

From `git clone` to **seeing a real benchmark result** in under five minutes.

This page runs *one* policy on *one* simulated environment and produces a
parquet row plus a rollout video on your own machine. It is the fastest way
to confirm your install works before you read `docs/REPRODUCE.md` (verify a
published cell) or `CONTRIBUTING.md` (add your own policy).

> Just want to *look* at results? You do not need to install anything — open
> the [live leaderboard](https://huggingface.co/spaces/thrmnn/embodimetry).
> This guide is for running the benchmark locally.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [The eval contract in one paragraph](#3-the-eval-contract-in-one-paragraph)
4. [Your first result](#4-your-first-result)
5. [What you just produced](#5-what-you-just-produced)
6. [Run the sweep](#6-run-the-sweep)
7. [Where to go next](#7-where-to-go-next)
8. [Common issues](#8-common-issues)

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| **Python 3.12** | The project pins `requires-python = ">=3.12"`. A conda env is recommended. |
| **`lerobot==0.5.1`** | Pulled in as a dependency. The pin is exact — a different release breaks the reproducibility contract. |
| **GPU** | A CUDA GPU is **recommended** for running cells. `run_one.py` defaults to `--device cuda`. CPU works (`--device cpu`) but is much slower and is not the configuration the leaderboard was measured on. |
| **Disk** | First run downloads the policy checkpoint from the HF Hub (tens to hundreds of MB depending on policy). |

The published sweep ran on an NVIDIA RTX 4060 Laptop (8 GB VRAM), 32 GB host
RAM, Ubuntu on WSL2. Any comparable CUDA box will do for this guide.

Create and activate the env:

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

## 2. Install

Clone the repo and install it in editable mode with **all extras** — the
extra is named `all` (it expands to `sim,viz,space,dev`, see
`pyproject.toml`):

```bash
git clone https://github.com/thrmnn/embodimetry.git
cd embodimetry
pip install -e ".[all]"
```

Verify the install and the `lerobot` pin in one shot:

```bash
python -c "import embodimetry, lerobot; print('embodimetry OK; lerobot', lerobot.__version__)"
```

Expected output:

```
embodimetry OK; lerobot 0.5.1
```

If `lerobot.__version__` is **not** `0.5.1`, stop here — fix the env before
running anything (see [Common issues](#6-common-issues)).

## 3. The eval contract in one paragraph

Everything the benchmark scores is reduced to one interface: a
**`PolicyCallable`** — `__call__(obs: dict) -> action: np.ndarray`, plus a
`reset()` invoked once per episode (a no-op for stateless policies). A
pretrained LeRobot checkpoint, a `no_op`/`random` baseline, a hand-written
controller, and a world-model planner running CEM at inference are *all* the
same callable, so they run the *same* eval loop and produce the *same*
parquet schema. A **cell** is a `(policy, env, seed, n_episodes)` tuple; the
seed deterministically drives the env reset and policy sampling, so any row
replays bit-for-bit. The loop pairs each callable with a `GymLikeEnv`
(`reset(seed)`, `step(action)`, `render()`), records the binary `success`
outcome per episode, and `embodimetry.stats` turns the pooled outcomes into a
Wilson 95% CI. That is the whole contract — see
[`docs/API.md`](API.md#module-eval) for the protocol and
[`docs/DESIGN.md`](DESIGN.md) § Methodology for the seeding rule.

## 4. Your first result

Run a single `(policy, env, seed)` **cell** with a small episode count.
`scripts/run_one.py` is the building block the full sweep dispatches to; with
`--n-episodes 5` it finishes in a couple of minutes on a GPU:

```bash
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --n-episodes 5
```

> The flag is `--n-episodes` (not `--episodes`). The default is 50; pass a
> small value for a quick smoke run. Run `python scripts/run_one.py --help`
> to see every flag.

Expected final log line (your success count will vary):

```
[run-one] policy=act env=aloha_transfer_cube seed=0 eps=5 success=3/5 rows_appended=5 out=results/results.parquet
```

The exit code tells you what happened:

| Exit | Meaning |
|---|---|
| `0` | Cell ran cleanly, 5 rows appended. |
| `2` | Cell ran but some episodes errored — rows still appended. |
| `3` | Policy not runnable (missing pinned `revision_sha`). |
| `4` | Missing runtime — `lerobot` or sim extras not installed. |
| `5` | Policy/env not in the registry, or env not in the policy's `env_compat`. |

**Want a zero-GPU, zero-download sanity check first?** Add `--dry-run`. It
resolves the policy and env through the registries and prints the cell it
*would* run — no weights, no sim, no torch import:

```bash
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --dry-run
```

## 5. What you just produced

The run wrote two artifacts under `results/` (gitignored):

**1. Parquet rows — `results/results.parquet`.** One row per episode. This is
the same schema every leaderboard number traces back to. Read it:

```bash
python -c "import pandas as pd; df = pd.read_parquet('results/results.parquet'); print(df[['policy','env','seed','episode_index','success','n_steps']])"
```

Each row carries the `(policy, env, seed, episode_index)` identity, the binary
`success` outcome, the step count, and timing fields. The leaderboard
success rate for a `(policy, env)` pair is the pooled outcome of its seed
cells — see `docs/REPRODUCE.md` § The contract.

**2. Rollout videos — `results/videos/*.mp4`.** One 256 px / 10 fps clip per
episode (skip them entirely with `--no-record-video`). Open any file to watch
the policy attempt the task — this is how the failure taxonomy
(`docs/FAILURE_TAXONOMY.md`) is labelled.

## 6. Run the sweep

One cell is the unit; a **sweep** dispatches the whole matrix. Start with the
smoke sweep (baselines only, minutes not hours) to confirm the pipeline end to
end:

```bash
make sweep-mini    # 2 baselines × 2 envs × 2 seeds × 25 episodes
```

The full benchmark runs serially — `scripts/run_sweep.py` dispatches one
`run_one.py` subprocess per cell, writes `results.parquet` incrementally, and a
`sweep_manifest.json` that survives `kill -9` (resume is automatic; a cell that
dies mid-run restarts from episode 0 at the next launch). v1 pins
`--max-parallel 1` (serial dispatch is the reproducible, RAM-safe default on
the 8 GB / 32 GB reference box); the flag exists for forward-compatible
concurrent scheduling but a value other than `1` is rejected in v1. The full
operating procedure — calibration, auto-downscope, the 18 GB cgroup cap, OOM
playbook — is in [`docs/RUNBOOK.md`](RUNBOOK.md) § Running a sweep, and the
dispatch/queue/resume mechanics in [`docs/ORCHESTRATION.md`](ORCHESTRATION.md).

## 7. Where to go next

| You want to… | Go to |
|---|---|
| Watch a sweep run live (progress, rollout preview, log tail) | `make dashboard` → http://127.0.0.1:7860 |
| **Verify** a published leaderboard cell is bit-for-bit reproducible | [`docs/REPRODUCE.md`](REPRODUCE.md) |
| Add your own pretrained policy to the benchmark | [`CONTRIBUTING.md`](../CONTRIBUTING.md) § Add a policy |
| Add a new simulated environment | [`docs/ENV_CONTRIBUTION_GUIDE.md`](ENV_CONTRIBUTION_GUIDE.md) |
| Run and operate the full overnight sweep | [`docs/RUNBOOK.md`](RUNBOOK.md) § Running a sweep |
| Understand the methodology (seeding, CIs, MDE, downscope) | [`docs/DESIGN.md`](DESIGN.md) |
| Browse all results without installing anything | [Live leaderboard](https://huggingface.co/spaces/thrmnn/embodimetry) |

## 8. Common issues

### MuJoCo / rendering fails headless

The Aloha and LIBERO envs render through MuJoCo, which needs an OpenGL
context. On a headless box or under WSL you may see `GLFW`, `EGL`, or
`DISPLAY`-related errors. Pick a rendering backend with `MUJOCO_GL`:

```bash
# Headless / server / CI — offscreen software rendering (most portable):
export MUJOCO_GL=osmesa

# NVIDIA GPU, no display — hardware offscreen rendering:
export MUJOCO_GL=egl
```

Under **WSLg** a display is available; export `DISPLAY=:0` instead and let
MuJoCo use the default GLX backend. Set the variable before launching
`run_one.py`, not after.

### `CUDA not available` / no GPU

`run_one.py` defaults to `--device cuda`. On a CPU-only box, force CPU
explicitly — it is slower and not the leaderboard configuration, but it runs:

```bash
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --n-episodes 5 --device cpu
```

If you *have* a GPU but `torch.cuda.is_available()` is `False`, your PyTorch
build is CPU-only. Reinstall a CUDA build of `torch` into the env.

### Wrong `lerobot` version

The benchmark pins `lerobot==0.5.1`. If `python -c "import lerobot; print(lerobot.__version__)"`
prints anything else, another package or a stale env pulled a different
release. Recreate the conda env and reinstall (`pip install -e ".[all]"`).
Version drift is the most common cause of a non-reproducing cell.

### `exit 5` — policy/env not compatible

Each policy declares an `env_compat` list. `act` runs only on
`aloha_transfer_cube`; `diffusion_policy` only on `pusht`; the LIBERO VLAs on
the four `libero_*` suites. The error message lists the valid envs. The full
matrix is in the README § v1 scope.

### `exit 4` — missing runtime

`lerobot` or the sim extras did not import. Confirm the env is active
(`conda activate lerobot`) and re-run `pip install -e ".[all]"` from the repo
root.

### Out-of-memory mid-cell

A single small `run_one.py` cell should not OOM on an 8 GB GPU. If it does,
your GPU is shared — close other CUDA processes. For full-sweep OOM handling
(cgroup memory cap, fp16 fallback, deferred Pi0 policies) see
`docs/RUNBOOK.md` § OOM playbook.
