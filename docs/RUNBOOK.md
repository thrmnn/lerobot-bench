# Runbook

Operational doc for running, resuming, publishing, and rolling back the
benchmark sweep and the public Space. Authoritative *how-to-do-the-thing*
companion to `docs/DESIGN.md` (the *what* and *why*).

## Contents

0. [GPU health & WSL2 GPU-PV desync (phase-0)](#gpu-health--wsl2-gpu-pv-desync-phase-0)
1. [Day 0 — calibration spike](#day-0--calibration-spike)
2. [Running a sweep](#running-a-sweep)
3. [Resume drill (cell-boundary)](#resume-drill-cell-boundary)
4. [OOM playbook](#oom-playbook)
5. [Publish to HF Hub dataset](#publish-to-hf-hub-dataset)
6. [Deploy + roll back the Space](#deploy--roll-back-the-space)
7. [Verify reproducibility of a published cell](#verify-reproducibility-of-a-published-cell)
8. [Release a new embodimetry version](#release-a-new-embodimetry-version)
9. [Prune stale agent worktrees](#prune-stale-agent-worktrees)

---

## GPU health & WSL2 GPU-PV desync (phase-0)

**Run this before any GPU task.** It is phase-0 of the gpu-task workflow:
no CUDA dispatch happens until the preflight is green.

```bash
scripts/gpu_preflight.sh                       # default 1500 MB headroom
scripts/gpu_preflight.sh --required-headroom-mb 2000
# or wrap a capped GPU launch so it fails fast on a dead GPU:
scripts/run_capped.sh --gpu-preflight 18G -- python scripts/run_one.py --policy act --env pusht --seed 0
```

### What it checks (and the exit codes)

| exit | check | meaning / action |
|------|-------|------------------|
| `0`  | all pass | healthy — dispatch is safe |
| `10` | `nvidia-smi -L` unreachable | driver channel down — **likely GPU-PV desync**; `wsl --shutdown` |
| `11` | `torch.cuda` unusable | nvidia-smi up but CUDA runtime crashed/saw no device — treat as desync; `wsl --shutdown` |
| `12` | free VRAM < headroom | another process holds the card — wait / free it before dispatching |

The torch check runs **in a subprocess** on purpose: a desynced adapter
SIGSEGVs inside the CUDA runtime, and out-of-process that crash is a
nonzero exit code here instead of taking down the caller. This converts
the cryptic ~30-min diagnosis from the 2026-06-09 incident into a
2-second clear error.

### Symptoms of a GPU-PV desync (incident 2026-06-09)

- `nvidia-smi` → "couldn't communicate with the NVIDIA driver".
- `torch.cuda.is_available()` / raw `cuInit(0)` **SIGSEGV** even in a
  fresh child process.
- `dmesg` spams `misc dxg: dxgk: dxgkio_query_adapter_info: Ioctl failed:
  -22` (and `dxgkio_destroy_paging_queue: Ioctl failed: -22`). `/dev/dxg`
  still exists, but the guest↔host channel returns `-22 EINVAL`.

### Root cause: near-OOM TDR

The host-side WSL2 virtual GPU adapter lost sync with Windows. The trigger
here was a JEPA-WM run pinning the 8 GB RTX 4060 to **~96% VRAM with
allocator thrash** — sustained near-OOM VRAM is a known WSL2 GPU TDR /
GPU-PV-desync trigger. (Other triggers: a mid-session Windows NVIDIA
driver auto-update, host sleep/resume.)

### VRAM headroom rule (prevention)

Keep **≥25% of the 8 GB card free** at all times. Concretely:

- The VRAM-budget scheduler default is **6000 MB** (lowered from 7000 on
  2026-06-11) — `embodimetry.vram_scheduler.DEFAULT_VRAM_BUDGET_MB`. 7000
  of 8192 MB is ~85% reserved *before* the CUDA context and allocator
  fragmentation, which is into the TDR danger band.
- Optionally run the watchdog with the **VRAM-ceiling guard**, which
  aborts a run that stays near-OOM too long:

  ```bash
  python scripts/watchdog.py --out results/watchdog.jsonl --pgid <PGID> \
    --vram-ceiling-pct 90 --vram-ceiling-seconds 120
  ```

  cgroup `MemoryMax` (see `scripts/run_capped.sh`) stays the **primary RAM
  defense**; this ceiling is an *additive* VRAM defense, not a replacement.

### Recovery (the only fix)

A GPU-PV desync is **not recoverable in-guest** (no GPU sysfs reset in
WSL; dxgkrnl is built-in; killing guest processes won't re-handshake). The
only fix is restarting the WSL2 VM:

```text
# from Windows (PowerShell/cmd):
wsl --shutdown          # ~8s, then reopen WSL and reconnect
```

⚠️ WSL2 runs all distros in one VM, so this also kills any parallel
SSH/work in the same distro — **checkpoint first; do not run it
autonomously.** After restart, re-run `scripts/gpu_preflight.sh`; it
should be green before you resume GPU work.

---

## Day 0 — calibration spike

```bash
conda activate lerobot
make install
huggingface-cli login          # write scope; needed for publish step
make calibrate                 # writes results/calibration-YYYYMMDD.json
```

Inspect `mean_ms_per_step`, `p95_ms`, `vram_peak_mb` per (policy, env). The
auto-downscope rule in `scripts/run_sweep.py` consumes this file at sweep
start. If a VLA policy's per-cell minimum already exceeds 3 hours,
**drop that policy from the v1 matrix** in `configs/sweep_full.yaml` —
do not silently truncate.

## Running a sweep

```bash
# Smoke (3 policies × 2 envs × 2 seeds × 25 episodes — minutes, not hours):
make sweep-mini

# Full (overnight; 2 nights budgeted):
make sweep-full SWEEP_NAME=$(date +%Y%m%d)
```

The full sweep emits a per-cell heartbeat to stdout and an append-only
`sweep.log` next to `results/<sweep>/results.parquet`. Tail with:

```bash
tail -f results/sweep-*/sweep.log
```

## Resume drill (cell-boundary)

Mid-cell death = cell restarts from episode 0 next run. This is by design
(see `docs/DESIGN.md` § Methodology — Resumability).

```bash
# 1. Kill the sweep process (or simulate WSL sleep / SIGKILL):
pkill -f run_sweep.py

# 2. Verify which cells are already in the parquet:
python -c "
import pandas as pd
df = pd.read_parquet('results/sweep-YYYYMMDD/results.parquet')
print(df.groupby(['policy','env','seed']).size())
"

# 3. Restart — same command. Already-complete cells are skipped:
make sweep-full SWEEP_NAME=YYYYMMDD
```

If the restart picks up a stale cell (because the parquet write was
truncated on SIGKILL), delete the partial cell's rows by `(policy, env,
seed)` and rerun:

```python
df = pd.read_parquet(path)
df = df[~((df.policy=='X') & (df.env=='Y') & (df.seed==Z))]
df.to_parquet(path, index=False)
```

## OOM playbook

Triggered by `torch.cuda.OutOfMemoryError` mid-cell.

1. **Single-cell OOM**: log the cell, drop to fp16 if not already, retry once. If still OOM, drop the policy from this sweep and add to `manifest.json#dropped_policies` with reason.
2. **VRAM creep across cells**: insert `torch.cuda.empty_cache()` between cells if not already (see `embodimetry.eval.run_cell`). If the creep persists, restart the process every N cells via the orchestrator.
3. **Pi0 specifically**: design decision is to drop, not quantize. See `docs/DESIGN.md` § Open Questions Q3.

## Publish to HF Hub dataset

```bash
huggingface-cli login                          # if not already
make publish SWEEP=results/sweep-YYYYMMDD
```

`scripts/publish_results.py` is idempotent: files already on Hub with
matching SHA are skipped. The dataset repo is
`thrmnn/embodimetry-v1`. Bump to `-v2` only on a breaking
schema change.

## Deploy + roll back the Space

`space/` is **not** a standalone repo — it lives inside this monorepo. A flat
`git push` of the parent would nest `app.py` under `space/` where the Space
can't find it, so the deploy uses a **subtree push** that lands `space/`'s
contents at the Space root.

One-time setup (create the Space + add the remote; auth as `thrmnn`):

```bash
huggingface-cli repo create embodimetry --type space --space_sdk gradio
git remote add hf-space https://huggingface.co/spaces/thrmnn/embodimetry
```

Then deploy (the target guards on the `hf-space` remote existing and prints
the two commands above if it doesn't):

```bash
make space-deploy
# == git subtree push --prefix space hf-space main
```

The Space reads the dataset by URL, so it renders non-empty only after the
dataset parquet is published (see Publish step). `space/requirements.txt`
pins the project at a GitHub SHA — confirm that SHA is an ancestor of `main`
before deploying.

Roll back if a push broke the Space. A subtree push has no plain
`main~1:main` ancestor on the remote, so re-push the previous good
`space/` tree by checking out an earlier parent commit and re-running the
subtree push (or push a revert commit through it):

```bash
git subtree push --prefix space hf-space main   # after reverting space/ on main
```

The Space is configured for free CPU tier; `space/requirements.txt`
pins `lerobot==0.5.1`. App boot time is monitored locally by:

```bash
cd space && GRADIO_SERVER_PORT=7860 python app.py
curl -I http://127.0.0.1:7860/
```

CI runs the same boot test on every push to `space/**` via
`.github/workflows/space-smoke.yml`.

## Verify reproducibility of a published cell

```bash
# Pulls the manifest from the published dataset, picks one
# (policy, env, seed, episode_index), reruns, compares success.
python scripts/verify_repro.py \
  --dataset thrmnn/embodimetry-v1 \
  --sweep YYYYMMDD \
  --policy diffusion_policy --env pusht --seed 0 --episode 0
```

Bit-equality is guaranteed only at cell boundaries. Within-cell episode
index is reproducible only if the cell starts from episode 0 (mid-cell
restart is documented as non-reproducible).

## Release a new embodimetry version

```bash
# 1. Bump three places in lock-step:
#      VERSION
#      src/embodimetry/__version__.py
#      pyproject.toml [project].version
# 2. Update CHANGELOG.md — move [Unreleased] entries under [X.Y.Z].
# 3. Commit, tag, push the tag — release.yml validates the three are equal.

git commit -am "chore(release): vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

Tags are immutable. A bad release is fixed by the next version, not by
moving the tag.

## Prune stale agent worktrees

Repo-modifying agents run in isolated worktrees under `.claude/worktrees/`.
Once their branches merge, the worktrees are dead weight. Clean them up:

```bash
make worktree-prune
```

It is **conservative by design** — it only touches worktrees under
`.claude/worktrees/`, and it **skips** any that are (a) the current one,
(b) have uncommitted changes, or (c) sit on a branch not yet merged into
`main`. Skipped worktrees are printed with the reason. It never deletes
branches, only the worktree checkout, then runs `git worktree prune` to
clear stale administrative entries.

Dry-run equivalent (just see what it would do, without removing):

```bash
git worktree list --porcelain
```
