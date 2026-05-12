# lerobot-bench

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code_style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> Public multi-policy benchmark for pretrained LeRobot policies on PushT, Aloha, and LIBERO sim envs.
> Multi-seed contract, bootstrap + Wilson CIs, MDE bounds, paired comparisons, failure taxonomy. Arxiv-grade writeup and upstream-ready eval module.

**Status: v1 in progress.** Calibration matrix complete (22 cells across 6 policies × 6 envs). Overnight sweep in flight at the time of writing. Pi0 family deferred to v1.1 (~30 GB host-RAM cold-load spike — see [paper Limitations](paper/main.tex)).

---

## TL;DR — what you get

Three artifacts, all open:

1. **Public leaderboard** — Hugging Face Space + Hub dataset `Theozinh0/lerobot-bench-results-v1`. Every per-episode outcome, every rollout MP4, queryable by `(policy, env, seed, episode)`.
2. **4-page arxiv writeup** — `paper/main.tex`. Methodology, related work, results, limitations. Every figure regenerated from `notebooks/01-write-finding.ipynb`.
3. **Upstream-ready eval pipeline** — `src/lerobot_bench/eval.py` extracted as `lerobot.eval.multi_seed` in a follow-up PR to `huggingface/lerobot`.

Two tools for running and inspecting it:

| | What | URL when local |
|---|---|---|
| 🟢 **`dashboard/`** | Local operator dashboard: live sweep progress, calibration inspector, rollout video preview, color-coded log tail | `make dashboard` → http://127.0.0.1:7860 |
| 🔵 **`space/`** | Public HF Space leaderboard, paired comparisons, failure taxonomy | `python space/app.py` |

---

## v1 scope

**6 policies × 6 envs (= 22 runnable cells after `env_compat` filter):**

| | pusht | aloha_transfer_cube | libero_spatial | libero_object | libero_goal | libero_10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `no_op` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `random` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `diffusion_policy` | ✓ | | | | | |
| `act` | | ✓ | | | | |
| `smolvla_libero` | | | ✓ | ✓ | ✓ | ✓ |
| `xvla_libero` | | | ✓ | ✓ | ✓ | ✓ |

**5 seeds × 50 episodes per cell** (N=250 binary outcomes per cell, with 2 cells auto-downscoped to 25 after calibration flagged slow inference). Pi0 family (`pi0_libero`, `pi0fast_libero`, `pi05_libero_finetuned_v044`) **deferred to v1.1** — they overflow the 32 GB WSL2 host budget during `from_pretrained` cold load (~30 GB CPU RAM peak under HF Transformers' default weight-conversion path). v1.1 paths: quantized weights or `accelerate device_map="auto"` streaming load.

---

## Methodology in 60 seconds

- **Seed contract.** Per-cell determinism via `(env_seed, action_seed, policy_seed)` triple derived from the cell's seed index. Re-running cell `(policy, env, seed=k)` reproduces the exact parquet rows.
- **Confidence intervals.** Wilson 95% on per-cell success rate; stratified bootstrap (10k resamples over seed × episode) for distributional summaries and paired deltas.
- **Minimum detectable effect (MDE).** Pre-computed per-cell from N=250 and the cell's empirical success rate. Headline findings cite deltas only where `|delta| > MDE` (see [`docs/MDE_TABLE.md`](docs/MDE_TABLE.md)).
- **Failure taxonomy.** Per-rollout categorical labeling against `docs/FAILURE_TAXONOMY.md`; labels live in `labels.json` alongside the MP4.
- **Auto-downscope.** Calibration (20 steps per cell) flags `mean_step_ms > 100` (slow) or `vram_peak_mb > 5500` (VRAM-pressured) and trims that cell's episode budget so the full sweep fits.
- **Safety.** All heavy workloads run under a kernel-enforced 18 GB cgroup memory cap via `scripts/run_capped.sh`. Pre-flight gate refuses launch when baseline RAM > 55% used to protect parallel tenants on the host.

Full design: [`docs/DESIGN.md`](docs/DESIGN.md). Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). MDE math: [`docs/MDE_TABLE.md`](docs/MDE_TABLE.md).

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/thrmnn/lerobot-bench.git
cd lerobot-bench

# Activate the lerobot conda env (Python 3.12, miniforge3)
conda activate lerobot

# Install in editable mode with sim + dev extras
pip install -e ".[all]"

# Smoke a single (policy, env, seed) cell — ~1 min
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --episodes 5
```

### Run the full sweep

```bash
# 1. Calibrate (~30 min — measures step latency + VRAM per cell)
make calibrate

# 2. Merge per-policy calibration JSONs (if you split the run)
python scripts/merge_calibration.py results/calibration-cheap.json \
    results/calibration-smolvla.json results/calibration-xvla.json \
    --out results/calibration-$(date +%Y-%m-%d).json

# 3. Generate sweep_full.yaml overrides from calibration
python scripts/auto_downscope.py results/calibration-$(date +%Y-%m-%d).json --apply

# 4. Launch under the 18 GB cgroup cap (overnight, ~8-15 hr)
scripts/launch_overnight_sweep.sh
```

### Watch progress

```bash
# Live operator dashboard
make dashboard
# → http://127.0.0.1:7860

# Or tail the log directly
tail -F logs/sweep-$(cat /tmp/lerobot-bench-sweep-ts).log
```

---

## Repo layout

```
lerobot-bench/
├── src/lerobot_bench/     # eval, stats, render, registries, checkpointing
├── scripts/               # entrypoints: calibrate, run_sweep, run_one, publish,
│                          #             merge_calibration, auto_downscope,
│                          #             run_capped, watchdog, launch_overnight_sweep
├── configs/               # policies.yaml, envs.yaml, sweep_full.yaml, sweep_mini.yaml
├── dashboard/             # local-first operator Gradio app
├── space/                 # public HF Space app (Gradio)
├── notebooks/             # 01-write-finding.ipynb (every paper figure)
├── paper/                 # main.tex + references.bib (4-page arxiv writeup)
├── tests/                 # 360+ tests (lint + mypy + pytest, all green on CI)
├── docs/                  # DESIGN, ARCHITECTURE, MDE_TABLE, FAILURE_TAXONOMY, RUNBOOK
└── results/               # gitignored — pushed to HF Hub dataset on publish
```

---

## Development

```bash
make install      # editable install with dev extras
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
make test         # pytest fast tier
make all          # lint + typecheck + test
make dashboard    # launch the local operator dashboard
make sweep        # run the full sweep (under the 18 GB cap, recommended)
```

Pre-commit hooks run ruff and the typecheck/test fast tier on every commit. CI on every push and PR.

---

## Reproducibility contract

Every leaderboard row is anchored to:
- The pinned `lerobot==0.5.1` PyPI release (recorded in `pyproject.toml`).
- A pinned commit SHA per policy checkpoint (`configs/policies.yaml`, validated by tests).
- A deterministic seeding contract documented in [`docs/DESIGN.md`](docs/DESIGN.md) § Methodology.
- Wilson + bootstrap CIs from `src/lerobot_bench/stats.py` (audited; see PR #30 commit).
- Cell-boundary checkpointing in `src/lerobot_bench/checkpointing.py` — `kill -9` during the sweep loses only the current cell.

Hardware reference: NVIDIA RTX 4060 Laptop (8 GB VRAM), 32 GB host RAM, Ubuntu on WSL2.

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

The arxiv writeup pre-print lands when the sweep completes and the parquet is published. Citation guidance will appear here at that point. Until then, please link to this repo.
