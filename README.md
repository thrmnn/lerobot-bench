<div align="center">

<img src="docs/assets/logo.svg" alt="lerobot-bench" width="420">

### A public, reproducible benchmark of pretrained LeRobot manipulation policies.

5 policies in v1 (plus xvla executed-but-deferred) × 6 sim envs · multi-seed contract · Wilson + bootstrap CIs · MDE bounds · paired comparisons · failure taxonomy.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code_style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/thrmnn/lerobot-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/thrmnn/lerobot-bench/actions/workflows/ci.yml)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-lerobot--bench-yellow)](https://huggingface.co/spaces/thrmnn/lerobot-bench)
<!-- TODO: dataset not yet uploaded — link 404s until v1.0.0 parquet is pushed; see task #100. -->
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-v1-yellow)](https://huggingface.co/datasets/thrmnn/lerobot-bench-v1)

**Quick links:** [Get started](docs/GETTING_STARTED.md) · [Live leaderboard](https://huggingface.co/spaces/thrmnn/lerobot-bench) · [Dataset](https://huggingface.co/datasets/thrmnn/lerobot-bench-v1) · [Paper](paper/main.tex) · [Contributing](CONTRIBUTING.md) · [Reproduce](docs/REPRODUCE.md)
<!-- Hero image: the public HF Space leaderboard. Captured by the maintainer once the Space is live; see docs/assets/README.md. The repo renders fine while this file is absent. -->
<picture>
  <img src="docs/assets/leaderboard.png" alt="lerobot-bench leaderboard: success rate with 95% confidence intervals for 6 pretrained LeRobot policies across 6 simulated manipulation environments" width="820">
</picture>

</div>

---

> Public multi-policy benchmark for pretrained LeRobot policies on PushT, Aloha, and LIBERO sim envs.
> Multi-seed contract, bootstrap + Wilson CIs, MDE bounds, paired comparisons, failure taxonomy. Arxiv-grade writeup and upstream-ready eval module.

**Status: v1 finalized (dataset version `v1.0.0`), with v1.0.1 methodology audit incorporated into framing.** Sweep complete: **107/107 cells dispatched, 0 failures** across 6 policies × 6 envs. Pi0 family deferred to v1.1 (~30 GB host-RAM cold-load spike — see [paper Limitations](paper/main.tex)). `xvla_libero` was executed but is **deferred from the v1 leaderboard** — two upstream Hub-artifact wiring bugs were patched in our loader but a third unresolved issue still produces 0% rollouts; see [`docs/DEFERRED_POLICIES.md`](docs/DEFERRED_POLICIES.md).

> **Headline (audit-aware read).** SmolVLA on `libero_10` measures **0.252** [0.202, 0.309] under the lerobot-bench v1 default protocol, against the **0.71** reported by Shukor et al. — but the v1.0.1 audit ([PR #84](https://github.com/thrmnn/lerobot-bench/pull/84) scope, [PR #89](https://github.com/thrmnn/lerobot-bench/pull/89) step cap) establishes this is a **single-task probe at a truncated step cap**, not a clean paper-replication number: the paper averages 10 tasks × 10 trials per suite (we ran `task_id=0` × 5 seeds × 50 episodes), and 74.8% of our failed `libero_10` episodes hit the 520-step cap vs. canonical LIBERO's 600. The 0.252 is real for that scope and is a **lower bound** at our cap; v1.1 closes both caveats via [PR #90](https://github.com/thrmnn/lerobot-bench/pull/90)'s selectable `--canonical` criterion and an all-10-tasks LIBERO sweep. See [Methodology caveats](#methodology-caveats-v101-audit) below.

---

## TL;DR — what you get

Three artifacts, all open:

1. **Public leaderboard** — Hugging Face Space + Hub dataset `thrmnn/lerobot-bench-v1` (v1.0.0, 107 cells, 0 failures). Every per-episode outcome, every rollout MP4, queryable by `(policy, env, seed, episode)`.

2. **4-page arxiv writeup** — `paper/main.tex`. Methodology, related work, results, limitations. Every figure regenerated from `notebooks/01-write-finding.ipynb`.
3. **Upstream-ready eval pipeline** — `src/lerobot_bench/eval.py` extracted as `lerobot.eval.multi_seed` in a follow-up PR to `huggingface/lerobot`.

Two tools for running and inspecting it:

| | What | URL when local |
|---|---|---|
| 🟢 **`dashboard/`** | Local operator dashboard: live sweep progress, calibration inspector, rollout video preview, color-coded log tail | `make dashboard` → http://127.0.0.1:7860 |
| 🔵 **`space/`** | Public HF Space leaderboard, paired comparisons, failure taxonomy | `python space/app.py` |

---

## v1 scope

**5 leaderboard policies + xvla executed-but-deferred × 6 envs (107 cells dispatched after `env_compat` filter, 0 failures):**

| | pusht | aloha_transfer_cube | libero_spatial | libero_object | libero_goal | libero_10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `no_op` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `random` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `diffusion_policy` | ✓ | | | | | |
| `act` | | ✓ | | | | |
| `smolvla_libero` | | | ✓ | ✓ | ✓ | ✓ |
| `xvla_libero` | | | 🅓 | 🅓 | 🅓 | 🅓 |

Legend: ✓ runnable cell in v1 leaderboard · 🅓 cell *executed* in the v1 sweep but **deferred from the leaderboard**; upstream Hub artifacts ship with wiring bugs (PR #71 + PR #74 patch two; a third manifestation remained unresolved in the v1 window). See [`docs/DEFERRED_POLICIES.md`](docs/DEFERRED_POLICIES.md).

**5 seeds × 50 episodes per cell** (N=250 binary outcomes per cell; 2 cells were auto-downscoped to 25 after calibration flagged slow inference). Pi0 family (`pi0_libero`, `pi0fast_libero`, `pi05_libero_finetuned_v044`) **deferred to v1.1** — they overflow the 32 GB WSL2 host budget during `from_pretrained` cold load (~30 GB CPU RAM peak under HF Transformers' default weight-conversion path). v1.1 paths: quantized weights or `accelerate device_map="auto"` streaming load. The `xvla_libero` deferral is documented alongside the pi-family in [`docs/DEFERRED_POLICIES.md`](docs/DEFERRED_POLICIES.md).

---

## Methodology caveats (v1.0.1 audit)

After v1.0 sweep completion we conducted a static methodology audit against each policy's source paper and each env's canonical protocol. Three mismatches were confirmed; all three **constrain what the headline cells mean** without invalidating the underlying measurements. Every v1 parquet row remains valid for the scope it was measured under — the audit reframes how cross-paper comparisons should be read.

| Audit | What we ran | What the paper / canonical protocol uses | Effect on the headline |
|---|---|---|---|
| [PR #84](https://github.com/thrmnn/lerobot-bench/pull/84) — SmolVLA task coverage | `task_id=0` × 5 seeds × 50 ep = 250 single-task episodes per LIBERO suite | 10 tasks × 10 trials per task = 100-ep suite averages (Shukor et al., Table 2) | The "0.71 → 0.252" gap on `libero_10` is **single-task vs. 10-task-averaged scope**, not an apples-to-apples replication gap. Holds as a single-task envelope claim only. |
| [PR #86](https://github.com/thrmnn/lerobot-bench/pull/86) — ACT inference settings | Hub default `temporal_ensemble_coeff=None`, `n_action_steps=100` | Paper: `coeff=0.01`, `n_action_steps=1` (overlapping-chunk weighted averaging, Zhao et al., Table I) | The 0.016 on `act × aloha_transfer_cube` is plausibly an inference-config artefact, not the architecture failing. **Probe with paper settings pending in v1.0.2.** |
| [PR #89](https://github.com/thrmnn/lerobot-bench/pull/89) — LIBERO step caps | `max_steps={spatial=280, object=280, goal=300, libero_10=520}` (lerobot defaults) | `max_steps=600` for all four suites (canonical LIBERO, Liu et al.) | 74.8% of failed `libero_10` episodes hit our cap → **all four LIBERO numbers are lower bounds at our caps**; `libero_10` is the most sensitive. |

[PR #90](https://github.com/thrmnn/lerobot-bench/pull/90) ships a selectable `--canonical` criterion on `scripts/run_one.py` and `scripts/run_sweep.py` that adopts the canonical step caps and the paper-canonical success rules for PushT and Aloha; v1.1 reruns the audit-affected cells under it. Full audit reports: [`docs/CLAIM_AUDIT_SMOLVLA.md`](docs/CLAIM_AUDIT_SMOLVLA.md), [`docs/INFERENCE_AUDIT.md`](docs/INFERENCE_AUDIT.md), [`docs/SUCCESS_CRITERION_AUDIT.md`](docs/SUCCESS_CRITERION_AUDIT.md), [`docs/CANONICAL_CRITERIA.md`](docs/CANONICAL_CRITERIA.md). Per-policy "paper vs. measured" notes are in [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md).

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

## Getting started

From `git clone` to a real benchmark result in three commands. Run them from
an activated Python 3.12 conda env (`conda activate lerobot`):

```bash
# 1. Clone and install (editable, all extras: sim + viz + space + dev)
git clone https://github.com/thrmnn/lerobot-bench.git && cd lerobot-bench
pip install -e ".[all]"

# 2. Run a single (policy, env, seed) cell — a couple of minutes on a GPU
python scripts/run_one.py --policy act --env aloha_transfer_cube --seed 0 --n-episodes 5
```

You just produced per-episode rows in `results/results.parquet` and rollout
MP4s in `results/videos/` — the same artifacts every leaderboard number is
built from.

**→ Full walkthrough, expected output, and common-issue fixes:
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).**

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

The arxiv writeup pre-print lands alongside the v1.0.0 dataset upload. Citation guidance will appear here at that point. Until then, please link to this repo.
<!-- TODO: replace with BibTeX once arxiv ID is assigned and the dataset at huggingface.co/datasets/thrmnn/lerobot-bench-v1 is live. -->
