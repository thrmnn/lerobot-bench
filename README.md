# lerobot-bench

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code_style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> Public multi-policy benchmark for pretrained LeRobot policies on PushT, Aloha, and Libero sim envs.
> Anchored on one defensible non-obvious finding, with a 4-page arxiv-grade writeup and an upstream PR.

**Status: pre-alpha (0.0.1).** The repo is bootstrapped, the design is locked, the implementation has not started yet.

---

## What this is

A reproducible benchmark of pretrained robotics policies (Diffusion Policy, ACT, SmolVLA, Pi0, plus a no-op baseline) evaluated head-to-head on three sim envs (PushT, Aloha, Libero), with multi-seed evaluation, bootstrap confidence intervals, and a public Hugging Face Space leaderboard.

The artifact has three pieces:
1. **The benchmark + leaderboard** — this repo plus a public HF Space.
2. **An arxiv-grade writeup** — `notebooks/01-write-finding.ipynb` becomes a 4-page LaTeX paper at ship time.
3. **An upstream PR** — the eval pipeline contributed back to `huggingface/lerobot` as a reusable module.

See `docs/CEO-PLAN.md` for the strategic framing and `docs/DESIGN.md` for the full technical design.

---

## Repo layout

```
lerobot-bench/
├── src/lerobot_bench/      # library code (eval, stats, render, registries)
├── scripts/                # entrypoints: calibrate, run_sweep, run_one, publish
├── configs/                # policy + env registry YAML / JSON
├── tests/                  # unit tests
├── notebooks/              # analysis + writeup
├── docs/                   # design doc, CEO plan, architecture
├── space/                  # HF Spaces app (Gradio) — separate git remote
└── results/                # gitignored — pushed to HF Hub dataset
```

## Quickstart (once implementation lands)

```bash
# Clone and install
git clone https://github.com/theoh-io/lerobot-bench.git
cd lerobot-bench

# Activate the existing lerobot conda env (Python 3.12, miniforge3)
conda activate lerobot

# Install in editable mode with sim + dev extras
pip install -e ".[all]"

# Smoke test
python -c "import lerobot_bench; print(lerobot_bench.__version__)"

# Run a single (policy, env, seed) cell — fast sanity check
python scripts/run_one.py --policy diffusion_policy --env pusht --seed 0 --episodes 5

# Run the full sweep (overnight on RTX 4060)
python scripts/run_sweep.py --config configs/full_sweep.yaml
```

## Development

```bash
make install      # editable install with dev extras
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
make test         # pytest
make all          # lint + typecheck + test
```

Pre-commit hooks run ruff on every commit:

```bash
pre-commit install
```

CI runs the same checks on every push and PR.

## Reproducibility

Every result in the leaderboard is anchored to:
- The exact `lerobot==0.5.1` PyPI release.
- A pinned commit SHA per policy checkpoint (recorded in `results/<sweep>/manifest.json`).
- A deterministic seeding contract documented in `docs/DESIGN.md` § Methodology.
- Bootstrap 95% CIs from 10,000 resamples over (seed, episode) outcomes.

Hardware reference: a single NVIDIA RTX 4060 Laptop (8GB VRAM), Ubuntu 22.04 on WSL2.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If this benchmark is useful to your work, citation guidance will appear here once the writeup is on arxiv. Until then, please link to this repo.
