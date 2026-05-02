# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/lerobot_bench/checkpointing.py` — cell-boundary parquet resume layer. `CellKey` + `ResumePlan` dataclasses; `load_results`, `plan_resume`, `append_cell_rows`, `drop_partial_cells`. Atomic append via `<path>.tmp.parquet` + `os.replace`; classifies cells as completed (exact `set(range(n_episodes))` match), partial (any other non-zero count, including missing/extra indices), or pending. Duplicate-key guard rejects double-writes before touching disk. 20 unit tests, pure pandas + pyarrow (no torch/env/GPU).
- `pandas-stubs>=2.2` to `[dev]` extras (mypy stubs for the new module).
- `src/lerobot_bench/render.py` — `render_episode` (frames -> 256x256 / 10 fps / H.264 MP4, ≤ 2 MiB cap) and `render_thumbnail_strip` (PNG preview) with frozen `RenderResult` carrying `bytes_written`, `frame_count`, `encoder_settings`, and `content_sha256`. Pure imageio.v3 + libx264 at fixed crf=23 (byte-identical reproducible in spike); oversize clips raise `RenderSizeError` after deleting the file. Resize is a small numpy bilinear sampler — no PIL/scipy dep added.
- `src/lerobot_bench/envs.py` — `EnvSpec` (frozen dataclass) and `EnvRegistry` with strict YAML loader. Default registry at `configs/envs.yaml` ships with PushT and Aloha; Libero is gated on the Day 1 install spike.
- `src/lerobot_bench/policies.py` — `PolicySpec` and `PolicyRegistry`. Pre-Day-0a entries with `revision_sha: null` load fine but fail `assert_runnable()` (explicit-not-silent substitution). Default registry at `configs/policies.yaml` ships with `no_op`, `random` (baselines, runnable), `diffusion_policy`, `act` (TODO Day 0a: lock revision SHAs).
- `types-PyYAML` to `[dev]` extras and pre-commit's mypy `additional_dependencies` so type-check passes against the YAML loaders.
- `src/lerobot_bench/stats.py` — `bootstrap_ci`, `paired_delta_bootstrap`, `paired_wilcoxon`, `cohens_h`, `wilson_ci`. Frozen dataclasses for results, RNG required as kwarg (no hidden global state). 20 unit tests against analytical references (Wilson convergence, identity-pairs Wilcoxon, closed-form Cohen's h, textbook Wilson CI value).
- `scipy>=1.13,<2.0` runtime dependency for Wilcoxon and Wilson interval z-quantile.
- Project-scoped Claude Code agent team under `.claude/agents/`: `bench-eval-engineer`, `stats-rigor-reviewer`, `render-pipeline-engineer`, `sweep-sre`, `spaces-frontend-engineer`, `researcher-writeup`, `upstream-contributor`, `devx-toolsmith`.
- Hooks (`.claude/settings.json`): auto-format Python files on edit, gate `git push` on `make all`, block writes to generated artifacts (`results/`, `*.parquet`, `*.mp4`, weight files), branch+dirty status on Stop.
- CI: tag-driven release workflow (`release.yml`), daily fresh-install smoke (`smoke.yml`), Gradio Space boot test (`space-smoke.yml`).
- `dependabot.yml` for GitHub Actions (Python deps intentionally pinned — `lerobot==0.5.1` is the reproducibility anchor).
- `docs/RUNBOOK.md` — sweep ops, resume drill, OOM playbook, publish + Space rollback.
- `docs/MODEL_CARDS.md` — per-policy template populated at Day 0a (revision SHAs) and Day 7 (failure taxonomy).
- `docs/NEXT_STEPS.md` — live execution checklist between CEO plan (strategy) and DESIGN (spec). PR-shaped chunks, Day 0a → ship.
- `src/lerobot_bench/cli.py` — CLI entrypoint stub (currently `--version` only); subcommands grow with `scripts/`.

### Changed
- `docs/CEO-PLAN.md` — appended an "Infrastructure (added 2026-04-30)" section noting the agent team, hooks, CI evolution, and operational docs. Strategy unchanged.

## [0.0.1] - 2026-04-29

### Added
- Initial repository scaffold.
- Project structure: `src/`, `tests/`, `scripts/`, `configs/`, `notebooks/`, `docs/`, `space/`.
- Build & tooling: `pyproject.toml` (PEP 621), ruff, mypy, pytest, pre-commit, Makefile.
- CI: GitHub Actions workflow for lint + type check + tests on Python 3.12.
- Documentation: design doc (`docs/DESIGN.md`), CEO plan (`docs/CEO-PLAN.md`), architecture stub (`docs/ARCHITECTURE.md`).
- Licensing: MIT.

### Notes
- No implementation code yet. Day 0 of the build per `docs/CEO-PLAN.md`.

[Unreleased]: https://github.com/thrmnn/lerobot-bench/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/thrmnn/lerobot-bench/releases/tag/v0.0.1
