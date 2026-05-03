# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/run_one.py` — single-cell CLI shell over `eval.run_cell_from_specs`. Pre-flight order: registry lookup (exit 5) → env compat (exit 5) → `policy.is_runnable()` (exit 3) → `import lerobot` (exit 4); cell execution exits 2 on any per-episode error with rows still appended atomically. `--dry-run` short-circuits before any torch/lerobot/render import (AST-guarded). `RunOneOutcome` dataclass surfaces decisions to tests; render module is lazy-imported inside `render_episodes_to_videos`. 12 tests using a `_fake_cell_result` builder + monkeypatched `eval.run_cell_from_specs`. `make run-one ARGS=...` target wired up.
- `src/lerobot_bench/eval.py` — eval orchestration core. `seed_everything` (numpy seeded immediately, torch + cuda seeded if importable, returns `seed_idx * 1000`), `run_cell` (full inner loop: per-cell seeding, per-episode `env.reset(seed=base+e)` + `policy.reset()`, render-after-step frame collection, success = `final_reward >= success_threshold`, per-episode exception capture), `CellResult` + `EpisodeResult` (frozen) with `to_rows()` matching `RESULT_SCHEMA` exactly, `_NoOpPolicy` + `_RandomPolicy` baselines wired through `load_policy(spec, action_shape=...)`. `load_policy` raises `RuntimeError` on non-runnable specs (Day 0a hint), `NotImplementedError` on pretrained until Day 0b lerobot factory wire-up; `load_env` lazy-imports gymnasium. 32 unit tests against `MockEnv` / `MockPolicy` (no torch/lerobot/gym needed); end-to-end seeding round-trip via the random baseline proves byte-identical action sequences across runs.
- `gymnasium.*` to mypy `ignore_missing_imports` overrides so `eval.load_env` type-checks before sim extras are installed.
- `scripts/calibrate.py` — Day 0b calibration spike scaffold. CLI + `CellTiming` / `CalibrationReport` dataclasses + `auto_downscope` rule (8 GB RTX 4060 thresholds) + `plan_cells` (ready / skipped / incompat) + `measure_cell` with lazy torch/lerobot imports and OOM catch. Exit codes 0/2/3/4 with one-line resume hints. Inner measurement loop is a Day-0b TODO returning `status="error"` until lerobot is locked; surrounding plumbing is unit-tested. 22 tests including a static AST guard that no top-level `import torch`/`import lerobot` ever creeps in. `make calibrate` target wired to the script.
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
- `src/lerobot_bench/render.py` — `render_episode` now walks an adaptive `RENDER_LADDER` of `(fps, crf)` rungs (`(10,23) -> (5,23) -> (5,28) -> (5,33)`) until the encoded clip fits under `MAX_BYTES`, instead of single-shot encoding at fixed `crf=23` and erroring on overshoot. Premortem mitigation for real Aloha episodes (1000+ steps) blowing the 2 MiB cap. The successful rung index is recorded in the new `EncoderSettings.rung_index` field (`-1` when the caller passes explicit `fps`/`crf` to bypass the ladder; `>=1` is "playback faster than wall-clock"). Pathological inputs that overshoot every rung still raise `RenderSizeError`, now carrying the full per-rung attempt log in the message and a structured `attempts` tuple. We do *not* drop input frames — lower fps means faster playback, which is the documented tradeoff over sample loss.
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
