# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/theoh-io/lerobot-bench/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/theoh-io/lerobot-bench/releases/tag/v0.0.1
