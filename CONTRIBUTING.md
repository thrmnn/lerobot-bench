# Contributing to lerobot-bench

This is a personal portfolio project, but PRs and issues are welcome.

## Development setup

Requires Python 3.12 and a recent CUDA-capable GPU for sim runs (CPU works for the leaderboard reader only).

```bash
git clone https://github.com/theoh-io/lerobot-bench.git
cd lerobot-bench
conda activate lerobot          # Python 3.12 env with lerobot 0.5.1
pip install -e ".[all]"
pre-commit install
```

Verify:

```bash
make all
```

## Workflow

- Branch from `main` with a short prefix: `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `test/`.
- Conventional Commits for messages: `feat(eval): add bootstrap CI helper`.
- One logical change per commit. Squash later if needed.
- Open a PR against `main`. CI must pass before review.

## Code style

- `ruff` for lint and format. Configured in `pyproject.toml`.
- `mypy --strict` for type checking. New code is fully typed.
- Line length 100.
- No unused imports, no dead code, no print debugging in library code (use `logging` or `rich`).

## Tests

- New code paths get a unit test in `tests/`.
- Sim-dependent tests are marked `@pytest.mark.sim` and excluded from default CI.
- GPU tests are marked `@pytest.mark.gpu`.
- Don't commit fixtures larger than 100 KB. Use Hub datasets for large data.

## Reporting issues

Include:
- Python version and OS (`python --version`, `uname -a`).
- `pip freeze | grep -E "lerobot|torch|numpy"`.
- Minimal reproducible example.
- Full traceback.

## License

By contributing, you agree your contribution is licensed under MIT.
