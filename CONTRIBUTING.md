# Contributing to lerobot-bench

This is a personal portfolio project, but PRs and issues are welcome.

## Development setup

Requires Python 3.12 and a recent CUDA-capable GPU for sim runs (CPU works for the leaderboard reader only).

```bash
git clone https://github.com/thrmnn/lerobot-bench.git
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

### Working in parallel / agent dispatch

Larger changes are often built by several authoring agents running
concurrently, each in its own git worktree and owning a non-overlapping set
of files (the path globs in [`.github/CODEOWNERS`](.github/CODEOWNERS)). The
operating convention — disjoint file ownership, worktree isolation,
`PYTHONPATH=$(pwd)/src` test runs, and the serial squash-merge drain into
strict-protected `main` — is documented in
[`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md). Read it before launching a
multi-agent wave.

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

## Add a policy

Onboarding a pretrained LeRobot policy is **one PR**: a single entry in
`configs/policies.yaml`. No Python changes. This walkthrough is exact —
copy the template, fill it, dry-run it, open the PR.

If you only want to *suggest* a policy without doing the work, open a
[Propose a policy](https://github.com/thrmnn/lerobot-bench/issues/new?template=propose-a-policy.yml)
issue instead.

### 1. The `configs/policies.yaml` entry

A policy entry is a YAML mapping under the top-level `policies:` list.
Validated by `lerobot_bench.policies.PolicyRegistry.from_yaml` —
unknown or missing fields fail CI with the loader's own error message.

```yaml
  - name: my_policy                       # required
    is_baseline: false                    # required
    env_compat:                           # required
      - pusht
    repo_id: org/my-policy-checkpoint     # required for non-baselines
    revision_sha: <40-char commit SHA>    # required for non-baselines
    fp_precision: bf16                    # optional (fp32 | fp16 | bf16)
    license: apache-2.0                   # optional but please fill it
    notes: "One line: what it is, DL count, license caveats."  # optional
    paper_reported_success:               # optional
      pusht: 0.65
    paper_reported_notes: >-              # optional
      Citation for the number above.
```

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Unique registry key. `snake_case`, no spaces. Used on the CLI (`--policy <name>`). |
| `is_baseline` | yes | `true` only for weight-free policies (`no_op`, `random`). Baselines **must not** set `repo_id` / `revision_sha` / `fp_precision`. |
| `env_compat` | yes | List of env names the policy can run, from `configs/envs.yaml` (`pusht`, `aloha_transfer_cube`, `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`). See below. |
| `repo_id` | non-baseline | HF Hub repo, e.g. `lerobot/diffusion_pusht`. |
| `revision_sha` | non-baseline | **Pinned** 40-char commit SHA — never a branch like `main`. See below. |
| `fp_precision` | optional | One of `fp32`, `fp16`, `bf16`. Omit to let the eval loop pick a default. |
| `license` | optional | SPDX-style id or upstream license name (`apache-2.0`, `mit`, `gemma`). If unknown, say so in `notes`. |
| `notes` | optional | Free-text one-liner. License caveats, download count, quirks. |
| `paper_reported_success` | optional | Mapping `env_name -> fraction in [0, 1]` (not percentages). Every key must appear in `env_compat`. Use `null` for an env the paper does not report. Powers the "delta-vs-published" panel. |
| `paper_reported_notes` | optional | Exact citation for the numbers above (paper, table, row, episode count). |

### 2. Getting the locked `revision_sha`

`revision_sha` **must** be a commit SHA, not a floating ref. A pinned SHA
is what makes a benchmark cell reproducible — `main` can change under you.

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('org/my-policy-checkpoint').sha)"
```

Equivalently, open the model page on the Hub, click the commit history,
and copy the full hash of the commit you want. Paste it verbatim into
`revision_sha`.

### 3. Setting `env_compat`

List every env the checkpoint was trained for and can actually run.
A policy is matched to an env only if the env name is in `env_compat`,
so an over-broad list will schedule cells that fail at load time.

- A Push-T diffusion policy → `[pusht]`.
- An ALOHA transfer-cube ACT policy → `[aloha_transfer_cube]`.
- A LIBERO-finetuned VLA → all four suites
  (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`):
  LeRobot's LIBERO wrapper exposes the same observation contract for
  every suite, so any libero-finetuned VLA runs on all four.

The valid env names are exactly the `name` values in `configs/envs.yaml`.

### 4. Dry-run it locally

A dry-run resolves the policy + env through the registries and prints
the cell it *would* run — no weights download, no sim, no GPU. It
catches a malformed entry, a bad env name, or a name not in `env_compat`:

```bash
python scripts/run_one.py --policy my_policy --env pusht --seed 0 --dry-run
```

Exit `0` means the entry is well-formed and the (policy, env) pair is
valid. Exit `5` means the policy/env is not in the registry or the env
is not in `env_compat`. (Exit `3`, "not runnable", is skipped under
`--dry-run` — that only fires on a real run with a missing
`revision_sha`.)

To validate the whole file the way CI does:

```bash
python -c "from pathlib import Path; from lerobot_bench.policies import PolicyRegistry; PolicyRegistry.from_yaml(Path('configs/policies.yaml')); print('OK')"
```

### 5. Open the PR

- Branch: `feat/policy-<name>`.
- Commit message: `feat(policies): add <name>`.
- The `validate-configs` workflow re-loads `configs/policies.yaml`
  through the registry on every PR that touches it — a bad entry fails
  CI with the loader's exact error.

### 6. What the maintainer does to admit it

After the entry merges, a maintainer with a GPU box runs calibration
and a real cell to confirm the checkpoint loads and produces sane
numbers:

```bash
python scripts/run_one.py --policy my_policy --env pusht --seed 0 --n-episodes 5
make calibrate          # measures per-cell wall-time for the sweep budget
```

If it calibrates cleanly the policy is included in the next sweep and
appears on the leaderboard. The pinned `revision_sha` guarantees the
admitted checkpoint is the one that was reviewed.

## Reporting issues

Please use the [issue forms](https://github.com/thrmnn/lerobot-bench/issues/new/choose):
**Bug report** for crashes, or **Result discrepancy** if a benchmark
number disagrees with a published claim.

Include:
- Python version and OS (`python --version`, `uname -a`).
- `pip freeze | grep -E "lerobot|torch|numpy"`.
- Minimal reproducible example.
- Full traceback.

## License

By contributing, you agree your contribution is licensed under MIT.
