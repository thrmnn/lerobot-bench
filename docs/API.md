# Python API reference

Hand-authored reference for the public `lerobot_bench` Python API — the
modules under `src/lerobot_bench/` that a contributor or downstream user
imports. It is kept accurate to the current code; if a signature here
disagrees with the source, the source wins — please file an issue.

Scope notes:

- This page documents the **public surface** — the registries, the eval
  core, the stats helpers, the renderer, and the checkpointing layer.
  Names prefixed with a single underscore (`_load_pretrained_policy`,
  `_DebatchedVecEnvAdapter`, `_NoOpPolicy`, …) are internal and may
  change without notice.
- `lerobot_bench` does not eagerly import `torch`, `lerobot`, or
  `gymnasium`. The eval module imports them lazily inside the functions
  that need them, so `import lerobot_bench` works in a torch-free
  environment (CI fast tier, the leaderboard reader).
- The runnable `scripts/` (`run_one`, `run_sweep`, `calibrate`,
  `publish_results`, `reproduce_cell`, …) are thin shells over this API
  and are documented in [`docs/RUNBOOK.md`](RUNBOOK.md) and
  [`docs/GETTING_STARTED.md`](GETTING_STARTED.md), not here.

## Contents

- [`lerobot_bench`](#package-lerobot_bench) — package root
- [`lerobot_bench.envs`](#module-envs) — sim env registry
- [`lerobot_bench.policies`](#module-policies) — policy registry
- [`lerobot_bench.eval`](#module-eval) — eval orchestration core
- [`lerobot_bench.stats`](#module-stats) — confidence intervals and paired tests
- [`lerobot_bench.render`](#module-render) — episode → MP4 renderer
- [`lerobot_bench.checkpointing`](#module-checkpointing) — cell-boundary resume
- [`lerobot_bench.cli`](#module-cli) — `lerobot-bench` command-line entrypoint

---

## Package `lerobot_bench`

```python
from lerobot_bench import __version__
```

The package root exports a single name, `__version__` (a string).
It is sourced from the standalone `lerobot_bench.__version__` module so
the version can be read without importing the package's dependencies.

---

## Module `envs`

`lerobot_bench.envs` — the sim env registry. Pure data; no env is
constructed here (that is `eval.load_env`'s job). The human-edited
source of truth is `configs/envs.yaml`.

### `class EnvSpec`

Frozen dataclass describing one sim env in the benchmark matrix.

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Registry key, e.g. `"pusht"`, `"libero_spatial"`. |
| `family` | `str` | Leaderboard grouping, e.g. `"pusht"`, `"aloha"`, `"libero"`. |
| `max_steps` | `int` | Per-episode step cap (positive). |
| `success_threshold` | `float` | An episode succeeds when its final-step reward reaches this. |
| `lerobot_module` | `str` | Dotted path to the env config inside `lerobot`. |
| `gym_id` | `str \| None` | Set for the gym construction path. |
| `gym_kwargs` | `tuple[tuple[str, Any], ...]` | Hashable kwargs for `gymnasium.make`. |
| `factory` | `str \| None` | Set for the factory construction path. |
| `factory_kwargs` | `tuple[tuple[str, Any], ...]` | Hashable kwargs for the factory call. |

Exactly one of `gym_id` / `factory` must be set — both-set or
neither-set raises `ValueError` from `__post_init__`. The two paths:

- **gym path** (`gym_id` set) — `eval.load_env` calls
  `gymnasium.make(gym_id, max_episode_steps=max_steps, **gym_kwargs_dict())`.
  Used for PushT and Aloha.
- **factory path** (`factory` set) — `eval.load_env` imports the dotted
  module `factory` and calls its `make_env`. Used for the LIBERO
  suites, which are not gym-registered.

Methods:

- **`gym_kwargs_dict() -> dict[str, Any]`** — materialize `gym_kwargs`
  as a fresh dict.
- **`factory_kwargs_dict() -> dict[str, Any]`** — materialize
  `factory_kwargs` as a fresh dict.
- **`uses_factory -> bool`** *(property)* — `True` iff this spec is
  constructed via the factory path.

### `class EnvRegistry`

An indexed collection of `EnvSpec`, loaded from YAML.

- **`EnvRegistry.from_yaml(path: Path | str) -> EnvRegistry`**
  *(classmethod)* — load and validate `configs/envs.yaml`. Raises
  `ValueError` on a malformed file, unknown/missing fields, or a
  duplicate env name.
- **`get(name: str) -> EnvSpec`** — look up one spec; raises `KeyError`
  (listing the available names) if absent.
- **`names() -> list[str]`** — sorted list of registered env names.
- **`by_family(family: str) -> list[EnvSpec]`** — all specs in a family.
- Supports `in`, `len()`, and iteration over the `EnvSpec` values.

---

## Module `policies`

`lerobot_bench.policies` — the policy registry. Pure data; no weights
are loaded here (that is `eval.load_policy`'s job). The human-edited
source of truth is `configs/policies.yaml`.

### `FpPrecision`

```python
FpPrecision = Literal["fp32", "fp16", "bf16"]
```

Type alias for the allowed floating-point precision values.

### `class PolicySpec`

Frozen dataclass describing one policy in the benchmark matrix.

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Registry key; used on the CLI as `--policy <name>`. |
| `is_baseline` | `bool` | `True` for weight-free baselines (`no_op`, `random`). |
| `env_compat` | `tuple[str, ...]` | Env names this policy can run. |
| `repo_id` | `str \| None` | HF Hub repo; required for non-baselines. |
| `revision_sha` | `str \| None` | Pinned 40-char commit SHA; required for non-baselines. |
| `fp_precision` | `FpPrecision \| None` | Optional precision hint. |
| `license` | `str \| None` | Optional SPDX-style license id. |
| `notes` | `str` | Free-text one-liner. |
| `paper_reported_success` | `dict[str, float \| None] \| None` | Per-env paper-reported success rates, for the delta-vs-published panel. |
| `paper_reported_notes` | `str` | Citation for the numbers above. |

Baselines must **not** carry `repo_id` / `revision_sha` /
`fp_precision` — the loader rejects that combination.

Methods:

- **`is_runnable() -> bool`** — `True` for a baseline, or for a
  pretrained spec with both `repo_id` and `revision_sha` set. A
  pre-lock-in entry with `revision_sha=None` returns `False`.
- **`assert_runnable() -> None`** — raise `ValueError` with a
  user-actionable message (naming the missing fields and the source
  location) if the spec is not runnable.

### `class PolicyRegistry`

An indexed collection of `PolicySpec`, loaded from YAML.

- **`PolicyRegistry.from_yaml(path: Path | str) -> PolicyRegistry`**
  *(classmethod)* — load and validate `configs/policies.yaml`. Raises
  `ValueError` on a malformed file, unknown/missing fields, an invalid
  `fp_precision`, a baseline carrying weights metadata, a
  `paper_reported_success` key not in `env_compat` or a value outside
  `[0, 1]`, or a duplicate policy name. This is the validator the
  `validate-configs` CI job runs.
- **`get(name: str) -> PolicySpec`** — look up one spec; raises
  `KeyError` (listing the available names) if absent.
- **`names() -> list[str]`** — sorted list of registered policy names.
- **`supporting(env_name: str) -> list[PolicySpec]`** — policies whose
  `env_compat` includes `env_name`.
- **`runnable() -> list[PolicySpec]`** — policies that pass
  `PolicySpec.is_runnable`.
- Supports `in`, `len()`, and iteration over the `PolicySpec` values.

---

## Module `eval`

`lerobot_bench.eval` — the eval orchestration core. Every `scripts/`
entrypoint (`calibrate`, `run_one`, `run_sweep`) is a thin shell over
`run_cell`. The seeding contract from
[`docs/DESIGN.md`](DESIGN.md) § Methodology is enforced here and nowhere
else. `torch`, `lerobot`, and `gymnasium` are imported lazily inside the
functions that use them.

### Protocols

- **`class PolicyCallable`** *(`typing.Protocol`)* — anything that maps a
  gym observation dict to an action ndarray. Requires
  `__call__(obs: dict) -> NDArray[floating]` and a `reset() -> None`
  called once at the start of each episode. Stateless policies may
  implement `reset` as a no-op.
- **`class GymLikeEnv`** *(`typing.Protocol`)* — the gymnasium-style
  slice the cell loop uses: `reset(*, seed)`,
  `step(action)`, `render()`, `close()`.

### Result types

#### `class EpisodeResult`

Frozen dataclass — one episode within a cell.

| Field | Type | Notes |
|---|---|---|
| `episode_index` | `int` | Index within the cell. |
| `success` | `bool` | `final_reward >= success_threshold`. |
| `return_` | `float` | Cumulative episode reward. |
| `n_steps` | `int` | Steps taken. |
| `wallclock_s` | `float` | Episode wall-clock seconds. |
| `frames` | `tuple[NDArray[uint8], ...]` | `(H, W, 3)` frames; empty when video is off or after streaming encode. |
| `final_reward` | `float` | Reward at the last step. |
| `error` | `str \| None` | Short stringified exception when the episode crashed; `None` otherwise. |
| `video_path` | `Path \| None` | MP4 path when the streaming encoder produced one. |
| `video_sha256` | `str` | Hex SHA-256 of the encoded MP4, or `""`. |

When `error` is set, `success=False` and the numeric fields are zeroed —
the cell continues and the row is preserved.

#### `class CellResult`

Frozen dataclass — the output of `run_cell`: a tuple of `EpisodeResult`
plus cell metadata (`policy`, `env`, `seed`, `code_sha`,
`lerobot_version`, `timestamp_utc`).

- **`success_rate -> float`** *(property)* — fraction of successful
  episodes (`0.0` for an empty cell).
- **`to_rows(*, video_sha256_per_episode=None) -> pd.DataFrame`** —
  convert to a DataFrame matching `checkpointing.RESULT_SCHEMA`, one row
  per episode. `video_sha256_per_episode`, if given, must be parallel to
  `episodes`; if `None`, the column is filled from each episode's own
  `video_sha256`.

### `seed_everything(seed_idx: int) -> int`

Apply the per-cell seeding contract. Seeds NumPy's global RNG
immediately, then lazy-imports `torch` and seeds its CPU and (if
available) CUDA generators. Returns the base seed (`seed_idx * 1000`).
Logs a warning rather than raising if `torch` is unavailable.

### `load_policy(spec, *, action_shape=None, device="cuda") -> PolicyCallable`

Resolve a `PolicySpec` to a callable policy.

- `spec` — the `PolicySpec` to load.
- `action_shape: tuple[int, ...] | None` — required for baselines (the
  action dim comes from the env, not the spec); used as a final
  shape-check for pretrained policies.
- `device: str` — Torch device for the pretrained branch.

Baselines resolve to a zero-action (`no_op`) or uniform-noise
(`random`) callable. Pretrained policies lazy-import `lerobot`, build
the pre/post-processor pipelines, recover legacy normalization stats
from safetensors buffers when the Hub repo predates the
processor-pipeline split, load the model with `from_pretrained`, and
wrap it as a `PolicyCallable`. Raises `RuntimeError` for a non-runnable
spec (missing `revision_sha`) before any heavy import.

### `load_env(spec: EnvSpec) -> GymLikeEnv`

Instantiate a gym-like env from an `EnvSpec`. Picks the gym path or the
factory path from `spec.uses_factory` (see [`envs`](#module-envs)).
Sim extras must be installed. The factory path additionally pre-creates
`~/.libero/config.yaml` so LIBERO's first import is non-interactive, and
wraps a size-1 vector env so the cell loop sees a single-env API. The
caller owns `env.close()`.

### `run_cell(...) -> CellResult`

The main entry point — run `n_episodes` for one `(policy, env,
seed_idx)` cell.

```python
run_cell(
    policy: PolicyCallable,
    env: GymLikeEnv,
    *,
    policy_name: str,
    env_spec: EnvSpec,
    seed_idx: int,
    n_episodes: int,
    record_video: bool = True,
    videos_dir: Path | None = None,
    code_sha: str | None = None,
    lerobot_version: str | None = None,
) -> CellResult
```

The seeding contract is applied once at cell start. For each episode:
`env.reset(seed=seed_idx*1000 + e)`, `policy.reset()`, optional first
`render()`, then the `policy → step → [render] → check-termination`
loop until `terminated` / `truncated` / step cap. Per-episode exceptions
are caught and recorded in `EpisodeResult.error`; the cell continues.

**Streaming MP4 encode.** When `record_video` and `videos_dir` are both
set, each episode's frames are encoded to
`videos_dir/"{policy}__{env}__seed{seed}__ep{K:03d}.mp4"` immediately
after the episode ends and the in-memory frames are dropped — this
bounds the working set to one episode's frames at a time. When
`record_video=True` but `videos_dir is None`, frames are kept on the
`EpisodeResult` for the caller to encode later.

`code_sha` and `lerobot_version` default to autodetection
(`git rev-parse HEAD` and `lerobot.__version__`). Raises `ValueError`
for a non-positive `n_episodes` or a negative `seed_idx`.

### `run_cell_from_specs(...) -> CellResult`

```python
run_cell_from_specs(
    policy_spec: PolicySpec,
    env_spec: EnvSpec,
    *,
    seed_idx: int,
    n_episodes: int,
    device: str = "cuda",
    record_video: bool = True,
    videos_dir: Path | None = None,
) -> CellResult
```

Convenience wrapper: build the env via `load_env`, sniff the action
shape from `env.action_space.shape`, load the policy via `load_policy`,
then call `run_cell`. The caller is responsible for `env.close()` once
this function returns. `videos_dir` is forwarded to `run_cell`.

---

## Module `stats`

`lerobot_bench.stats` — statistical helpers behind every leaderboard
claim. Every stochastic function takes an explicit RNG or seed (no
hidden global state); same inputs + same RNG state → same output. The
unit of resampling is the **episode** — pass the flat array of
`5 × n_episodes_per_seed` binary outcomes, never per-seed means. See
[`docs/DESIGN.md`](DESIGN.md) § Methodology and
[`docs/MDE_TABLE.md`](MDE_TABLE.md).

### Result types

- **`class BootstrapResult`** *(frozen dataclass)* — `mean`, `lo`, `hi`,
  `n_resamples`, `ci`. Returned by `bootstrap_ci` and
  `paired_delta_bootstrap`.
- **`class WilcoxonResult`** *(frozen dataclass)* — `statistic`,
  `pvalue`, `n_pairs`, `n_zero_diffs`. `n_zero_diffs` is reported
  separately because Wilcoxon drops zero-difference pairs, which
  materially changes the effective sample size for binary outcomes.

### `bootstrap_ci(outcomes, *, n_resamples=10_000, ci=0.95, rng) -> BootstrapResult`

Percentile-bootstrap CI over a flat 1-D bool array of per-episode
outcomes (`True` = success). Resamples `len(outcomes)` indices with
replacement `n_resamples` times and takes the `ci`-percentile interval
of the resampled means. `rng` (a `numpy.random.Generator`) is required.

### `paired_delta_bootstrap(a, b, *, n_resamples=10_000, ci=0.95, rng) -> BootstrapResult`

Paired percentile-bootstrap of the success-rate delta `mean(a) −
mean(b)`. `a` and `b` are 1-D bool arrays of equal shape, paired by
index — `a[i]` and `b[i]` must come from the same
`(seed_idx, episode_idx)`. Each resample draws shared indices into both
arrays. Use this for cross-cell comparisons only when the seeding
contract was identical and `n_episodes_per_seed` matches.

### `paired_wilcoxon(a, b) -> WilcoxonResult`

Two-sided paired Wilcoxon signed-rank test on index-paired per-episode
outcomes. Ties (`a[i] == b[i]`) are dropped per the standard
convention. Returns `pvalue=1.0` and `statistic=NaN` when every pair is
a tie.

### `cohens_h(p1: float, p2: float) -> float`

Cohen's *h* effect size for the difference between two proportions:
`h = 2·arcsin(√p1) − 2·arcsin(√p2)`. Conventional reading: `|h| < 0.2`
small, `< 0.5` medium, `≥ 0.8` large. Report it alongside every
Δsuccess — a "significant" small-effect delta should not be framed as a
meaningful improvement. Raises `ValueError` if either argument is
outside `[0, 1]`.

### `wilson_ci(successes, n, *, ci=0.95) -> tuple[float, float]`

Wilson score interval for a Bernoulli proportion (Wilson 1927), the
closed-form `(lo, hi)` sanity reference the bootstrap converges to at
large `n`. Raises `ValueError` for `n <= 0`, `successes` outside
`[0, n]`, or `ci` outside `(0, 1)`.

### `wilson_halfwidth_at_p(p, n, *, alpha=0.05) -> float`

Wilson score interval half-width at proportion `p` and `n` trials,
computed as `(hi − lo) / 2` for `wilson_ci(round(p·n), n)`. This is the
function behind [`docs/MDE_TABLE.md`](MDE_TABLE.md) and the leaderboard
"inconclusive at this N" gate. Closed-form, no RNG.

### `mcnemar_paired(b, c, *, exact=True) -> tuple[float, float]`

Paired McNemar test on the two discordant cells of a 2×2 table: `b` =
(A succeeds, B fails) pairs, `c` = (A fails, B succeeds) pairs. With
`exact=True` and `b + c <= 25` it runs an exact two-sided binomial test
(`statistic` is `min(b, c)`); otherwise a chi-square with continuity
correction (`statistic` is the χ² value). Returns `(0.0, 1.0)` when
there are no discordant pairs. Raises `ValueError` for negative `b`/`c`.

### `bootstrap_pivotal_ci(values, *, alpha=0.05, n_resamples=10_000, seed=0) -> tuple[float, float]`

Pivotal ("basic") bootstrap CI on the mean of a 1-D `values` array
(bool or float). The pivotal form reflects the percentile interval
through the point estimate (Efron & Tibshirani 1993, Eq. 13.5),
correcting the percentile interval's bias under an asymmetric sampling
distribution. Takes a deterministic integer `seed` (constructs
`numpy.random.Generator(PCG64(seed))` internally). Pass the flat
per-episode outcome array, not per-seed means.

### `paired_diff_ci(a, b, *, alpha=0.05, n_resamples=10_000, seed=0) -> tuple[float, float]`

Pivotal paired-bootstrap CI on the mean of `a − b`, with `a` and `b`
index-paired and equal-shaped. Resamples index pairs to preserve
within-pair correlation. Same deterministic-`seed` contract as
`bootstrap_pivotal_ci`. Use when the seeding contract was identical and
`n_episodes_per_seed` matches across cells.

---

## Module `render`

`lerobot_bench.render` — episode-frame → small MP4 renderer for the
leaderboard Space. Uses `imageio.v3` end-to-end (no `ffmpeg` shell-out)
and a pure-numpy bilinear resizer.

### Module constants

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_FPS` | `10` | Default output frame rate. |
| `DEFAULT_SIZE` | `256` | Default square output edge in pixels. |
| `DEFAULT_CRF` | `23` | Default libx264 constant rate factor. |
| `MAX_BYTES` | `2 * 1024 * 1024` | Per-clip size cap (2 MiB). |
| `RENDER_LADDER` | `((10,23),(5,23),(5,28),(5,33))` | `(fps, crf)` rungs the encoder walks until a clip fits under `MAX_BYTES`. |

### `class EncoderSettings`

Frozen dataclass of the encoder knobs burned into a result: `fps`,
`size`, `codec`, `pixel_format`, `crf`, `rung_index`. `rung_index` is
the `RENDER_LADDER` index that produced a fit, or `-1` when the ladder
was bypassed by an explicit `fps`/`crf`. A `rung_index >= 1` means
playback is faster than wall-clock (expect frame jumps). For PNG
thumbnails `codec="png"` and `fps=crf=rung_index=0`.

### `class RenderResult`

Frozen dataclass returned by every renderer call: `path`,
`bytes_written`, `frame_count`, `encoder_settings`, and
`content_sha256` (hex SHA-256 of the on-disk file bytes — folded into
the parquet `video_sha256` column).

### `class RenderSizeError(RuntimeError)`

Raised when an encoded clip exceeds `MAX_BYTES`. Carries `path`, `size`,
`limit`, and `attempts` (the full per-rung log in ladder mode). The
file is removed before the exception is raised — no partial artifact is
left behind.

### `render_episode(frames, out_path, *, fps=None, size=DEFAULT_SIZE, crf=None) -> RenderResult`

Encode a `(T, H, W, 3)` uint8 RGB array to a small MP4 at `out_path`
(parent directory must already exist). With `fps` and `crf` both `None`
(the default) the encoder walks `RENDER_LADDER` until the output fits
under `MAX_BYTES`. Passing an explicit `fps` or `crf` bypasses the
ladder for a single-shot encode (`rung_index = -1`). Frames are resized
to `size × size` via the numpy bilinear resizer; `size` must be even.
Raises `ValueError`/`TypeError` on malformed frames and
`RenderSizeError` if every rung overshoots.

### `render_thumbnail_strip(frames, out_path, *, n_thumbs=6, thumb_size=96) -> RenderResult`

Write a horizontal PNG strip of `n_thumbs` evenly-spaced frames from a
`(T, H, W, 3)` uint8 array. If `T < n_thumbs` it uses every frame. Each
thumbnail is `thumb_size × thumb_size`; the strip is
`(thumb_size, n_used · thumb_size, 3)`. The returned `EncoderSettings`
uses `codec="png"`.

---

## Module `checkpointing`

`lerobot_bench.checkpointing` — the cell-boundary resume layer for the
sweep orchestrator. A "cell" is `(policy, env, seed)`. Resume
granularity is the cell boundary because mid-cell resume is not
bit-reproducible (the Torch generator advances across episodes). Pure
`pandas` + `pyarrow` — no torch, no env, no policy loading.

### Module constants

- **`RESULT_SCHEMA: tuple[str, ...]`** — the canonical column order of
  every `results.parquet` row: `policy`, `env`, `seed`,
  `episode_index`, `success`, `return_`, `n_steps`, `wallclock_s`,
  `video_sha256`, `code_sha`, `lerobot_version`, `timestamp_utc`.

### `class CellKey`

Frozen, hashable dataclass identifying a `(policy: str, env: str,
seed: int)` cell.

### `class ResumePlan`

Frozen dataclass — the output of `plan_resume`. Fields:

- `completed_cells: frozenset[CellKey]` — cells with a clean full set of
  episode rows; skipped on resume.
- `partial_cells: frozenset[CellKey]` — cells with some but not a clean
  full set of rows; must be dropped before re-running.
- `pending_cells: frozenset[CellKey]` — requested cells with zero rows.
- `rows_loaded: int` — total rows read from the parquet.

### `load_results(parquet_path: Path) -> pd.DataFrame`

Read an existing `results.parquet`. Returns an empty DataFrame with
exactly `RESULT_SCHEMA` columns if the file is missing; raises
`ValueError` if the file exists but its columns do not match the schema.
Columns are reordered to canonical order.

### `plan_resume(parquet_path, *, requested_cells, n_episodes) -> ResumePlan`

Inspect the parquet and classify each requested cell as completed,
partial, or pending. A cell is **completed** iff its `episode_index` set
equals `set(range(n_episodes))` exactly — any other non-zero count
(missing index, over-write) is **partial**. Cells in the parquet that
are not in `requested_cells` are ignored, so a sweep can be re-shaped
without losing prior work. Raises `ValueError` for non-positive
`n_episodes`.

### `append_cell_rows(parquet_path, new_rows) -> int`

Atomically append `new_rows` (a DataFrame whose columns must be exactly
`RESULT_SCHEMA`) to the parquet — write to a `.tmp.parquet` sibling,
then `os.replace` into place. Guards against duplicate
`(policy, env, seed, episode_index)` keys. Returns the total row count.
A no-op returning the existing count when `new_rows` is empty. Raises
`ValueError` on wrong columns or a duplicate key.

### `drop_partial_cells(parquet_path, cells) -> int`

Remove every row belonging to the given `cells`, used to clean partial
cells before re-running them. Returns the number of rows removed; a
no-op (returns `0`) if the file is missing or `cells` is empty. Uses the
same atomic write strategy as `append_cell_rows`.

---

## Module `cli`

`lerobot_bench.cli` — the `lerobot-bench` console entrypoint, wired into
`[project.scripts]` in `pyproject.toml`.

- **`build_parser() -> argparse.ArgumentParser`** — construct the
  argument parser. Currently exposes only `--version`; subcommands are
  added as the eval library lands.
- **`main(argv: Sequence[str] | None = None) -> int`** — parse `argv`
  (defaults to `sys.argv`), print help, and return an exit code.

The day-to-day benchmark workflows are driven by the `scripts/`
entrypoints, not this CLI — see [`docs/RUNBOOK.md`](RUNBOOK.md).
