"""Pure-Python helpers for the lerobot-bench Gradio Space.

Lives next to ``space/app.py`` but contains no Gradio import. The point
of the split is testability: the project's CI runs against
``tests/test_space.py``, which imports these helpers and asserts on
their behaviour against synthetic parquet data. Gradio is heavy and
not in the repo's ``[dev]`` extras; keeping it out of this module is
what lets the fast pytest job exercise the Space's data layer.

Responsibilities:

1. **Load + cache the published parquet.** ``load_results_df`` reads
   the Hub-hosted ``results.parquet`` (or a local override path used by
   tests) and validates the column set against the canonical
   :data:`lerobot_bench.checkpointing.RESULT_SCHEMA`. The result is
   cached so a tab switch doesn't re-fetch.

2. **Aggregate to leaderboard rows.** ``compute_leaderboard_table``
   groups the per-episode rows by ``(policy, env)`` and emits one row
   per cell with success rate and a Wilson score CI half-width
   (computed via :func:`lerobot_bench.stats.wilson_ci`). The aggregate
   is sorted so that the policy with the highest **overall mean
   success rate** appears first; within each policy, rows are ordered
   alphabetically by env so the table reads as a policy ranking.

3. **Paired comparisons.** ``compute_paired_table`` builds a per-env
   delta table between two policies, with pivotal-bootstrap CIs on the
   delta and a per-cell MDE threshold derived from
   :func:`lerobot_bench.stats.wilson_halfwidth_at_p`. ``narrate_top_finding``
   returns the auto-generated headline sentence under that table.

4. **Failure taxonomy renderer.** ``parse_failure_taxonomy_md`` reads
   the markdown source and returns the list of category names; the
   Gradio "Failures" tab joins those against the (optional)
   ``failure_label`` column in the parquet to produce per-cell counts,
   or falls back to an empty-state message.

5. **Format video URLs.** ``format_video_url`` produces the *direct*
   Hub raw-content URL for one episode's MP4. The Space never proxies
   video bytes — Gradio fetches them straight from Hub. This keeps the
   free-CPU Space well under the per-request memory limit.

A standalone :func:`render_methodology_md` returns the markdown for the
Methodology tab. It's plain text so the test suite can assert the
key methodology terms (seed, Wilson, bootstrap, lerobot version pin)
are present without depending on the disk layout of ``docs/``.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.stats import paired_diff_ci, wilson_ci, wilson_halfwidth_at_p

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #

# HF Hub dataset that hosts the published parquet + MP4 grid.
# Mirrors the value used by scripts/publish_results.py and
# docs/RUNBOOK.md. Bumped in lock-step on a breaking schema change.
HUB_DATASET_REPO = "thrmnn/lerobot-bench-results-v1"

# Direct raw-content URL prefix on the Hub. ``resolve/main`` is what
# Hub returns the actual file bytes for; the alternative ``blob/main``
# returns the HTML preview page, which Gradio's ``gr.Video`` cannot
# play. See docs/DESIGN.md § Video render policy.
HUB_RAW_PREFIX = f"https://huggingface.co/datasets/{HUB_DATASET_REPO}/resolve/main"

# Default location of the parquet inside the dataset repo. Override
# via ``load_results_df(parquet_url=...)`` in tests against synthetic
# parquet files on disk.
DEFAULT_PARQUET_URL = f"{HUB_RAW_PREFIX}/results.parquet"

# Wilson CI confidence level used for the leaderboard table. Mirrors
# the bootstrap CI level in stats.py — 95% is the only value the
# methodology copy mentions, so changing this here without updating
# the methodology tab would be a silent inconsistency.
LEADERBOARD_CI = 0.95

# Canonical leaderboard table columns. The Gradio component is wired
# to this exact tuple (see ``app.py``); changing the order here
# requires updating ``app.py`` too.
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "n_episodes",
    "n_successes",
    "success_rate",
    "ci_half_width",
    "ci_low",
    "ci_high",
)


# --------------------------------------------------------------------- #
# Data loading                                                          #
# --------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _read_parquet_cached(source: str) -> pd.DataFrame:
    """Read parquet from a URL or local path; cached at module level.

    Separated from :func:`load_results_df` so the schema validation +
    column reordering happens on every call (cheap) but the actual
    network / disk read is amortised. The cache is keyed on the
    string source — tests pass synthetic local paths and get a fresh
    cache entry per file.
    """
    return pd.read_parquet(source)


def clear_results_cache() -> None:
    """Drop the cached parquet read.

    Wired to the manual refresh button in ``app.py``: a reviewer
    re-running an updated sweep upstream wants to pull fresh data
    without restarting the Space cold. Tests also call this between
    runs so synthetic-parquet writes don't bleed across cases.
    """
    _read_parquet_cached.cache_clear()


def load_results_df(parquet_url: str | Path | None = None) -> pd.DataFrame:
    """Load the published episode-level results parquet.

    Args:
        parquet_url: HTTP URL or local path. Defaults to
            :data:`DEFAULT_PARQUET_URL` (the published dataset on Hub).
            Tests pass a local path to a synthetic parquet.

    Returns:
        A DataFrame with exactly :data:`RESULT_SCHEMA` columns, in
        canonical order. An empty DataFrame is returned (rather than
        raising) if the source is missing locally — this lets the
        Space render an empty leaderboard with a "no data yet" hint
        instead of crashing during cold start.

    Raises:
        ValueError: if the parquet is reachable but its column set
            does not match :data:`RESULT_SCHEMA`. Schema drift between
            the orchestrator and the Space is a real bug worth
            surfacing loudly.
    """
    source = str(parquet_url) if parquet_url is not None else DEFAULT_PARQUET_URL

    # Local-path missing-file fast path: return an empty frame rather
    # than letting pyarrow raise a FileNotFoundError. Hub URLs cannot
    # be checked this cheaply; if the Hub fetch fails the underlying
    # exception bubbles up and the Space's gradio handler renders the
    # error inline.
    if not source.startswith(("http://", "https://")) and not Path(source).exists():
        return _empty_results_df()

    df = _read_parquet_cached(source)

    actual = set(df.columns)
    expected = set(RESULT_SCHEMA)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"results parquet at {source!r} has wrong columns: missing={missing} extra={extra}"
        )
    return df[list(RESULT_SCHEMA)]


def _empty_results_df() -> pd.DataFrame:
    """Build an empty DataFrame with exactly RESULT_SCHEMA columns.

    Mirrors the dtype map used by ``checkpointing._empty_results_df``
    so downstream code (groupby on ``policy`` / ``env``) does not
    explode on object-dtype empty columns.
    """
    return pd.DataFrame({col: pd.Series(dtype=_dtype_for(col)) for col in RESULT_SCHEMA})


def _dtype_for(col: str) -> str:
    if col in {"seed", "episode_index", "n_steps"}:
        return "int64"
    if col in {"return_", "wallclock_s"}:
        return "float64"
    if col == "success":
        return "bool"
    return "object"


# --------------------------------------------------------------------- #
# Leaderboard aggregation                                               #
# --------------------------------------------------------------------- #


def compute_leaderboard_table(
    df: pd.DataFrame,
    *,
    ci: float = LEADERBOARD_CI,
) -> pd.DataFrame:
    """Aggregate episode-level rows to one ``(policy, env)`` cell per row.

    Each row reports:

    * ``n_episodes`` — total episodes summed across seeds for the cell.
    * ``n_successes`` — count of ``success == True``.
    * ``success_rate`` — n_successes / n_episodes (0..1).
    * ``ci_low`` / ``ci_high`` — Wilson score interval bounds at
      ``ci`` confidence.
    * ``ci_half_width`` — ``(ci_high - ci_low) / 2``. Reported because
      the table is more readable as ``rate ± half_width`` than as a
      pair of bounds.

    Sorted so policies are **ranked by their overall mean success rate
    across envs** (highest first); within each policy, rows are
    alphabetical by env. This produces a top-to-bottom policy ranking
    rather than a flat list of cells, which is what reviewers actually
    read for. Deterministic — the Space renders the same order on
    every reload.

    Empty input returns an empty frame with :data:`LEADERBOARD_COLUMNS`
    so the Gradio table component doesn't choke on a missing column
    list.
    """
    if df.empty:
        return pd.DataFrame({col: [] for col in LEADERBOARD_COLUMNS})

    # groupby on the two-level key. ``observed=True`` would only matter
    # for categorical columns and these are object/string; the default
    # is fine.
    grouped = df.groupby(["policy", "env"], sort=False, dropna=False)
    rows: list[dict[str, object]] = []
    for (policy, env), cell in grouped:
        n = len(cell)
        if n == 0:
            continue
        successes = int(cell["success"].sum())
        rate = successes / n
        lo, hi = wilson_ci(successes, n, ci=ci)
        half = (hi - lo) / 2.0
        rows.append(
            {
                "policy": str(policy),
                "env": str(env),
                "n_episodes": n,
                "n_successes": successes,
                "success_rate": float(rate),
                "ci_half_width": float(half),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )

    out = pd.DataFrame(rows, columns=list(LEADERBOARD_COLUMNS))
    # Rank policies by their overall mean success rate (descending),
    # then sort envs alphabetically within each policy. A separate
    # ``_policy_mean`` column drives the sort and is dropped before
    # return so the public column set stays stable.
    policy_means = out.groupby("policy")["success_rate"].mean()
    out["_policy_mean"] = out["policy"].map(policy_means)
    out = out.sort_values(
        ["_policy_mean", "policy", "env"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )
    out = out.drop(columns=["_policy_mean"])
    return out


# --------------------------------------------------------------------- #
# Browse Rollouts dropdowns                                             #
# --------------------------------------------------------------------- #


def list_unique(df: pd.DataFrame, column: str) -> list[str]:
    """Return the sorted unique string values of ``column`` in ``df``.

    Used to populate the Browse-Rollouts dropdowns. Returns an empty
    list on an empty frame — the dropdowns then render with no
    options and the page emits a "no data yet" message.
    """
    if df.empty or column not in df.columns:
        return []
    values = df[column].dropna().unique().tolist()
    # Cast to str then sort: seeds are ints, the rest are strings;
    # dropdowns are uniformly textual in Gradio so we normalise here.
    return sorted({str(v) for v in values})


def filter_episodes(
    df: pd.DataFrame,
    *,
    policy: str,
    env: str,
    seed: int | str,
) -> pd.DataFrame:
    """Return rows matching ``(policy, env, seed)``, sorted by episode.

    ``seed`` accepts a string (the value Gradio dropdowns emit) or an
    int and coerces to int internally. An empty frame on no match —
    the caller uses ``df.empty`` to drive the "no rollout for this
    combination" message.
    """
    if df.empty:
        return df

    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        return df.iloc[0:0]

    mask = (df["policy"] == policy) & (df["env"] == env) & (df["seed"] == seed_int)
    return df.loc[mask].sort_values("episode_index", kind="stable")


# --------------------------------------------------------------------- #
# Video URL formatting                                                  #
# --------------------------------------------------------------------- #


def format_video_url(policy: str, env: str, seed: int, episode: int) -> str:
    """Return the direct Hub URL for one episode's MP4.

    The Space never proxies the bytes — Gradio's ``gr.Video`` is given
    this URL and fetches it from Hub directly. This is documented in
    DESIGN.md § Video render policy as the load-bearing choice for
    free-CPU compatibility.

    The URL pattern mirrors the directory layout written by
    ``lerobot_bench.render`` + ``scripts/publish_results.py``::

        videos/<policy>/<env>/seed<N>/episode<E>.mp4

    Filenames use zero-padded indices for clean sort order on the
    Hub file browser, but the Hub URL resolver is not picky about
    leading zeros — keep it simple and match the writer.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if episode < 0:
        raise ValueError(f"episode must be non-negative, got {episode}")
    return f"{HUB_RAW_PREFIX}/videos/{policy}/{env}/seed{seed}/episode{episode}.mp4"


def episode_metadata(row: pd.Series) -> dict[str, object]:
    """Project one parquet row into the JSON metadata block under a video.

    Pulls success / return / steps / wallclock so the reviewer sees
    "did this episode succeed and at what cost" alongside the playback
    without scrolling. Renamed for human-readable display; the
    underlying parquet column names mirror RESULT_SCHEMA.
    """
    return {
        "policy": str(row.get("policy", "")),
        "env": str(row.get("env", "")),
        "seed": int(row.get("seed", 0)),
        "episode_index": int(row.get("episode_index", 0)),
        "success": bool(row.get("success", False)),
        "return": float(row.get("return_", 0.0)),
        "n_steps": int(row.get("n_steps", 0)),
        "wallclock_s": float(row.get("wallclock_s", 0.0)),
    }


# --------------------------------------------------------------------- #
# Methodology tab content                                               #
# --------------------------------------------------------------------- #


def render_methodology_md() -> str:
    """Return the markdown shown on the Methodology tab.

    Mirrors ``docs/DESIGN.md`` § Methodology in headline form. Kept as
    a Python constant rather than a separate file so the Space ships
    as one app + one helper module — easier to deploy from the HF
    Spaces git remote without dragging in the whole repo's docs tree.

    The test suite asserts that the four load-bearing terms appear
    here verbatim (``seed``, ``Wilson``, ``bootstrap``, the lerobot
    version pin). Add new sections freely; do not silently drop those
    words.
    """
    return (
        "## Methodology\n"
        "\n"
        "This Space renders the published results of the **lerobot-bench**\n"
        "multi-policy benchmark. All numbers come from a pre-computed sweep\n"
        "on `lerobot==0.5.1`; the Space itself runs no policy inference\n"
        "and uses no GPU.\n"
        "\n"
        "### Sweep contract\n"
        "\n"
        "* **Episodes per cell:** 5 seeds × 50 episodes = 250 binary\n"
        "  outcomes per `(policy, env)` cell, unless the auto-downscope\n"
        "  rule (per-cell wall-clock budget) reduced it. Per-cell episode\n"
        "  counts surface in the Leaderboard table's `n_episodes` column.\n"
        "* **Seeding:** at the start of each cell, `numpy`, `torch`, and\n"
        "  `torch.cuda` are seeded with `seed_idx * 1000`. Per episode\n"
        "  `e`, the env is reset with `seed = seed_idx * 1000 + e`. Policy\n"
        "  stochasticity inherits the torch generator and is **not**\n"
        "  re-seeded per episode — mid-cell resume is therefore not\n"
        "  bit-reproducible. The orchestrator only resumes at cell\n"
        "  boundaries.\n"
        "* **Success threshold:** the per-env standard reward threshold\n"
        "  from `lerobot.envs.<env>.config.SUCCESS_REWARD`. Episodes\n"
        "  succeed iff their final reward meets or exceeds it.\n"
        "\n"
        "### Confidence intervals\n"
        "\n"
        "The Leaderboard table reports the Wilson score interval at 95% on\n"
        "the per-cell success rate; `±` is the half-width of that\n"
        "interval. Wilson is closed-form, faster than the bootstrap, and\n"
        "the bootstrap interval (used in the upstream notebook for the\n"
        "headline finding) converges to it at the per-cell sample sizes\n"
        "this benchmark runs. Cross-cell comparisons (Δsuccess + paired\n"
        "Wilcoxon, Cohen's h effect size) live in the analysis notebook;\n"
        "the Space surfaces only the per-cell intervals.\n"
        "\n"
        "### Reading the Browse-Rollouts tab\n"
        "\n"
        "Pick `(policy, env, seed)` and the Space lists all episodes for\n"
        "that cell. Each video is loaded from the Hub dataset directly\n"
        "(`gr.Video` receives a Hub `resolve/main/...mp4` URL — the Space\n"
        "never proxies video bytes through itself). First-frame latency on\n"
        "the free CPU tier is typically 5-10 seconds; subsequent plays are\n"
        "served from your browser cache.\n"
        "\n"
        "### Reproducibility pointer\n"
        "\n"
        "Code: <https://github.com/thrmnn/lerobot-bench>. Each parquet row\n"
        "carries `code_sha`, `lerobot_version`, and `video_sha256` so any\n"
        "row can be replayed end-to-end with `python scripts/run_one.py\n"
        "--policy <p> --env <e> --seed <n>`. See the repo's `docs/RUNBOOK.md`\n"
        "for the full repro recipe.\n"
    )


# --------------------------------------------------------------------- #
# Paired-comparison tab                                                 #
# --------------------------------------------------------------------- #


# Bootstrap iteration count for the paired CI on Δsuccess. Mirrors the
# analysis-notebook default (DESIGN.md § Methodology) so the Space's
# numbers match the published headline finding to within Monte-Carlo
# noise. ``paired_diff_ci`` is closed under a deterministic seed so the
# table is bit-stable across reloads.
PAIRED_N_RESAMPLES = 10_000

# Seed used to drive the pivotal bootstrap CI on paired deltas. Same
# integer the analysis notebook uses; locks the Space's CI bounds so a
# refresh produces the same numbers as the paper's headline figure.
PAIRED_BOOTSTRAP_SEED = 0

# Significance level used for the paired comparison CI on Δsuccess. 95%
# CI matches LEADERBOARD_CI; the alpha-style call into stats.py wants
# 1 - ci.
PAIRED_ALPHA = 1.0 - LEADERBOARD_CI

# Canonical column order for the paired-comparison table. The Gradio
# DataFrame component renders columns in this order. ``clears_MDE`` is
# a bool so the UI can colour-code the chip without parsing strings.
PAIRED_COLUMNS: tuple[str, ...] = (
    "env",
    "n_A",
    "n_B",
    "success_rate_A",
    "ci_half_width_A",
    "success_rate_B",
    "ci_half_width_B",
    "delta",
    "ci_low_delta",
    "ci_high_delta",
    "MDE",
    "clears_MDE",
)


def compute_paired_table(
    df: pd.DataFrame,
    *,
    policy_a: str,
    policy_b: str,
    envs: list[str] | None = None,
    ci: float = LEADERBOARD_CI,
    n_resamples: int = PAIRED_N_RESAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compute the per-env paired comparison between two policies.

    For each env where both policies have rows, the row reports:

    * ``success_rate_A`` / ``ci_half_width_A`` — A's marginal success
      rate with Wilson 95% half-width.
    * ``success_rate_B`` / ``ci_half_width_B`` — same for B.
    * ``delta`` — ``success_rate_A - success_rate_B``.
    * ``ci_low_delta`` / ``ci_high_delta`` — pivotal-bootstrap 95% CI
      on ``delta``, computed via
      :func:`lerobot_bench.stats.paired_diff_ci`. Pairing is by
      ``(seed, episode_index)`` — if the per-(seed, episode) pairs do
      not align between cells, that env falls back to the
      cell sizes' min and is logged.
    * ``MDE`` — per-cell minimum detectable difference at the cell's
      ``max(p_hat_a, p_hat_b)``, derived from
      :func:`lerobot_bench.stats.wilson_halfwidth_at_p` (2·HW under
      the looser Wilson-band rule from ``docs/MDE_TABLE.md`` §4).
    * ``clears_MDE`` — ``|delta| >= MDE``. The UI renders this as a
      coloured chip.

    Envs where one of the policies has no rows are excluded with a
    log line — the caller's "no data" message wins over an empty row.

    Args:
        df: episode-level results frame (canonical RESULT_SCHEMA).
        policy_a / policy_b: policy names to compare. ``a == b`` is
            allowed (produces an all-zero delta column) but pointless.
        envs: subset of envs to include; ``None`` means "every env
            where both policies have rows".
        ci: CI level for the marginal Wilson half-widths and the
            paired bootstrap on Δ. Default 0.95 mirrors the rest of
            the Space.
        n_resamples / seed: passthrough to
            :func:`paired_diff_ci`; same seed → same CI bounds.

    Returns:
        DataFrame with columns :data:`PAIRED_COLUMNS`. Empty input,
        identical policies with no overlapping env, and "both policies
        absent" all return an empty frame with the canonical column
        set so the Gradio table renders the header row regardless.
    """
    empty = pd.DataFrame({col: [] for col in PAIRED_COLUMNS})

    if df.empty or "policy" not in df.columns or "env" not in df.columns:
        return empty

    df_a = df[df["policy"] == policy_a]
    df_b = df[df["policy"] == policy_b]

    if df_a.empty and df_b.empty:
        return empty

    envs_a = set(df_a["env"].unique().tolist()) if not df_a.empty else set()
    envs_b = set(df_b["env"].unique().tolist()) if not df_b.empty else set()
    shared = sorted(envs_a & envs_b)

    if envs is not None:
        # Restrict to the requested envs but still drop those one side
        # is missing. Log each drop so the caller knows the table is
        # narrower than the dropdown implied.
        requested = list(envs)
        missing = [e for e in requested if e not in shared]
        for env_missing in missing:
            logger.info(
                "paired-comparison: dropping env %r — %r has %d rows, %r has %d rows",
                env_missing,
                policy_a,
                int((df_a["env"] == env_missing).sum()),
                policy_b,
                int((df_b["env"] == env_missing).sum()),
            )
        envs_to_use = [e for e in requested if e in shared]
    else:
        envs_to_use = shared

    if not envs_to_use:
        return empty

    rows: list[dict[str, object]] = []
    for env in envs_to_use:
        cell_a = df_a[df_a["env"] == env]
        cell_b = df_b[df_b["env"] == env]

        n_a = len(cell_a)
        n_b = len(cell_b)
        if n_a == 0 or n_b == 0:
            # Defensive: shouldn't hit this after the shared filter,
            # but a stray NaN in env could slip through.
            logger.info("paired-comparison: skipping env %r (n_a=%d, n_b=%d)", env, n_a, n_b)
            continue

        successes_a = int(cell_a["success"].sum())
        successes_b = int(cell_b["success"].sum())
        p_a = successes_a / n_a
        p_b = successes_b / n_b

        lo_a, hi_a = wilson_ci(successes_a, n_a, ci=ci)
        lo_b, hi_b = wilson_ci(successes_b, n_b, ci=ci)
        hw_a = (hi_a - lo_a) / 2.0
        hw_b = (hi_b - lo_b) / 2.0

        a_arr, b_arr = _pair_outcomes(cell_a, cell_b)
        if a_arr.size == 0:
            # No alignable pairs at all — fall back to the marginal
            # rates only. ``delta`` is well-defined; the CI is left
            # NaN so the UI can show a dash.
            delta = p_a - p_b
            ci_lo = float("nan")
            ci_hi = float("nan")
        else:
            delta = float(a_arr.mean() - b_arr.mean())
            ci_lo, ci_hi = paired_diff_ci(
                a_arr,
                b_arr,
                alpha=PAIRED_ALPHA,
                n_resamples=n_resamples,
                seed=seed,
            )

        # MDE bound at the higher of the two cells' p̂. Pulled from
        # docs/MDE_TABLE.md §4 — the per-cell threshold is 2·HW at
        # max(p̂_a, p̂_b), not a flat 0.124. We use the larger N
        # because the cells' Ns can differ after auto-downscope.
        mde_n = min(n_a, n_b)
        mde = 2.0 * wilson_halfwidth_at_p(max(p_a, p_b), mde_n, alpha=PAIRED_ALPHA)
        clears = bool(abs(delta) >= mde) if mde > 0 else False

        rows.append(
            {
                "env": str(env),
                "n_A": n_a,
                "n_B": n_b,
                "success_rate_A": float(p_a),
                "ci_half_width_A": float(hw_a),
                "success_rate_B": float(p_b),
                "ci_half_width_B": float(hw_b),
                "delta": float(delta),
                "ci_low_delta": float(ci_lo),
                "ci_high_delta": float(ci_hi),
                "MDE": float(mde),
                "clears_MDE": clears,
            }
        )

    if not rows:
        return empty

    out = pd.DataFrame(rows, columns=list(PAIRED_COLUMNS))
    # Sort by absolute delta descending so the most discriminating
    # envs surface at the top of the table.
    out = out.assign(_abs_delta=out["delta"].abs())
    out = out.sort_values(
        ["_abs_delta", "env"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )
    out = out.drop(columns=["_abs_delta"])
    return out


def _pair_outcomes(
    cell_a: pd.DataFrame,
    cell_b: pd.DataFrame,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Align ``cell_a`` and ``cell_b`` on ``(seed, episode_index)`` pairs.

    Both inputs are row-wise episode frames for a single ``(policy, env)``
    cell. Returns ``(a, b)`` 1-D boolean arrays containing only the
    pairs that exist in both — the bootstrap requires equal-length
    paired arrays. Order is deterministic: sorted by ``(seed,
    episode_index)`` ascending.

    Returns two empty arrays if there is no overlap.
    """
    cols = ["seed", "episode_index", "success"]
    if not all(c in cell_a.columns for c in cols) or not all(c in cell_b.columns for c in cols):
        return np.array([], dtype=bool), np.array([], dtype=bool)

    merged = (
        cell_a[cols]
        .merge(
            cell_b[cols],
            on=["seed", "episode_index"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        .sort_values(["seed", "episode_index"], kind="stable")
    )

    a = merged["success_a"].to_numpy(dtype=bool, copy=False)
    b = merged["success_b"].to_numpy(dtype=bool, copy=False)
    return a, b


def narrate_top_finding(
    paired_df: pd.DataFrame,
    *,
    policy_a: str,
    policy_b: str,
) -> str:
    """Return the human-readable sentence under the paired-comparison table.

    Picks the env with the largest absolute delta whose ``clears_MDE``
    bit is set; if none clear, falls back to an honest "no env clears
    the MDE bound at this N" sentence. The sentence is plain prose so
    a reviewer can read it without scrubbing the table.

    Empty input → an empty-state sentence directing the reviewer to
    pick policies.
    """
    if paired_df.empty:
        return "_Pick two policies above to compute their per-env Δsuccess._"

    clears = paired_df[paired_df["clears_MDE"].astype(bool)]
    if clears.empty:
        # No env clears its MDE bound. Report the largest observed
        # delta and call it inconclusive at this N, per MDE_TABLE.md §5.
        biggest_idx = int(paired_df["delta"].abs().idxmax())
        biggest = paired_df.loc[biggest_idx]
        env = biggest["env"]
        delta = float(biggest["delta"])
        mde = float(biggest["MDE"])
        n = int(min(biggest["n_A"], biggest["n_B"]))
        sign = "outperforms" if delta > 0 else "underperforms"
        return (
            f"No env clears the MDE bound. Largest observed delta is on "
            f"`{env}`: `{policy_a}` {sign} `{policy_b}` by "
            f"{abs(delta):.3f}, but this is below the MDE of "
            f"{mde:.3f} at N={n} — inconclusive at this sample size."
        )

    # Pick the env with the largest |delta| among those that clear MDE.
    # idxmax returns the original DataFrame index; positional lookup via
    # ``loc`` keeps the row whether or not the frame was re-indexed.
    biggest_idx = int(clears["delta"].abs().idxmax())
    biggest = clears.loc[biggest_idx]
    env = biggest["env"]
    delta = float(biggest["delta"])
    ci_lo = float(biggest["ci_low_delta"])
    ci_hi = float(biggest["ci_high_delta"])
    mde = float(biggest["MDE"])
    n = int(min(biggest["n_A"], biggest["n_B"]))
    sign = "outperforms" if delta > 0 else "underperforms"
    ci_str = "" if np.isnan(ci_lo) or np.isnan(ci_hi) else f" (95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}])"
    return (
        f"On `{env}`, `{policy_a}` {sign} `{policy_b}` by "
        f"{abs(delta):.3f}{ci_str}; this clears the MDE bound of "
        f"{mde:.3f} for N={n}."
    )


# --------------------------------------------------------------------- #
# Failure taxonomy tab                                                  #
# --------------------------------------------------------------------- #


# Canonical six-mode label strings. Mirrors docs/FAILURE_TAXONOMY.md
# "Mode labels" section verbatim — the chart legend keys on these and
# the labeling protocol forbids inventing new sub-categories.
FAILURE_MODES: tuple[str, ...] = (
    "trajectory_overshoot",
    "gripper_slip",
    "timeout",
    "wrong_object",
    "premature_release",
    "drift",
)

# Path to the canonical taxonomy doc, resolved at import time so the
# Space can fall back to the bundled snapshot if docs/ is not on disk
# (the HF Space deploy ships only ``space/`` + the lerobot_bench
# package). Tests pass an explicit path.
_DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "docs" / "FAILURE_TAXONOMY.md"


def parse_failure_taxonomy_md(
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Parse the failure-taxonomy markdown into a list of categories.

    Reads the doc at ``path`` (or the repo's canonical
    ``docs/FAILURE_TAXONOMY.md`` when ``None``) and pulls out the six
    ``### N. <Title>`` headings as the canonical mode list. Each
    returned dict has::

        {
            "name": "Trajectory overshoot",
            "label": "trajectory_overshoot",
            "summary": "<first non-empty paragraph after the heading>",
        }

    ``label`` is the snake_case identifier the CSV template uses (see
    :data:`FAILURE_MODES`); it's matched positionally against the
    headings in document order.

    Returns an empty list if the file doesn't exist — the caller
    renders an empty-state message in that case.
    """
    src_path = Path(path) if path is not None else _DEFAULT_TAXONOMY_PATH
    if not src_path.exists():
        logger.info("failure-taxonomy doc not found at %s", src_path)
        return []

    text = src_path.read_text(encoding="utf-8")

    # Match ``### <N>. <Title>`` headings. The doc currently numbers
    # them 1..6 in document order; the regex is permissive on the
    # number range so a 7th mode added later picks up automatically.
    pattern = re.compile(r"^###\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    categories: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        # Pull the first non-empty paragraph after the heading as the
        # summary. We look between this heading and the next match
        # (or EOF for the last one).
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        summary = _first_paragraph(body)

        # Snake_case label: positional mapping if available, else a
        # best-effort lower-snake of the heading. The doc-canonical
        # mapping wins because the CSV legend keys on the snake form.
        label = FAILURE_MODES[i] if i < len(FAILURE_MODES) else _slugify(title)

        categories.append({"name": title, "label": label, "summary": summary})

    return categories


def _first_paragraph(body: str) -> str:
    """Return the first non-empty paragraph from a markdown body.

    Strips the leading ``**Definition.**`` bold marker the taxonomy
    uses for the lead line so the summary reads naturally.
    """
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Strip the "**Definition.** " preamble if present — the
        # taxonomy uses it on every mode and the summary reads
        # better without it.
        chunk = re.sub(r"^\*\*Definition\.\*\*\s*", "", chunk)
        # Collapse hard wraps to a single line so the markdown
        # renderer treats the summary as one paragraph.
        return " ".join(line.strip() for line in chunk.splitlines() if line.strip())
    return ""


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def compute_failure_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell counts of labeled failures, one row per (policy, env, mode).

    Reads the optional ``failure_label`` column from the results
    parquet. When the column is missing or empty, returns an empty
    frame with columns ``(policy, env, mode, count)`` — the caller
    renders the empty-state message instead of a chart.

    Only rows where ``success == False`` AND ``failure_label`` is one
    of the canonical :data:`FAILURE_MODES` contribute. Labels outside
    that set are dropped with a log line (the labeling protocol
    forbids inventing new modes).
    """
    columns = ("policy", "env", "mode", "count")
    empty = pd.DataFrame({c: [] for c in columns})

    if df.empty or "failure_label" not in df.columns:
        return empty

    # Filter to failed episodes with a recognised label.
    failed = df[~df["success"].astype(bool)]
    labeled = failed[failed["failure_label"].notna()]
    if labeled.empty:
        return empty

    canonical = set(FAILURE_MODES)
    out_of_set = labeled[~labeled["failure_label"].isin(canonical)]
    if not out_of_set.empty:
        unique_unknown = sorted(out_of_set["failure_label"].astype(str).unique().tolist())
        logger.info(
            "failure-taxonomy: dropping %d rows with unknown labels: %s",
            len(out_of_set),
            unique_unknown,
        )
    labeled = labeled[labeled["failure_label"].isin(canonical)]
    if labeled.empty:
        return empty

    grouped = (
        labeled.groupby(["policy", "env", "failure_label"], sort=True)
        .size()
        .reset_index(name="count")
        .rename(columns={"failure_label": "mode"})
    )
    return grouped[list(columns)].sort_values(
        ["policy", "env", "mode"], ignore_index=True, kind="stable"
    )


def render_failure_panel_markdown(
    categories: list[dict[str, str]],
    counts: pd.DataFrame,
) -> str:
    """Top-of-panel markdown: the six categories + a status line.

    Always shows the bulleted category list (so the panel is useful
    pre-labels). Below it: either a "no labels yet" empty state or a
    one-line summary of how many labels have been ingested.
    """
    if not categories:
        return (
            "_Failure taxonomy doc not found. See_ "
            "<https://github.com/thrmnn/lerobot-bench/blob/main/docs/FAILURE_TAXONOMY.md>."
        )

    lines = ["### Categories", ""]
    for cat in categories:
        summary = cat["summary"]
        # Trim very long summaries to one line at render time so the
        # bullet list stays scannable.
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."
        lines.append(f"- **{cat['name']}** (`{cat['label']}`) — {summary}")

    lines.append("")
    if counts.empty:
        lines.append(
            "_No failure labels yet. Use the local dashboard to label_ "
            "_rollouts; this panel will populate once `labels.json` files_ "
            "_exist in the published dataset._"
        )
    else:
        n_labels = int(counts["count"].sum())
        n_cells = int(counts.groupby(["policy", "env"]).ngroups)
        lines.append(f"_Loaded **{n_labels}** labeled failures across **{n_cells}** cells._")

    return "\n".join(lines)


# --------------------------------------------------------------------- #
# v1 status badge                                                       #
# --------------------------------------------------------------------- #


def render_v1_status(manifest: dict[str, Any] | None = None) -> str:
    """One-line status badge for the leaderboard header.

    When a parsed sweep manifest is passed, derives the running /
    completed / pending counts from it. Otherwise returns the
    hardcoded "v1 in progress" copy — the manifest is not yet
    published to the Hub dataset at the time of writing (see
    ``scripts/publish_results.py``), so the hardcoded form is the
    expected path on the live Space.
    """
    if manifest is None:
        return (
            "**v1 status** · 6 policies × 6 envs · 22/22 cells calibrated · "
            "sweep running · Pi0 family deferred to v1.1"
        )

    cells = manifest.get("cells", [])
    by_status: dict[str, int] = {}
    for c in cells:
        s = str(c.get("status", "unknown"))
        by_status[s] = by_status.get(s, 0) + 1
    completed = by_status.get("completed", 0)
    total = len(cells)
    return (
        f"**v1 status** · {completed}/{total} cells completed · "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        + " · Pi0 family deferred to v1.1"
    )
