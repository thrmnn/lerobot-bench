"""Pure-Python helpers for the lerobot-bench Gradio Space.

Lives next to ``space/app.py`` but contains no Gradio import. The point
of the split is testability: the project's CI runs against
``tests/test_space.py``, which imports these helpers and asserts on
their behaviour against synthetic parquet data. Gradio is heavy and
not in the repo's ``[dev]`` extras; keeping it out of this module is
what lets the fast pytest job exercise the Space's data layer.

Three responsibilities:

1. **Load + cache the published parquet.** ``load_results_df`` reads
   the Hub-hosted ``results.parquet`` (or a local override path used by
   tests) and validates the column set against the canonical
   :data:`lerobot_bench.checkpointing.RESULT_SCHEMA`. The result is
   cached so a tab switch doesn't re-fetch.

2. **Aggregate to leaderboard rows.** ``compute_leaderboard_table``
   groups the per-episode rows by ``(policy, env)`` and emits one row
   per cell with success rate and a Wilson score CI half-width
   (computed via :func:`lerobot_bench.stats.wilson_ci`). The aggregate
   is sortable; default order is mean success descending. Empty input
   returns an empty frame with the canonical column set so the Gradio
   table doesn't crash on first paint.

3. **Format video URLs.** ``format_video_url`` produces the *direct*
   Hub raw-content URL for one episode's MP4. The Space never proxies
   video bytes — Gradio fetches them straight from Hub. This keeps the
   free-CPU Space well under the per-request memory limit.

A standalone :func:`render_methodology_md` returns the markdown for the
Methodology tab. It's plain text so the test suite can assert the
key methodology terms (seed, Wilson, bootstrap, lerobot version pin)
are present without depending on the disk layout of ``docs/``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.stats import wilson_ci

# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #

# HF Hub dataset that hosts the published parquet + MP4 grid.
# Mirrors the value used by scripts/publish_results.py and
# docs/RUNBOOK.md. Bumped in lock-step on a breaking schema change.
HUB_DATASET_REPO = "Theozinh0/lerobot-bench-results-v1"

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

    Sorted by ``success_rate`` descending, then ``policy`` then ``env``
    for ties — deterministic so the Space renders the same order on
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
    # Sort: highest success first, then policy / env for stability.
    out = out.sort_values(
        ["success_rate", "policy", "env"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )
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
