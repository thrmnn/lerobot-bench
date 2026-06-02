"""Pure-Python helpers for the local-first sweep dashboard.

Lives next to ``dashboard/app.py`` but contains no Gradio import. Same
split rationale as ``space/_helpers.py``: the pytest job exercises this
module directly, and Gradio is not in the project's ``[dev]`` extras.

Three tabs, three concerns:

1. **Sweep progress** -- :func:`load_sweep_manifests` discovers every
   ``results/**/sweep_manifest.json`` and the matching results.parquet,
   :func:`build_progress_table` rolls the manifest cells up to a
   ``(policy, env)`` grid with per-cell status + seeds-completed + ETA.
2. **Calibration inspector** -- :func:`load_latest_calibration` reads
   the newest ``results/calibration-*.json``, :func:`build_calibration_table`
   projects the dataclass to a table with the auto-downscope reason
   reverse-engineered from the timing thresholds.
3. **Rollout preview** -- :func:`scan_video_index` walks every videos
   directory (local + the Windows-mounted Robotics-Data drive) and
   returns a cached map of ``(policy, env, seed, episode) -> path``.
   Dropdown options come from :func:`video_index_options`; lookups go
   through :func:`find_video_path`.

The dashboard reads from disk only -- no Hub fetches, no policy
inference. Cell statuses come straight out of the manifest JSON, so
the dashboard is correct exactly when the manifest is, which is the
same contract the operator already relies on for resume.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Reuse the audited Wilson primitives from ``src/lerobot_bench/stats.py``
# rather than reimplementing the score interval here. The dashboard runs
# from the repo so ``src/`` is importable; in the slim test env it is put
# on the path by ``conftest`` / ``PYTHONPATH``. We add it defensively so
# ``python dashboard/app.py`` works from inside the dir too.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lerobot_bench.envs import EnvRegistry, EnvSpec  # noqa: E402
from lerobot_bench.leaderboard_filter import (  # noqa: E402
    V1_POLICIES,
    filter_to_v1_policies,
)
from lerobot_bench.policies import PolicyRegistry, PolicySpec  # noqa: E402
from lerobot_bench.stats import wilson_ci, wilson_halfwidth_at_p  # noqa: E402

# ``V1_POLICIES`` + ``filter_to_v1_policies`` are imported (not redefined)
# so this dashboard and the Gradio Space share one v1 policy gate. Both are
# re-exported for back-compat with ``from _helpers import ...`` call sites.
__all__ = ["V1_POLICIES", "filter_to_v1_policies"]

# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #

# Repo root resolved relative to this file. The dashboard is launched
# from anywhere (typically ``make dashboard`` from the repo root, but
# also ``python dashboard/app.py`` from inside the dir), so we anchor
# all default paths off this constant rather than ``Path.cwd()``.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default directories scanned for sweep state. Mirrors the layout in
# ``configs/sweep_*.yaml`` (``results/sweep-{mini,full}/...``).
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Extra video roots. The Windows-mounted drive is enumerated lazily and
# its directory listing is cached -- metadata ops over the 9p mount are
# slow even when the file streams themselves are fine.
DEFAULT_VIDEO_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "results",
    Path.home() / "Robotics-Data" / "lerobot-bench-videos",
)

# Manifest cell statuses, mirrored from ``scripts/run_sweep.py``. Kept
# as a local constant so the dashboard can be imported without dragging
# the scripts package into the dashboard's import graph (and therefore
# its requirements).
STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
KNOWN_STATUSES = frozenset({STATUS_PENDING, STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED})

# Per-cell roll-up status surfaced in the progress table. Distinct from
# the manifest-row status because one (policy, env) grid cell rolls up
# multiple seeds.
CELL_STATUS_QUEUED = "queued"  # all seeds pending
CELL_STATUS_RUNNING = "running"  # at least one seed started, not all done
CELL_STATUS_DONE = "done"  # every seed completed (or skipped)
CELL_STATUS_FAILED = "failed"  # at least one seed failed
CELL_STATUS_SKIPPED = "skipped"  # every seed skipped (incompat or pre-resumed)

# Calibration auto-downscope thresholds. Sourced from
# ``scripts/calibrate.py`` -- we keep duplicates here so the dashboard
# can derive a human-readable reason without importing ``scripts.*``
# (the scripts package imports ``lerobot_bench`` which would force
# heavyweight transitive deps just to render a table).
SLOW_MS_PER_STEP_THRESHOLD = 100.0
VERY_SLOW_MS_PER_STEP_THRESHOLD = 500.0
HIGH_VRAM_THRESHOLD_MB = 5500.0
VERY_HIGH_VRAM_THRESHOLD_MB = 7000.0

# --------------------------------------------------------------------- #
# Statistical-rigor constants (council audit P0)                        #
# --------------------------------------------------------------------- #
#
# A Wilson score interval at small N spans most of [0, 1] and misleads
# the operator into reading noise as signal. Below this many episodes
# the success-rate / CI columns render "—" instead of a number. The
# threshold is a judgement call, not a power calculation: at n=25 the
# Wilson half-width at p=0.5 is ~0.18, wide but no longer absurd.
MIN_N_FOR_SUCCESS_RATE = 25

# Placeholder rendered wherever a statistic is suppressed (small-N gate,
# missing raw data). A single constant so the UI is uniform.
STAT_PLACEHOLDER = "—"

# Per-seed dispersion above which the 5-seed budget is likely too small
# to pin the cell's success rate. Flagged red in the progress table.
SEED_SPREAD_FLAG_THRESHOLD = 0.2

# Wilson "inconclusive band" 2·HW at p=0.5, N=250 -- the v1 reference
# line on the half-width-vs-N plot. Sourced from ``docs/MDE_TABLE.md``
# § TL;DR ("Wilson inconclusive band 2·HW at p=0.5, N=250 = 0.1230").
# This is the smallest paired Δ two N=250 cells could produce from
# sampling noise alone; a running cell whose *single-cell* half-width
# is still above half of this is nowhere near resolving a real effect.
V1_INCONCLUSIVE_BAND = 0.123

# Latency-skew flag for the calibration table. The auto-downscope rule
# keys on the *mean* step time; when the p95/mean ratio is large the
# mean understates the tail and the budget derived from it is fragile.
LATENCY_SKEW_FLAG_RATIO = 3.0

# Progress-table columns. Wired into the Gradio Dataframe in ``app.py``.
# The three rigor columns (success_rate_so_far, wilson_ci_so_far,
# seed_spread) come from ``results.parquet``; the rest are manifest-only.
PROGRESS_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "status",
    "seeds_done",
    "seeds_total",
    "episodes_done",
    "episodes_total",
    "success_rate_so_far",
    "wilson_ci_so_far",
    "seed_spread",
    "last_update_utc",
    "eta_minutes",
)

# Calibration-table columns. ``reason`` is derived locally; everything
# else is straight from the CellTiming dataclass on disk. ``std_step_ms``
# and ``latency_skew`` are the council rigor additions.
CALIBRATION_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "status",
    "mean_step_ms",
    "p95_step_ms",
    "std_step_ms",
    "n_steps",
    "latency_skew",
    "vram_peak_mb",
    "recommended_seeds",
    "recommended_episodes",
    "reason",
)


# --------------------------------------------------------------------- #
# Sweep manifest loading                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepRun:
    """One discovered sweep run on disk.

    Identified by the directory containing ``sweep_manifest.json``. The
    operator typically has at most a handful of these
    (``results/sweep-mini/``, ``results/sweep-full/``); the dropdown
    label uses the directory basename so it's recognisable at a glance.
    """

    name: str
    manifest_path: Path
    results_path: Path | None
    started_utc: str | None
    finished_utc: str | None
    manifest: dict[str, Any]

    @property
    def label(self) -> str:
        """Human-readable label for the dropdown.

        Includes the finished marker so the operator can spot an
        in-progress sweep at a glance.
        """
        marker = "[running]" if self.finished_utc is None else "[done]"
        return f"{self.name} {marker}"


def discover_sweep_runs(results_root: Path = DEFAULT_RESULTS_DIR) -> list[SweepRun]:
    """Find every ``sweep_manifest.json`` under ``results_root``.

    Returns runs sorted by ``started_utc`` descending so the most recent
    sweep is first in the dropdown. Missing root returns ``[]`` -- the
    dashboard renders an empty progress table with a hint rather than
    crashing.
    """
    if not results_root.exists():
        return []

    runs: list[SweepRun] = []
    for manifest_path in sorted(results_root.rglob("sweep_manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        if not isinstance(data, dict):
            continue

        # results.parquet sits next to the manifest -- mirror of
        # ``run_sweep.manifest_path_for``. The file may not exist yet
        # if the sweep just started; that's fine, the manifest alone
        # drives the progress table.
        candidate_results = manifest_path.parent / "results.parquet"
        results_path = candidate_results if candidate_results.exists() else None

        runs.append(
            SweepRun(
                name=manifest_path.parent.name,
                manifest_path=manifest_path,
                results_path=results_path,
                started_utc=data.get("started_utc"),
                finished_utc=data.get("finished_utc"),
                manifest=data,
            )
        )

    runs.sort(key=lambda r: r.started_utc or "", reverse=True)
    return runs


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a manifest JSON. Returns ``{}`` on missing/invalid file.

    A missing manifest is not an error -- the operator may have pointed
    the dashboard at a results dir that's still being populated. The
    progress tab then renders an empty table.
    """
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("manifest %s unreadable: %s", manifest_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------- #
# Per-episode results parquet (with stale-data fallback)                #
# --------------------------------------------------------------------- #
#
# ``results.parquet`` is appended to *while the sweep runs*, so a read
# can race a write and surface a truncated / invalid file. PR #42's
# intent was a shared ``StaleDataCache``; it never landed, so this
# module ships its own tiny last-good cache: a read failure falls back
# to the last successfully parsed frame for that path rather than
# blanking the rigor columns mid-sweep.


@dataclass
class _StaleDataCache:
    """Per-path cache of the last successfully read DataFrame.

    Keyed on the absolute parquet path. ``load_results_parquet`` writes
    a fresh frame on every clean read and reads back the last-good frame
    when pyarrow chokes on a half-written file. Module-level singleton
    (:data:`_RESULTS_CACHE`); the dashboard is single-process so no lock
    is needed.
    """

    by_path: dict[str, pd.DataFrame] = field(default_factory=dict)


_RESULTS_CACHE = _StaleDataCache()


def load_results_parquet(results_path: Path | None) -> pd.DataFrame | None:
    """Read a per-episode ``results.parquet``, tolerating a mid-write file.

    The sweep appends rows to the parquet as cells finish, so a read on
    the dashboard's 5 s tick can land on a partially flushed file.
    pyarrow raises ``ArrowInvalid`` (an ``OSError`` subclass in recent
    builds, but we catch both) on such a file; we then fall back to the
    last frame this function returned cleanly for the same path.

    Args:
        results_path: path to ``results.parquet`` (or ``None`` -- a
            sweep that just started has a manifest but no parquet yet).

    Returns:
        The per-episode DataFrame, or ``None`` when no good read has
        ever succeeded for this path (cold start, parquet absent).
    """
    if results_path is None:
        return None
    key = str(results_path)
    if not results_path.exists():
        return _RESULTS_CACHE.by_path.get(key)
    try:
        df = pd.read_parquet(results_path)
    except Exception as exc:
        # pyarrow.lib.ArrowInvalid is not importable without pinning the
        # pyarrow internal module path; it subclasses Exception (and on
        # some builds OSError). Catch broadly, log, and serve last-good.
        cached = _RESULTS_CACHE.by_path.get(key)
        logger.warning(
            "results parquet %s unreadable (%s); %s",
            results_path,
            exc,
            "serving last-good frame" if cached is not None else "no last-good frame",
        )
        return cached
    df = filter_to_v1_policies(df)
    _RESULTS_CACHE.by_path[key] = df
    return df


def clear_results_cache() -> None:
    """Drop the stale-parquet last-good cache. Used by tests for isolation."""
    _RESULTS_CACHE.by_path.clear()


@dataclass(frozen=True)
class CellEpisodeStats:
    """Episode-level success summary for one ``(policy, env)`` cell.

    Computed from the *flat list of per-episode outcomes* -- the unit is
    the episode, never the per-seed mean (DESIGN.md § Methodology,
    "pseudo-replication" guard). ``wilson_lo/hi`` is the closed-form
    Wilson 95% score interval from :func:`lerobot_bench.stats.wilson_ci`.

    ``seed_spread`` is ``max - min`` of the per-seed success rates and
    is only defined once two or more seeds have episodes on disk;
    ``None`` below that.
    """

    n_episodes: int
    n_success: int
    success_rate: float
    wilson_lo: float
    wilson_hi: float
    n_seeds: int
    seed_spread: float | None


def compute_cell_episode_stats(
    df: pd.DataFrame | None,
    *,
    policy: str,
    env: str,
) -> CellEpisodeStats | None:
    """Summarise the per-episode outcomes for one cell from the parquet.

    Slices ``df`` to ``(policy, env)`` and computes the episode-level
    success rate, the Wilson 95% CI over the flat outcome list, and the
    per-seed spread. Returns ``None`` when the cell has no rows yet.

    The Wilson CI is over **all** episodes in the cell pooled across
    seeds -- not a CI on the 5 per-seed means. ``seed_spread`` carries
    the per-seed dispersion separately so the operator still sees
    between-seed variance without it contaminating the CI.

    Args:
        df: the per-episode results frame (schema: ``policy, env, seed,
            episode_index, success, ...``).
        policy: cell policy key.
        env: cell env key.

    Returns:
        A :class:`CellEpisodeStats`, or ``None`` if the cell is absent
        from ``df`` (no episodes written yet).
    """
    if df is None or df.empty:
        return None
    needed = {"policy", "env", "seed", "success"}
    if not needed.issubset(df.columns):
        return None
    cell = df[(df["policy"] == policy) & (df["env"] == env)]
    n = len(cell)
    if n == 0:
        return None

    outcomes = cell["success"].astype(bool)
    n_success = int(outcomes.sum())
    success_rate = n_success / n
    wilson_lo, wilson_hi = wilson_ci(n_success, n)

    # Per-seed spread: max - min of each seed's success rate. Defined
    # only with >= 2 seeds; one seed has zero dispersion by construction.
    per_seed = cell.groupby("seed")["success"].mean()
    n_seeds = int(per_seed.size)
    seed_spread: float | None = None
    if n_seeds >= 2:
        seed_spread = float(per_seed.max() - per_seed.min())

    return CellEpisodeStats(
        n_episodes=n,
        n_success=n_success,
        success_rate=float(success_rate),
        wilson_lo=float(wilson_lo),
        wilson_hi=float(wilson_hi),
        n_seeds=n_seeds,
        seed_spread=seed_spread,
    )


def format_success_rate_cell(stats: CellEpisodeStats | None) -> str:
    """Render the ``success_rate_so_far`` column value.

    Small-N gate: below :data:`MIN_N_FOR_SUCCESS_RATE` episodes the
    point estimate is suppressed to :data:`STAT_PLACEHOLDER`. A success
    rate is *never* shown without its CI in :func:`format_wilson_ci_cell`
    -- the two columns gate on the same threshold so they appear and
    disappear together (council veto: "never a rate without a CI").
    """
    if stats is None or stats.n_episodes < MIN_N_FOR_SUCCESS_RATE:
        return STAT_PLACEHOLDER
    return f"{stats.success_rate:.2f} ({stats.n_success}/{stats.n_episodes})"


def format_wilson_ci_cell(stats: CellEpisodeStats | None) -> str:
    """Render the ``wilson_ci_so_far`` column as ``[lo, hi]`` to 2 dp.

    Gated identically to :func:`format_success_rate_cell`: a Wilson CI
    at n<25 spans ~0.5 of the unit interval and misleads, so it shows
    :data:`STAT_PLACEHOLDER` until the cell has enough episodes.
    """
    if stats is None or stats.n_episodes < MIN_N_FOR_SUCCESS_RATE:
        return STAT_PLACEHOLDER
    return f"[{stats.wilson_lo:.2f}, {stats.wilson_hi:.2f}]"


def format_seed_spread_cell(stats: CellEpisodeStats | None) -> str:
    """Render the ``seed_spread`` column, flagging high between-seed spread.

    ``seed_spread`` is ``max - min`` of the per-seed success rates and
    is only meaningful once >= 2 seeds have episodes on disk; below that
    it shows :data:`STAT_PLACEHOLDER`. When the spread exceeds
    :data:`SEED_SPREAD_FLAG_THRESHOLD` a ``⚠`` prefix flags that the
    5-seed budget is likely too small to pin this cell's rate.

    Returns a plain string; the Gradio Dataframe styles the warning
    via the ``⚠`` glyph rather than CSS (the component does not expose
    per-cell colour without a styler, and the glyph survives copy-paste
    into the operator's notes).
    """
    if stats is None or stats.seed_spread is None:
        return STAT_PLACEHOLDER
    spread = stats.seed_spread
    if spread > SEED_SPREAD_FLAG_THRESHOLD:
        return f"⚠ {spread:.2f}"
    return f"{spread:.2f}"


# --------------------------------------------------------------------- #
# Wilson CI half-width vs N plot                                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class HalfWidthCurve:
    """Data for the "Wilson CI half-width vs N" plot of one cell.

    ``n_values`` / ``halfwidths`` are the closed-form Wilson 95%
    half-widths at the cell's *current* p-hat for every N from 1 to
    :attr:`n_current`. The curve is what the operator's CI will look
    like as the cell accrues episodes, assuming the success rate holds.

    ``is_running`` distinguishes the live cell from the fallback case
    (no cell running -> most recently completed cell), which the plot
    title states explicitly so the operator is not misled.
    """

    policy: str
    env: str
    p_hat: float
    n_current: int
    n_values: list[int]
    halfwidths: list[float]
    is_running: bool


def select_plot_cell(
    table: pd.DataFrame,
    results_df: pd.DataFrame | None,
) -> tuple[str, str, bool] | None:
    """Pick the cell for the half-width plot: running, else most-recent-done.

    Prefers the first ``running`` cell that has *episodes on disk*
    (failures sort above running, so the first such running row is the
    live one). A cell flips to ``running`` the moment its first seed
    dispatches -- before any episode is written -- so a running cell
    with zero parquet rows is skipped and the selector falls back to
    the most recently completed cell. Returns ``(policy, env,
    is_running)`` or ``None`` when no cell has episodes yet.
    """
    if table.empty or results_df is None or results_df.empty:
        return None

    def _has_episodes(policy: str, env: str) -> bool:
        return compute_cell_episode_stats(results_df, policy=policy, env=env) is not None

    running = table[table["status"] == CELL_STATUS_RUNNING]
    for row in running.itertuples(index=False):
        if _has_episodes(str(row.policy), str(row.env)):
            return str(row.policy), str(row.env), True

    done = table[table["status"] == CELL_STATUS_DONE]
    for row in reversed(list(done.itertuples(index=False))):
        if _has_episodes(str(row.policy), str(row.env)):
            return str(row.policy), str(row.env), False

    return None


def build_halfwidth_curve(
    table: pd.DataFrame,
    results_df: pd.DataFrame | None,
) -> HalfWidthCurve | None:
    """Build the half-width-vs-N curve for the running (or last-done) cell.

    Picks the cell via :func:`select_plot_cell`, reads its current
    success rate from the parquet, and evaluates
    :func:`lerobot_bench.stats.wilson_halfwidth_at_p` at that fixed
    ``p_hat`` for every ``N`` in ``1..n_current``. Returns ``None`` when
    no cell qualifies (cold start, empty parquet) -- the caller then
    renders an empty-state message instead of a plot.
    """
    selected = select_plot_cell(table, results_df)
    if selected is None:
        return None
    policy, env, is_running = selected
    stats = compute_cell_episode_stats(results_df, policy=policy, env=env)
    if stats is None:
        return None

    n_values = list(range(1, stats.n_episodes + 1))
    halfwidths = [wilson_halfwidth_at_p(stats.success_rate, n) for n in n_values]
    return HalfWidthCurve(
        policy=policy,
        env=env,
        p_hat=stats.success_rate,
        n_current=stats.n_episodes,
        n_values=n_values,
        halfwidths=halfwidths,
        is_running=is_running,
    )


# --------------------------------------------------------------------- #
# Sweep progress aggregation                                            #
# --------------------------------------------------------------------- #


def build_progress_table(
    manifest: dict[str, Any],
    *,
    now_utc: dt.datetime | None = None,
    results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Roll the manifest's per-seed cells up to one row per ``(policy, env)``.

    Status logic (first match wins):

    * any ``failed`` seed in the cell -> ``failed``
    * every seed in ``{completed, skipped}`` -> ``done`` (or ``skipped``
      if *every* seed is ``skipped`` -- the cell never ran)
    * any seed ``started_utc`` set -> ``running``
    * otherwise -> ``queued``

    ETA is computed from completed seeds: ``mean_seed_wall_s *
    remaining_seeds``. Failed seeds are excluded from the mean (they
    finish faster than a successful seed). The estimate is rough by
    design; it's accurate enough for an operator deciding "should I
    leave this overnight or kill it".

    When ``results_df`` (the per-episode parquet) is supplied, three
    rigor columns are filled per cell from the *episode-level* outcomes:
    ``success_rate_so_far``, ``wilson_ci_so_far`` (closed-form Wilson
    95% interval), and ``seed_spread``. Cells with fewer than
    :data:`MIN_N_FOR_SUCCESS_RATE` episodes show :data:`STAT_PLACEHOLDER`
    -- a Wilson CI at small N misleads. ``results_df=None`` leaves all
    three as the placeholder (cold start, parquet not on disk yet).
    """
    cells = manifest.get("cells", [])
    if not isinstance(cells, list) or not cells:
        return pd.DataFrame({col: [] for col in PROGRESS_COLUMNS})

    now = now_utc or dt.datetime.now(dt.UTC)

    # Group rows by (policy, env). Order of insertion is preserved by
    # Python dicts -- and the manifest already orders cells by dispatch
    # order, so the resulting table reads top-to-bottom in the same
    # order the operator sees them scroll past in the log.
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in cells:
        if not isinstance(entry, dict):
            continue
        policy = entry.get("policy")
        env = entry.get("env")
        if not isinstance(policy, str) or not isinstance(env, str):
            continue
        by_cell.setdefault((policy, env), []).append(entry)

    rows: list[dict[str, Any]] = []
    for (policy, env), seeds in by_cell.items():
        stats = (
            compute_cell_episode_stats(results_df, policy=policy, env=env)
            if results_df is not None
            else None
        )
        rows.append(_summarise_cell(policy=policy, env=env, seeds=seeds, now=now, stats=stats))

    out = pd.DataFrame(rows, columns=list(PROGRESS_COLUMNS))
    # Sort: failures first (operator wants to see those), then running,
    # then queued, then done/skipped. Stable secondary sort on policy/env.
    status_order = {
        CELL_STATUS_FAILED: 0,
        CELL_STATUS_RUNNING: 1,
        CELL_STATUS_QUEUED: 2,
        CELL_STATUS_DONE: 3,
        CELL_STATUS_SKIPPED: 4,
    }
    out["_sort_key"] = out["status"].map(lambda s: status_order.get(s, 99))
    out = out.sort_values(
        ["_sort_key", "policy", "env"],
        kind="stable",
        ignore_index=True,
    ).drop(columns=["_sort_key"])
    return out


def _summarise_cell(
    *,
    policy: str,
    env: str,
    seeds: list[dict[str, Any]],
    now: dt.datetime,
    stats: CellEpisodeStats | None = None,
) -> dict[str, Any]:
    """Roll up the per-seed manifest entries for one (policy, env) cell.

    ``stats`` carries the episode-level success summary from the parquet
    (``None`` when the parquet is absent); it fills the three rigor
    columns. The manifest alone drives status / seeds / ETA.
    """
    n_total = len(seeds)
    statuses = [str(s.get("status", "")) for s in seeds]
    n_failed = sum(1 for s in statuses if s == STATUS_FAILED)
    n_completed = sum(1 for s in statuses if s == STATUS_COMPLETED)
    n_skipped = sum(1 for s in statuses if s == STATUS_SKIPPED)
    n_pending_or_running = n_total - (n_failed + n_completed + n_skipped)

    # Determine cell-level status.
    if n_failed > 0:
        cell_status = CELL_STATUS_FAILED
    elif n_completed + n_skipped == n_total and n_skipped == n_total:
        cell_status = CELL_STATUS_SKIPPED
    elif n_completed + n_skipped == n_total:
        cell_status = CELL_STATUS_DONE
    elif any(s.get("started_utc") for s in seeds):
        cell_status = CELL_STATUS_RUNNING
    else:
        cell_status = CELL_STATUS_QUEUED

    episodes_total = sum(int(s.get("n_episodes") or 0) for s in seeds)
    # Episodes-done is approximate: completed seeds contribute their
    # full n_episodes; pending/running/failed contribute 0. The exact
    # count for an in-flight seed would require reading the parquet,
    # which we deliberately defer to keep the progress tick cheap.
    episodes_done = sum(
        int(s.get("n_episodes") or 0) for s in seeds if s.get("status") == STATUS_COMPLETED
    )

    # Last update: max of started_utc / finished_utc across seeds.
    timestamps: list[str] = []
    for s in seeds:
        for key in ("finished_utc", "started_utc"):
            v = s.get(key)
            if isinstance(v, str) and v:
                timestamps.append(v)
    last_update_utc = max(timestamps) if timestamps else ""

    eta_minutes = _estimate_eta_minutes(seeds=seeds, n_remaining=n_pending_or_running, now=now)

    return {
        "policy": policy,
        "env": env,
        "status": cell_status,
        "seeds_done": n_completed + n_skipped,
        "seeds_total": n_total,
        "episodes_done": episodes_done,
        "episodes_total": episodes_total,
        "success_rate_so_far": format_success_rate_cell(stats),
        "wilson_ci_so_far": format_wilson_ci_cell(stats),
        "seed_spread": format_seed_spread_cell(stats),
        "last_update_utc": last_update_utc,
        "eta_minutes": eta_minutes,
    }


def _estimate_eta_minutes(
    *,
    seeds: list[dict[str, Any]],
    n_remaining: int,
    now: dt.datetime,
) -> float:
    """Per-seed wall-clock mean * remaining seeds, in minutes.

    Returns ``0.0`` if we have no completed seed to base the estimate
    on (the cell is either fully queued or fully done). Returns
    ``0.0`` on the "all done" case too -- there is no remaining work
    so the eta is structurally zero.
    """
    if n_remaining <= 0:
        return 0.0

    completed_durations_s: list[float] = []
    for s in seeds:
        if s.get("status") != STATUS_COMPLETED:
            continue
        started = _parse_iso(s.get("started_utc"))
        finished = _parse_iso(s.get("finished_utc"))
        if started is None or finished is None:
            continue
        delta = (finished - started).total_seconds()
        if delta > 0:
            completed_durations_s.append(delta)

    if not completed_durations_s:
        # No baseline yet -- a currently-running seed could still let us
        # estimate, but only roughly. Use elapsed-so-far on the latest
        # started seed if we have one.
        running_elapsed = _elapsed_running_seconds(seeds, now=now)
        if running_elapsed is None:
            return 0.0
        # Treat the running elapsed as a single-data-point mean; this
        # over-estimates if the cell is nearly done, but the operator
        # would rather over-estimate than under-estimate at this stage.
        return (running_elapsed * n_remaining) / 60.0

    mean_s = sum(completed_durations_s) / len(completed_durations_s)
    return (mean_s * n_remaining) / 60.0


def _elapsed_running_seconds(seeds: list[dict[str, Any]], *, now: dt.datetime) -> float | None:
    """Return seconds since the most recently started running seed, if any."""
    candidates: list[dt.datetime] = []
    for s in seeds:
        if s.get("status") != STATUS_PENDING:
            continue
        started = _parse_iso(s.get("started_utc"))
        if started is not None:
            candidates.append(started)
    if not candidates:
        return None
    most_recent = max(candidates)
    return max(0.0, (now - most_recent).total_seconds())


def _parse_iso(value: Any) -> dt.datetime | None:
    """Parse an ISO 8601 string into a tz-aware UTC datetime.

    Returns ``None`` on missing / unparseable input. The manifest writer
    emits ``isoformat(timespec="seconds")`` strings; we accept the
    trailing ``Z`` form too in case a future writer normalises to it.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.rstrip("Z")
    # ``fromisoformat`` handles ``+00:00`` but not bare ``Z``; we strip
    # the Z and treat the result as UTC. If parsing fails for any other
    # reason we silently drop the value -- the dashboard shouldn't
    # crash on a malformed manifest.
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


# --------------------------------------------------------------------- #
# Calibration                                                           #
# --------------------------------------------------------------------- #


def find_latest_calibration(results_root: Path = DEFAULT_RESULTS_DIR) -> Path | None:
    """Return the newest ``results/calibration-*.json``, or ``None``.

    Filenames are ``calibration-YYYYMMDD.json`` (mirrored exactly by
    ``scripts/calibrate.py``), so lexicographic sort == chronological
    sort. Missing directory returns ``None`` -- a fresh checkout has
    no calibration on disk yet, which the dashboard treats as a
    "run calibration first" state rather than an error.
    """
    if not results_root.exists():
        return None
    candidates = sorted(results_root.glob("calibration-*.json"))
    return candidates[-1] if candidates else None


def load_calibration_report(path: Path) -> dict[str, Any]:
    """Read a calibration JSON. Returns ``{}`` on missing/invalid file.

    Same defensive shape as :func:`load_manifest`: the dashboard
    surfaces an empty calibration table on disk read failure rather
    than crashing the whole tab.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("calibration %s unreadable: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _calibration_std_step_ms(cell: dict[str, Any]) -> str:
    """Render the ``std_step_ms`` column from a calibration cell.

    Returns the std-dev of per-step latency when the cell carries raw
    step times, else :data:`STAT_PLACEHOLDER`.

    TODO(scripts/calibrate.py): the calibration JSON currently records
    only ``mean_ms_per_step`` / ``p95_ms_per_step`` -- not the raw
    per-step list -- so ``std_step_ms`` cannot be computed and shows
    ``"—"``. Persisting a ``step_ms`` array (or a precomputed
    ``std_ms_per_step``) from the latency probe in ``calibrate.py``
    would let this column light up without a dashboard change.
    """
    raw = cell.get("step_ms")
    if isinstance(raw, list) and len(raw) >= 2:
        series = pd.Series([float(x) for x in raw], dtype="float64")
        return f"{float(series.std(ddof=1)):.2f}"
    precomputed = cell.get("std_ms_per_step")
    if isinstance(precomputed, (int, float)):
        return f"{float(precomputed):.2f}"
    return STAT_PLACEHOLDER


def _latency_skew_flag(*, mean_ms: float, p95_ms: float) -> str:
    """Flag ``⚠ skewed`` when the p95/mean latency ratio is large.

    The auto-downscope rule keys the seed/episode budget on the *mean*
    step time. When ``p95/mean > LATENCY_SKEW_FLAG_RATIO`` the mean
    badly understates the tail, so a budget derived from it can blow
    the wall-clock estimate on the slow episodes. Returns the empty
    string when latency is well-behaved (or mean is zero, which means
    the cell never ran -- nothing to flag).
    """
    if mean_ms <= 0.0:
        return ""
    if p95_ms / mean_ms > LATENCY_SKEW_FLAG_RATIO:
        return "⚠ skewed"
    return ""


def build_calibration_table(report: dict[str, Any]) -> pd.DataFrame:
    """Project a calibration report's ``cells`` list into a table.

    Empty input returns an empty frame with :data:`CALIBRATION_COLUMNS`
    so the Gradio Dataframe component doesn't choke on a missing column
    list during cold start.

    The ``reason`` column is derived locally via :func:`downscope_reason`
    -- the JSON itself does not carry a reason field; the auto-downscope
    rule's bucket is reverse-engineered from the timing thresholds.

    Rigor columns: ``std_step_ms`` is the per-step latency std-dev;
    ``n_steps`` is the probe length the mean/p95 are computed over (a
    20-step probe is a thin sample, so the operator should see it); and
    ``latency_skew`` flags ``⚠ skewed`` when ``p95/mean > 3``, meaning
    the auto-downscope rule -- which keys on the *mean* -- understates
    the tail. The calibration JSON shipped by ``scripts/calibrate.py``
    currently records only ``mean_ms_per_step`` / ``p95_ms_per_step`` /
    ``n_steps_measured``, not the raw per-step times, so ``std_step_ms``
    renders :data:`STAT_PLACEHOLDER` until that script persists them.
    """
    cells = report.get("cells", [])
    if not isinstance(cells, list) or not cells:
        return pd.DataFrame({col: [] for col in CALIBRATION_COLUMNS})

    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        recommended = cell.get("recommended") if isinstance(cell.get("recommended"), dict) else None
        seeds = int(recommended["seeds"]) if recommended and "seeds" in recommended else 0
        episodes = int(recommended["episodes"]) if recommended and "episodes" in recommended else 0
        mean_ms = float(cell.get("mean_ms_per_step") or 0.0)
        p95_ms = float(cell.get("p95_ms_per_step") or 0.0)
        rows.append(
            {
                "policy": str(cell.get("policy", "")),
                "env": str(cell.get("env", "")),
                "status": str(cell.get("status", "")),
                "mean_step_ms": mean_ms,
                "p95_step_ms": p95_ms,
                "std_step_ms": _calibration_std_step_ms(cell),
                "n_steps": int(cell.get("n_steps_measured") or 0),
                "latency_skew": _latency_skew_flag(mean_ms=mean_ms, p95_ms=p95_ms),
                "vram_peak_mb": float(cell.get("vram_peak_mb") or 0.0),
                "recommended_seeds": seeds,
                "recommended_episodes": episodes,
                "reason": downscope_reason(cell),
            }
        )

    out = pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))
    # Sort: surface the cells that got downscoped (or dropped) first.
    # Within each bucket, sort by VRAM descending so heavy policies
    # cluster together.
    status_order = {"oom": 0, "error": 1, "skipped": 2, "ok": 3}
    out["_sort_key"] = out["status"].map(lambda s: status_order.get(s, 99))
    out = out.sort_values(
        ["_sort_key", "vram_peak_mb"],
        ascending=[True, False],
        kind="stable",
        ignore_index=True,
    ).drop(columns=["_sort_key"])
    return out


def downscope_reason(cell: dict[str, Any]) -> str:
    """Human-readable reason explaining what the auto-downscope rule did.

    Mirrors the decision table in
    ``scripts/calibrate.py::auto_downscope`` -- if you change one,
    change both, or the dashboard will silently lie. The reason is
    derived from the timing fields rather than persisted in the JSON
    because the JSON shape is owned by the calibration script and
    re-emitting it with an extra field would be a schema change.
    """
    status = str(cell.get("status", ""))
    if status == "oom":
        return "OOM during calibration -- drop cell from sweep"
    if status == "error":
        err = str(cell.get("error") or "unknown")[:80]
        return f"calibration error: {err}"
    if status == "skipped":
        return "policy not runnable (pre-Day-0a)"
    if status != "ok":
        return f"unknown status {status!r}"

    mean_ms = float(cell.get("mean_ms_per_step") or 0.0)
    vram_mb = float(cell.get("vram_peak_mb") or 0.0)

    if vram_mb > VERY_HIGH_VRAM_THRESHOLD_MB:
        return f"VRAM pressure ({vram_mb:.0f} MB > {VERY_HIGH_VRAM_THRESHOLD_MB:.0f}) -- cut seeds"
    if mean_ms > VERY_SLOW_MS_PER_STEP_THRESHOLD:
        return f"very slow ({mean_ms:.0f} ms/step) -- cut seeds"
    if vram_mb > HIGH_VRAM_THRESHOLD_MB:
        return f"high VRAM ({vram_mb:.0f} MB) -- cut episodes"
    if mean_ms > SLOW_MS_PER_STEP_THRESHOLD:
        return f"slow ({mean_ms:.0f} ms/step) -- cut episodes"
    return "within budget"


# --------------------------------------------------------------------- #
# Video index                                                           #
# --------------------------------------------------------------------- #


# Map (policy, env, seed, episode) -> filesystem path. Built once per
# refresh; the Windows-mount listing is the slow part so the cache key
# is the tuple of root paths (so callers can re-scan after copying new
# videos in).
@dataclass(frozen=True)
class VideoIndex:
    """In-memory index of episode video files.

    ``by_key`` maps the canonical episode key to a single path. When
    multiple roots contain the same key (e.g. both ``results/`` and
    the Robotics-Data mount have a copy) the first-scanned wins --
    operator can flip the root order in :data:`DEFAULT_VIDEO_ROOTS` to
    pick the other.
    """

    by_key: dict[tuple[str, str, int, int], Path]
    scanned_at_ms: int
    roots: tuple[Path, ...]

    @property
    def n_videos(self) -> int:
        return len(self.by_key)


def _video_cache_token() -> int:
    """Cache-busting token bumped by :func:`clear_video_cache`.

    ``lru_cache`` is keyed on the function arguments only, so we
    include this token in the call to :func:`scan_video_index` to let
    the manual refresh button drop the cache without touching the
    private ``cache_clear`` of the lru.
    """
    return _VIDEO_CACHE_TOKEN[0]


_VIDEO_CACHE_TOKEN = [0]


def clear_video_cache() -> None:
    """Drop the cached video index.

    Wired to the manual refresh button on the rollout-preview tab.
    Bumping the token causes the next :func:`scan_video_index` call to
    miss the lru cache and re-scan the disks.
    """
    _VIDEO_CACHE_TOKEN[0] += 1
    _scan_video_index_cached.cache_clear()


@lru_cache(maxsize=8)
def _scan_video_index_cached(
    roots_tuple: tuple[str, ...],
    token: int,
) -> VideoIndex:
    """Cached worker. Argument shape is hashable so ``lru_cache`` is happy."""
    roots = tuple(Path(p) for p in roots_tuple)
    by_key: dict[tuple[str, str, int, int], Path] = {}
    for root in roots:
        if not root.exists():
            continue
        # ``rglob`` is fine on local disk; on the Windows mount it's
        # the slow path but only on first scan -- the lru caches the
        # result and subsequent UI ticks return instantly.
        for mp4 in root.rglob("*.mp4"):
            parsed = parse_video_filename(mp4.name)
            if parsed is None:
                continue
            key = parsed
            # First write wins -- skip duplicates from later roots so
            # the index is deterministic across re-scans.
            if key not in by_key:
                by_key[key] = mp4
    return VideoIndex(
        by_key=by_key,
        scanned_at_ms=int(time.time() * 1000),
        roots=roots,
    )


def scan_video_index(
    roots: Iterable[Path] = DEFAULT_VIDEO_ROOTS,
) -> VideoIndex:
    """Walk ``roots`` for ``*.mp4`` files and return a cached index.

    Filename convention is ``{policy}__{env}__seed{seed}__ep{idx:03d}.mp4``
    (matches ``scripts/run_one.py``). Files that don't parse are
    skipped silently -- the operator might have other MP4s in the
    same tree (rendered figures, etc) and we don't want to surface
    them in the rollout dropdown.

    The lru cache is keyed on ``roots`` and a module-level token; call
    :func:`clear_video_cache` to force a re-scan.
    """
    roots_tuple = tuple(str(p) for p in roots)
    return _scan_video_index_cached(roots_tuple, _video_cache_token())


def parse_video_filename(name: str) -> tuple[str, str, int, int] | None:
    """Parse ``{policy}__{env}__seed{seed}__ep{ep}.mp4`` -> 4-tuple.

    Returns ``None`` on any deviation from the schema. Both ``seed``
    and ``ep`` may be zero-padded; the regex tolerates either form
    because ``int()`` of ``"007"`` is fine.
    """
    if not name.endswith(".mp4"):
        return None
    stem = name[:-4]
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    policy, env, seed_part, ep_part = parts
    if not seed_part.startswith("seed") or not ep_part.startswith("ep"):
        return None
    try:
        seed = int(seed_part[len("seed") :])
        ep = int(ep_part[len("ep") :])
    except ValueError:
        return None
    if not policy or not env:
        return None
    return (policy, env, seed, ep)


def video_index_options(index: VideoIndex) -> dict[str, list[str]]:
    """Return per-axis sorted unique values for the dropdown cascade.

    Keys are ``"policy"``, ``"env"``, ``"seed"``, ``"episode"``. Values
    are sorted lists of strings (dropdowns are uniformly textual; the
    callers cast to int as needed in :func:`find_video_path`).
    """
    policies: set[str] = set()
    envs: set[str] = set()
    seeds: set[int] = set()
    episodes: set[int] = set()
    for policy, env, seed, ep in index.by_key:
        policies.add(policy)
        envs.add(env)
        seeds.add(seed)
        episodes.add(ep)
    return {
        "policy": sorted(policies),
        "env": sorted(envs),
        "seed": [str(s) for s in sorted(seeds)],
        "episode": [str(e) for e in sorted(episodes)],
    }


def find_video_path(
    index: VideoIndex,
    *,
    policy: str | None,
    env: str | None,
    seed: int | str | None,
    episode: int | str | None,
) -> Path | None:
    """Look up the MP4 path for one episode. Returns ``None`` on miss.

    Accepts the str values that Gradio dropdowns emit; coerces seed +
    episode to int. Missing / unparseable inputs return ``None`` so
    the UI can fall back to a "pick all four dropdowns" message.
    """
    if not policy or not env or seed in (None, "") or episode in (None, ""):
        return None
    try:
        seed_int = int(seed)  # type: ignore[arg-type]
        ep_int = int(episode)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return index.by_key.get((policy, env, seed_int, ep_int))


# --------------------------------------------------------------------- #
# Misc helpers                                                          #
# --------------------------------------------------------------------- #


def format_relative_time(iso_ts: str, *, now_utc: dt.datetime | None = None) -> str:
    """Render an ISO timestamp as a "5 min ago"-style string.

    Returns the empty string on empty / unparseable input so the
    Dataframe cell renders blank rather than the literal "unknown".
    """
    parsed = _parse_iso(iso_ts)
    if parsed is None:
        return ""
    now = now_utc or dt.datetime.now(dt.UTC)
    delta_s = max(0.0, (now - parsed).total_seconds())
    if delta_s < 60:
        return f"{int(delta_s)}s ago"
    if delta_s < 3600:
        return f"{int(delta_s / 60)}m ago"
    if delta_s < 86400:
        return f"{int(delta_s / 3600)}h ago"
    return f"{int(delta_s / 86400)}d ago"


def env_dashboard_results_dir() -> Path:
    """Allow operators to point the dashboard at a non-default results dir.

    The dashboard launcher accepts ``DASHBOARD_RESULTS_DIR`` as an
    environment variable so a user can dev against ``results/sweep-staging/``
    or wherever without editing code. Defaults to :data:`DEFAULT_RESULTS_DIR`.
    """
    override = os.environ.get("DASHBOARD_RESULTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_RESULTS_DIR


# --------------------------------------------------------------------- #
# Tab 4: Live event log                                                 #
# --------------------------------------------------------------------- #
#
# The sweep driver (``scripts/run_sweep.py``) tees its Python ``logging``
# output to ``logs/sweep-YYYYMMDD-HHMMSS.log``. The dashboard's event-log
# tab tails that file, classifies each line into a coarse bucket, and
# renders it as colour-coded HTML so the operator can watch a long sweep
# scroll past without flipping back to the terminal.
#
# Helpers in this section are gradio-free (same contract as the rest of
# the module); the Gradio wiring lives in ``app.py`` and only formats
# the classified output. Patterns are pinned local constants so the
# colour coding is deterministic across operator machines.

# Default location of sweep logs. The driver writes ``logs/sweep-*.log``
# at the repo root. ``DASHBOARD_LOGS_DIR`` overrides this for operators
# who keep their logs on a different disk (mirrors the existing
# ``DASHBOARD_RESULTS_DIR`` knob).
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"

# Tail window. 200 lines covers ~the last ~10 cells of a sweep at the
# current verbosity; long enough to see context around an error without
# blowing up the textbox the browser has to lay out every 2 s.
DEFAULT_LOG_TAIL_LINES = 200

# Maximum bytes we'll read off the end of a log file. Sweep logs are
# typically <10 MB but tracebacks from a stuck cell can balloon them;
# capping the read keeps the 2 s refresh tick from doing unbounded I/O
# if the file grows mid-session.
_LOG_TAIL_MAX_BYTES = 2 * 1024 * 1024  # 2 MB tail window

# Line classification. Order matters: more-specific patterns first
# (BREACH wins over ERROR even when both substrings appear). Each
# pattern is matched with ``re.search`` against the full line.
LogLineCategory = Literal["dispatch", "success", "error", "breach", "other"]

_LOG_BREACH_RE = re.compile(r"\bBREACH\b|\bwatchdog:\s*BREACH\b", re.IGNORECASE)
_LOG_ERROR_RE = re.compile(
    r"\bTraceback\b|\bRuntimeError\b|\bOOMError\b|\bKilled\b|\bOOM\b|"
    r"\[ERROR\]|\bERROR\s*:|\bFAILED\b|\bexit=-?\d+\)",
    re.IGNORECASE,
)
_LOG_DISPATCH_RE = re.compile(
    r"\bdispatch\b\s+\S+/\S+/seed\d+|\brun-sweep:\s*\[\d+/\d+\]\s+dispatch"
)
_LOG_SUCCESS_RE = re.compile(r"success_rate\s*=")


def classify_log_line(line: str) -> LogLineCategory:
    """Bucket a single log line into one of the five categories.

    The buckets feed both the colour coding and the header counters.
    The classifier is deliberately permissive -- a real error
    traceback that doesn't match the canonical phrases still falls
    through to ``"other"`` which the UI renders in the default colour;
    no exception is raised. Matching is done with ``re.search`` so the
    timestamp prefix on each line is ignored.

    Premortem: the watchdog's ``BREACH`` log line technically *also*
    contains the word "error" in some failure paths; we want it
    bucketed as ``"breach"`` because that's the actionable signal for
    the operator (cgroup OOM != Python error). Hence the explicit
    BREACH-first ordering.
    """
    if not line:
        return "other"
    if _LOG_BREACH_RE.search(line):
        return "breach"
    if _LOG_ERROR_RE.search(line):
        return "error"
    if _LOG_SUCCESS_RE.search(line):
        return "success"
    if _LOG_DISPATCH_RE.search(line):
        return "dispatch"
    return "other"


def discover_sweep_logs(logs_root: Path | None = None) -> list[Path]:
    """List every ``sweep-*.log`` under ``logs_root``, newest mtime first.

    Returns ``[]`` if the directory doesn't exist -- a fresh checkout
    has no sweeps yet, which the dashboard renders as an empty
    dropdown rather than an exception. Sorting by mtime (not by
    filename) handles the ``sweep-YYYYMMDD-HHMMSS.log`` schema and
    also correctly orders pre-2026 logs whose names lack the
    timestamp suffix.
    """
    root = logs_root or env_dashboard_logs_dir()
    if not root.exists():
        return []
    candidates = [p for p in root.glob("sweep-*.log") if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def tail_log_lines(path: Path, n: int = DEFAULT_LOG_TAIL_LINES) -> list[str]:
    """Return the last ``n`` lines of ``path`` as a list of strings.

    Missing file -> ``[]``. We read at most :data:`_LOG_TAIL_MAX_BYTES`
    off the tail to bound I/O on a runaway log. Trailing newline on
    the file is dropped before splitting so the last entry isn't an
    empty string.

    Premortem: ``Path.read_text().splitlines()[-n:]`` would work but
    streams the entire file through memory; the seek-from-end approach
    keeps memory bounded even when an operator points the dashboard at
    a 500 MB log from a previous failed sweep.
    """
    if not path.exists() or not path.is_file():
        return []
    try:
        size = path.stat().st_size
        read_n = min(size, _LOG_TAIL_MAX_BYTES)
        with path.open("rb") as fh:
            if size > read_n:
                fh.seek(size - read_n)
            blob = fh.read()
    except OSError as exc:
        logger.warning("could not tail %s: %s", path, exc)
        return []
    text = blob.decode("utf-8", errors="replace")
    # Drop a leading partial line when we seeked into the middle of one.
    if path.stat().st_size > _LOG_TAIL_MAX_BYTES:
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    lines = text.splitlines()
    if n <= 0:
        return lines
    return lines[-n:]


def summarize_log(lines: Iterable[str]) -> dict[str, int]:
    """Count each :func:`classify_log_line` bucket across ``lines``.

    Keys are the five literal categories plus ``"total"`` so callers
    can show ``N dispatched``/``M completed``/``X errors``/``Y breaches``
    headers without recomputing. Returns zero counts for every key on
    an empty iterable so the header always renders.
    """
    counts: dict[str, int] = {
        "dispatch": 0,
        "success": 0,
        "error": 0,
        "breach": 0,
        "other": 0,
        "total": 0,
    }
    for line in lines:
        counts[classify_log_line(line)] += 1
        counts["total"] += 1
    return counts


# Colours used by :func:`format_log_lines_html`. Picked to match the
# usual terminal palette so the screenshots in the operator's notes are
# readable in both light and dark Gradio themes.
_LOG_LINE_COLOURS: dict[str, str] = {
    "dispatch": "#3b82f6",  # blue
    "success": "#16a34a",  # green
    "error": "#dc2626",  # red
    "breach": "#dc2626",  # red (same hue as error -- still actionable)
    "other": "inherit",
}


def format_log_lines_html(
    lines: Iterable[str],
    *,
    categories: Iterable[str] | None = None,
) -> str:
    """Render classified log lines as a single colour-coded HTML string.

    ``categories`` is the set of buckets to *include* in the output --
    e.g. ``{"error", "breach"}`` to filter to just the actionable lines.
    ``None`` means "show everything". The returned string is the
    ``innerHTML`` for a ``gr.HTML`` component (no surrounding ``<pre>``
    is emitted; the caller wraps the block so the textbox keeps its
    monospace styling from the parent component).

    Each line is HTML-escaped before colour-wrapping so a Python
    traceback containing ``<class 'X'>`` doesn't accidentally inject
    markup into the DOM.
    """
    allowed: set[str] | None = set(categories) if categories is not None else None
    out: list[str] = []
    for raw_line in lines:
        cat = classify_log_line(raw_line)
        if allowed is not None and cat not in allowed:
            continue
        colour = _LOG_LINE_COLOURS.get(cat, "inherit")
        escaped = html.escape(raw_line)
        out.append(f'<span style="color:{colour}">{escaped}</span>')
    return "\n".join(out)


def env_dashboard_logs_dir() -> Path:
    """Allow operators to point the dashboard at a non-default logs dir.

    Mirrors :func:`env_dashboard_results_dir`: the ``DASHBOARD_LOGS_DIR``
    environment variable overrides the default. Defaults to the repo
    root's ``logs/`` directory.
    """
    override = os.environ.get("DASHBOARD_LOGS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_LOGS_DIR


# --------------------------------------------------------------------- #
# Explainability layer: project context, methodology, per-tab intros    #
# --------------------------------------------------------------------- #
#
# The data tabs are already plenty; the operator's complaint is that
# the numbers on screen need framing. The helpers below produce *only*
# markdown -- the gradio wiring in ``app.py`` is a thin shell that
# renders the strings into ``gr.Markdown`` / ``gr.Accordion`` blocks.
# Keeping them here means the tests (which can't import gradio) cover
# the prose surface directly.
#
# Three slots:
#
# * **persistent_header_markdown** -- the one-screen elevator pitch
#   that renders above the tab strip on every tab. Includes a live
#   ``cell N/total`` badge wired through :func:`compute_manifest_progress`.
# * **per_tab_intro_markdown** -- the "What this tab shows" block that
#   sits inside an open accordion at the top of each tab. Pulled into
#   the About tab too so the operator has a single reference of all
#   four tabs' purpose.
# * **methodology_markdown** -- the About tab body. Mirrors the seeding
#   contract, CI math, MDE bounds, and v1 scope from ``paper/main.tex``.
#
# Thresholds are interpolated from the module-level constants
# (``SLOW_MS_PER_STEP_THRESHOLD`` et al.) so the prose drifts only when
# the underlying calibration rule changes -- the per-tab-intro test
# pins this by reading the constants and asserting they appear in the
# rendered string.

# v1 scope numbers. Sourced from ``configs/sweep_full.yaml`` (6 policies
# x 6 envs minus the env_compat-pruned cells == 22 runnable, with 5
# seeds each == 110 seed-entries). Kept as constants so the header
# badge has a stable denominator even before the manifest is on disk.
#
# ``V1_POLICIES`` is imported at module top from
# ``lerobot_bench.leaderboard_filter`` (shared with the Gradio Space).
# xvla_libero is intentionally absent: deferred to v1.1 (PR #76 — two
# patched + one unresolved Hub-JSON processor bugs). The published
# parquet still carries xvla rows for reproducibility, but
# :func:`load_results_parquet` drops them before any leaderboard
# aggregate touches them.
V1_ENVS: tuple[str, ...] = (
    "pusht",
    "aloha_transfer_cube",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
V1_RUNNABLE_CELLS = 22
V1_SEEDS_PER_CELL = 5
V1_EPISODES_PER_SEED = 50
V1_TOTAL_SEED_ENTRIES = V1_RUNNABLE_CELLS * V1_SEEDS_PER_CELL  # 110

# Tags used by :func:`compute_manifest_progress` -- match the
# ``manifest["cells"][k]["status"]`` schema from ``scripts/run_sweep.py``.
ManifestStatusCounts = dict[str, int]


def compute_manifest_progress(
    manifest_path: Path | None,
) -> ManifestStatusCounts:
    """Count seed-entries in a sweep manifest by status.

    Returns a dict with keys ``{"completed", "failed", "skipped",
    "pending", "running", "total"}``. ``running`` is the subset of
    ``pending`` entries that have a non-empty ``started_utc`` (the
    sweep driver flips ``pending -> completed/failed/skipped`` rather
    than introducing a separate ``running`` state, so we recover the
    in-flight count from the started_utc field).

    Returns all-zero counts on missing/unparseable manifest -- the
    persistent header then degrades to "no in-flight sweep" rather
    than throwing on cold start.
    """
    counts: ManifestStatusCounts = {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "pending": 0,
        "running": 0,
        "total": 0,
    }
    if manifest_path is None or not manifest_path.exists():
        return counts
    data = load_manifest(manifest_path)
    cells = data.get("cells", [])
    if not isinstance(cells, list):
        return counts
    for entry in cells:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", ""))
        counts["total"] += 1
        if status == STATUS_COMPLETED:
            counts["completed"] += 1
        elif status == STATUS_FAILED:
            counts["failed"] += 1
        elif status == STATUS_SKIPPED:
            counts["skipped"] += 1
        elif status == STATUS_PENDING:
            counts["pending"] += 1
            if entry.get("started_utc"):
                counts["running"] += 1
    return counts


def _format_progress_badge(counts: ManifestStatusCounts) -> str:
    """Render the cell ``N/total`` chip for the persistent header.

    Three states:

    * No manifest discovered -> "_no sweep on disk yet_".
    * Manifest present, every seed completed/skipped -> "all 110 done".
    * Mid-flight -> ``M/N done (R running, F failed)``.
    """
    total = counts["total"]
    done = counts["completed"] + counts["skipped"]
    failed = counts["failed"]
    running = counts["running"]
    denom = total or V1_TOTAL_SEED_ENTRIES
    if total == 0:
        return "_no sweep on disk yet_"
    if done == total and failed == 0:
        return f"**all {done}/{denom} seeds complete**"
    parts = [f"**{done}/{denom}** done"]
    if running > 0:
        parts.append(f"{running} running")
    if failed > 0:
        parts.append(f"**{failed} failed**")
    return f"sweep in progress -- {', '.join(parts)}"


def persistent_header_markdown(
    *,
    results_dir: Path | None = None,
    logs_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> str:
    """Return the markdown block rendered above the tab strip.

    Renders on every tab. Designed for <10 s "what is this?" parsing:
    one-sentence project pitch, v1 scope, live progress badge, links to
    the long-form docs.

    Arguments default to the environment-resolved values (so the live
    UI matches what the operator's shell sees) but accept overrides
    for tests.
    """
    rdir = results_dir if results_dir is not None else env_dashboard_results_dir()
    ldir = logs_dir if logs_dir is not None else env_dashboard_logs_dir()
    if manifest_path is None:
        # Best-effort: the newest sweep manifest on disk. None is fine.
        runs = discover_sweep_runs(rdir)
        manifest_path = runs[0].manifest_path if runs else None
    counts = compute_manifest_progress(manifest_path)
    badge = _format_progress_badge(counts)

    policies_str = ", ".join(f"`{p}`" for p in V1_POLICIES)
    return (
        "# lerobot-bench -- local operator dashboard\n"
        "\n"
        "Public reproducible benchmark of pretrained LeRobot policies "
        f"({policies_str}) on 6 sim envs (PushT, Aloha-transfer-cube, "
        "LIBERO x4). "
        f"**v1 scope:** {V1_RUNNABLE_CELLS} cells x {V1_SEEDS_PER_CELL} "
        f"seeds x {V1_EPISODES_PER_SEED} episodes "
        f"({V1_TOTAL_SEED_ENTRIES} seed-entries, ~12 hr sweep on RTX 4060 laptop). "
        "**Pi0 family deferred to v1.1** (~30 GB cold-load RAM; see Limitations).\n"
        "\n"
        f"| Reading results from | `{rdir}` |\n"
        "|---|---|\n"
        f"| Reading logs from | `{ldir}` |\n"
        f"| v1 status | {badge} |\n"
        "\n"
        "Reference: "
        "[`paper/main.tex`](paper/main.tex) (Limitations) - "
        "[`docs/MDE_TABLE.md`](docs/MDE_TABLE.md) - "
        "[`docs/FAILURE_TAXONOMY.md`](docs/FAILURE_TAXONOMY.md) - "
        "see the **About** tab for the 60-second methodology brief."
    )


def resolved_paths_banner_markdown(
    *,
    results_dir: Path | None = None,
    logs_dir: Path | None = None,
) -> str:
    """One-line markdown showing the resolved env-var values.

    Sits directly under the persistent header so a misconfigured
    ``DASHBOARD_RESULTS_DIR`` (the previous footgun: pointing at an
    empty worktree results dir) is obvious before the operator clicks
    into a tab and stares at empty tables.
    """
    rdir = results_dir if results_dir is not None else env_dashboard_results_dir()
    ldir = logs_dir if logs_dir is not None else env_dashboard_logs_dir()
    return f"_Resolved paths_ - `DASHBOARD_RESULTS_DIR={rdir}` - `DASHBOARD_LOGS_DIR={ldir}`"


# Per-tab intros. The four entries match the four data tabs by exact
# label. Strings interpolate the calibration thresholds so the prose
# stays in sync with ``downscope_reason``. The About tab embeds the
# same strings so the operator has them in one place.
def per_tab_intro_markdown(tab: str) -> str:
    """Return the "What this tab shows" markdown for ``tab``.

    ``tab`` is one of ``{"progress", "calibration", "rollouts",
    "events"}``. Unknown keys fall back to the empty string; callers
    should treat that as a bug in the wire-up rather than swallow it.
    """
    if tab == "progress":
        return (
            "**What this tab shows.** A live grid of every "
            f"`(policy, env)` cell in the v1 sweep ({V1_RUNNABLE_CELLS} "
            f"cells x {V1_SEEDS_PER_CELL} seeds = "
            f"{V1_TOTAL_SEED_ENTRIES} seed-entries). It reads "
            "`results/<run>/sweep_manifest.json` and rolls the per-seed "
            "entries up to a per-cell status / seeds-done / ETA row.\n"
            "\n"
            "**How to read it.** `status` is the cell-level roll-up: "
            "`queued` (nothing started), `running` (some seeds in flight), "
            "`done` (every seed completed or skipped), `failed` (any seed "
            "failed). Failures sort to the top -- if you see red, that's "
            "where to look first. `eta_minutes` is the mean wall-clock of "
            "completed seeds times remaining seeds; rough by design, useful "
            'for the "leave overnight?" call.\n'
            "\n"
            "**Good shape.** Most cells should march `queued -> running -> "
            "done` in dispatch order without any `failed`. A single failed "
            "cell does not abort the sweep -- the driver continues so "
            "you can triage in the morning."
        )
    if tab == "calibration":
        return (
            "**What is calibration?** Before launching the full sweep, "
            "`scripts/calibrate.py` runs each `(policy, env)` pair for "
            "20 steps to measure per-step inference latency and peak "
            "VRAM. The `auto_downscope` rule then cuts a cell's episode "
            f"budget when it's too slow (`mean_step_ms > "
            f"{SLOW_MS_PER_STEP_THRESHOLD:.0f}`) or VRAM-pressured "
            f"(`vram_peak_mb > {HIGH_VRAM_THRESHOLD_MB:.0f}`), keeping "
            "the overnight sweep under wall-clock + GPU budget.\n"
            "\n"
            "**How to read this table.** Each row is one cell. "
            "`status=ok` means the calibration succeeded; the "
            "`reason` column states whether the cell is `within budget`, "
            "cut on episodes (`mean_step_ms > "
            f"{SLOW_MS_PER_STEP_THRESHOLD:.0f}` or `vram_peak_mb > "
            f"{HIGH_VRAM_THRESHOLD_MB:.0f}`), or cut on seeds "
            f"(`mean_step_ms > {VERY_SLOW_MS_PER_STEP_THRESHOLD:.0f}` "
            f"or `vram_peak_mb > {VERY_HIGH_VRAM_THRESHOLD_MB:.0f}`). "
            "`recommended_episodes < 50` means auto-downscope fired and "
            "trimmed the cell from the default 50.\n"
            "\n"
            "**Good shape.** At least one cell per policy passes within "
            "budget. If many cells are `oom` or have `cut seeds` in the "
            "reason column, the matrix is too aggressive for this "
            "hardware and you should re-run calibration on a smaller "
            "policy roster (or upgrade the GPU)."
        )
    if tab == "rollouts":
        return (
            "**What this tab shows.** A four-dropdown cascade "
            "(`policy`, `env`, `seed`, `episode`) into the MP4 archive "
            "of every episode rendered by the sweep. The file lookup "
            "scans `results/**/videos/` plus the optional "
            "Robotics-Data Windows mount; missing combinations show "
            '"no rollout for this combination" rather than crashing.\n'
            "\n"
            "**How to read it.** Pick a policy that interests you and "
            "skim seed 0 ep 0 of a few envs. The rollout MP4s are the "
            "single best way to understand why a policy's success rate "
            "is what it is -- a 0.42 success rate from a policy that "
            "mostly looks coherent is a different story from 0.42 with "
            "random thrashing.\n"
            "\n"
            "**Good shape.** Every cell on the **Sweep progress** tab "
            "with `status=done` should have 50 episodes available here. "
            "If a `done` cell is missing rollouts, video recording was "
            "disabled at sweep time or the Windows mount is unreachable."
        )
    if tab == "events":
        return (
            "**What this tab shows.** A tail of the active "
            "`logs/sweep-*.log` file, colour-coded by line category: "
            "blue = dispatch, green = success (per-seed `success_rate=`), "
            "red = error or watchdog BREACH. Refreshes every 2 s; the "
            "category-filter radio narrows the tail to just the lines "
            "you care about.\n"
            "\n"
            "**How to read it.** Most of the time you want `all`. "
            "Switch to `error` or `breach` if you're chasing a failure "
            "or watching for a cgroup OOM (the watchdog emits a BREACH "
            "line when the WSL2 RAM cap is exceeded). The header above "
            "the tail counts dispatched / completed / errors / breaches "
            "across the visible window.\n"
            "\n"
            "**Good shape.** Steady cadence of blue `dispatch` lines "
            "followed by green `success_rate=` summaries, no red. One "
            "red `FAILED (exit=-9)` is usually an OOM kill -- check "
            "the **Calibration** tab and consider re-running the "
            "downscope on that cell."
        )
    return ""


def methodology_markdown() -> str:
    """Return the markdown body for the About tab.

    Single source of truth for the explainability prose: pulled in by
    ``app.py`` for tab 5 and by tests for the H2-headings check. Tracks
    ``paper/main.tex`` Methods + Limitations sections at a high level
    -- the paper is the authoritative reference and this tab links out
    to it rather than restating the LaTeX in markdown.
    """
    return f"""## What is lerobot-bench?

lerobot-bench is a public reproducible benchmark of pretrained policies
from the HuggingFace LeRobot stack. The v1 roster runs
{len(V1_POLICIES)} policies ({", ".join(V1_POLICIES)}) on
{len(V1_ENVS)} sim envs (PushT, Aloha-transfer-cube, and the four
LIBERO task suites: spatial, object, goal, 10). After dropping
incompatible `(policy, env)` pairs via each policy's declared
`env_compat`, the v1 sweep covers **{V1_RUNNABLE_CELLS} runnable cells
x {V1_SEEDS_PER_CELL} seeds x {V1_EPISODES_PER_SEED} episodes per seed
= {V1_RUNNABLE_CELLS * V1_SEEDS_PER_CELL * V1_EPISODES_PER_SEED}
binary outcomes**.

The artifact has three parts: this evaluation harness, a public Hub
dataset (`thrmnn/lerobot-bench-v1`) with every per-episode
outcome plus an MP4 of every rollout, and a 4-page arXiv writeup.
This dashboard is the **operator** view: live, local, no Hub fetches.
The public-facing leaderboard lives separately under `space/`.

## Methodology in 60 seconds

- **Seeding contract.** Each `(policy, env, seed)` cell sets
  `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, and
  the env's own seed; rollout `i` then derives its episode seed as
  `seed * 1000 + i`. Re-running any cell reproduces its 50 outcomes
  to the bit on the same `lerobot==0.5.1` pin.
- **Confidence intervals.** Per-cell success is reported with two
  cross-checked 95% CIs: a closed-form Wilson score interval (Wilson,
  1927) and a percentile bootstrap (Efron, 1979) over the per-episode
  binary outcomes, 2000 resamples. The two agree to <0.005 across the
  full N=250 grid; the leaderboard shows Wilson, the forest plot uses
  bootstrap for visual symmetry.
- **Minimum detectable effect.** The Wilson half-width at p=0.5,
  N=250 is **0.0615** (inconclusive band 2.HW = 0.123). The empirical
  paired-MDE at 80% power is **0.15** at rho in {{0, 0.3}}. Any
  cross-cell delta below 0.15 is labeled inconclusive on the
  leaderboard. See [docs/MDE_TABLE.md](docs/MDE_TABLE.md) for the
  full table.
- **Paired comparisons.** Cross-policy claims on a shared env use a
  paired bootstrap on episode-level outcome differences plus a paired
  Wilcoxon signed-rank test. Headline claims cite only the planned
  comparisons; exploratory ones are tabulated for transparency.

## How a sweep works

```
[1] calibrate                 -> results/calibration-YYYYMMDD.json
        (20-step latency + VRAM probe per cell)
        |
        v
[2] auto_downscope --apply    -> configs/sweep_full.yaml overrides
        (cuts episode count for slow/VRAM-pressured cells)
        |
        v
[3] run_sweep --config sweep_full.yaml
        |   per-cell dispatch -> scripts/run_one.py
        |   writes:
        |     results/sweep-full/sweep_manifest.json   (this dashboard)
        |     results/sweep-full/results.parquet       (leaderboard)
        |     results/sweep-full/videos/*.mp4          (Rollouts tab)
        |     logs/sweep-YYYYMMDD-HHMMSS.log           (Event log tab)
        v
[4] publish_results -> HF Hub dataset
        (the public Space then renders that dataset)
```

## v1 scope and known limits

- **Simulation only.** Every number is sim. lerobot-bench is a
  screening tool, not a substitute for hardware evaluation.
- **Single-hardware bias on wall-clock.** Latency is measured on a
  single RTX 4060 laptop (8 GB VRAM). The policy *ranking* is
  portable; absolute ms/step differs by GPU.
- **Pi0 family deferred to v1.1.** The pi0 / pi0fast / pi0.5
  checkpoints (PaliGemma-3B backbone) require ~30 GB host RAM during
  cold load under HuggingFace Transformers' default `from_pretrained`
  path. The 32 GB WSL2 laptop used for v1 cannot fit that with other
  tenants running. v1.1 plans to onboard them with a quantized
  checkpoint or `accelerate`'s `device_map="auto"` streaming load.
- **Sparse matrix.** The `(policy, env)` grid is sparse by design --
  a cell is N/A only when no public Hub checkpoint exists. The
  shared-env set for paired comparisons is {{LIBERO-spatial, -object,
  -goal, -10}} for the v1 VLAs.
- **Episode budget.** N=250 (5 seeds x 50 episodes) is the GPU
  budget, not a power calculation. The MDE is documented above; the
  leaderboard labels any delta below the MDE as inconclusive.

## Reading this dashboard

**Sweep progress.** {per_tab_intro_markdown("progress")}

**Calibration inspector.** {per_tab_intro_markdown("calibration")}

**Rollout preview.** {per_tab_intro_markdown("rollouts")}

**Live event log.** {per_tab_intro_markdown("events")}
"""


# Column glossaries -- short one-line "X means Y" notes rendered below
# each ``gr.Dataframe`` since Gradio's native headers don't support
# tooltips. Kept here so the prose stays alongside the column
# constants above.
def column_glossary_markdown(tab: str) -> str:
    """Markdown of column glossaries for the named tab.

    ``tab`` is one of ``{"progress", "calibration"}``. Other tabs
    don't ship dataframes; callers asking for them get ``""``.
    """
    if tab == "progress":
        return (
            "**Column glossary:** "
            "`policy` / `env` = the cell key - "
            "`status` = `queued`/`running`/`done`/`failed`/`skipped` cell roll-up - "
            f"`seeds_done/seeds_total` = completed seeds out of {V1_SEEDS_PER_CELL} - "
            "`episodes_done/episodes_total` = approximate; counted from completed seeds "
            f"(50 each by default) - "
            "`success_rate_so_far` = episode-level success over the parquet rows on disk "
            f"(`—` until n >= {MIN_N_FOR_SUCCESS_RATE}; a Wilson CI at smaller N misleads) - "
            "`wilson_ci_so_far` = closed-form Wilson 95% interval `[lo, hi]` over the flat "
            "per-episode outcomes (never a rate without its CI) - "
            "`seed_spread` = `max - min` of the per-seed success rates once >= 2 seeds have "
            f"episodes; **`⚠` flags `seed_spread > {SEED_SPREAD_FLAG_THRESHOLD:.1f}` -> "
            "5-seed budget likely insufficient for this cell** - "
            "`last_update_utc` = most recent started_utc / finished_utc on any seed in the cell - "
            "`eta_minutes` = mean wall-clock of completed seeds x remaining seeds (rough)."
        )
    if tab == "calibration":
        return (
            "**Column glossary:** "
            "`status` = `ok`/`oom`/`error`/`skipped` from the calibration JSON - "
            "`mean_step_ms` / `p95_step_ms` = per-step inference latency over the probe - "
            "`std_step_ms` = per-step latency std-dev (`—` until `scripts/calibrate.py` "
            "persists raw step times) - "
            "`n_steps` = probe length the mean/p95 are computed over (a 20-step probe is a "
            "thin sample) - "
            f"`latency_skew` = **`⚠ skewed` when `p95/mean > {LATENCY_SKEW_FLAG_RATIO:.0f}` -> "
            "auto-downscope keys on the mean and may be fragile** - "
            "`vram_peak_mb` = peak GPU memory observed during the probe - "
            "`recommended_seeds` / `recommended_episodes` = output of `auto_downscope` "
            f"(defaults {V1_SEEDS_PER_CELL} / {V1_EPISODES_PER_SEED}; trimmed for slow / VRAM-pressured cells) - "
            "`reason` = human-readable explanation derived from the timing thresholds "
            f"(slow >{SLOW_MS_PER_STEP_THRESHOLD:.0f} ms/step, "
            f"very slow >{VERY_SLOW_MS_PER_STEP_THRESHOLD:.0f}; "
            f"high VRAM >{HIGH_VRAM_THRESHOLD_MB:.0f} MB, "
            f"very high >{VERY_HIGH_VRAM_THRESHOLD_MB:.0f})."
        )
    return ""


# --------------------------------------------------------------------- #
# Scientific-context panels: Policies + Envs tabs (council audit P0)     #
# --------------------------------------------------------------------- #
#
# A reviewer landing on the dashboard cannot tell what `xvla_libero` is
# or what `libero_10` tests. These helpers render a markdown card per
# policy / per env so the numbers on the other tabs have scientific
# context. They are gradio-free (same contract as the rest of this
# module); ``app.py`` renders the returned strings into ``gr.Markdown``.
#
# Policy cards pull architecture + training-data prose out of
# ``docs/MODEL_CARDS.md`` (parsed below); env cards pull task prose from
# the hand-authored ``_ENV_CONTEXT`` constant dict (the env YAML is read
# per-cell by a live sweep, so we deliberately do *not* extend its
# schema -- the constant-dict route keeps the running sweep untouched).

DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"
DEFAULT_MODEL_CARDS = REPO_ROOT / "docs" / "MODEL_CARDS.md"

# Delta-chip colour thresholds for paper-reported-vs-ours. |Δ| below
# DELTA_GREEN_MAX is a clean reproduction; between green and yellow max
# is a notable-but-plausible gap; above is a red flag worth a caveat.
DELTA_GREEN_MAX = 0.05
DELTA_YELLOW_MAX = 0.15

# Hand-authored scientific context for each v1 env. Sourced from the
# env's original paper (PushT: Chi et al. 2023; Aloha: Zhao et al. 2023;
# LIBERO suites: Liu et al. 2023) and cross-checked against
# ``paper/main.tex`` Related Work. Kept here rather than in
# ``configs/envs.yaml`` so the running sweep -- which reads the env YAML
# per cell at dispatch -- is unaffected by this descriptive layer.
_ENV_CONTEXT: dict[str, dict[str, str]] = {
    "pusht": {
        "task": (
            "Push a T-shaped block across a planar table into a fixed "
            "target pose using a circular end-effector. Success is a "
            "coverage threshold: the block must overlap the goal region."
        ),
        "obs": "96x96 RGB agent-view image + 2-D end-effector (x, y) position.",
        "discriminates": (
            "Fine-grained contact-rich planar control and recovery from "
            "the block slipping off the pusher."
        ),
        "source": (
            'Chi et al. 2023, "Diffusion Policy: Visuomotor Policy '
            'Learning via Action Diffusion" (RSS 2023); gym-pusht env.'
        ),
    },
    "aloha_transfer_cube": {
        "task": (
            "Bimanual cube transfer: one arm picks a cube from the table "
            "and hands it to the second arm, which must receive and hold "
            "it. Success requires a completed mid-air transfer."
        ),
        "obs": "480x640 top-camera RGB + 14-D joint state (two 7-DoF arms).",
        "discriminates": (
            "Bimanual coordination and gripper timing -- the hand-off "
            "fails on early/late release or a missed grasp."
        ),
        "source": (
            'Zhao et al. 2023, "Learning Fine-Grained Bimanual '
            'Manipulation with Low-Cost Hardware" (RSS 2023); gym-aloha '
            "AlohaTransferCube env."
        ),
    },
    "libero_spatial": {
        "task": (
            "Pick-and-place where the target object's identity is fixed "
            "but its spatial layout (and that of distractors) varies "
            "across episodes; the instruction names a spatial relation."
        ),
        "obs": ("Agent-view + wrist-camera RGB + proprioceptive state (Franka Panda 7-DoF arm)."),
        "discriminates": (
            'Spatial grounding -- resolving "the bowl next to the '
            'plate"-style instructions under layout shift.'
        ),
        "source": (
            'Liu et al. 2023, "LIBERO: Benchmarking Knowledge Transfer '
            'for Lifelong Robot Learning" (NeurIPS 2023); LIBERO-Spatial '
            "suite, task 0."
        ),
    },
    "libero_object": {
        "task": (
            "Pick-and-place where the spatial layout is fixed but the "
            "target object identity varies across episodes; the "
            "instruction names which object to manipulate."
        ),
        "obs": ("Agent-view + wrist-camera RGB + proprioceptive state (Franka Panda 7-DoF arm)."),
        "discriminates": (
            "Object grounding -- visually distinguishing and selecting "
            "the named object among distractors."
        ),
        "source": ('Liu et al. 2023, "LIBERO" (NeurIPS 2023); LIBERO-Object suite, task 0.'),
    },
    "libero_goal": {
        "task": (
            "Manipulation where the object set and layout are fixed but "
            "the goal varies across episodes; the instruction specifies "
            "which of several goals to achieve."
        ),
        "obs": ("Agent-view + wrist-camera RGB + proprioceptive state (Franka Panda 7-DoF arm)."),
        "discriminates": (
            "Goal grounding -- conditioning behaviour on the instruction "
            "rather than memorising one fixed outcome."
        ),
        "source": ('Liu et al. 2023, "LIBERO" (NeurIPS 2023); LIBERO-Goal suite, task 0.'),
    },
    "libero_10": {
        "task": (
            "Long-horizon compositional tasks (the LIBERO-Long / "
            "LIBERO-10 suite): multi-stage instructions chaining several "
            "pick-place-and-manipulate subgoals into one episode."
        ),
        "obs": ("Agent-view + wrist-camera RGB + proprioceptive state (Franka Panda 7-DoF arm)."),
        "discriminates": (
            "Long-horizon credit assignment and recovery -- one dropped "
            "subgoal fails the whole episode."
        ),
        "source": (
            'Liu et al. 2023, "LIBERO" (NeurIPS 2023); LIBERO-Long ("LIBERO-10") suite, task 0.'
        ),
    },
}


@lru_cache(maxsize=1)
def load_policy_registry(path: str | None = None) -> PolicyRegistry:
    """Load the v1 :class:`PolicyRegistry` from ``configs/policies.yaml``.

    Cached: the registry is pure data and the dashboard process is
    single-tenant, so one load per session is plenty. ``path`` is the
    cache key; ``None`` resolves to the shipped config.
    """
    return PolicyRegistry.from_yaml(path or DEFAULT_POLICIES_YAML)


@lru_cache(maxsize=1)
def load_env_registry(path: str | None = None) -> EnvRegistry:
    """Load the v1 :class:`EnvRegistry` from ``configs/envs.yaml``.

    Cached for the same reason as :func:`load_policy_registry`.
    """
    return EnvRegistry.from_yaml(path or DEFAULT_ENVS_YAML)


def policy_dropdown_choices(registry: PolicyRegistry | None = None) -> list[str]:
    """Return the sorted policy names for the Policies-tab dropdown."""
    reg = registry if registry is not None else load_policy_registry()
    return reg.names()


def env_dropdown_choices(registry: EnvRegistry | None = None) -> list[str]:
    """Return the sorted env names for the Envs-tab dropdown."""
    reg = registry if registry is not None else load_env_registry()
    return reg.names()


@lru_cache(maxsize=1)
def _parse_model_cards(path_str: str) -> dict[str, str]:
    """Parse ``docs/MODEL_CARDS.md`` into ``{normalized-name: section-body}``.

    The file is one ``## Heading`` section per policy. We key the
    section bodies on a normalized form of the heading (lower-cased,
    parenthetical qualifiers stripped) so a YAML policy name can be
    matched against the prose heading despite formatting differences
    (``smolvla_libero`` vs ``## SmolVLA (libero finetune)``).

    Returns ``{}`` if the file is missing -- the policy card then
    renders the architecture / training-data lines as ``"—"``.
    """
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("model cards %s unreadable: %s", path, exc)
        return {}
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = _normalize_card_heading(line[3:])
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def _normalize_card_heading(heading: str) -> str:
    """Normalize a model-card H2 heading to a stable lookup key.

    Drops a trailing parenthetical (``(libero finetune)``), lower-cases,
    and collapses whitespace -- so ``Pi0.5 (libero finetune v0.4.4)``
    and a YAML name like ``pi05`` can be brought together by the
    caller's own normalization of the YAML name.
    """
    base = re.sub(r"\(.*?\)", "", heading)
    return re.sub(r"\s+", " ", base).strip().lower()


def _model_card_section(policy_name: str) -> str:
    """Best-effort lookup of a policy's ``docs/MODEL_CARDS.md`` section body.

    The YAML policy names (``smolvla_libero``, ``pi05_libero_finetuned_v044``)
    do not match the prose headings (``SmolVLA``, ``Pi0.5``) one-to-one,
    so we try a small ladder of normalizations. Returns ``""`` when no
    section matches -- the card then shows ``"—"`` for the prose lines.
    """
    sections = _parse_model_cards(str(DEFAULT_MODEL_CARDS))
    if not sections:
        return ""
    # Candidate keys, most-specific first.
    raw = policy_name.lower()
    candidates = [raw, raw.replace("_", " ")]
    # Strip common suffixes the YAML carries but the prose heading omits.
    stripped = re.sub(r"[_\s]*(libero[_\s]*)?finetuned[_\s]*v[\d.]+$", "", raw)
    stripped = re.sub(r"[_\s]*libero$", "", stripped)
    candidates.append(stripped.replace("_", " ").strip())
    # ``pi05`` -> ``pi0.5``, ``pi0fast`` -> ``pi0fast`` (heading is "Pi0Fast").
    candidates.append(stripped.replace("pi05", "pi0.5").replace("_", " ").strip())
    candidates.append("no-op" if raw == "no_op" else raw)
    for key in candidates:
        if key and key in sections:
            return sections[key]
    return ""


def _extract_card_field(section_body: str, field_label: str) -> str:
    """Pull a single ``- **Label**: value`` bullet from a model-card section.

    Returns ``""`` when the bullet is absent so the caller can render
    the :data:`STAT_PLACEHOLDER`.
    """
    for line in section_body.splitlines():
        m = re.match(rf"\s*-\s*\*\*{re.escape(field_label)}\*\*\s*:\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return ""


def policy_architecture_line(policy_name: str) -> str:
    """One-liner: architecture family + training data, from the model card.

    Synthesised from the model-card section's ``Source paper`` /
    ``Source`` / ``Envs supported`` bullets. Returns :data:`STAT_PLACEHOLDER`
    when the card has no usable prose (keeps the panel honest rather
    than inventing an architecture string).
    """
    body = _model_card_section(policy_name)
    if not body:
        return STAT_PLACEHOLDER
    paper = _extract_card_field(body, "Source paper")
    source = _extract_card_field(body, "Source")
    envs = _extract_card_field(body, "Envs supported")
    purpose = _extract_card_field(body, "Purpose")
    parts: list[str] = []
    if paper:
        parts.append(f"Architecture/source: {paper}")
    elif source:
        parts.append(f"Source: {source}")
    elif purpose:
        parts.append(purpose)
    if envs:
        parts.append(f"Trained for: {envs}")
    return " · ".join(parts) if parts else STAT_PLACEHOLDER


def delta_chip(paper: float | None, ours: float | None) -> tuple[str, str]:
    """Return ``(delta_text, chip_emoji)`` for a paper-vs-ours comparison.

    * Either side missing -> ``("—", "")`` (no chip; nothing to compare).
    * Both present -> signed delta plus a colour chip keyed on ``|Δ|``:
      green when ``|Δ| < DELTA_GREEN_MAX``, yellow up to
      ``DELTA_YELLOW_MAX``, red above. The chip is a coloured circle
      emoji so it survives copy-paste into the operator's notes (the
      same rationale as the ``⚠`` glyphs elsewhere in this module).
    """
    if paper is None or ours is None:
        return STAT_PLACEHOLDER, ""
    delta = ours - paper
    abs_d = abs(delta)
    if abs_d < DELTA_GREEN_MAX:
        chip = "🟢"
    elif abs_d <= DELTA_YELLOW_MAX:
        chip = "🟡"
    else:
        chip = "🔴"
    return f"{delta:+.3f}", chip


def _our_success_rate(
    results_df: pd.DataFrame | None,
    *,
    policy: str,
    env: str,
) -> float | None:
    """Our re-run success rate for one cell, or ``None`` when not on disk.

    Thin wrapper over :func:`compute_cell_episode_stats` so the policy
    card and the test suite share one definition of "our number".
    """
    stats = compute_cell_episode_stats(results_df, policy=policy, env=env)
    return stats.success_rate if stats is not None else None


def build_policy_card_markdown(
    policy_name: str,
    *,
    registry: PolicyRegistry | None = None,
    results_df: pd.DataFrame | None = None,
) -> str:
    """Render the markdown science card for one policy.

    Sections: identity (repo @ short-SHA, license, baseline flag),
    architecture/training prose from ``docs/MODEL_CARDS.md``, a
    paper-reported-vs-ours table with a colour-chipped delta per env,
    the ``paper_reported_notes`` caveat, and the policy ``notes``.

    ``results_df`` is the per-episode results parquet (loaded via
    :func:`load_results_parquet`); when a cell has no rows yet the
    "ours" column shows ``(pending)``. ``None`` -> every cell pending.
    """
    reg = registry if registry is not None else load_policy_registry()
    try:
        spec: PolicySpec = reg.get(policy_name)
    except KeyError:
        return f"_Unknown policy `{policy_name}`._"

    short_sha = spec.revision_sha[:12] if spec.revision_sha else STAT_PLACEHOLDER
    repo = spec.repo_id or STAT_PLACEHOLDER
    license_str = spec.license or STAT_PLACEHOLDER
    kind = "baseline (no weights)" if spec.is_baseline else "pretrained checkpoint"
    fp = spec.fp_precision or STAT_PLACEHOLDER

    lines: list[str] = [
        f"## {spec.name}",
        "",
        f"| Repo | `{repo}` |",
        "|---|---|",
        f"| Revision | `{short_sha}` |",
        f"| License | {license_str} |",
        f"| Kind | {kind} |",
        f"| Inference precision | {fp} |",
        "",
        "**Architecture & training data.** "
        + (policy_architecture_line(spec.name) or STAT_PLACEHOLDER),
        "",
    ]

    # Paper-reported vs our re-run, one row per supported env.
    lines.append("### Paper-reported vs. our re-run")
    lines.append("")
    paper_map = spec.paper_reported_success or {}
    if not paper_map:
        lines.append(
            "_No published reference for this policy "
            + ("(baseline)." if spec.is_baseline else "on its supported envs.")
            + "_"
        )
    else:
        lines.append("| Env | Paper | Ours | Δ (ours−paper) | |")
        lines.append("|---|---|---|---|---|")
        for env in spec.env_compat:
            paper_val = paper_map.get(env)
            ours_val = _our_success_rate(results_df, policy=spec.name, env=env)
            paper_cell = f"{paper_val:.3f}" if paper_val is not None else STAT_PLACEHOLDER
            if ours_val is None:
                ours_cell = "(pending)"
                delta_cell, chip = STAT_PLACEHOLDER, ""
            else:
                ours_cell = f"{ours_val:.3f}"
                delta_cell, chip = delta_chip(paper_val, ours_val)
            lines.append(f"| `{env}` | {paper_cell} | {ours_cell} | {delta_cell} | {chip} |")
        lines.append("")
        lines.append(
            "_Chip: 🟢 |Δ| < "
            f"{DELTA_GREEN_MAX:.2f} · 🟡 {DELTA_GREEN_MAX:.2f}–{DELTA_YELLOW_MAX:.2f} "
            f"· 🔴 > {DELTA_YELLOW_MAX:.2f}. `(pending)` = cell not yet in the "
            "results parquet._"
        )

    if spec.paper_reported_notes:
        lines.append("")
        lines.append(f"> **Reference caveat.** {spec.paper_reported_notes}")

    if spec.notes:
        lines.append("")
        lines.append(f"**Notes.** {spec.notes}")

    return "\n".join(lines)


def build_env_card_markdown(
    env_name: str,
    *,
    registry: EnvRegistry | None = None,
) -> str:
    """Render the markdown science card for one env.

    Combines the runtime fields from :class:`EnvSpec` (``max_steps``,
    ``success_threshold``, family, construction path) with the
    hand-authored task / observation / discrimination prose from
    :data:`_ENV_CONTEXT`. Missing prose degrades to ``"—"`` rather
    than failing -- the runtime fields always render.
    """
    reg = registry if registry is not None else load_env_registry()
    try:
        spec: EnvSpec = reg.get(env_name)
    except KeyError:
        return f"_Unknown env `{env_name}`._"

    ctx = _ENV_CONTEXT.get(env_name, {})
    task = ctx.get("task", STAT_PLACEHOLDER)
    obs = ctx.get("obs", STAT_PLACEHOLDER)
    discriminates = ctx.get("discriminates", STAT_PLACEHOLDER)
    source = ctx.get("source", STAT_PLACEHOLDER)
    construction = "factory" if spec.uses_factory else "gym"

    lines: list[str] = [
        f"## {spec.name}",
        "",
        f"**Task.** {task}",
        "",
        f"| Family | {spec.family} |",
        "|---|---|",
        f"| Max steps per episode | {spec.max_steps} |",
        f"| Success threshold | {spec.success_threshold:g} |",
        f"| Construction path | {construction} |",
        "",
        f"**Observation.** {obs}",
        "",
        f"**What it discriminates.** {discriminates}",
        "",
        f"**Source.** {source}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Representative-rollout episode selection (Rollout-preview tab)         #
# --------------------------------------------------------------------- #
#
# The rollout tab previously defaulted episode selection to first-
# alphabetical, which over-samples the episode with index 0 -- not a
# representative view of the cell. The selector below picks the
# *representative* episode: the one whose success matches the cell's
# modal outcome and whose step count is closest to the cell median.
# A second mode picks the *best* episode (cherry-pick warning attached
# in the UI). All three modes fall back to first-on-disk when the cell
# has no parquet rows.

EPISODE_SELECT_REPRESENTATIVE = "representative"
EPISODE_SELECT_BEST = "best"
EPISODE_SELECT_FIRST = "first"
EPISODE_SELECT_MODES: tuple[str, ...] = (
    EPISODE_SELECT_REPRESENTATIVE,
    EPISODE_SELECT_BEST,
    EPISODE_SELECT_FIRST,
)


def select_representative_episode(
    results_df: pd.DataFrame | None,
    *,
    policy: str,
    env: str,
    seed: int | str,
    mode: str = EPISODE_SELECT_REPRESENTATIVE,
    available_episodes: Iterable[int] | None = None,
) -> int | None:
    """Pick an episode index for one ``(policy, env, seed)`` cell.

    Modes:

    * ``representative`` -- among the cell's episodes, the one whose
      ``success`` equals the cell's modal outcome and whose ``n_steps``
      is closest to the cell median. Ties on distance break to the
      lowest episode index for determinism.
    * ``best`` -- the highest-return episode; with binary success and
      no return column we proxy "best" as a *successful* episode with
      the shortest ``n_steps`` (a fast success), falling back to the
      shortest episode overall when none succeed.
    * ``first`` -- the lowest episode index on disk (legacy default).

    Returns ``None`` when the parquet has no rows for the cell *and*
    ``available_episodes`` is empty/None -- the caller then keeps the
    dropdown's first-on-disk value. When parquet rows are missing but
    ``available_episodes`` is supplied, returns ``min(available)`` so
    every mode degrades to first-on-disk rather than blanking the view.
    """
    fallback: int | None = None
    if available_episodes is not None:
        eps = sorted(int(e) for e in available_episodes)
        fallback = eps[0] if eps else None

    if results_df is None or results_df.empty:
        return fallback
    needed = {"policy", "env", "seed", "episode_index", "success"}
    if not needed.issubset(results_df.columns):
        return fallback

    cell = results_df[
        (results_df["policy"] == policy)
        & (results_df["env"] == env)
        & (results_df["seed"].astype(str) == str(seed))
    ]
    if cell.empty:
        return fallback

    if mode == EPISODE_SELECT_FIRST:
        return int(cell["episode_index"].min())

    has_steps = "n_steps" in cell.columns
    cell = cell.sort_values("episode_index", kind="stable")

    if mode == EPISODE_SELECT_BEST:
        successes = cell[cell["success"].astype(bool)]
        pool = successes if not successes.empty else cell
        if has_steps:
            best_idx = pool["n_steps"].astype(float).idxmin()
            return int(pool.loc[best_idx, "episode_index"])
        return int(pool["episode_index"].min())

    # Representative: modal outcome, then closest to median step count.
    outcomes = cell["success"].astype(bool)
    modal_outcome = bool(outcomes.mean() >= 0.5)
    modal = cell[outcomes == modal_outcome]
    if modal.empty:
        modal = cell
    if has_steps:
        median_steps = float(cell["n_steps"].astype(float).median())
        dist = (modal["n_steps"].astype(float) - median_steps).abs()
        # Stable sort already applied; idxmin breaks ties to first row.
        chosen = modal.loc[dist.idxmin()]
        return int(chosen["episode_index"])
    return int(modal["episode_index"].min())


# --------------------------------------------------------------------- #
# Accordion open-state policy                                           #
# --------------------------------------------------------------------- #
#
# Council audit item 7: the "What this tab shows" accordions should be
# open on the first visit of a session (so a reviewer sees the framing
# without hunting for it) but collapsed on subsequent tab switches (so
# returning operators don't get wall-of-text fatigue every time they
# flip tabs). The helper below is a pure function that decides the
# next open-state from a visit counter -- ``app.py`` stores the
# counter in ``gr.State`` and increments it on every tab render.


def should_accordion_be_open(visit_count: int) -> bool:
    """Return True iff this is the operator's first visit in the session.

    ``visit_count`` is the count *before* this visit -- i.e. ``0`` on
    the first render of the session. Callers should increment the
    counter as part of the same callback so the next visit sees ``1``
    and the accordion stays collapsed.

    The function is intentionally trivial; lifting it to its own
    helper means the test suite can pin the policy without spinning up
    Gradio (the test job has no Gradio install).
    """
    return visit_count <= 0


# --------------------------------------------------------------------- #
# Sweep-progress row click -> Rollout-preview drill-down                #
# --------------------------------------------------------------------- #
#
# Council audit item 8: clicking a row on the Sweep progress table
# should jump to the Rollout-preview tab pre-populated with that
# cell's ``(policy, env)``. The helper below extracts the click
# target from the Gradio ``SelectData`` payload + table snapshot,
# tolerating empty / out-of-range / non-actionable rows. Drill-down
# defaults: seed=0 (every cell has seed 0; we don't try to pick a
# "representative" seed without rollout data), episode=None (let the
# rollout tab's own default fire).
#
# The handler is parameterised on the column list rather than the
# index of "policy"/"env" so Agent A's pending column additions
# don't silently shift the lookup.


@dataclass(frozen=True)
class RowClickTarget:
    """Result of resolving a Sweep-progress row-click event.

    * ``actionable``: True only when the row has a policy + env and is
      not in a status (``skipped``) that has no rollouts. Callers use
      this to decide whether to switch tabs or surface a warning.
    * ``policy`` / ``env``: extracted from the row; empty strings on
      non-actionable rows.
    * ``seed``: drill-down default; currently always ``"0"`` so the
      rollout-preview dropdowns repopulate to a known-good combination.
    * ``warning``: empty on success, one-line markdown on failure.
    """

    actionable: bool
    policy: str
    env: str
    seed: str
    warning: str


# Cell statuses that should not navigate to the rollout-preview tab on
# row-click. ``skipped`` cells have no rollouts on disk; ``queued`` cells
# may have none yet but the operator may still want to peek, so we let
# them through. Failed cells have partial rollouts -- worth a look.
_NON_ACTIONABLE_ROW_STATUSES: frozenset[str] = frozenset({CELL_STATUS_SKIPPED})


def extract_row_click_target(
    table: pd.DataFrame,
    row_index: int | None,
    *,
    columns: Iterable[str] | None = None,
) -> RowClickTarget:
    """Resolve a Sweep-progress row-click into a drill-down target.

    ``row_index`` is the row clicked (from ``gr.SelectData.index[0]``);
    ``columns`` is the list of column names on the live table -- pass
    the current column list so Agent A's pending column additions
    don't break the lookup. Defaults to :data:`PROGRESS_COLUMNS`.

    Returns a :class:`RowClickTarget` with ``actionable=False`` and a
    populated ``warning`` on every failure mode: empty table, out-of-
    range index, missing policy / env column, or a row whose status is
    in :data:`_NON_ACTIONABLE_ROW_STATUSES`.
    """
    col_list = list(columns) if columns is not None else list(PROGRESS_COLUMNS)

    if table is None or len(table) == 0:
        return RowClickTarget(
            actionable=False,
            policy="",
            env="",
            seed="0",
            warning="_No rows in the progress table -- nothing to drill into yet._",
        )

    if row_index is None or row_index < 0 or row_index >= len(table):
        return RowClickTarget(
            actionable=False,
            policy="",
            env="",
            seed="0",
            warning=f"_Row index `{row_index}` is out of range for this table._",
        )

    # Tolerate either the live DataFrame (columns attribute matches
    # ``col_list``) or a column-renamed snapshot. We prefer the column
    # *name* lookup, falling back to positional access on a numpy-style
    # row when the column isn't present.
    row = table.iloc[row_index]
    policy = _row_value(row, "policy", col_list)
    env = _row_value(row, "env", col_list)
    status = _row_value(row, "status", col_list)

    if not policy or not env:
        return RowClickTarget(
            actionable=False,
            policy="",
            env="",
            seed="0",
            warning=(
                "_That row is missing `policy` or `env` -- the manifest may "
                "be mid-write. Try again in a few seconds._"
            ),
        )

    if status in _NON_ACTIONABLE_ROW_STATUSES:
        return RowClickTarget(
            actionable=False,
            policy=policy,
            env=env,
            seed="0",
            warning=(
                f"_Cell `{policy}` / `{env}` is `{status}` -- no rollouts "
                "to preview. Pick a `done` / `running` / `failed` row instead._"
            ),
        )

    return RowClickTarget(
        actionable=True,
        policy=policy,
        env=env,
        seed="0",
        warning="",
    )


def _row_value(row: Any, key: str, col_list: list[str]) -> str:
    """Pull a string value out of a Series-like row by column name.

    Falls back to positional lookup via ``col_list`` for the case
    where the caller passed a raw numpy row (from
    ``df.values[row_index]``) rather than a Series. Returns the empty
    string on any KeyError / IndexError so the call site can branch
    on truthiness.
    """
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        try:
            idx = col_list.index(key)
            value = row[idx]
        except (ValueError, IndexError, TypeError):
            return ""
    if value is None:
        return ""
    return str(value)


# --------------------------------------------------------------------- #
# Stale-data resilience: last-good cache + warning escalation           #
# --------------------------------------------------------------------- #
#
# Council audit item 9: when ``load_manifest`` / ``find_latest_calibration``
# / parquet reads fail (mid-write, briefly missing, ArrowInvalid), the
# progress / calibration tables currently repaint *empty*, which looks
# like the sweep died. Replace that with a "last-good" cache: on
# success, the cache is updated and the result rendered; on failure,
# the cached value is rendered with a small warning markdown, and
# after three consecutive failures the warning escalates to a louder
# "filesystem error" message.
#
# Test surface: the cache + warning policy are pure (no Gradio). Tests
# in ``tests/test_dashboard.py`` simulate the failure modes by passing
# a loader callable that raises -- no real filesystem manipulation.


# How many consecutive failures before the warning escalates.
STALE_DATA_ESCALATE_AFTER = 3


@dataclass
class StaleDataCache:
    """Last-good-value cache with consecutive-failure tracking.

    One instance per panel that needs the protection (progress table,
    calibration table). The cache is module-level state in ``app.py``
    but isolated here so tests can construct a fresh instance per
    case and pin the escalation rule without leaking state across
    tests.

    The cache stores the last successfully-loaded value plus its UTC
    timestamp. On a failed load, callers get back the cached value
    plus a warning markdown describing the failure -- after
    :data:`STALE_DATA_ESCALATE_AFTER` consecutive failures the
    warning gets louder so the operator notices it's not a one-off.
    """

    last_value: Any = None
    last_success_utc: str = ""
    consecutive_failures: int = 0

    def record_success(self, value: Any, *, now_utc: dt.datetime | None = None) -> None:
        """Update the cache after a successful load."""
        self.last_value = value
        now = now_utc or dt.datetime.now(dt.UTC)
        self.last_success_utc = now.isoformat(timespec="seconds")
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Increment the failure counter (does not touch ``last_value``)."""
        self.consecutive_failures += 1

    def warning_markdown(self) -> str:
        """Return the markdown warning to render below the table.

        Empty string on success (no consecutive failures); soft warning
        on 1-2 failures; loud warning on 3+. The soft warning interpolates
        the last-success timestamp so the operator can tell how stale
        the displayed data is.
        """
        if self.consecutive_failures <= 0:
            return ""
        if self.consecutive_failures < STALE_DATA_ESCALATE_AFTER:
            ts = self.last_success_utc or "unknown"
            return (
                f"_Last refresh failed (file may be mid-write). Showing "
                f"data from {ts}; retrying in 5s._"
            )
        return "**File system error. Check disk space and try `make dashboard` again.**"


def load_with_stale_fallback(
    cache: StaleDataCache,
    loader: Any,
    *,
    empty_factory: Any | None = None,
    now_utc: dt.datetime | None = None,
) -> tuple[Any, str]:
    """Run ``loader``; on failure return the cache + a warning string.

    ``loader`` is a zero-argument callable returning the loaded value;
    any exception is caught and the cache's failure counter bumped.
    ``empty_factory`` produces the value to return when the cache has
    no prior success (zero-argument callable returning e.g. an empty
    DataFrame). Returns ``(value_to_render, warning_markdown)``.

    Premortem: ``loader`` exceptions cover the three failure modes
    cited in the audit (``OSError`` on truncated read, ``JSONDecodeError``
    on mid-write JSON, ``pyarrow.lib.ArrowInvalid`` on truncated
    parquet). Catching the broad ``Exception`` here is deliberate --
    the alternative is a brittle catch list that misses the next
    surprise from upstream.
    """
    try:
        value = loader()
    except Exception as exc:
        logger.warning("stale-data fallback: loader raised %s: %s", type(exc).__name__, exc)
        cache.record_failure()
        if cache.last_value is not None:
            return cache.last_value, cache.warning_markdown()
        # First-ever load failed; fall through to the empty factory so
        # the UI still renders a canonical-shape table.
        if empty_factory is not None:
            return empty_factory(), cache.warning_markdown()
        return None, cache.warning_markdown()

    cache.record_success(value, now_utc=now_utc)
    return value, ""


# --------------------------------------------------------------------- #
# Mission Control: one-screen sweep supervision                          #
# --------------------------------------------------------------------- #
#
# The Mission Control tab is the default landing tab: a single glanceable
# screen the operator leaves open (often on a tablet over tailnet) to
# supervise a ~20-hour sweep without clicking around. Four stacked
# sections, each backed by a pure helper here so the Gradio wiring in
# ``app.py`` stays a thin shell.
#
# 1. At-a-glance health -- :func:`compute_mission_kpis` rolls the
#    manifest into the KPI strip (cells done / failed / running / ETA /
#    state) plus the green/amber/red health banner.
# 2. Live results forming -- :func:`build_live_leaderboard` aggregates
#    the per-episode parquet to a per-policy mean success rate + Wilson
#    CI as cells finish.
# 3. Anomaly alerts -- :func:`run_anomaly_review` wraps the incremental
#    checks from ``scripts/review_results.py`` against the live parquet.
# 4. Resource + throttle -- :func:`read_system_memory`,
#    :func:`find_sweep_pid`, :func:`read_throttle_state` surface RAM,
#    the sweep's cgroup memory cap, and the cgroup freeze state.

# Health-banner severities. The banner is the single most prominent
# element on the Mission Control screen; an operator glancing from
# across the room reads the colour, not the text.
HEALTH_GREEN = "green"
HEALTH_AMBER = "amber"
HEALTH_RED = "red"

# Sweep-state labels surfaced in the KPI strip. THROTTLED is reported
# when the sweep's cgroup is frozen (the watchdog froze it on a RAM
# breach); RUNNING / DONE come straight from the manifest roll-up.
SWEEP_STATE_RUNNING = "RUNNING"
SWEEP_STATE_THROTTLED = "THROTTLED-frozen"
SWEEP_STATE_DONE = "DONE"
SWEEP_STATE_IDLE = "IDLE"


@dataclass(frozen=True)
class MissionKPIs:
    """At-a-glance health numbers for the Mission Control KPI strip.

    Computed purely from the sweep manifest -- no parquet read -- so the
    strip repaints cheaply on the 5 s tick even when the parquet is
    mid-write. ``denom`` falls back to :data:`V1_TOTAL_SEED_ENTRIES`
    when the manifest is smaller than the v1 target (a mini sweep, or a
    manifest still being populated) so the ``N/110`` reads stably.
    """

    cells_done: int
    cells_failed: int
    cells_running: int
    cells_total: int
    denom: int
    percent_done: float
    running_label: str
    elapsed_label: str
    eta_label: str
    state: str
    health: str
    health_message: str


def _running_cell_label(manifest: dict[str, Any]) -> str:
    """Name the (policy, env, seed) of the most recently started seed.

    Returns ``"—"`` when nothing is in flight. The driver flips
    ``pending -> completed`` rather than marking a ``running`` status,
    so a running seed is a ``pending`` entry with a ``started_utc``.
    """
    best_ts = ""
    best: dict[str, Any] | None = None
    cells = manifest.get("cells", [])
    if not isinstance(cells, list):
        return STAT_PLACEHOLDER
    for entry in cells:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != STATUS_PENDING:
            continue
        ts = entry.get("started_utc")
        if isinstance(ts, str) and ts > best_ts:
            best_ts, best = ts, entry
    if best is None:
        return STAT_PLACEHOLDER
    seed = best.get("seed_idx", best.get("seed", "?"))
    return f"{best.get('policy', '?')} / {best.get('env', '?')} / seed {seed}"


def _format_duration(seconds: float) -> str:
    """Render a wall-clock duration as ``Hh Mm`` (or ``Mm`` under 1 h)."""
    if seconds <= 0:
        return STAT_PLACEHOLDER
    total_min = int(seconds // 60)
    hours, minutes = divmod(total_min, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _sweep_elapsed_seconds(manifest: dict[str, Any], now: dt.datetime) -> float:
    """Seconds since the sweep's ``started_utc`` (0.0 if unparseable)."""
    started = _parse_iso(manifest.get("started_utc"))
    if started is None:
        return 0.0
    return max(0.0, (now - started).total_seconds())


def _sweep_eta_seconds(manifest: dict[str, Any], now: dt.datetime) -> float:
    """Whole-sweep ETA: mean completed-seed wall-clock x remaining seeds.

    Pools every completed seed in the manifest for the mean (failed
    seeds excluded -- they finish faster and would bias the estimate
    low). Returns 0.0 when nothing has completed yet (no baseline) or
    when no seed is outstanding.
    """
    cells = manifest.get("cells", [])
    if not isinstance(cells, list):
        return 0.0
    durations: list[float] = []
    n_remaining = 0
    for entry in cells:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status in (STATUS_PENDING,):
            n_remaining += 1
        elif status == STATUS_COMPLETED:
            started = _parse_iso(entry.get("started_utc"))
            finished = _parse_iso(entry.get("finished_utc"))
            if started is not None and finished is not None:
                delta = (finished - started).total_seconds()
                if delta > 0:
                    durations.append(delta)
    if n_remaining == 0 or not durations:
        return 0.0
    return (sum(durations) / len(durations)) * n_remaining


def compute_mission_kpis(
    manifest: dict[str, Any],
    *,
    now_utc: dt.datetime | None = None,
    throttled: bool = False,
) -> MissionKPIs:
    """Roll a sweep manifest into the Mission Control KPI strip.

    Counts are over *cells* -- the ``(policy, env)`` grid, not the
    per-seed entries -- because that's the "N/110" the operator thinks
    in. (110 is the v1 seed-entry count; the cell denominator is
    :data:`V1_RUNNABLE_CELLS`. We surface seed-entries via the manifest
    badge elsewhere; here the KPI strip counts cells for the headline.)

    The health banner is the operator's primary signal:

    * **red** -- any failed cell, OR the sweep's cgroup is frozen.
    * **amber** -- a seed has been running far longer than its cell's
      completed seeds (a stuck cell), or no manifest at all.
    * **green** -- cells marching through cleanly.

    Args:
        manifest: parsed ``sweep_manifest.json`` dict.
        now_utc: pinned clock for deterministic ETA in tests.
        throttled: True when :func:`read_throttle_state` reports the
            sweep's cgroup is frozen -- forces the red banner + the
            ``THROTTLED-frozen`` state label.
    """
    now = now_utc or dt.datetime.now(dt.UTC)
    table = build_progress_table(manifest, now_utc=now)

    if table.empty:
        return MissionKPIs(
            cells_done=0,
            cells_failed=0,
            cells_running=0,
            cells_total=0,
            denom=V1_RUNNABLE_CELLS,
            percent_done=0.0,
            running_label=STAT_PLACEHOLDER,
            elapsed_label=STAT_PLACEHOLDER,
            eta_label=STAT_PLACEHOLDER,
            state=SWEEP_STATE_IDLE,
            health=HEALTH_AMBER,
            health_message=("No sweep running — no manifest on disk. Start one with `make sweep`."),
        )

    status_counts = table["status"].value_counts().to_dict()
    cells_done = int(status_counts.get(CELL_STATUS_DONE, 0)) + int(
        status_counts.get(CELL_STATUS_SKIPPED, 0)
    )
    cells_failed = int(status_counts.get(CELL_STATUS_FAILED, 0))
    cells_running = int(status_counts.get(CELL_STATUS_RUNNING, 0))
    cells_total = len(table)
    denom = cells_total if cells_total >= V1_RUNNABLE_CELLS else V1_RUNNABLE_CELLS
    percent = (cells_done / denom * 100.0) if denom else 0.0

    finished = bool(manifest.get("finished_utc"))
    all_done = cells_done == cells_total and cells_failed == 0

    if throttled:
        state = SWEEP_STATE_THROTTLED
    elif finished or all_done:
        state = SWEEP_STATE_DONE
    elif cells_running > 0:
        state = SWEEP_STATE_RUNNING
    else:
        state = SWEEP_STATE_RUNNING if not all_done else SWEEP_STATE_DONE

    eta_label = _format_duration(_sweep_eta_seconds(manifest, now))
    eta_phrase = f"ETA ~{eta_label}" if eta_label != STAT_PLACEHOLDER else "ETA unknown"

    if throttled:
        health = HEALTH_RED
        health_message = (
            f"Sweep needs you — FROZEN by the RAM watchdog. "
            f"{cells_done}/{denom} cells done. Free host memory or raise the cgroup cap."
        )
    elif cells_failed > 0:
        health = HEALTH_RED
        health_message = (
            f"Sweep needs you — {cells_done}/{denom} cells done · "
            f"{cells_failed} failed. Triage the failed rows in the cell grid below."
        )
    elif all_done:
        health = HEALTH_GREEN
        health_message = (
            f"Sweep complete — all {cells_done}/{denom} cells done · 0 failed · nothing needs you."
        )
    else:
        health = HEALTH_GREEN
        health_message = (
            f"Sweep healthy — {cells_done}/{denom} cells done · "
            f"0 failed · {eta_phrase} · nothing needs you."
        )

    return MissionKPIs(
        cells_done=cells_done,
        cells_failed=cells_failed,
        cells_running=cells_running,
        cells_total=cells_total,
        denom=denom,
        percent_done=percent,
        running_label=_running_cell_label(manifest),
        elapsed_label=_format_duration(_sweep_elapsed_seconds(manifest, now)),
        eta_label=eta_label,
        state=state,
        health=health,
        health_message=health_message,
    )


@dataclass(frozen=True)
class LeaderboardRow:
    """One policy's live aggregate for the Mission Control mini-leaderboard.

    ``success_rate`` and the Wilson CI are over the *flat per-episode
    outcomes* pooled across every completed cell of the policy (same
    pseudo-replication guard as :func:`compute_cell_episode_stats`).
    ``n_cells`` is how many ``(policy, env)`` cells have episodes on
    disk so far -- the operator reads this as "how settled is this row".
    """

    policy: str
    success_rate: float
    wilson_lo: float
    wilson_hi: float
    n_episodes: int
    n_cells: int


def build_live_leaderboard(results_df: pd.DataFrame | None) -> list[LeaderboardRow]:
    """Aggregate the per-episode parquet into a per-policy live leaderboard.

    Pools every episode of a policy across all its envs/seeds into one
    flat outcome list, computes the success rate + Wilson 95% CI, and
    sorts best-first. Returns ``[]`` when the parquet is absent / empty
    so the caller renders a "results forming once the first cell
    finishes" empty state.

    The unit is the episode, never the per-cell mean -- a policy with
    two cells of wildly different N still gets a single honest pooled
    rate (cells weight by episode count, which is the intended
    behaviour for a "results forming" glance).
    """
    if results_df is None or results_df.empty:
        return []
    needed = {"policy", "env", "success"}
    if not needed.issubset(results_df.columns):
        return []

    rows: list[LeaderboardRow] = []
    for policy, group in results_df.groupby("policy", sort=False):
        outcomes = group["success"].astype(bool)
        n = len(outcomes)
        if n == 0:
            continue
        n_success = int(outcomes.sum())
        lo, hi = wilson_ci(n_success, n)
        n_cells = int(group.groupby(["env"]).ngroups)
        rows.append(
            LeaderboardRow(
                policy=str(policy),
                success_rate=n_success / n,
                wilson_lo=float(lo),
                wilson_hi=float(hi),
                n_episodes=n,
                n_cells=n_cells,
            )
        )
    rows.sort(key=lambda r: r.success_rate, reverse=True)
    return rows


# Mission Control mini-leaderboard columns.
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "policy",
    "success_rate",
    "wilson_95_ci",
    "episodes",
    "cells_done",
)


def leaderboard_dataframe(rows: list[LeaderboardRow]) -> pd.DataFrame:
    """Project :func:`build_live_leaderboard` rows into a Gradio Dataframe.

    Empty input returns the canonical-column empty frame so the Gradio
    component never chokes on a missing header list at cold start. The
    success rate is rendered as a percent string and the CI as
    ``[lo, hi]`` so the row reads at a glance.
    """
    if not rows:
        return pd.DataFrame({c: [] for c in LEADERBOARD_COLUMNS})
    return pd.DataFrame(
        [
            {
                "policy": r.policy,
                "success_rate": f"{r.success_rate:.1%}",
                "wilson_95_ci": f"[{r.wilson_lo:.2f}, {r.wilson_hi:.2f}]",
                "episodes": r.n_episodes,
                "cells_done": r.n_cells,
            }
            for r in rows
        ],
        columns=list(LEADERBOARD_COLUMNS),
    )


@dataclass(frozen=True)
class AnomalyReport:
    """Result of running the ``review_results.py`` checks on the live parquet.

    ``ok`` is True when every completed cell looks healthy. ``lines`` is
    a flat list of one-line strings -- one per (flagged cell x flag) --
    ready to render. ``error`` is set when the review could not run at
    all (no parquet, mid-write read, bad config); the panel then shows a
    neutral "review unavailable" state rather than a false all-clear.
    """

    ok: bool
    n_cells_reviewed: int
    n_cells_flagged: int
    lines: list[str]
    error: str


def run_anomaly_review(
    results_path: Path | None,
    *,
    policies_yaml: Path | None = None,
    envs_yaml: Path | None = None,
) -> AnomalyReport:
    """Run the incremental sweep anomaly checks against the live parquet.

    Reuses ``scripts/review_results.py`` -- :func:`review_cells` plus the
    five checks (far-from-paper, baseline-above-floor, never-succeeds,
    seed-disagreement, degenerate). The parquet is read through
    :func:`load_results_parquet` so a mid-write file serves the
    last-good frame instead of crashing the panel.

    Degrades gracefully: a missing parquet, an unparseable config, or an
    import failure for the scripts package all yield an
    :class:`AnomalyReport` with ``ok=False`` and a populated ``error``
    -- never a raised exception, never a false "no anomalies".

    Args:
        results_path: path to the sweep ``results.parquet``.
        policies_yaml / envs_yaml: registry overrides (tests). Default
            to the shipped ``configs/*.yaml``.
    """
    df = load_results_parquet(results_path)
    if df is None or df.empty:
        return AnomalyReport(
            ok=True,
            n_cells_reviewed=0,
            n_cells_flagged=0,
            lines=[],
            error="no per-episode results on disk yet",
        )

    try:
        from scripts.review_results import review_cells
    except ImportError as exc:
        return AnomalyReport(
            ok=False,
            n_cells_reviewed=0,
            n_cells_flagged=0,
            lines=[],
            error=f"anomaly review unavailable (scripts import failed: {exc})",
        )

    try:
        policies = load_policy_registry(str(policies_yaml) if policies_yaml is not None else None)
        envs = load_env_registry(str(envs_yaml) if envs_yaml is not None else None)
    except (FileNotFoundError, ValueError) as exc:
        return AnomalyReport(
            ok=False,
            n_cells_reviewed=0,
            n_cells_flagged=0,
            lines=[],
            error=f"anomaly review unavailable (config load failed: {exc})",
        )

    required = {"policy", "env", "seed", "episode_index", "success", "n_steps", "wallclock_s"}
    if not required.issubset(df.columns):
        # The review's per-cell checks need n_steps / wallclock_s; a
        # results frame missing them is not an error worth a red panel,
        # just an "incomplete schema" note.
        return AnomalyReport(
            ok=True,
            n_cells_reviewed=0,
            n_cells_flagged=0,
            lines=[],
            error="results parquet missing columns the anomaly checks need",
        )

    try:
        reviews = review_cells(df, policies, envs)
    except Exception as exc:
        logger.warning("anomaly review raised %s: %s", type(exc).__name__, exc)
        return AnomalyReport(
            ok=False,
            n_cells_reviewed=0,
            n_cells_flagged=0,
            lines=[],
            error=f"anomaly review errored ({type(exc).__name__})",
        )

    flagged = [r for r in reviews if r.flagged]
    lines: list[str] = []
    for r in flagged:
        prefix = f"{r.policy} x {r.env} x seed {r.seed}"
        for flag in r.flags:
            lines.append(f"{prefix}: {flag}")
    return AnomalyReport(
        ok=not flagged,
        n_cells_reviewed=len(reviews),
        n_cells_flagged=len(flagged),
        lines=lines,
        error="",
    )


# --------------------------------------------------------------------- #
# Resource + throttle visibility                                        #
# --------------------------------------------------------------------- #
#
# The sweep runs under a cgroup v2 memory cap (scripts/run_capped.sh on
# this WSL2 host). Mission Control surfaces host RAM, the sweep's cgroup
# memory cap + current usage, and the cgroup freeze state -- the same
# files the watchdog uses. Every reader here degrades to ``None`` /
# ``STAT_PLACEHOLDER`` when the sweep isn't running or the files aren't
# readable; nothing raises.

# cgroup v2 root on this host. The sweep's cgroup is resolved per-PID
# from ``/proc/<pid>/cgroup`` so this constant is only the mount point.
_CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class SystemMemory:
    """Host RAM snapshot from ``/proc/meminfo``, in bytes.

    ``available`` is ``MemAvailable`` -- the kernel's own estimate of
    allocatable memory, which is the number the operator should watch
    (free + reclaimable cache), not raw ``MemFree``.
    """

    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float


def read_system_memory(meminfo_path: Path = Path("/proc/meminfo")) -> SystemMemory | None:
    """Read host RAM totals from ``/proc/meminfo``.

    Returns ``None`` on any read failure (non-Linux, sandboxed) so the
    caller renders ``"—"`` rather than crashing the Mission Control
    refresh tick. ``meminfo_path`` is injectable for tests.
    """
    try:
        text = meminfo_path.read_text()
    except OSError:
        return None
    fields: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        # values are "<number> kB"
        val = parts[1].strip().split()
        if not val:
            continue
        try:
            fields[key] = int(val[0]) * 1024
        except ValueError:
            continue
    total = fields.get("MemTotal")
    available = fields.get("MemAvailable")
    if total is None or available is None or total <= 0:
        return None
    used = max(0, total - available)
    return SystemMemory(
        total_bytes=total,
        available_bytes=available,
        used_bytes=used,
        percent_used=used / total * 100.0,
    )


def find_sweep_pid(proc_root: Path = Path("/proc")) -> int | None:
    """Find the PID of the running ``run_sweep.py`` process, if any.

    Scans ``/proc/<pid>/cmdline`` for a python invocation whose argv
    contains ``run_sweep.py``. Returns the lowest matching PID (the
    parent, not a forked child) or ``None`` when the sweep isn't
    running. Never raises -- a process exiting mid-scan just gets
    skipped.
    """
    if not proc_root.exists():
        return None
    matches: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        # cmdline args are NUL-separated.
        argv = cmdline.split(b"\x00")
        if any(b"run_sweep.py" in arg for arg in argv):
            matches.append(int(entry.name))
    if not matches:
        return None
    return min(matches)


def _resolve_cgroup_dir(pid: int, proc_root: Path = Path("/proc")) -> Path | None:
    """Resolve a PID's cgroup v2 directory from ``/proc/<pid>/cgroup``.

    The cgroup v2 line is ``0::/path/relative/to/the/cgroup/mount``.
    Returns the absolute directory under :data:`_CGROUP_ROOT`, or
    ``None`` when the file is unreadable or has no v2 line.
    """
    try:
        text = (proc_root / str(pid) / "cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        # cgroup v2 unified hierarchy: hierarchy-ID 0, empty controller.
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2].lstrip("/")
            return _CGROUP_ROOT / rel
    return None


def _read_int_file(path: Path) -> int | None:
    """Read a single-integer cgroup file. ``None`` on miss / ``max``."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw in ("", "max"):
        # ``memory.max`` is literally "max" when uncapped.
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class ThrottleState:
    """Sweep cgroup memory + freeze state for the Mission Control panel.

    Every field degrades to ``None`` when the sweep isn't running or the
    cgroup files aren't readable; the renderer then shows ``"—"``.

    * ``running`` -- a ``run_sweep.py`` PID was found.
    * ``frozen`` -- ``cgroup.freeze`` is ``1`` (the watchdog froze the
      sweep on a RAM breach). ``None`` when undiscoverable.
    * ``memory_current`` / ``memory_max`` -- ``memory.current`` /
      ``memory.max`` in bytes; ``memory_max`` is ``None`` when uncapped.
    """

    running: bool
    pid: int | None
    frozen: bool | None
    memory_current: int | None
    memory_max: int | None

    @property
    def state_label(self) -> str:
        """One-word state for the KPI strip: RUNNING / FROZEN / not running."""
        if not self.running:
            return "not running"
        if self.frozen:
            return "FROZEN"
        return "RUNNING"


def read_throttle_state(proc_root: Path = Path("/proc")) -> ThrottleState:
    """Resolve the sweep's cgroup freeze + memory state.

    Finds the ``run_sweep.py`` PID via :func:`find_sweep_pid`, resolves
    its cgroup v2 directory, and reads ``cgroup.freeze`` (0=running,
    1=frozen), ``memory.current`` and ``memory.max``. Returns a
    :class:`ThrottleState` with every field ``None`` when the sweep is
    absent or the files are unreadable -- never raises, so the Mission
    Control tick is safe on a non-Linux host or a sandboxed env.

    ``proc_root`` is injectable so tests can point at a synthetic
    ``/proc`` layout.
    """
    pid = find_sweep_pid(proc_root)
    if pid is None:
        return ThrottleState(
            running=False,
            pid=None,
            frozen=None,
            memory_current=None,
            memory_max=None,
        )
    cgroup_dir = _resolve_cgroup_dir(pid, proc_root)
    if cgroup_dir is None:
        return ThrottleState(
            running=True,
            pid=pid,
            frozen=None,
            memory_current=None,
            memory_max=None,
        )
    freeze_raw = _read_int_file(cgroup_dir / "cgroup.freeze")
    frozen = None if freeze_raw is None else bool(freeze_raw)
    return ThrottleState(
        running=True,
        pid=pid,
        frozen=frozen,
        memory_current=_read_int_file(cgroup_dir / "memory.current"),
        memory_max=_read_int_file(cgroup_dir / "memory.max"),
    )


def format_bytes_gb(value: int | None) -> str:
    """Render a byte count as ``N.N GB``; :data:`STAT_PLACEHOLDER` on None."""
    if value is None:
        return STAT_PLACEHOLDER
    return f"{value / 1024**3:.1f} GB"


# --------------------------------------------------------------------- #
# Cross-run registry + selector (monitoring layer, item 1)              #
# --------------------------------------------------------------------- #
#
# The Status tab historically hardwired ``runs[0]`` -- the newest sweep.
# That is the right default but makes the dashboard useless for
# comparing this run against last week's. The helpers below turn the run
# list into a dropdown-friendly registry and resolve a *selected* run by
# name, falling back to newest when the selection is stale.
#
# PROPOSED run-registry record shape (documented in docs/MONITORING.md,
# pending user confirmation): the dropdown carries the run *name* (the
# directory basename) as its value; everything else is recovered by
# re-discovering on disk so the dropdown never holds a stale Path.


def run_selector_choices(results_root: Path | None = None) -> list[str]:
    """Return the run *names* for the Status-tab run selector, newest first.

    Values are the directory basenames (``discover_sweep_runs`` already
    sorts newest-first by ``started_utc``). The basename is the dropdown
    value so the selection survives a re-discovery even when a run's
    ``finished_utc`` flips mid-session. Empty list on a missing/empty
    results dir.
    """
    root = results_root if results_root is not None else env_dashboard_results_dir()
    return [run.name for run in discover_sweep_runs(root)]


def run_selector_label_map(results_root: Path | None = None) -> dict[str, str]:
    """Map run name -> human label (with ``[running]``/``[done]`` marker).

    Insertion-ordered (newest first) so a caller can build
    ``(label, value)`` dropdown choices directly while keeping the bare
    name as the stable selection value.
    """
    root = results_root if results_root is not None else env_dashboard_results_dir()
    return {run.name: run.label for run in discover_sweep_runs(root)}


def resolve_selected_run(
    selected_name: str | None,
    results_root: Path | None = None,
) -> SweepRun | None:
    """Return the :class:`SweepRun` for ``selected_name``, or newest on miss.

    The selector stores a run *name*. ``None`` (first paint) or a name no
    longer on disk falls back to the newest run -- the same auto-newest
    behaviour the Status tab had before the selector. Returns ``None``
    only when there are no runs at all.
    """
    root = results_root if results_root is not None else env_dashboard_results_dir()
    runs = discover_sweep_runs(root)
    if not runs:
        return None
    if selected_name:
        for run in runs:
            if run.name == selected_name:
                return run
    return runs[0]


# --------------------------------------------------------------------- #
# Compare tab: paper-vs-measured AND WM-vs-VLA on shared axes           #
# --------------------------------------------------------------------- #
#
# Item 2. Two comparison modes share one table shape because both are
# "two success rates per (policy, env) cell + a colour-chipped delta".
# Reuses :func:`delta_chip` + :func:`_our_success_rate` (no new delta
# math) so the chip thresholds stay identical to the policy cards.

COMPARE_COLUMNS: tuple[str, ...] = ("env", "left", "right", "delta", "")

COMPARE_MODE_PAPER = "paper-vs-measured"
COMPARE_MODE_WM_VLA = "wm-vs-vla"


def build_paper_vs_measured_table(
    policy_name: str,
    *,
    registry: PolicyRegistry | None = None,
    results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per supported env: paper-reported vs our re-run + delta chip.

    Left is the paper-reported rate (from the policy's
    ``paper_reported_success`` map, sourced from ``docs/MODEL_CARDS.md``
    -- read-only here); right is our pooled re-run rate from the parquet.
    The chip is :func:`delta_chip` so it matches the policy-card buckets.
    Cells with no parquet rows show ``(pending)``; envs with no paper
    reference show :data:`STAT_PLACEHOLDER` on the left. Unknown policy
    returns the canonical-column empty frame.
    """
    reg = registry if registry is not None else load_policy_registry()
    try:
        spec: PolicySpec = reg.get(policy_name)
    except KeyError:
        return pd.DataFrame({c: [] for c in COMPARE_COLUMNS})

    paper_map = spec.paper_reported_success or {}
    rows: list[dict[str, Any]] = []
    for env in spec.env_compat:
        paper_val = paper_map.get(env)
        ours_val = _our_success_rate(results_df, policy=spec.name, env=env)
        left = f"{paper_val:.3f}" if paper_val is not None else STAT_PLACEHOLDER
        if ours_val is None:
            right, delta_cell, chip = "(pending)", STAT_PLACEHOLDER, ""
        else:
            right = f"{ours_val:.3f}"
            delta_cell, chip = delta_chip(paper_val, ours_val)
        rows.append({"env": env, "left": left, "right": right, "delta": delta_cell, "": chip})
    return pd.DataFrame(rows, columns=list(COMPARE_COLUMNS))


def build_wm_vs_vla_table(
    wm_policy: str | None,
    vla_policy: str | None,
    *,
    results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """WM-planner vs VLA on the shared envs both ran, + delta chip.

    Both rates come from the parquet via :func:`_our_success_rate`; rows
    are the *intersection* of the two policies' envs with episodes on
    disk (a paired comparison needs the same env on both sides). Left is
    the WM planner, right is the VLA baseline; the chip keys on
    ``|right - left|``.

    v1 has no WM rows, so this returns the canonical-column empty frame
    until a WM policy lands in ``results.parquet``.
    """
    if not wm_policy or not vla_policy or results_df is None or results_df.empty:
        return pd.DataFrame({c: [] for c in COMPARE_COLUMNS})
    if "env" not in results_df.columns or "policy" not in results_df.columns:
        return pd.DataFrame({c: [] for c in COMPARE_COLUMNS})

    wm_envs = set(results_df[results_df["policy"] == wm_policy]["env"].unique())
    vla_envs = set(results_df[results_df["policy"] == vla_policy]["env"].unique())
    shared = sorted(wm_envs & vla_envs)

    rows: list[dict[str, Any]] = []
    for env in shared:
        wm_val = _our_success_rate(results_df, policy=wm_policy, env=env)
        vla_val = _our_success_rate(results_df, policy=vla_policy, env=env)
        left = f"{wm_val:.3f}" if wm_val is not None else STAT_PLACEHOLDER
        right = f"{vla_val:.3f}" if vla_val is not None else STAT_PLACEHOLDER
        # delta_chip(paper=vla, ours=wm) gives ours-paper == wm-vla; the
        # chip colour keys on |delta| (symmetric), so the subtraction
        # direction only sets the printed sign.
        delta_cell, chip = delta_chip(vla_val, wm_val)
        rows.append({"env": env, "left": left, "right": right, "delta": delta_cell, "": chip})
    return pd.DataFrame(rows, columns=list(COMPARE_COLUMNS))


def wm_policy_names(results_df: pd.DataFrame | None) -> list[str]:
    """Policy names in the parquet that are NOT v1 leaderboard policies.

    Heuristic stand-in for "world-model planners run as policies" until a
    real ``kind`` column lands: any policy present that is outside
    :data:`V1_POLICIES`. In v1 this is empty (the parquet is filtered to
    v1 policies on load), which is why WM-vs-VLA shows an empty state.
    PROVISIONAL -- see docs/MONITORING.md.
    """
    if results_df is None or results_df.empty or "policy" not in results_df.columns:
        return []
    present = list(dict.fromkeys(str(p) for p in results_df["policy"].tolist()))
    return [p for p in present if p not in set(V1_POLICIES)]


# --------------------------------------------------------------------- #
# Failure drill-down (monitoring layer, item 3)                         #
# --------------------------------------------------------------------- #
#
# The parquet has NO ``failure_mode`` column yet (a real one is a future
# schema bump -- see docs/MONITORING.md and docs/FAILURE_TAXONOMY.md).
# The drill-down degrades GRACEFULLY: it shows what the per-episode rows
# DO carry -- success / length (n_steps) distributions and the cap-hit
# rate (failed episodes that ran to the env ``max_steps``, the closest
# label-free proxy to a "timeout" failure mode) -- plus MP4 links via
# the flat video naming.

FAILURE_DISTRIBUTION_COLUMNS: tuple[str, ...] = ("metric", "value")


@dataclass(frozen=True)
class CellFailureSummary:
    """Label-free failure drill-down for one ``(policy, env)`` cell.

    Computed entirely from columns the v1 parquet already carries
    (``success``, ``n_steps``); ``cap_hit_rate`` is the fraction of
    *failed* episodes whose ``n_steps`` reached the env ``max_steps``.
    ``None`` fields mean the column was absent -- the renderer shows
    :data:`STAT_PLACEHOLDER` rather than inventing a number.
    """

    n_episodes: int
    n_failures: int
    success_rate: float
    mean_steps: float | None
    median_steps: float | None
    max_steps: int | None
    cap_hit_failures: int | None
    cap_hit_rate: float | None


def compute_cell_failure_summary(
    results_df: pd.DataFrame | None,
    *,
    policy: str,
    env: str,
    max_steps: int | None = None,
) -> CellFailureSummary | None:
    """Summarise the label-free failure signals for one cell.

    ``max_steps`` is the env's step cap (from the :class:`EnvSpec`); when
    supplied, a *failed* episode whose ``n_steps >= max_steps`` is a
    cap-hit (timeout proxy). ``None`` when the cell has no rows -- the
    drill-down then renders a "no episodes for this cell" empty state.
    """
    if results_df is None or results_df.empty:
        return None
    if not {"policy", "env", "success"}.issubset(results_df.columns):
        return None
    cell = results_df[(results_df["policy"] == policy) & (results_df["env"] == env)]
    n = len(cell)
    if n == 0:
        return None

    outcomes = cell["success"].astype(bool)
    n_success = int(outcomes.sum())
    n_failures = n - n_success
    success_rate = n_success / n

    mean_steps = median_steps = None
    cap_hit_failures = cap_hit_rate = None
    if "n_steps" in cell.columns:
        steps = cell["n_steps"].astype(float)
        mean_steps = float(steps.mean())
        median_steps = float(steps.median())
        if max_steps is not None and max_steps > 0:
            failed = cell[~outcomes]
            if not failed.empty:
                failed_steps = failed["n_steps"].astype(float)
                cap_hit_failures = int((failed_steps >= max_steps).sum())
                cap_hit_rate = cap_hit_failures / len(failed)
            else:
                cap_hit_failures = 0
                cap_hit_rate = 0.0

    return CellFailureSummary(
        n_episodes=n,
        n_failures=n_failures,
        success_rate=float(success_rate),
        mean_steps=mean_steps,
        median_steps=median_steps,
        max_steps=max_steps,
        cap_hit_failures=cap_hit_failures,
        cap_hit_rate=cap_hit_rate,
    )


def failure_summary_table(summary: CellFailureSummary | None) -> pd.DataFrame:
    """Project a :class:`CellFailureSummary` into a 2-column metric table.

    Empty (canonical-column) frame on ``None`` so the Gradio Dataframe
    renders cleanly when the cell has no episodes yet.
    """
    if summary is None:
        return pd.DataFrame({c: [] for c in FAILURE_DISTRIBUTION_COLUMNS})

    def _opt(value: float | None, fmt: str) -> str:
        return format(value, fmt) if value is not None else STAT_PLACEHOLDER

    rows: list[dict[str, str]] = [
        {"metric": "Episodes", "value": str(summary.n_episodes)},
        {"metric": "Failures", "value": str(summary.n_failures)},
        {"metric": "Success rate", "value": f"{summary.success_rate:.1%}"},
        {"metric": "Mean steps", "value": _opt(summary.mean_steps, ".0f")},
        {"metric": "Median steps", "value": _opt(summary.median_steps, ".0f")},
        {
            "metric": "Env step cap",
            "value": (
                str(summary.max_steps) if summary.max_steps is not None else STAT_PLACEHOLDER
            ),
        },
        {
            "metric": "Cap-hit failures (timeout proxy)",
            "value": (
                str(summary.cap_hit_failures)
                if summary.cap_hit_failures is not None
                else STAT_PLACEHOLDER
            ),
        },
        {
            "metric": "Cap-hit rate (of failures)",
            "value": _opt(summary.cap_hit_rate, ".1%"),
        },
    ]
    return pd.DataFrame(rows, columns=list(FAILURE_DISTRIBUTION_COLUMNS))


def cell_max_steps(env: str, *, registry: EnvRegistry | None = None) -> int | None:
    """Look up the env's ``max_steps`` for the cap-hit (timeout) proxy.

    ``None`` for an unknown env so the drill-down omits the cap-hit
    metric rather than crashing.
    """
    reg = registry if registry is not None else load_env_registry()
    try:
        return int(reg.get(env).max_steps)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


def failed_episode_video_links(
    index: VideoIndex,
    results_df: pd.DataFrame | None,
    *,
    policy: str,
    env: str,
    limit: int = 12,
) -> list[tuple[str, Path]]:
    """Return ``(label, path)`` for failed-episode MP4s of one cell.

    Joins the failed rows of the parquet against the on-disk video index
    (flat naming ``{policy}__{env}__seed{seed}__ep{NNN}.mp4``). Labels
    read ``seed{S} ep{E}``. Capped at ``limit`` so a many-failure cell
    doesn't render a wall of links. ``[]`` when no failed episode has a
    video on disk.
    """
    if results_df is None or results_df.empty:
        return []
    if not {"policy", "env", "seed", "episode_index", "success"}.issubset(results_df.columns):
        return []
    cell = results_df[
        (results_df["policy"] == policy)
        & (results_df["env"] == env)
        & (~results_df["success"].astype(bool))
    ]
    if cell.empty:
        return []
    cell = cell.sort_values(["seed", "episode_index"], kind="stable")
    links: list[tuple[str, Path]] = []
    for row in cell.itertuples(index=False):
        path = find_video_path(
            index,
            policy=policy,
            env=env,
            seed=int(row.seed),
            episode=int(row.episode_index),
        )
        if path is not None:
            links.append((f"seed{int(row.seed)} ep{int(row.episode_index):03d}", path))
        if len(links) >= limit:
            break
    return links


# --------------------------------------------------------------------- #
# Slow-lane training visibility (monitoring layer, item 4)              #
# --------------------------------------------------------------------- #
#
# A WM/JEPA training run on a single offline laptop has no wandb. The
# PROPOSED contract (docs/MONITORING.md, pending user confirmation) is a
# per-run append-only JSONL at ``results/wm-runs/<run_id>/progress.jsonl``
# whose records are minimal: {ts, run_id, step, metric, value}. The
# Training tab tails the *latest* such file if present; absent => a
# friendly empty state. ``scripts/wm_run_log.py`` is the tiny offline-
# first writer (it imports nothing from this module).

WM_RUNS_SUBDIR = "wm-runs"
WM_PROGRESS_FILENAME = "progress.jsonl"

# Columns surfaced in the Training tab's tail table.
WM_PROGRESS_COLUMNS: tuple[str, ...] = ("step", "metric", "value", "ts")


def wm_runs_root(results_root: Path | None = None) -> Path:
    """Directory holding per-run training-progress subdirs.

    ``results/wm-runs/<run_id>/progress.jsonl``. Anchored on the same
    env-resolved results dir as the sweep state.
    """
    root = results_root if results_root is not None else env_dashboard_results_dir()
    return root / WM_RUNS_SUBDIR


def discover_wm_progress_files(results_root: Path | None = None) -> list[Path]:
    """List every ``wm-runs/*/progress.jsonl``, newest mtime first.

    ``[]`` when the ``wm-runs`` dir is absent -- the common v1 case
    (no WM training has run), which the Training tab renders as a
    friendly empty state.
    """
    root = wm_runs_root(results_root)
    if not root.exists():
        return []
    files = [p / WM_PROGRESS_FILENAME for p in root.iterdir() if p.is_dir()]
    files = [f for f in files if f.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_wm_progress(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    """Parse the last ``limit`` JSONL records of a training-progress file.

    Each line is one ``{ts, run_id, step, metric, value}`` record;
    malformed lines are skipped silently (a half-written final line is
    expected mid-append). ``[]`` on a missing file. Newest records last,
    capped to ``limit`` from the tail.
    """
    if not path.exists() or not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("wm progress %s unreadable: %s", path, exc)
        return []
    records: list[dict[str, Any]] = []
    for line in raw[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def wm_progress_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Project training-progress records into the Training-tab table.

    Canonical-column empty frame on no records. Missing keys render
    :data:`STAT_PLACEHOLDER` so a partially-written record still shows
    its step/metric without crashing.
    """
    if not records:
        return pd.DataFrame({c: [] for c in WM_PROGRESS_COLUMNS})
    rows = [
        {
            "step": rec.get("step", STAT_PLACEHOLDER),
            "metric": str(rec.get("metric", STAT_PLACEHOLDER)),
            "value": rec.get("value", STAT_PLACEHOLDER),
            "ts": str(rec.get("ts", STAT_PLACEHOLDER)),
        }
        for rec in records
    ]
    return pd.DataFrame(rows, columns=list(WM_PROGRESS_COLUMNS))


def wm_progress_summary(records: list[dict[str, Any]], *, run_id: str = "") -> str:
    """One-line markdown summary above the Training-tab table.

    States run id, latest step, and which metrics are present. Empty
    records yield a friendly "no records yet" line.
    """
    label = f"`{run_id}` — " if run_id else ""
    if not records:
        return f"{label}_no progress records yet (the file is empty or just created)._"
    steps = [r.get("step") for r in records if isinstance(r.get("step"), (int, float))]
    metrics = sorted({str(r.get("metric")) for r in records if r.get("metric") is not None})
    last_step = max(steps) if steps else STAT_PLACEHOLDER
    metric_str = ", ".join(f"`{m}`" for m in metrics) if metrics else STAT_PLACEHOLDER
    return (
        f"{label}**{len(records)}** record(s) · latest step **{last_step}** · metrics: {metric_str}"
    )
