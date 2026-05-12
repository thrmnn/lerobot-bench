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
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

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

# Progress-table columns. Wired into the Gradio Dataframe in ``app.py``.
PROGRESS_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "status",
    "seeds_done",
    "seeds_total",
    "episodes_done",
    "episodes_total",
    "last_update_utc",
    "eta_minutes",
)

# Calibration-table columns. ``reason`` is derived locally; everything
# else is straight from the CellTiming dataclass on disk.
CALIBRATION_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "status",
    "mean_step_ms",
    "p95_step_ms",
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
# Sweep progress aggregation                                            #
# --------------------------------------------------------------------- #


def build_progress_table(
    manifest: dict[str, Any],
    *,
    now_utc: dt.datetime | None = None,
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
        rows.append(_summarise_cell(policy=policy, env=env, seeds=seeds, now=now))

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
) -> dict[str, Any]:
    """Roll up the per-seed manifest entries for one (policy, env) cell."""
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


def build_calibration_table(report: dict[str, Any]) -> pd.DataFrame:
    """Project a calibration report's ``cells`` list into a table.

    Empty input returns an empty frame with :data:`CALIBRATION_COLUMNS`
    so the Gradio Dataframe component doesn't choke on a missing column
    list during cold start.

    The ``reason`` column is derived locally via :func:`downscope_reason`
    -- the JSON itself does not carry a reason field; the auto-downscope
    rule's bucket is reverse-engineered from the timing thresholds.
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
        rows.append(
            {
                "policy": str(cell.get("policy", "")),
                "env": str(cell.get("env", "")),
                "status": str(cell.get("status", "")),
                "mean_step_ms": float(cell.get("mean_ms_per_step") or 0.0),
                "p95_step_ms": float(cell.get("p95_ms_per_step") or 0.0),
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
