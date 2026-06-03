"""Cell-boundary resume layer for the sweep orchestrator.

A "cell" is the tuple ``(policy, env, seed)``. The sweep contract (see
``docs/DESIGN.md`` § Methodology) is **5 seeds × 50 episodes** per
cell, producing one ``results.parquet`` row per episode. Mid-cell
resume is **not** bit-reproducible because the torch generator advances
across episodes within a cell — so this module's resume granularity is
the cell boundary: a cell either ran to completion (skip on resume) or
is restarted from episode 0.

This module is pure ``pandas`` + ``pyarrow``. No torch, no env, no
policy loading — keeps it cheap to import and easy to test in CI
without GPU.

The orchestrator (``scripts/run_sweep.py``) uses :func:`plan_resume` to
decide which cells to run, calls :func:`drop_partial_cells` on partials
before re-running them, and flushes each completed cell with
:func:`append_cell_rows`. Atomicity is preserved by writing to a
``.tmp.parquet`` sibling and then ``os.replace``-ing it into place.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Columns required in every results.parquet row. This is the strict
# gate: a parquet missing any of these (or carrying an unknown extra) is
# rejected by load_results. Order is canonical; mirrored in the
# DataFrame written to disk.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "policy",
    "env",
    "seed",
    "episode_index",
    "success",
    "return_",
    "n_steps",
    "wallclock_s",
    "video_sha256",
    "code_sha",
    "lerobot_version",
    "timestamp_utc",
)

# Columns the loader tolerates as absent: a parquet predating their
# introduction (the published Hub dataset, `make reproduce`, the live
# Space) loads fine and the missing column is back-filled with the
# sentinel in OPTIONAL_DEFAULTS. New optional columns MUST be appended
# here (never to REQUIRED_COLUMNS) so old parquets stay loadable.
#
# * ``errored`` (audit H3) -- True for rows produced by a crashed
#   episode (OOM, env death). Distinguishes a crash from a legit
#   task failure and makes plan_resume re-run the cell.
# * ``eval_run_id`` (audit M5) -- provenance handle minted once per
#   sweep invocation; foundation for the future MLflow/DVC layer.
#   Named to NOT collide with the WM-training JSONL 'run_id'.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "errored",
    "eval_run_id",
)

# Sentinel back-fill value per optional column, used when an older
# parquet is loaded without it.
OPTIONAL_DEFAULTS: dict[str, bool | str] = {
    "errored": False,
    "eval_run_id": "",
}

# Full canonical column order = required followed by optional.
RESULT_SCHEMA: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Subset of RESULT_SCHEMA that uniquely identifies a row.
_ROW_KEY: tuple[str, ...] = ("policy", "env", "seed", "episode_index")

# Subset of RESULT_SCHEMA that identifies a cell (used by drop_partial_cells).
_CELL_KEY: tuple[str, ...] = ("policy", "env", "seed")


@dataclass(frozen=True)
class CellKey:
    """Identifies a ``(policy, env, seed)`` cell. Equality + hashable."""

    policy: str
    env: str
    seed: int


@dataclass(frozen=True)
class ResumePlan:
    """Output of :func:`plan_resume` — what the orchestrator should do next.

    ``completed_cells`` are cells already at full ``n_episodes`` rows with
    a clean ``set(episode_index) == set(range(n_episodes))``. They are
    skipped on resume.

    ``partial_cells`` are cells with at least one row but not a clean full
    set (mid-cell crash, missing index, or unexpected over-write). They
    must be dropped via :func:`drop_partial_cells` before the
    orchestrator restarts them from episode 0.

    ``pending_cells`` are cells with zero rows in the parquet — fresh
    work.
    """

    completed_cells: frozenset[CellKey]
    partial_cells: frozenset[CellKey]
    pending_cells: frozenset[CellKey]
    rows_loaded: int


def _empty_results_df() -> pd.DataFrame:
    """Build an empty DataFrame with exactly ``RESULT_SCHEMA`` columns."""
    return pd.DataFrame({col: pd.Series(dtype=_dtype_for(col)) for col in RESULT_SCHEMA})


def _dtype_for(col: str) -> str:
    """Canonical dtype per schema column. Used only to seed empty frames.

    Dtypes after a roundtrip through pyarrow are determined by pyarrow,
    not this map; the map exists so an empty DataFrame still has a
    sensible numeric dtype on ``seed`` / ``episode_index`` rather than
    object.
    """
    if col in {"seed", "episode_index", "n_steps"}:
        return "int64"
    if col in {"return_", "wallclock_s"}:
        return "float64"
    if col in {"success", "errored"}:
        return "bool"
    return "object"


def load_results(parquet_path: Path) -> pd.DataFrame:
    """Read existing ``results.parquet``.

    Returns an empty DataFrame with exactly ``RESULT_SCHEMA`` columns if
    the file is missing.

    Schema gate (audit C2 -- back-compat split): a file is rejected with
    :class:`ValueError` only if it is **missing a REQUIRED column** or
    carries an **unknown extra column** (not in REQUIRED ∪ OPTIONAL).
    Missing OPTIONAL columns are tolerated and back-filled with their
    :data:`OPTIONAL_DEFAULTS` sentinel, so a parquet written before an
    optional column existed (the published Hub dataset, ``make
    reproduce``, the live Space) still loads.
    """
    if not parquet_path.exists():
        return _empty_results_df()

    df = pd.read_parquet(parquet_path)
    actual = set(df.columns)
    missing_required = sorted(set(REQUIRED_COLUMNS) - actual)
    extra = sorted(actual - set(RESULT_SCHEMA))
    if missing_required or extra:
        raise ValueError(
            f"unexpected columns in {parquet_path}: missing={missing_required}, extra={extra}"
        )

    # Back-fill any absent optional columns with their sentinel so
    # downstream code can rely on every RESULT_SCHEMA column existing.
    for col in OPTIONAL_COLUMNS:
        if col not in actual:
            df[col] = OPTIONAL_DEFAULTS[col]

    # Reorder to canonical schema order so downstream code can rely on it.
    return df[list(RESULT_SCHEMA)]


def plan_resume(
    parquet_path: Path,
    *,
    requested_cells: Iterable[CellKey],
    n_episodes: int,
) -> ResumePlan:
    """Inspect the parquet and decide which cells to skip vs run.

    A cell is **completed** iff it has rows whose ``episode_index`` set
    equals ``set(range(n_episodes))`` exactly **and no row is flagged
    ``errored``**. Any other non-zero row count classifies the cell as
    **partial** — including the corruption cases of a missing index
    inside the range, an unexpected over-write past ``n_episodes - 1``,
    or a full-index cell that contains an errored row (audit H3: an
    OOM/env-crash episode is indistinguishable from a legit failure on
    ``success`` alone, so the ``errored`` flag forces a re-run instead
    of silently treating the crashed cell as done). A cell with no rows
    is **pending**.

    Cells in the existing parquet that are *not* in ``requested_cells``
    are ignored — this lets a sweep be re-shaped without losing prior
    work.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    requested_set = frozenset(requested_cells)
    df = load_results(parquet_path)
    rows_loaded = len(df)

    completed: set[CellKey] = set()
    partial: set[CellKey] = set()

    if rows_loaded > 0:
        # Build a dict of cell -> set(episode_index) for O(1) lookup below.
        # Iterating zips directly avoids fighting groupby's loose Hashable types.
        # `errored_cells` collects any cell with at least one crashed row so
        # a full-index-but-errored cell still re-runs (audit H3).
        index_sets: dict[CellKey, set[int]] = {}
        errored_cells: set[CellKey] = set()
        for policy, env, seed, ep, errored in zip(
            df["policy"],
            df["env"],
            df["seed"],
            df["episode_index"],
            df["errored"],
            strict=True,
        ):
            key = CellKey(policy=str(policy), env=str(env), seed=int(seed))
            index_sets.setdefault(key, set()).add(int(ep))
            if bool(errored):
                errored_cells.add(key)

        expected_indices = set(range(n_episodes))
        for cell in requested_set:
            seen = index_sets.get(cell)
            if seen is None:
                continue
            if seen == expected_indices and cell not in errored_cells:
                completed.add(cell)
            else:
                partial.add(cell)

    pending = requested_set - completed - partial

    return ResumePlan(
        completed_cells=frozenset(completed),
        partial_cells=frozenset(partial),
        pending_cells=frozenset(pending),
        rows_loaded=rows_loaded,
    )


def append_cell_rows(parquet_path: Path, new_rows: pd.DataFrame) -> int:
    """Atomically append ``new_rows`` to ``parquet_path``.

    Strategy: load existing rows, validate no duplicate ``(policy, env,
    seed, episode_index)`` keys, concat, write to a sibling
    ``.tmp.parquet``, then ``os.replace`` into the final path
    (atomic on POSIX). If ``new_rows`` is empty, this is a no-op and
    returns the existing row count.

    Schema gate mirrors :func:`load_results` (audit C2): ``new_rows``
    must carry every REQUIRED column and no unknown extras; absent
    OPTIONAL columns are back-filled with their
    :data:`OPTIONAL_DEFAULTS` sentinel so a caller built against an
    older schema still appends cleanly.

    Raises :class:`ValueError` if ``new_rows`` is missing a required
    column, carries an unknown extra, or if any row would duplicate an
    existing ``(policy, env, seed, episode_index)`` tuple.
    """
    actual = set(new_rows.columns)
    missing_required = sorted(set(REQUIRED_COLUMNS) - actual)
    extra = sorted(actual - set(RESULT_SCHEMA))
    if missing_required or extra:
        raise ValueError(f"new_rows has wrong columns: missing={missing_required}, extra={extra}")

    # Back-fill absent optional columns before concat so the written
    # parquet always has the full canonical schema.
    new_rows = new_rows.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in actual:
            new_rows[col] = OPTIONAL_DEFAULTS[col]

    existing = load_results(parquet_path)

    if len(new_rows) == 0:
        return len(existing)

    # Duplicate-key guard. Build a set of tuples from existing, then
    # check membership for each new row. Faster than a merge for the
    # row counts we expect (≤ 250 per cell).
    existing_keys: set[tuple[str, str, int, int]] = {
        (str(p), str(e), int(s), int(i))
        for p, e, s, i in zip(
            existing["policy"],
            existing["env"],
            existing["seed"],
            existing["episode_index"],
            strict=True,
        )
    }
    duplicates: list[tuple[str, str, int, int]] = []
    for p, e, s, i in zip(
        new_rows["policy"],
        new_rows["env"],
        new_rows["seed"],
        new_rows["episode_index"],
        strict=True,
    ):
        key = (str(p), str(e), int(s), int(i))
        if key in existing_keys:
            duplicates.append(key)
    if duplicates:
        raise ValueError(f"duplicate (policy, env, seed, episode_index) keys: {duplicates}")

    # Reorder new_rows to canonical column order before concat so the
    # resulting parquet schema is stable.
    new_rows_ordered = new_rows[list(RESULT_SCHEMA)]
    combined = pd.concat([existing, new_rows_ordered], ignore_index=True)

    _atomic_write_parquet(parquet_path, combined)
    return len(combined)


def drop_partial_cells(parquet_path: Path, cells: Iterable[CellKey]) -> int:
    """Remove all rows belonging to the given cells.

    Used to clean up partial cells before re-running them. Returns the
    number of rows removed. No-op (returns 0) if the parquet file is
    missing or ``cells`` is empty.
    """
    cells_list = list(cells)
    if not cells_list or not parquet_path.exists():
        return 0

    df = load_results(parquet_path)
    if len(df) == 0:
        return 0

    cell_tuples = {(c.policy, c.env, c.seed) for c in cells_list}
    # Build the per-row cell tuple via zip — avoids a MultiIndex round trip
    # and keeps dtypes intact across the boolean mask.
    keep_mask = pd.Series(
        [
            (str(p), str(e), int(s)) not in cell_tuples
            for p, e, s in zip(df["policy"], df["env"], df["seed"], strict=True)
        ],
        index=df.index,
    )
    removed = int((~keep_mask).sum())
    if removed == 0:
        return 0

    remaining = df[keep_mask].reset_index(drop=True)
    _atomic_write_parquet(parquet_path, remaining)
    return removed


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write ``df`` to ``path`` via a tmp sibling + ``os.replace``.

    The tmp file is always cleaned up: on success ``os.replace`` removes
    it (it becomes the final file); on exception we delete it ourselves
    so a stale ``.tmp.parquet`` can't shadow a future write.
    """
    tmp_path = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("failed to clean up tmp parquet at %s", tmp_path)
        raise
