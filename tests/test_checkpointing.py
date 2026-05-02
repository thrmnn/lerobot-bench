"""Tests for ``lerobot_bench.checkpointing``.

Pure pandas + pyarrow — no torch, no env, no GPU. These run in default CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lerobot_bench.checkpointing import (
    RESULT_SCHEMA,
    CellKey,
    append_cell_rows,
    drop_partial_cells,
    load_results,
    plan_resume,
)


def _row(
    policy: str,
    env: str,
    seed: int,
    ep: int,
    **kw: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": ep,
        "success": True,
        "return_": 1.0,
        "n_steps": 10,
        "wallclock_s": 0.5,
        "video_sha256": "",
        "code_sha": "abc",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-01T00:00:00Z",
    }
    base.update(kw)
    return base


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(RESULT_SCHEMA))


# --------------------------------------------------------------------- #
# load_results                                                          #
# --------------------------------------------------------------------- #


def test_load_results_missing_file_returns_empty_with_schema(tmp_path: Path) -> None:
    df = load_results(tmp_path / "missing.parquet")
    assert tuple(df.columns) == RESULT_SCHEMA
    assert len(df) == 0


def test_load_results_existing_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.parquet"
    rows = [_row("p1", "e1", 0, i) for i in range(3)]
    _df(rows).to_parquet(path, index=False)

    df = load_results(path)
    assert len(df) == 3
    assert list(df["episode_index"]) == [0, 1, 2]
    assert df["seed"].dtype == "int64"


def test_load_results_wrong_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=r"unexpected columns") as excinfo:
        load_results(path)
    msg = str(excinfo.value)
    assert "missing=" in msg
    assert "extra=" in msg
    assert "policy" in msg
    assert "foo" in msg


# --------------------------------------------------------------------- #
# plan_resume                                                           #
# --------------------------------------------------------------------- #


def test_plan_resume_no_file_all_pending(tmp_path: Path) -> None:
    cells = [CellKey("p", "e", s) for s in range(3)]
    plan = plan_resume(tmp_path / "missing.parquet", requested_cells=cells, n_episodes=5)
    assert plan.completed_cells == frozenset()
    assert plan.partial_cells == frozenset()
    assert plan.pending_cells == frozenset(cells)
    assert plan.rows_loaded == 0


def test_plan_resume_one_completed_cell(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    rows = [_row("A", "env", 0, i) for i in range(5)]
    _df(rows).to_parquet(path, index=False)

    a = CellKey("A", "env", 0)
    b = CellKey("B", "env", 0)
    plan = plan_resume(path, requested_cells=[a, b], n_episodes=5)

    assert plan.completed_cells == frozenset({a})
    assert plan.partial_cells == frozenset()
    assert plan.pending_cells == frozenset({b})
    assert plan.rows_loaded == 5


def test_plan_resume_partial_cell_classified_correctly(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    rows = [_row("A", "env", 0, i) for i in (0, 1, 2)]
    _df(rows).to_parquet(path, index=False)

    a = CellKey("A", "env", 0)
    plan = plan_resume(path, requested_cells=[a], n_episodes=5)
    assert plan.completed_cells == frozenset()
    assert plan.partial_cells == frozenset({a})
    assert plan.pending_cells == frozenset()


def test_plan_resume_missing_episode_index_treated_as_partial(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    # 4 rows but with a gap: indices {0,1,2,4} when n_episodes=5.
    rows = [_row("A", "env", 0, i) for i in (0, 1, 2, 4)]
    _df(rows).to_parquet(path, index=False)

    a = CellKey("A", "env", 0)
    plan = plan_resume(path, requested_cells=[a], n_episodes=5)
    assert plan.partial_cells == frozenset({a})
    assert plan.completed_cells == frozenset()


def test_plan_resume_extra_episode_indices_treated_as_partial(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    # n_episodes=5 but parquet has 6 rows with indices {0..5}.
    rows = [_row("A", "env", 0, i) for i in range(6)]
    _df(rows).to_parquet(path, index=False)

    a = CellKey("A", "env", 0)
    plan = plan_resume(path, requested_cells=[a], n_episodes=5)
    assert plan.partial_cells == frozenset({a})
    assert plan.completed_cells == frozenset()


def test_plan_resume_ignores_unrequested_cells(tmp_path: Path) -> None:
    """Cells in the parquet but not in requested_cells must not appear in any bucket."""
    path = tmp_path / "r.parquet"
    rows = [_row("A", "env", 0, i) for i in range(5)] + [_row("Z", "env", 9, i) for i in range(5)]
    _df(rows).to_parquet(path, index=False)

    a = CellKey("A", "env", 0)
    plan = plan_resume(path, requested_cells=[a], n_episodes=5)
    assert plan.completed_cells == frozenset({a})
    assert plan.pending_cells == frozenset()
    assert plan.partial_cells == frozenset()
    # rows_loaded reflects raw row count, not just requested.
    assert plan.rows_loaded == 10


def test_plan_resume_rejects_zero_n_episodes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_episodes must be positive"):
        plan_resume(tmp_path / "x.parquet", requested_cells=[], n_episodes=0)


# --------------------------------------------------------------------- #
# append_cell_rows                                                      #
# --------------------------------------------------------------------- #


def test_append_cell_rows_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    rows = _df([_row("A", "env", 0, i) for i in range(5)])
    total = append_cell_rows(path, rows)
    assert total == 5
    assert path.exists()
    df = load_results(path)
    assert len(df) == 5
    assert list(df["episode_index"]) == [0, 1, 2, 3, 4]


def test_append_cell_rows_to_existing(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    a_rows = _df([_row("A", "env", 0, i) for i in range(5)])
    b_rows = _df([_row("B", "env", 0, i) for i in range(5)])
    assert append_cell_rows(path, a_rows) == 5
    assert append_cell_rows(path, b_rows) == 10

    df = load_results(path)
    assert len(df) == 10
    assert set(df["policy"]) == {"A", "B"}


def test_append_cell_rows_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    rows = _df([_row("A", "env", 0, i) for i in range(5)])
    append_cell_rows(path, rows)

    file_size_before = path.stat().st_size
    dup = _df([_row("A", "env", 0, i) for i in range(5)])
    with pytest.raises(ValueError, match=r"duplicate \(policy, env, seed, episode_index\) keys"):
        append_cell_rows(path, dup)

    # File is byte-identical: write was rejected before any tmp swap.
    assert path.stat().st_size == file_size_before
    df = load_results(path)
    assert len(df) == 5


def test_append_cell_rows_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    bad = pd.DataFrame({"policy": ["A"], "env": ["e"], "seed": [0]})
    with pytest.raises(ValueError, match=r"new_rows has wrong columns"):
        append_cell_rows(path, bad)
    assert not path.exists()


def test_append_cell_rows_empty_new_rows_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    seed_rows = _df([_row("A", "env", 0, i) for i in range(3)])
    append_cell_rows(path, seed_rows)
    mtime_before = path.stat().st_mtime_ns

    empty = _df([])
    total = append_cell_rows(path, empty)
    assert total == 3
    # File untouched — mtime unchanged.
    assert path.stat().st_mtime_ns == mtime_before


def test_append_cell_rows_empty_no_existing_file_returns_zero(tmp_path: Path) -> None:
    """Empty new_rows + missing file is a no-op that returns 0 (existing row count)."""
    path = tmp_path / "r.parquet"
    total = append_cell_rows(path, _df([]))
    assert total == 0
    assert not path.exists()


# --------------------------------------------------------------------- #
# drop_partial_cells                                                    #
# --------------------------------------------------------------------- #


def test_drop_partial_cells_removes_only_target(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    a_rows = _df([_row("A", "env", 0, i) for i in range(5)])
    b_rows = _df([_row("B", "env", 0, i) for i in range(5)])
    append_cell_rows(path, a_rows)
    append_cell_rows(path, b_rows)

    removed = drop_partial_cells(path, [CellKey("A", "env", 0)])
    assert removed == 5

    df = load_results(path)
    assert len(df) == 5
    assert set(df["policy"]) == {"B"}


def test_drop_partial_cells_missing_file_returns_zero(tmp_path: Path) -> None:
    removed = drop_partial_cells(tmp_path / "nope.parquet", [CellKey("A", "e", 0)])
    assert removed == 0


def test_drop_partial_cells_empty_cells_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "r.parquet"
    rows = _df([_row("A", "env", 0, i) for i in range(3)])
    append_cell_rows(path, rows)

    mtime_before = path.stat().st_mtime_ns
    removed = drop_partial_cells(path, [])
    assert removed == 0
    assert path.stat().st_mtime_ns == mtime_before


# --------------------------------------------------------------------- #
# Atomicity                                                             #
# --------------------------------------------------------------------- #


def test_atomicity_simulated(tmp_path: Path) -> None:
    """A pre-existing stale .tmp.parquet sibling must not corrupt the write."""
    path = tmp_path / "r.parquet"
    tmp_sibling = path.with_suffix(".tmp.parquet")
    # Plant a stale tmp file that a prior crashed write might leave behind.
    tmp_sibling.write_bytes(b"garbage that is not parquet")

    rows = _df([_row("A", "env", 0, i) for i in range(3)])
    total = append_cell_rows(path, rows)
    assert total == 3

    # After successful os.replace the tmp sibling no longer exists.
    assert not tmp_sibling.exists()
    df = load_results(path)
    assert len(df) == 3
    assert list(df["episode_index"]) == [0, 1, 2]
