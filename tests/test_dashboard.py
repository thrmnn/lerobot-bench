"""Tests for ``dashboard/_helpers.py``.

Mirrors the pattern in ``tests/test_space.py``: we don't import the
Gradio-bearing ``dashboard/app.py`` here because Gradio is not in the
project's ``[dev]`` extras (lives only in ``[space]``). The helpers
module is gradio-free by design, so we can exercise its full surface
in CI without dragging in the heavy dep.

Coverage:

* calibration JSON -> table renderer is pure (mocked file, asserts
  columns + downscope reason buckets).
* sweep-manifest -> progress-table snapshot from a synthetic
  ``results/sweep-*/sweep_manifest.json`` layout.
* video file scanner handles missing dirs gracefully + parses the
  ``{policy}__{env}__seed{N}__ep{NNN}.mp4`` filename schema.
* AST guard: ``_helpers.py`` does not import gradio at module load.
"""

from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# dashboard/ and space/ both ship a top-level ``_helpers`` module --
# directly importing ``_helpers`` after sys.path manipulation would
# collide with ``tests/test_space.py``'s same trick (whichever test
# file collects first wins). Load the dashboard's _helpers under a
# unique module name via importlib.util so the two suites coexist.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_DIR = _REPO_ROOT / "dashboard"
_HELPERS_PATH = _DASHBOARD_DIR / "_helpers.py"

_spec = importlib.util.spec_from_file_location("dashboard_helpers", _HELPERS_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"could not build module spec for {_HELPERS_PATH}"
)
_dashboard_helpers = importlib.util.module_from_spec(_spec)
sys.modules["dashboard_helpers"] = _dashboard_helpers
_spec.loader.exec_module(_dashboard_helpers)

CALIBRATION_COLUMNS = _dashboard_helpers.CALIBRATION_COLUMNS
CELL_STATUS_DONE = _dashboard_helpers.CELL_STATUS_DONE
CELL_STATUS_FAILED = _dashboard_helpers.CELL_STATUS_FAILED
CELL_STATUS_QUEUED = _dashboard_helpers.CELL_STATUS_QUEUED
CELL_STATUS_RUNNING = _dashboard_helpers.CELL_STATUS_RUNNING
CELL_STATUS_SKIPPED = _dashboard_helpers.CELL_STATUS_SKIPPED
HIGH_VRAM_THRESHOLD_MB = _dashboard_helpers.HIGH_VRAM_THRESHOLD_MB
PROGRESS_COLUMNS = _dashboard_helpers.PROGRESS_COLUMNS
SLOW_MS_PER_STEP_THRESHOLD = _dashboard_helpers.SLOW_MS_PER_STEP_THRESHOLD
VERY_HIGH_VRAM_THRESHOLD_MB = _dashboard_helpers.VERY_HIGH_VRAM_THRESHOLD_MB
VERY_SLOW_MS_PER_STEP_THRESHOLD = _dashboard_helpers.VERY_SLOW_MS_PER_STEP_THRESHOLD
build_calibration_table = _dashboard_helpers.build_calibration_table
build_progress_table = _dashboard_helpers.build_progress_table
classify_log_line = _dashboard_helpers.classify_log_line
clear_video_cache = _dashboard_helpers.clear_video_cache
column_glossary_markdown = _dashboard_helpers.column_glossary_markdown
compute_manifest_progress = _dashboard_helpers.compute_manifest_progress
discover_sweep_logs = _dashboard_helpers.discover_sweep_logs
discover_sweep_runs = _dashboard_helpers.discover_sweep_runs
downscope_reason = _dashboard_helpers.downscope_reason
find_latest_calibration = _dashboard_helpers.find_latest_calibration
find_video_path = _dashboard_helpers.find_video_path
format_log_lines_html = _dashboard_helpers.format_log_lines_html
format_relative_time = _dashboard_helpers.format_relative_time
load_calibration_report = _dashboard_helpers.load_calibration_report
load_manifest = _dashboard_helpers.load_manifest
methodology_markdown = _dashboard_helpers.methodology_markdown
parse_video_filename = _dashboard_helpers.parse_video_filename
per_tab_intro_markdown = _dashboard_helpers.per_tab_intro_markdown
persistent_header_markdown = _dashboard_helpers.persistent_header_markdown
resolved_paths_banner_markdown = _dashboard_helpers.resolved_paths_banner_markdown
scan_video_index = _dashboard_helpers.scan_video_index
summarize_log = _dashboard_helpers.summarize_log
tail_log_lines = _dashboard_helpers.tail_log_lines
video_index_options = _dashboard_helpers.video_index_options
V1_RUNNABLE_CELLS = _dashboard_helpers.V1_RUNNABLE_CELLS
V1_SEEDS_PER_CELL = _dashboard_helpers.V1_SEEDS_PER_CELL
V1_TOTAL_SEED_ENTRIES = _dashboard_helpers.V1_TOTAL_SEED_ENTRIES

# --------------------------------------------------------------------- #
# Helpers to build synthetic manifests / calibration reports            #
# --------------------------------------------------------------------- #


def _manifest_entry(
    *,
    policy: str,
    env: str,
    seed: int,
    status: str,
    n_episodes: int = 50,
    started_utc: str | None = None,
    finished_utc: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "seed_idx": seed,
        "n_episodes": n_episodes,
        "status": status,
        "exit_code": exit_code,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "stderr_tail": "",
        "skip_reason": "",
    }


def _manifest(cells: list[dict[str, Any]], *, finished: str | None = None) -> dict[str, Any]:
    return {
        "started_utc": "2026-05-12T03:00:00+00:00",
        "finished_utc": finished,
        "code_sha": "deadbeefcafe",
        "lerobot_version": "0.5.1",
        "config_path": "configs/sweep_full.yaml",
        "cells": cells,
    }


def _cal_cell(
    *,
    policy: str,
    env: str,
    status: str = "ok",
    mean_ms: float = 50.0,
    p95_ms: float = 60.0,
    vram_mb: float = 2000.0,
    seeds: int = 5,
    episodes: int = 50,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "n_steps_measured": 20,
        "mean_ms_per_step": mean_ms,
        "p95_ms_per_step": p95_ms,
        "vram_peak_mb": vram_mb,
        "status": status,
        "error": error,
        "recommended": {"seeds": seeds, "episodes": episodes} if status == "ok" else None,
    }


# --------------------------------------------------------------------- #
# Progress table                                                        #
# --------------------------------------------------------------------- #


def test_build_progress_table_empty_input_returns_canonical_columns() -> None:
    """Premortem: missing/empty manifest must still render canonical columns."""
    table = build_progress_table({})
    assert list(table.columns) == list(PROGRESS_COLUMNS)
    assert len(table) == 0

    table2 = build_progress_table({"cells": []})
    assert list(table2.columns) == list(PROGRESS_COLUMNS)
    assert len(table2) == 0


def test_build_progress_table_snapshot_synthetic_layout() -> None:
    """Snapshot test of the progress roll-up against a hand-crafted manifest.

    Two policies x one env, three seeds each: ``smolvla`` has 1 done +
    1 running + 1 pending; ``xvla`` has 1 failed + 2 done. The
    snapshot verifies the cell-status logic and the seeds-done count.
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="smolvla",
                env="pusht",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(
                policy="smolvla",
                env="pusht",
                seed=1,
                status="pending",
                started_utc="2026-05-12T03:10:00+00:00",
            ),
            _manifest_entry(
                policy="smolvla",
                env="pusht",
                seed=2,
                status="pending",
            ),
            _manifest_entry(
                policy="xvla",
                env="pusht",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:20:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(
                policy="xvla",
                env="pusht",
                seed=1,
                status="failed",
                started_utc="2026-05-12T03:20:00+00:00",
                finished_utc="2026-05-12T03:21:00+00:00",
                exit_code=4,
            ),
            _manifest_entry(
                policy="xvla",
                env="pusht",
                seed=2,
                status="completed",
                started_utc="2026-05-12T03:21:00+00:00",
                finished_utc="2026-05-12T03:31:00+00:00",
                exit_code=0,
            ),
        ]
    )

    # Pin "now" so the ETA computation is deterministic.
    now = dt.datetime(2026, 5, 12, 3, 11, 0, tzinfo=dt.UTC)
    table = build_progress_table(manifest, now_utc=now)

    assert list(table.columns) == list(PROGRESS_COLUMNS)
    assert len(table) == 2

    by_policy = {row.policy: row for row in table.itertuples(index=False)}

    # xvla cell: 1 failed seed -> cell-status "failed", sorted first.
    xvla = by_policy["xvla"]
    assert xvla.status == CELL_STATUS_FAILED
    assert xvla.seeds_total == 3
    assert xvla.seeds_done == 2  # two completed seeds
    assert xvla.episodes_total == 150

    # smolvla cell: 1 done + 1 running + 1 pending -> "running".
    smolvla = by_policy["smolvla"]
    assert smolvla.status == CELL_STATUS_RUNNING
    assert smolvla.seeds_total == 3
    assert smolvla.seeds_done == 1
    # ETA: mean of the one completed seed (10 min) * 2 remaining = 20 min.
    assert smolvla.eta_minutes == pytest.approx(20.0, rel=0.05)


def test_build_progress_table_all_queued_renders_zero_eta() -> None:
    """A fresh sweep with no seeds started yet -> ETA is 0 (no baseline)."""
    manifest = _manifest(
        [
            _manifest_entry(policy="p", env="e", seed=0, status="pending"),
            _manifest_entry(policy="p", env="e", seed=1, status="pending"),
        ]
    )
    table = build_progress_table(manifest)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["status"] == CELL_STATUS_QUEUED
    assert row["seeds_done"] == 0
    assert row["eta_minutes"] == pytest.approx(0.0)


def test_build_progress_table_all_skipped_is_skipped_status() -> None:
    """All seeds skipped (incompat) -> cell-status surfaces as ``skipped``."""
    manifest = _manifest(
        [
            _manifest_entry(policy="p", env="aloha", seed=0, status="skipped"),
            _manifest_entry(policy="p", env="aloha", seed=1, status="skipped"),
        ]
    )
    table = build_progress_table(manifest)
    assert len(table) == 1
    assert table.iloc[0]["status"] == CELL_STATUS_SKIPPED


def test_build_progress_table_done_when_completed_and_skipped_mix() -> None:
    """Mix of completed + skipped (no failures, none pending) -> done."""
    manifest = _manifest(
        [
            _manifest_entry(
                policy="p",
                env="e",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:05:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(policy="p", env="e", seed=1, status="skipped"),
        ]
    )
    table = build_progress_table(manifest)
    assert len(table) == 1
    assert table.iloc[0]["status"] == CELL_STATUS_DONE


def test_load_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing manifest -> {}, never raises."""
    assert load_manifest(tmp_path / "nope.json") == {}


def test_load_manifest_invalid_json_returns_empty(tmp_path: Path) -> None:
    """Malformed manifest -> {}, dashboard tab still renders."""
    bad = tmp_path / "sweep_manifest.json"
    bad.write_text("{ this is : not json")
    assert load_manifest(bad) == {}


def test_discover_sweep_runs_synthetic_layout(tmp_path: Path) -> None:
    """Two runs in two subdirs are discovered, newest first."""
    older_dir = tmp_path / "sweep-old"
    older_dir.mkdir()
    (older_dir / "sweep_manifest.json").write_text(
        json.dumps(_manifest([], finished="2026-05-10T00:00:00+00:00"))
    )

    newer_dir = tmp_path / "sweep-new"
    newer_dir.mkdir()
    # Newer started_utc.
    newer = _manifest([])
    newer["started_utc"] = "2026-05-12T03:00:00+00:00"
    (newer_dir / "sweep_manifest.json").write_text(json.dumps(newer))

    runs = discover_sweep_runs(tmp_path)
    assert [r.name for r in runs] == ["sweep-new", "sweep-old"]
    # Labels include the running/done marker.
    assert "[done]" in runs[1].label
    assert "[running]" in runs[0].label  # newer has no finished_utc


def test_discover_sweep_runs_missing_root_returns_empty(tmp_path: Path) -> None:
    """Pointing at a non-existent results root -> [] (not an exception)."""
    assert discover_sweep_runs(tmp_path / "no-such-dir") == []


# --------------------------------------------------------------------- #
# Calibration table                                                     #
# --------------------------------------------------------------------- #


def test_build_calibration_table_empty_input_returns_canonical_columns() -> None:
    """Pure renderer contract: empty report -> empty table with canonical cols."""
    table = build_calibration_table({})
    assert list(table.columns) == list(CALIBRATION_COLUMNS)
    assert len(table) == 0


def test_build_calibration_table_columns_and_row_count(tmp_path: Path) -> None:
    """Mock the file via a written JSON: the renderer projects every cell."""
    report = {
        "timestamp_utc": "2026-05-12T03:00:00+00:00",
        "git_sha": "abc123",
        "lerobot_version": "0.5.1",
        "cells": [
            _cal_cell(policy="cheap", env="pusht", mean_ms=20.0, vram_mb=500.0),
            _cal_cell(
                policy="smolvla", env="pusht", mean_ms=150.0, vram_mb=3000.0
            ),  # slow ms -> cut episodes
            _cal_cell(
                policy="xvla", env="aloha", mean_ms=80.0, vram_mb=6800.0
            ),  # high vram -> cut episodes
            _cal_cell(policy="pi0_huge", env="libero", status="oom"),  # OOM -> drop
        ],
    }
    path = tmp_path / "calibration-20260512.json"
    path.write_text(json.dumps(report))
    loaded = load_calibration_report(path)
    table = build_calibration_table(loaded)

    assert list(table.columns) == list(CALIBRATION_COLUMNS)
    assert len(table) == 4

    # The OOM cell sorts first (status_order: oom -> error -> skipped -> ok).
    assert table.iloc[0]["policy"] == "pi0_huge"
    assert table.iloc[0]["status"] == "oom"
    assert "OOM" in table.iloc[0]["reason"]

    # Spot-check the cheap cell.
    cheap = table[table["policy"] == "cheap"].iloc[0]
    assert cheap["recommended_seeds"] == 5
    assert cheap["recommended_episodes"] == 50
    assert cheap["reason"] == "within budget"


def test_downscope_reason_buckets_match_thresholds() -> None:
    """Verify the reason string aligns with each auto_downscope bucket."""
    # ok + within budget
    assert (
        downscope_reason(_cal_cell(policy="p", env="e", mean_ms=10.0, vram_mb=100.0))
        == "within budget"
    )
    # ok + slow ms -> cut episodes
    reason = downscope_reason(
        _cal_cell(policy="p", env="e", mean_ms=SLOW_MS_PER_STEP_THRESHOLD + 1.0, vram_mb=100.0)
    )
    assert "cut episodes" in reason

    # ok + very slow ms -> cut seeds
    reason = downscope_reason(
        _cal_cell(policy="p", env="e", mean_ms=VERY_SLOW_MS_PER_STEP_THRESHOLD + 1.0, vram_mb=100.0)
    )
    assert "cut seeds" in reason

    # ok + high vram -> cut episodes
    reason = downscope_reason(
        _cal_cell(policy="p", env="e", mean_ms=10.0, vram_mb=HIGH_VRAM_THRESHOLD_MB + 1.0)
    )
    assert "cut episodes" in reason
    assert "VRAM" in reason or "vram" in reason.lower()

    # ok + very high vram -> cut seeds
    reason = downscope_reason(
        _cal_cell(policy="p", env="e", mean_ms=10.0, vram_mb=VERY_HIGH_VRAM_THRESHOLD_MB + 1.0)
    )
    assert "cut seeds" in reason

    # status=oom
    assert "OOM" in downscope_reason(_cal_cell(policy="p", env="e", status="oom"))

    # status=skipped
    assert "not runnable" in downscope_reason(_cal_cell(policy="p", env="e", status="skipped"))

    # status=error -- includes the error message tail
    err_cell = _cal_cell(policy="p", env="e", status="error", error="missing runtime: lerobot")
    assert "lerobot" in downscope_reason(err_cell)


def test_find_latest_calibration_picks_newest_by_filename(tmp_path: Path) -> None:
    """Filename date prefix is the ordering key (no need to read JSON)."""
    (tmp_path / "calibration-20260501.json").write_text("{}")
    (tmp_path / "calibration-20260512.json").write_text("{}")
    (tmp_path / "calibration-20260506.json").write_text("{}")

    latest = find_latest_calibration(tmp_path)
    assert latest is not None
    assert latest.name == "calibration-20260512.json"


def test_find_latest_calibration_missing_returns_none(tmp_path: Path) -> None:
    assert find_latest_calibration(tmp_path / "nope") is None
    assert find_latest_calibration(tmp_path) is None  # exists but empty


def test_load_calibration_report_malformed_returns_empty(tmp_path: Path) -> None:
    bad = tmp_path / "calibration-bad.json"
    bad.write_text("not-json {{")
    assert load_calibration_report(bad) == {}


# --------------------------------------------------------------------- #
# Video index                                                           #
# --------------------------------------------------------------------- #


def test_parse_video_filename_canonical() -> None:
    """The exact schema written by ``scripts/run_one.py``."""
    parsed = parse_video_filename("smolvla__pusht__seed0__ep007.mp4")
    assert parsed == ("smolvla", "pusht", 0, 7)


def test_parse_video_filename_no_zero_pad() -> None:
    parsed = parse_video_filename("cheap__aloha__seed12__ep3.mp4")
    assert parsed == ("cheap", "aloha", 12, 3)


def test_parse_video_filename_rejects_garbage() -> None:
    """Random MP4s in the same tree must not pollute the index."""
    assert parse_video_filename("rendered_figure.mp4") is None
    assert parse_video_filename("smolvla_pusht_seed0_ep1.mp4") is None  # single underscores
    assert parse_video_filename("p__e__seedX__ep0.mp4") is None  # non-int seed
    assert parse_video_filename("not-an-mp4.png") is None
    assert parse_video_filename("__e__seed0__ep0.mp4") is None  # empty policy


def test_scan_video_index_missing_roots_returns_empty(tmp_path: Path) -> None:
    """Premortem: the Windows-mount might be unmounted. Don't crash."""
    clear_video_cache()
    index = scan_video_index(roots=[tmp_path / "no-such-dir", tmp_path / "also-missing"])
    assert index.n_videos == 0
    assert index.by_key == {}


def test_scan_video_index_synthetic_layout(tmp_path: Path) -> None:
    """A handful of canonical MP4s on disk -> indexed; garbage MP4 skipped."""
    clear_video_cache()
    videos_dir = tmp_path / "sweep-mini" / "videos"
    videos_dir.mkdir(parents=True)
    (videos_dir / "smolvla__pusht__seed0__ep000.mp4").write_bytes(b"")
    (videos_dir / "smolvla__pusht__seed0__ep001.mp4").write_bytes(b"")
    (videos_dir / "cheap__aloha__seed1__ep042.mp4").write_bytes(b"")
    # Garbage file in the same tree (shouldn't be indexed).
    (videos_dir / "some_plot.mp4").write_bytes(b"")

    index = scan_video_index(roots=[tmp_path])
    assert index.n_videos == 3

    options = video_index_options(index)
    assert options["policy"] == ["cheap", "smolvla"]
    assert options["env"] == ["aloha", "pusht"]
    # Seeds are stringified for the dropdown but sorted as ints.
    assert options["seed"] == ["0", "1"]


def test_find_video_path_hit_and_miss(tmp_path: Path) -> None:
    """The dropdown lookup hits when the file exists; otherwise returns None."""
    clear_video_cache()
    videos_dir = tmp_path / "sweep-mini" / "videos"
    videos_dir.mkdir(parents=True)
    target = videos_dir / "smolvla__pusht__seed0__ep007.mp4"
    target.write_bytes(b"")

    index = scan_video_index(roots=[tmp_path])
    hit = find_video_path(index, policy="smolvla", env="pusht", seed="0", episode="7")
    assert hit == target

    miss = find_video_path(index, policy="xvla", env="pusht", seed="0", episode="7")
    assert miss is None

    # Bad seed string -> None, not crash.
    assert find_video_path(index, policy="smolvla", env="pusht", seed="abc", episode="7") is None

    # Missing dropdown values -> None.
    assert find_video_path(index, policy=None, env="pusht", seed="0", episode="7") is None
    assert find_video_path(index, policy="smolvla", env="pusht", seed=None, episode="7") is None


def test_scan_video_index_caches_by_root(tmp_path: Path) -> None:
    """Two scans with the same roots reuse the cache; clear_video_cache resets it."""
    clear_video_cache()
    videos_dir = tmp_path / "v"
    videos_dir.mkdir()
    (videos_dir / "p__e__seed0__ep0.mp4").write_bytes(b"")

    first = scan_video_index(roots=[tmp_path])
    second = scan_video_index(roots=[tmp_path])
    assert first is second  # same lru entry

    # Add a new file post-scan; without clearing, the cache hides it.
    (videos_dir / "p__e__seed0__ep1.mp4").write_bytes(b"")
    cached = scan_video_index(roots=[tmp_path])
    assert cached.n_videos == 1  # stale

    clear_video_cache()
    fresh = scan_video_index(roots=[tmp_path])
    assert fresh.n_videos == 2


# --------------------------------------------------------------------- #
# Misc helpers                                                          #
# --------------------------------------------------------------------- #


def test_format_relative_time_renders_recent() -> None:
    now = dt.datetime(2026, 5, 12, 3, 30, 0, tzinfo=dt.UTC)
    assert format_relative_time("2026-05-12T03:29:30+00:00", now_utc=now) == "30s ago"
    assert format_relative_time("2026-05-12T03:25:00+00:00", now_utc=now) == "5m ago"
    assert format_relative_time("2026-05-12T01:30:00+00:00", now_utc=now) == "2h ago"
    assert format_relative_time("2026-05-10T03:30:00+00:00", now_utc=now) == "2d ago"


def test_format_relative_time_empty_and_invalid_input() -> None:
    assert format_relative_time("") == ""
    assert format_relative_time("garbage") == ""


# --------------------------------------------------------------------- #
# Event log: line classification / discovery / tail / summary           #
# --------------------------------------------------------------------- #


def test_classify_log_line_dispatches() -> None:
    """Each canonical log shape -> the right bucket.

    The dispatch line shape matches ``scripts/run_sweep.py``'s
    ``[i/N] dispatch policy/env/seedK`` format; ``success_rate=`` is
    the run-one summary line; FAILED + Traceback + Killed all bucket
    as ``"error"``; watchdog ``BREACH`` is its own bucket.
    """
    dispatch = (
        "2026-05-12 17:11:35,207 [INFO] run-sweep: [1/110] dispatch "
        "act/aloha_transfer_cube/seed0 (n_episodes=50, timeout_s=14400.0)"
    )
    assert classify_log_line(dispatch) == "dispatch"

    success = (
        "2026-05-12 17:25:14,103 [INFO] run-one: success_rate=0.42 (21/50) "
        "mean_return=12.5 wall_s=684.1"
    )
    assert classify_log_line(success) == "success"

    failed = (
        "2026-05-12 17:23:06,884 [WARNING] run-sweep:     -> FAILED (exit=-9). "
        "Continuing to next cell."
    )
    assert classify_log_line(failed) == "error"

    traceback = 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>'
    assert classify_log_line(traceback) == "error"

    killed = "2026-05-12 17:30:00,000 [ERROR] run-one: subprocess Killed by OOM"
    assert classify_log_line(killed) == "error"

    breach = "2026-05-12 17:11:34,001 [INFO] watchdog: BREACH cap=18G used=18.5G"
    assert classify_log_line(breach) == "breach"


def test_classify_log_line_handles_unknown() -> None:
    """A pre-flight banner / blank line / random text -> "other"."""
    assert classify_log_line("") == "other"
    assert classify_log_line("pre-flight: RAM total=32096MB used=3264MB") == "other"
    assert classify_log_line("----") == "other"
    assert classify_log_line("2026-05-12 17:00:00 [INFO] config: loaded") == "other"


def test_classify_log_line_breach_wins_over_error() -> None:
    """A BREACH line that *also* contains "error" still buckets as breach.

    Premortem: a watchdog message that mentions an inner Python error
    string ("BREACH ... last error: ...") must not be miscoloured as
    a generic ERROR -- the operator distinguishes a cgroup OOM (action:
    cap-up) from a Python traceback (action: fix code).
    """
    line = "2026-05-12 17:00:00,000 [INFO] watchdog: BREACH cap=18G last_error=foo"
    assert classify_log_line(line) == "breach"


def test_discover_sweep_logs_sorts_newest_first(tmp_path: Path) -> None:
    """Files are mtime-sorted, newest first, regardless of filename order.

    We intentionally use mtimes that don't match lexicographic order so
    the assertion catches the (wrong) fallback to filename sorting.
    """
    logs_dir = tmp_path
    a = logs_dir / "sweep-A.log"
    b = logs_dir / "sweep-B.log"
    c = logs_dir / "sweep-C.log"
    a.write_text("dispatch act/x/seed0\n")
    b.write_text("dispatch act/x/seed1\n")
    c.write_text("dispatch act/x/seed2\n")

    import os as _os

    # B is newest, A is middle, C is oldest -- on purpose alphabet vs mtime mismatch.
    _os.utime(c, (1, 1000))
    _os.utime(a, (1, 2000))
    _os.utime(b, (1, 3000))

    discovered = discover_sweep_logs(logs_dir)
    assert [p.name for p in discovered] == ["sweep-B.log", "sweep-A.log", "sweep-C.log"]


def test_discover_sweep_logs_missing_root_returns_empty(tmp_path: Path) -> None:
    """Pointing at a non-existent log root -> [] (not an exception)."""
    assert discover_sweep_logs(tmp_path / "no-such-dir") == []


def test_discover_sweep_logs_ignores_non_sweep_files(tmp_path: Path) -> None:
    """Only ``sweep-*.log`` is picked; other files (calibrate logs, .gz) are dropped."""
    (tmp_path / "sweep-20260512.log").write_text("x")
    (tmp_path / "calibrate-20260512.log").write_text("x")
    (tmp_path / "sweep-20260511.log.gz").write_bytes(b"\x1f\x8b")
    (tmp_path / "README.md").write_text("x")
    names = [p.name for p in discover_sweep_logs(tmp_path)]
    assert names == ["sweep-20260512.log"]


def test_tail_log_lines_handles_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing file -> [] without raising."""
    assert tail_log_lines(tmp_path / "nope.log") == []
    assert tail_log_lines(tmp_path / "nope.log", n=10) == []


def test_tail_log_lines_returns_last_n(tmp_path: Path) -> None:
    """Tail returns exactly the last n lines; smaller files return everything."""
    log = tmp_path / "sweep.log"
    log.write_text("\n".join(f"line {i}" for i in range(20)) + "\n")

    assert tail_log_lines(log, n=5) == [f"line {i}" for i in range(15, 20)]
    assert tail_log_lines(log, n=100) == [f"line {i}" for i in range(20)]


def test_tail_log_lines_handles_empty_file(tmp_path: Path) -> None:
    """An empty log file -> [] (not a list with one empty string)."""
    log = tmp_path / "sweep.log"
    log.write_text("")
    assert tail_log_lines(log, n=10) == []


def test_summarize_log_counts() -> None:
    """Synthetic 10-line log -> expected per-bucket counts.

    Mix: 4 dispatch, 2 success, 2 error, 1 breach, 1 other. The total
    equals the input length so the header counter never drifts.
    """
    lines = [
        "2026-05-12 17:00:00 [INFO] run-sweep: [1/10] dispatch act/pusht/seed0 (n=50)",
        "2026-05-12 17:00:01 [INFO] run-sweep: [2/10] dispatch act/pusht/seed1 (n=50)",
        "2026-05-12 17:00:02 [INFO] run-sweep: [3/10] dispatch act/pusht/seed2 (n=50)",
        "2026-05-12 17:00:03 [INFO] run-sweep: [4/10] dispatch act/pusht/seed3 (n=50)",
        "2026-05-12 17:05:00 [INFO] run-one: success_rate=0.42 (21/50)",
        "2026-05-12 17:05:30 [INFO] run-one: success_rate=0.55 (27/50)",
        "2026-05-12 17:10:00 [WARNING] run-sweep: -> FAILED (exit=-9)",
        "2026-05-12 17:11:00 [ERROR] Traceback (most recent call last):",
        "2026-05-12 17:12:00 [INFO] watchdog: BREACH cap=18G",
        "----",
    ]
    counts = summarize_log(lines)
    assert counts == {
        "dispatch": 4,
        "success": 2,
        "error": 2,
        "breach": 1,
        "other": 1,
        "total": 10,
    }


def test_summarize_log_empty_returns_zero_counts() -> None:
    """An empty iterable yields zero counts for every key (never KeyError)."""
    counts = summarize_log([])
    assert counts == {
        "dispatch": 0,
        "success": 0,
        "error": 0,
        "breach": 0,
        "other": 0,
        "total": 0,
    }


def test_event_log_tab_renders_synthetic_log() -> None:
    """Snapshot-style: 3 lines -> colour-wrapped HTML in the expected order.

    The renderer is purely a function of (lines, categories), so this
    is a deterministic snapshot. We assert each colour appears once and
    the original (HTML-escaped) text is present in the output.
    """
    lines = [
        "2026-05-12 17:11:35,207 [INFO] run-sweep: [1/110] dispatch act/x/seed0 (n=50)",
        "2026-05-12 17:25:14,103 [INFO] run-one: success_rate=0.42 (21/50)",
        "2026-05-12 17:30:00,000 [INFO] watchdog: BREACH cap=18G",
    ]
    rendered = format_log_lines_html(lines)
    # One <span> per line, in order.
    assert rendered.count("<span") == 3
    # Each canonical colour appears.
    assert "#3b82f6" in rendered  # dispatch (blue)
    assert "#16a34a" in rendered  # success (green)
    assert "#dc2626" in rendered  # breach (red)
    # Original line text survives (modulo HTML-escaping, which doesn't
    # touch this content because there are no <, >, & characters here).
    for line in lines:
        assert line in rendered


def test_event_log_tab_filters_by_category() -> None:
    """``categories={"error", "breach"}`` -> only those lines render."""
    lines = [
        "dispatch act/x/seed0",
        "run-one: success_rate=0.5 (5/10)",
        "watchdog: BREACH cap=18G",
        "[ERROR] Traceback",
    ]
    rendered = format_log_lines_html(lines, categories={"error", "breach"})
    assert rendered.count("<span") == 2
    assert "BREACH" in rendered
    assert "Traceback" in rendered
    assert "success_rate" not in rendered
    assert "dispatch" not in rendered


def test_event_log_tab_html_escapes_traceback_brackets() -> None:
    """A Python traceback with ``<class 'X'>`` must not inject markup.

    Premortem: an unescaped traceback line containing ``<script>`` could
    inject JS into the dashboard. We verify the angle brackets get
    escaped before the colour span is added.
    """
    line = "Traceback: <class 'RuntimeError'>: oh no"
    rendered = format_log_lines_html([line])
    # The literal ``<class`` must NOT appear as raw markup in the output;
    # it should be ``&lt;class``.
    assert "&lt;class" in rendered
    assert "<class 'RuntimeError'>" not in rendered


# --------------------------------------------------------------------- #
# Explainability layer: persistent header, per-tab intros, About tab    #
# --------------------------------------------------------------------- #


def test_persistent_header_shows_current_results_dir(tmp_path: Path) -> None:
    """The persistent header must contain the resolved results-dir string.

    Premortem: a misconfigured ``DASHBOARD_RESULTS_DIR`` (pointing at
    an empty worktree results/) is the most common dashboard failure
    mode; the persistent header is the first place the operator sees
    it. We pin the string contents so a refactor that drops the path
    from the header gets caught immediately.
    """
    fake_results = tmp_path / "fake-results"
    fake_results.mkdir()
    fake_logs = tmp_path / "fake-logs"
    fake_logs.mkdir()

    rendered = persistent_header_markdown(
        results_dir=fake_results,
        logs_dir=fake_logs,
    )
    # The resolved path literally appears so a wrong-dir misconfig is obvious.
    assert str(fake_results) in rendered
    assert str(fake_logs) in rendered
    # Project pitch lives in the header too -- one-screen elevator pitch.
    assert "lerobot-bench" in rendered
    assert "Pi0" in rendered or "pi0" in rendered.lower()
    # v1 scope numbers must surface for the "what is this?" answer.
    assert str(V1_RUNNABLE_CELLS) in rendered
    assert str(V1_SEEDS_PER_CELL) in rendered

    # Resolved-paths banner mirrors the same values one row down so
    # the operator can spot a wrong-dir misconfig without parsing the
    # full header.
    banner = resolved_paths_banner_markdown(
        results_dir=fake_results,
        logs_dir=fake_logs,
    )
    assert "DASHBOARD_RESULTS_DIR" in banner
    assert "DASHBOARD_LOGS_DIR" in banner
    assert str(fake_results) in banner
    assert str(fake_logs) in banner


def test_persistent_header_progress_badge_with_in_flight_manifest(
    tmp_path: Path,
) -> None:
    """An in-flight manifest -> the header badge counts done / running / failed.

    Two seeds: one completed, one running (started_utc set but
    pending). The badge should say "1/110 done" plus "1 running".
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="pusht",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(
                policy="act",
                env="pusht",
                seed=1,
                status="pending",
                started_utc="2026-05-12T03:10:00+00:00",
            ),
        ]
    )
    run_dir = tmp_path / "sweep-full"
    run_dir.mkdir()
    manifest_path = run_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    counts = compute_manifest_progress(manifest_path)
    assert counts["completed"] == 1
    assert counts["running"] == 1
    assert counts["pending"] == 1
    assert counts["total"] == 2

    rendered = persistent_header_markdown(
        results_dir=tmp_path,
        logs_dir=tmp_path,
        manifest_path=manifest_path,
    )
    # The denominator falls back to the v1 110-entry target when the
    # manifest's total < v1 (which is the case for this 2-entry
    # synthetic manifest); the running / failed counters still show.
    assert "1/" in rendered
    assert "running" in rendered


def test_compute_manifest_progress_handles_missing_manifest(tmp_path: Path) -> None:
    """Missing manifest -> zero counts, never raises."""
    counts = compute_manifest_progress(tmp_path / "no-such-file.json")
    assert counts == {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "pending": 0,
        "running": 0,
        "total": 0,
    }
    # None path -> same shape.
    assert compute_manifest_progress(None)["total"] == 0


def test_per_tab_intro_thresholds_match_calibrate_py() -> None:
    """The calibration tab intro must reference the actual auto_downscope thresholds.

    Pulled either from ``scripts.calibrate`` (preferred -- one source
    of truth) or via ``ast`` parsing if the script can't be imported
    in the test env (which is the case in the dashboard's slim test
    environment). The rendered prose must contain the live integer
    value so a future threshold tweak forces a docs update.
    """
    # First try importing the real constant from scripts/calibrate.py.
    # The scripts package imports lerobot_bench which has heavy deps;
    # fall back to AST extraction if the import fails.
    expected_slow_ms: float | None = None
    expected_high_vram: float | None = None
    try:
        from scripts.calibrate import (  # type: ignore[import-not-found]
            HIGH_VRAM_THRESHOLD_MB as _real_high_vram,  # noqa: N811
        )
        from scripts.calibrate import (
            SLOW_MS_PER_STEP_THRESHOLD as _real_slow_ms,  # noqa: N811
        )

        expected_slow_ms = float(_real_slow_ms)
        expected_high_vram = float(_real_high_vram)
    except ImportError:
        # Fallback: parse the constants out of scripts/calibrate.py
        # via ast so the test still pins the source of truth without
        # importing the heavy lerobot_bench package.
        src = (_REPO_ROOT / "scripts" / "calibrate.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if not isinstance(node.value, ast.Constant):
                    continue
                if target.id == "SLOW_MS_PER_STEP_THRESHOLD":
                    expected_slow_ms = float(node.value.value)
                elif target.id == "HIGH_VRAM_THRESHOLD_MB":
                    expected_high_vram = float(node.value.value)

    assert expected_slow_ms is not None, "could not resolve SLOW_MS_PER_STEP_THRESHOLD"
    assert expected_high_vram is not None, "could not resolve HIGH_VRAM_THRESHOLD_MB"

    # Dashboard copy of the constants must match the source of truth.
    assert pytest.approx(expected_slow_ms) == SLOW_MS_PER_STEP_THRESHOLD
    assert pytest.approx(expected_high_vram) == HIGH_VRAM_THRESHOLD_MB

    # Rendered intro must mention the live integer value so a future
    # threshold tweak forces a docs update.
    intro = per_tab_intro_markdown("calibration")
    assert f"{expected_slow_ms:.0f}" in intro
    assert f"{expected_high_vram:.0f}" in intro
    # Also mention the very-slow / very-high tier so the operator sees
    # both downscope buckets named in the prose.
    assert f"{VERY_SLOW_MS_PER_STEP_THRESHOLD:.0f}" in intro
    assert f"{VERY_HIGH_VRAM_THRESHOLD_MB:.0f}" in intro


def test_per_tab_intro_all_four_tabs_render() -> None:
    """Each of the four data tabs has a non-empty intro markdown block.

    Plus an unknown tab key returns the empty string (not an
    exception) -- callers should treat that as a wiring bug, not a
    runtime error.
    """
    for tab in ("progress", "calibration", "rollouts", "events"):
        rendered = per_tab_intro_markdown(tab)
        assert len(rendered) > 100, f"intro for tab={tab!r} is suspiciously short"
        # All four start with the "What this tab shows" framing.
        assert "What" in rendered and "tab" in rendered
        # All four name a "Good shape" expectation (the operator-facing
        # "is it healthy?" answer).
        assert "Good shape" in rendered or "good" in rendered.lower()

    assert per_tab_intro_markdown("unknown-tab") == ""


def test_column_glossary_progress_and_calibration() -> None:
    """Both data-bearing tabs render a one-line glossary; other tabs none."""
    progress = column_glossary_markdown("progress")
    assert "policy" in progress and "env" in progress
    assert "eta_minutes" in progress
    assert "seeds_done" in progress

    calibration = column_glossary_markdown("calibration")
    assert "vram_peak_mb" in calibration
    assert "mean_step_ms" in calibration
    # The reason-column note must reference the same thresholds the
    # intro mentions -- catches a drift between intro and glossary.
    assert f"{SLOW_MS_PER_STEP_THRESHOLD:.0f}" in calibration
    assert f"{HIGH_VRAM_THRESHOLD_MB:.0f}" in calibration

    # Tabs without dataframes get the empty string.
    assert column_glossary_markdown("rollouts") == ""
    assert column_glossary_markdown("events") == ""


def test_about_tab_renders_methodology_sections() -> None:
    """The About tab must contain every required H2 section.

    Each H2 anchors a specific operator question (What is this? How
    does the methodology work? What are the limits?). A missing H2
    means the at-a-glance reading guide has a gap.
    """
    rendered = methodology_markdown()
    required_h2 = [
        "## What is lerobot-bench?",
        "## Methodology in 60 seconds",
        "## How a sweep works",
        "## v1 scope and known limits",
        "## Reading this dashboard",
    ]
    for heading in required_h2:
        assert heading in rendered, f"About tab missing required H2: {heading!r}"

    # Key methodological keywords must appear so the test fails if the
    # 60-second brief is silently truncated.
    for keyword in (
        "Wilson",
        "bootstrap",
        "seed",
        "MDE",
        "paired",
    ):
        assert keyword in rendered, f"About tab missing methodology keyword: {keyword!r}"

    # v1 scope numbers must be cited.
    assert str(V1_RUNNABLE_CELLS) in rendered
    assert str(V1_TOTAL_SEED_ENTRIES) in rendered or "110" in rendered

    # The pi0 deferral with the RAM reason is one of the explicit
    # requirements from the requesting spec.
    assert "pi0" in rendered.lower() or "Pi0" in rendered
    assert "30" in rendered  # the ~30 GB cold-load number


def test_dashboard_readme_exists_and_has_sections() -> None:
    """dashboard/README.md must exist with the required H2 sections."""
    readme_path = _DASHBOARD_DIR / "README.md"
    assert readme_path.exists(), f"dashboard/README.md must exist at {readme_path}"
    body = readme_path.read_text()

    required_sections = [
        "## Launch",
        "## Configure",
        "## Tabs",
        "## Public Space vs local dashboard",
        "## Empty-state checklist",
    ]
    for heading in required_sections:
        assert heading in body, f"dashboard/README.md missing section: {heading!r}"

    # The configure section must mention both env vars by name -- the
    # operator copy-paste path for fixing a wrong-dir misconfig.
    assert "DASHBOARD_RESULTS_DIR" in body
    assert "DASHBOARD_LOGS_DIR" in body


# --------------------------------------------------------------------- #
# AST guard                                                             #
# --------------------------------------------------------------------- #


def test_helpers_does_not_import_gradio() -> None:
    """Static guard mirroring tests/test_space.py.

    Gradio is in ``[space]`` extras, not ``[dev]``, so a stray
    ``import gradio`` in ``_helpers.py`` would break the fast pytest
    job on machines without the Spaces dep installed. AST-level catch
    so the error names the offending module.
    """
    src = (_DASHBOARD_DIR / "_helpers.py").read_text()
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "gradio":
                    offenders.append(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "gradio"
        ):
            offenders.append(node.module)
    assert not offenders, (
        f"dashboard/_helpers.py must not import gradio at module load; found {offenders}"
    )


# --------------------------------------------------------------------- #
# Council audit P0 item 7 / 8 / 9 — UX wins                             #
# --------------------------------------------------------------------- #


def test_should_accordion_be_open_first_visit_only() -> None:
    """``should_accordion_be_open`` returns True only on the first visit
    of a session; subsequent calls (visit_count >= 1) return False so
    revisit users do not see the wall-of-text framing again."""
    fn = _dashboard_helpers.should_accordion_be_open
    assert fn(0) is True
    assert fn(1) is False
    assert fn(5) is False
    assert fn(-1) is True  # defensive: negative behaves like first visit


def test_extract_row_click_target_resolves_policy_env_on_normal_row() -> None:
    """A click on a row with ``policy``/``env``/``status=running`` resolves
    to an actionable RowClickTarget with seed='0' default."""
    import pandas as pd

    extract = _dashboard_helpers.extract_row_click_target
    table = pd.DataFrame(
        [
            {"policy": "act", "env": "aloha_transfer_cube", "status": CELL_STATUS_RUNNING},
            {"policy": "smolvla_libero", "env": "libero_spatial", "status": CELL_STATUS_DONE},
        ],
        columns=list(PROGRESS_COLUMNS),
    )
    target = extract(table, row_index=1)
    assert target.actionable is True
    assert target.policy == "smolvla_libero"
    assert target.env == "libero_spatial"
    assert target.seed == "0"
    assert target.warning == ""


def test_extract_row_click_target_skipped_row_is_non_actionable() -> None:
    """Skipped cells have no rollouts on disk; the handler should not
    navigate, and should surface a warning."""
    import pandas as pd

    extract = _dashboard_helpers.extract_row_click_target
    table = pd.DataFrame(
        [{"policy": "act", "env": "pusht", "status": CELL_STATUS_SKIPPED}],
        columns=list(PROGRESS_COLUMNS),
    )
    target = extract(table, row_index=0)
    assert target.actionable is False
    assert "skipped" in target.warning.lower() or "no rollouts" in target.warning.lower() or target.warning


def test_extract_row_click_target_out_of_range_returns_warning() -> None:
    """Clicking an out-of-range or empty index yields a non-actionable
    result with a clear warning instead of an IndexError."""
    import pandas as pd

    extract = _dashboard_helpers.extract_row_click_target
    empty_table = pd.DataFrame(columns=list(PROGRESS_COLUMNS))
    assert extract(empty_table, row_index=0).actionable is False

    one_row = pd.DataFrame(
        [{"policy": "act", "env": "pusht", "status": CELL_STATUS_QUEUED}],
        columns=list(PROGRESS_COLUMNS),
    )
    target = extract(one_row, row_index=99)
    assert target.actionable is False
    assert "out of range" in target.warning.lower() or target.warning


def test_stale_data_cache_falls_back_to_last_good_on_loader_failure() -> None:
    """``load_with_stale_fallback`` returns the cache + soft warning on
    a loader exception, after a prior success has been recorded."""
    cache = _dashboard_helpers.StaleDataCache()
    load = _dashboard_helpers.load_with_stale_fallback

    def good_loader() -> str:
        return "good-value"

    def bad_loader() -> str:
        raise OSError("file mid-write")

    value, warning = load(cache, good_loader)
    assert value == "good-value"
    assert warning == ""

    value, warning = load(cache, bad_loader)
    assert value == "good-value"  # last-good
    assert "Last refresh failed" in warning


def test_stale_data_cache_escalates_after_three_failures() -> None:
    """After ``STALE_DATA_ESCALATE_AFTER`` consecutive failures the
    warning escalates to the loud "filesystem error" message."""
    cache = _dashboard_helpers.StaleDataCache()
    load = _dashboard_helpers.load_with_stale_fallback

    threshold = _dashboard_helpers.STALE_DATA_ESCALATE_AFTER

    def good_loader() -> str:
        return "v"

    def bad_loader() -> str:
        raise OSError("nope")

    # Seed cache with a good value first so we can see the cache survive.
    load(cache, good_loader)

    for i in range(threshold):
        _, warning = load(cache, bad_loader)
        if i < threshold - 1:
            assert "Last refresh failed" in warning
        else:
            assert "File system error" in warning, (
                f"escalated warning expected at failure #{i + 1}, got: {warning!r}"
            )
