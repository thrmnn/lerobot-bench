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
clear_video_cache = _dashboard_helpers.clear_video_cache
discover_sweep_runs = _dashboard_helpers.discover_sweep_runs
downscope_reason = _dashboard_helpers.downscope_reason
find_latest_calibration = _dashboard_helpers.find_latest_calibration
find_video_path = _dashboard_helpers.find_video_path
format_relative_time = _dashboard_helpers.format_relative_time
load_calibration_report = _dashboard_helpers.load_calibration_report
load_manifest = _dashboard_helpers.load_manifest
parse_video_filename = _dashboard_helpers.parse_video_filename
scan_video_index = _dashboard_helpers.scan_video_index
video_index_options = _dashboard_helpers.video_index_options

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
