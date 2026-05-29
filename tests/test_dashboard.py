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

import pandas as pd
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
build_halfwidth_curve = _dashboard_helpers.build_halfwidth_curve
build_progress_table = _dashboard_helpers.build_progress_table
classify_log_line = _dashboard_helpers.classify_log_line
clear_results_cache = _dashboard_helpers.clear_results_cache
compute_cell_episode_stats = _dashboard_helpers.compute_cell_episode_stats
format_seed_spread_cell = _dashboard_helpers.format_seed_spread_cell
format_success_rate_cell = _dashboard_helpers.format_success_rate_cell
format_wilson_ci_cell = _dashboard_helpers.format_wilson_ci_cell
load_results_parquet = _dashboard_helpers.load_results_parquet
select_plot_cell = _dashboard_helpers.select_plot_cell
MIN_N_FOR_SUCCESS_RATE = _dashboard_helpers.MIN_N_FOR_SUCCESS_RATE
SEED_SPREAD_FLAG_THRESHOLD = _dashboard_helpers.SEED_SPREAD_FLAG_THRESHOLD
STAT_PLACEHOLDER = _dashboard_helpers.STAT_PLACEHOLDER
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

# Mission Control: KPI strip, live leaderboard, anomaly review, throttle.
build_live_leaderboard = _dashboard_helpers.build_live_leaderboard
compute_mission_kpis = _dashboard_helpers.compute_mission_kpis
format_bytes_gb = _dashboard_helpers.format_bytes_gb
HEALTH_AMBER = _dashboard_helpers.HEALTH_AMBER
HEALTH_GREEN = _dashboard_helpers.HEALTH_GREEN
HEALTH_RED = _dashboard_helpers.HEALTH_RED
leaderboard_dataframe = _dashboard_helpers.leaderboard_dataframe
LEADERBOARD_COLUMNS = _dashboard_helpers.LEADERBOARD_COLUMNS
read_system_memory = _dashboard_helpers.read_system_memory
read_throttle_state = _dashboard_helpers.read_throttle_state
run_anomaly_review = _dashboard_helpers.run_anomaly_review
SWEEP_STATE_DONE = _dashboard_helpers.SWEEP_STATE_DONE
SWEEP_STATE_RUNNING = _dashboard_helpers.SWEEP_STATE_RUNNING
SWEEP_STATE_THROTTLED = _dashboard_helpers.SWEEP_STATE_THROTTLED

# Scientific-context panels (Policies + Envs tabs, representative rollout).
build_env_card_markdown = _dashboard_helpers.build_env_card_markdown
build_policy_card_markdown = _dashboard_helpers.build_policy_card_markdown
delta_chip = _dashboard_helpers.delta_chip
DELTA_GREEN_MAX = _dashboard_helpers.DELTA_GREEN_MAX
DELTA_YELLOW_MAX = _dashboard_helpers.DELTA_YELLOW_MAX
env_dropdown_choices = _dashboard_helpers.env_dropdown_choices
load_env_registry = _dashboard_helpers.load_env_registry
load_policy_registry = _dashboard_helpers.load_policy_registry
policy_dropdown_choices = _dashboard_helpers.policy_dropdown_choices
select_representative_episode = _dashboard_helpers.select_representative_episode
EPISODE_SELECT_BEST = _dashboard_helpers.EPISODE_SELECT_BEST
EPISODE_SELECT_FIRST = _dashboard_helpers.EPISODE_SELECT_FIRST
EPISODE_SELECT_REPRESENTATIVE = _dashboard_helpers.EPISODE_SELECT_REPRESENTATIVE

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


def test_per_tab_intro_rebuilt_tabs_render() -> None:
    """The rebuilt dashboard's intro-bearing tabs render non-empty blocks.

    After the 7->3 rebuild the dashboard only renders the per-tab intro
    for Pre-flight (``calibration``) and Rollouts (``rollouts``); the
    progress / events prose now lives inline on the Status screen. An
    unknown tab key still returns the empty string (a wiring bug, not a
    runtime error).
    """
    for tab in ("calibration", "rollouts"):
        rendered = per_tab_intro_markdown(tab)
        assert len(rendered) > 100, f"intro for tab={tab!r} is suspiciously short"
        assert "What" in rendered and "tab" in rendered
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
# Statistical-rigor: per-episode parquet stats, CI columns, plots        #
# --------------------------------------------------------------------- #


def _results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a synthetic per-episode results frame (parquet schema subset)."""
    return pd.DataFrame(rows, columns=["policy", "env", "seed", "episode_index", "success"])


def _cell_rows(
    *,
    policy: str,
    env: str,
    per_seed_successes: list[int],
    n_episodes_per_seed: int,
) -> list[dict[str, Any]]:
    """Synthesise per-episode rows: ``per_seed_successes[k]`` wins for seed k."""
    rows: list[dict[str, Any]] = []
    for seed, n_success in enumerate(per_seed_successes):
        for ep in range(n_episodes_per_seed):
            rows.append(
                {
                    "policy": policy,
                    "env": env,
                    "seed": seed,
                    "episode_index": ep,
                    "success": ep < n_success,
                }
            )
    return rows


def test_compute_cell_episode_stats_pools_episodes_not_seed_means() -> None:
    """The Wilson CI is over the flat episode list, not the 5 per-seed means.

    Council veto: bootstrapping / CIs over per-seed means is
    pseudo-replication. We assert n_episodes == 5*10 (the flat count)
    and that the success rate equals total_successes / total_episodes.
    """
    df = _results_frame(
        _cell_rows(
            policy="act",
            env="pusht",
            per_seed_successes=[2, 4, 6, 8, 10],
            n_episodes_per_seed=10,
        )
    )
    stats = compute_cell_episode_stats(df, policy="act", env="pusht")
    assert stats is not None
    assert stats.n_episodes == 50  # flat 5*10, NOT 5 seed-means
    assert stats.n_success == 30
    assert stats.success_rate == pytest.approx(0.6)
    # Wilson CI brackets the point estimate and is a sub-interval of [0,1].
    assert 0.0 <= stats.wilson_lo < stats.success_rate < stats.wilson_hi <= 1.0
    # seed_spread = max(1.0) - min(0.2) = 0.8 over the 5 per-seed rates.
    assert stats.n_seeds == 5
    assert stats.seed_spread == pytest.approx(0.8)


def test_compute_cell_episode_stats_absent_cell_returns_none() -> None:
    """A cell with no rows in the parquet yields None (cold start)."""
    df = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[5], n_episodes_per_seed=10)
    )
    assert compute_cell_episode_stats(df, policy="xvla", env="pusht") is None
    assert compute_cell_episode_stats(None, policy="act", env="pusht") is None


def test_success_rate_and_ci_columns_gated_below_min_n() -> None:
    """Small-N gate: below MIN_N_FOR_SUCCESS_RATE both columns show "—".

    A success rate is never shown without its CI; the two columns gate
    on the same threshold, so they appear and disappear together.
    """
    # 1 seed x (MIN_N - 1) episodes -> just under the gate.
    n_below = MIN_N_FOR_SUCCESS_RATE - 1
    df_small = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[3], n_episodes_per_seed=n_below)
    )
    small = compute_cell_episode_stats(df_small, policy="act", env="pusht")
    assert small is not None and small.n_episodes == n_below
    assert format_success_rate_cell(small) == STAT_PLACEHOLDER
    assert format_wilson_ci_cell(small) == STAT_PLACEHOLDER

    # At exactly MIN_N the columns light up.
    df_ok = _results_frame(
        _cell_rows(
            policy="act",
            env="pusht",
            per_seed_successes=[MIN_N_FOR_SUCCESS_RATE // 2],
            n_episodes_per_seed=MIN_N_FOR_SUCCESS_RATE,
        )
    )
    ok = compute_cell_episode_stats(df_ok, policy="act", env="pusht")
    assert ok is not None
    rate_cell = format_success_rate_cell(ok)
    ci_cell = format_wilson_ci_cell(ok)
    assert rate_cell != STAT_PLACEHOLDER
    assert ci_cell != STAT_PLACEHOLDER
    # CI is formatted "[lo, hi]" to 2 decimals.
    assert ci_cell.startswith("[") and ci_cell.endswith("]") and ci_cell.count(",") == 1
    lo_str, hi_str = ci_cell.strip("[]").split(", ")
    assert len(lo_str.split(".")[1]) == 2 and len(hi_str.split(".")[1]) == 2

    # None stats (no parquet) -> placeholder, never a bare number.
    assert format_success_rate_cell(None) == STAT_PLACEHOLDER
    assert format_wilson_ci_cell(None) == STAT_PLACEHOLDER


def test_seed_spread_red_flag_threshold() -> None:
    """seed_spread > 0.2 gets the ⚠ flag; <= 0.2 and <2 seeds do not."""
    # Two seeds, spread = 1.0 - 0.0 = 1.0 -> flagged.
    df_wide = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[0, 10], n_episodes_per_seed=10)
    )
    wide = compute_cell_episode_stats(df_wide, policy="act", env="pusht")
    assert wide is not None and wide.seed_spread == pytest.approx(1.0)
    flagged = format_seed_spread_cell(wide)
    assert "⚠" in flagged

    # Two seeds, spread = 0.1 (5 vs 6 of 10) -> below threshold, no flag.
    df_tight = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[5, 6], n_episodes_per_seed=10)
    )
    tight = compute_cell_episode_stats(df_tight, policy="act", env="pusht")
    assert tight is not None and tight.seed_spread == pytest.approx(0.1)
    assert tight.seed_spread <= SEED_SPREAD_FLAG_THRESHOLD
    assert "⚠" not in format_seed_spread_cell(tight)

    # One seed -> seed_spread undefined -> placeholder, no flag.
    df_one = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[5], n_episodes_per_seed=10)
    )
    one = compute_cell_episode_stats(df_one, policy="act", env="pusht")
    assert one is not None and one.seed_spread is None
    assert format_seed_spread_cell(one) == STAT_PLACEHOLDER


def test_build_progress_table_fills_rigor_columns_from_parquet() -> None:
    """build_progress_table wires the three rigor columns from results_df."""
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="pusht",
                seed=s,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            )
            for s in range(5)
        ]
    )
    df = _results_frame(
        _cell_rows(
            policy="act",
            env="pusht",
            per_seed_successes=[10, 10, 10, 10, 10],
            n_episodes_per_seed=10,
        )
    )
    table = build_progress_table(manifest, results_df=df)
    row = table.iloc[0]
    assert "success_rate_so_far" in table.columns
    assert "wilson_ci_so_far" in table.columns
    assert "seed_spread" in table.columns
    # 50/50 successes -> rate 1.00, all seeds identical -> spread 0.00.
    assert row["success_rate_so_far"].startswith("1.00")
    assert row["wilson_ci_so_far"].startswith("[")
    assert row["seed_spread"] == "0.00"

    # results_df=None leaves all three columns as the placeholder.
    table_no_df = build_progress_table(manifest)
    assert table_no_df.iloc[0]["success_rate_so_far"] == STAT_PLACEHOLDER
    assert table_no_df.iloc[0]["wilson_ci_so_far"] == STAT_PLACEHOLDER
    assert table_no_df.iloc[0]["seed_spread"] == STAT_PLACEHOLDER


def test_calibration_table_skew_flag_and_std_placeholder(tmp_path: Path) -> None:
    """Latency-skew flag fires on p95/mean > 3; std_step_ms is "—" without raw times."""
    report = {
        "timestamp_utc": "2026-05-12T03:00:00+00:00",
        "git_sha": "abc123",
        "cells": [
            # p95/mean = 1500/200 = 7.5 -> skewed.
            _cal_cell(policy="dp", env="pusht", mean_ms=200.0, p95_ms=1500.0),
            # p95/mean = 60/50 = 1.2 -> not skewed.
            _cal_cell(policy="cheap", env="pusht", mean_ms=50.0, p95_ms=60.0),
        ],
    }
    path = tmp_path / "calibration-20260512.json"
    path.write_text(json.dumps(report))
    table = build_calibration_table(load_calibration_report(path))

    assert "std_step_ms" in table.columns
    assert "n_steps" in table.columns
    assert "latency_skew" in table.columns

    dp = table[table["policy"] == "dp"].iloc[0]
    cheap = table[table["policy"] == "cheap"].iloc[0]
    assert "skewed" in dp["latency_skew"]
    assert cheap["latency_skew"] == ""
    # n_steps comes straight from n_steps_measured (20 in _cal_cell).
    assert dp["n_steps"] == 20
    # The calibration JSON carries no raw step times -> std is "—".
    assert dp["std_step_ms"] == STAT_PLACEHOLDER
    assert cheap["std_step_ms"] == STAT_PLACEHOLDER


def test_calibration_table_std_step_ms_from_raw_step_times(tmp_path: Path) -> None:
    """When a cell carries raw ``step_ms``, std_step_ms is computed."""
    cell = _cal_cell(policy="p", env="e", mean_ms=10.0, p95_ms=12.0)
    cell["step_ms"] = [8.0, 10.0, 12.0, 10.0, 10.0]
    report = {"cells": [cell]}
    table = build_calibration_table(report)
    std_cell = table.iloc[0]["std_step_ms"]
    assert std_cell != STAT_PLACEHOLDER
    # Sample std (ddof=1) of [8,10,12,10,10] is ~1.41.
    assert float(std_cell) == pytest.approx(1.41, abs=0.05)


def test_load_results_parquet_stale_fallback(tmp_path: Path) -> None:
    """A mid-write (corrupt) parquet falls back to the last-good frame."""
    clear_results_cache()
    parquet = tmp_path / "results.parquet"
    df = _results_frame(
        _cell_rows(policy="act", env="pusht", per_seed_successes=[5], n_episodes_per_seed=10)
    )
    df.to_parquet(parquet)

    good = load_results_parquet(parquet)
    assert good is not None and len(good) == 10

    # Simulate a half-written file: overwrite with garbage bytes.
    parquet.write_bytes(b"PAR1\x00\x00not-a-valid-parquet")
    stale = load_results_parquet(parquet)
    # The corrupt read serves the last-good frame, not None / not a crash.
    assert stale is not None and len(stale) == 10

    # Cold start (no good read ever) -> None, no crash.
    clear_results_cache()
    assert load_results_parquet(tmp_path / "never-existed.parquet") is None
    assert load_results_parquet(None) is None


def test_build_halfwidth_curve_falls_back_when_no_cell_running() -> None:
    """No running cell with episodes -> the plot uses the last done cell.

    Also covers the cold-start case: no cell with episodes at all -> None,
    so the caller renders an empty-state plot rather than crashing.
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="pusht",
                seed=s,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            )
            for s in range(5)
        ]
        # A running cell with no episodes on disk yet.
        + [
            _manifest_entry(
                policy="xvla",
                env="pusht",
                seed=0,
                status="pending",
                started_utc="2026-05-12T03:20:00+00:00",
            )
        ]
    )
    df = _results_frame(
        _cell_rows(
            policy="act",
            env="pusht",
            per_seed_successes=[6, 6, 6, 6, 6],
            n_episodes_per_seed=10,
        )
    )
    table = build_progress_table(manifest, results_df=df)

    # xvla is "running" but has zero parquet rows -> selector falls back.
    selected = select_plot_cell(table, df)
    assert selected is not None
    policy, env, is_running = selected
    assert (policy, env) == ("act", "pusht")
    assert is_running is False  # the fallback (done) cell, not the running one

    curve = build_halfwidth_curve(table, df)
    assert curve is not None
    assert curve.n_current == 50
    assert curve.is_running is False
    assert len(curve.n_values) == len(curve.halfwidths) == 50
    # Half-width shrinks ~1/sqrt(N): it ends well below its start. It is
    # NOT strictly monotone -- wilson_halfwidth_at_p discretises via
    # round(p*n), so small-N rounding produces local bumps; we only
    # assert the trend, which is the operator-facing claim.
    assert curve.halfwidths[-1] < curve.halfwidths[0] / 2.0

    # Cold start: parquet present but empty -> no curve, no crash.
    empty_df = _results_frame([])
    assert build_halfwidth_curve(table, empty_df) is None
    assert build_halfwidth_curve(table, None) is None


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
    assert (
        "skipped" in target.warning.lower()
        or "no rollouts" in target.warning.lower()
        or target.warning
    )


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


# --------------------------------------------------------------------- #
# Scientific-context panels: Policies + Envs tabs                        #
# --------------------------------------------------------------------- #


def test_policy_card_renders_for_every_v1_policy() -> None:
    """Every policy in the shipped configs/policies.yaml renders a card.

    The card must surface the repo, license, the short revision SHA,
    and the paper-vs-ours heading -- a reviewer's minimum context. We
    pass ``results_df=None`` so every cell shows ``(pending)``.
    """
    registry = load_policy_registry()
    names = policy_dropdown_choices(registry)
    assert len(names) >= 6, "expected the full v1 policy roster"
    for name in names:
        card = build_policy_card_markdown(name, registry=registry, results_df=None)
        assert card.startswith(f"## {name}"), f"card for {name!r} missing its heading"
        assert "Paper-reported vs. our re-run" in card
        spec = registry.get(name)
        if spec.repo_id:
            assert spec.repo_id in card
        if spec.is_baseline:
            assert "baseline" in card.lower()


def test_policy_card_paper_vs_ours_pending_and_delta() -> None:
    """A VLA policy's card shows ``(pending)`` with no parquet, a delta with one."""
    registry = load_policy_registry()
    # xvla_libero carries paper_reported_success for all 4 libero suites.
    no_data = build_policy_card_markdown("xvla_libero", registry=registry, results_df=None)
    assert "(pending)" in no_data
    # The paper number for libero_spatial (0.982) must appear.
    assert "0.982" in no_data

    # Now supply a results frame for one of its cells.
    df = _results_frame(
        _cell_rows(
            policy="xvla_libero",
            env="libero_spatial",
            per_seed_successes=[50, 50, 50, 50, 50],
            n_episodes_per_seed=50,
        )
    )
    with_data = build_policy_card_markdown("xvla_libero", registry=registry, results_df=df)
    # 250/250 success -> our 1.000 rendered; libero_spatial no longer pending.
    assert "1.000" in with_data
    # The other three libero suites still pending.
    assert "(pending)" in with_data


def test_delta_chip_color_thresholds() -> None:
    """delta_chip's colour buckets honour the green / yellow / red cutoffs."""
    # |Δ| below the green max -> green chip.
    _, chip = delta_chip(0.90, 0.90 + DELTA_GREEN_MAX / 2)
    assert chip == "🟢"
    # |Δ| at the yellow band -> yellow chip.
    _, chip = delta_chip(0.90, 0.90 - (DELTA_GREEN_MAX + DELTA_YELLOW_MAX) / 2)
    assert chip == "🟡"
    # |Δ| above the yellow max -> red chip.
    _, chip = delta_chip(0.90, 0.90 - (DELTA_YELLOW_MAX + 0.1))
    assert chip == "🔴"
    # Exactly at the green/yellow boundary -> yellow (green is strict <).
    _, chip = delta_chip(0.50, 0.50 + DELTA_GREEN_MAX)
    assert chip == "🟡"
    # Either side missing -> placeholder, no chip.
    text, chip = delta_chip(None, 0.5)
    assert text == STAT_PLACEHOLDER and chip == ""
    text, chip = delta_chip(0.5, None)
    assert text == STAT_PLACEHOLDER and chip == ""
    # Delta text is signed to 3 dp.
    text, _ = delta_chip(0.80, 0.83)
    assert text == "+0.030"
    text, _ = delta_chip(0.80, 0.77)
    assert text == "-0.030"


def test_env_card_renders_for_every_v1_env() -> None:
    """Every env in the shipped configs/envs.yaml renders an informative card.

    The card must carry the runtime fields (max_steps, success
    threshold) and the hand-authored task / observation / source prose
    -- no ``"—"`` placeholders for the v1 envs, which are all covered
    by the _ENV_CONTEXT constant dict.
    """
    registry = load_env_registry()
    names = env_dropdown_choices(registry)
    assert set(names) >= {
        "pusht",
        "aloha_transfer_cube",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    }
    for name in names:
        card = build_env_card_markdown(name, registry=registry)
        assert card.startswith(f"## {name}")
        spec = registry.get(name)
        assert str(spec.max_steps) in card
        assert "**Task.**" in card
        assert "**Observation.**" in card
        assert "**Source.**" in card
        # Every v1 env has hand-authored context -> no placeholder prose.
        assert STAT_PLACEHOLDER not in card, f"env card for {name!r} has a placeholder"


def test_policy_and_env_card_unknown_name() -> None:
    """An unknown policy / env name yields a graceful message, not a crash."""
    assert "Unknown policy" in build_policy_card_markdown("not_a_policy")
    assert "Unknown env" in build_env_card_markdown("not_an_env")


# --------------------------------------------------------------------- #
# Representative-rollout episode selection                               #
# --------------------------------------------------------------------- #


def _rollout_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-episode frame with an n_steps column for the rollout selector."""
    return pd.DataFrame(
        rows, columns=["policy", "env", "seed", "episode_index", "success", "n_steps"]
    )


def test_select_representative_episode_picks_modal_outcome_and_median_steps() -> None:
    """Representative = modal-outcome episode closest to the cell's median steps.

    Five episodes for one cell: 4 successes, 1 failure (modal outcome =
    success). Among the 4 successes, the one whose n_steps is closest
    to the cell's median step count is the representative.
    """
    rows = [
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 0, "success": True, "n_steps": 100},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 1, "success": True, "n_steps": 200},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 2, "success": True, "n_steps": 150},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 3, "success": True, "n_steps": 400},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 4, "success": False, "n_steps": 90},
    ]
    df = _rollout_frame(rows)
    # Median of [100, 200, 150, 400, 90] = 150 -> episode 2 (success, 150 steps).
    chosen = select_representative_episode(
        df, policy="p", env="e", seed=0, mode=EPISODE_SELECT_REPRESENTATIVE
    )
    assert chosen == 2


def test_select_representative_episode_best_mode_is_fast_success() -> None:
    """Best = a successful episode with the shortest step count (fast success)."""
    rows = [
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 0, "success": False, "n_steps": 50},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 1, "success": True, "n_steps": 300},
        {"policy": "p", "env": "e", "seed": 0, "episode_index": 2, "success": True, "n_steps": 120},
    ]
    df = _rollout_frame(rows)
    chosen = select_representative_episode(
        df, policy="p", env="e", seed=0, mode=EPISODE_SELECT_BEST
    )
    # Episode 2 is the shortest *successful* episode -- episode 0 is faster
    # but is a failure, so it must not win "best".
    assert chosen == 2

    # First mode -> lowest episode index regardless of outcome / steps.
    first = select_representative_episode(
        df, policy="p", env="e", seed=0, mode=EPISODE_SELECT_FIRST
    )
    assert first == 0


def test_select_representative_episode_falls_back_with_no_parquet_rows() -> None:
    """No parquet rows for the cell -> fall back to first available on disk."""
    df = _rollout_frame(
        [
            {
                "policy": "p",
                "env": "e",
                "seed": 0,
                "episode_index": 0,
                "success": True,
                "n_steps": 100,
            }
        ]
    )
    # Cell (other, e, 0) is absent from the parquet: fall back to min(available).
    chosen = select_representative_episode(
        df,
        policy="other",
        env="e",
        seed=0,
        mode=EPISODE_SELECT_REPRESENTATIVE,
        available_episodes=[7, 3, 11],
    )
    assert chosen == 3

    # No parquet at all + no available episodes -> None (keep dropdown value).
    assert select_representative_episode(None, policy="p", env="e", seed=0) is None
    # No parquet + available episodes -> first on disk.
    assert (
        select_representative_episode(
            None, policy="p", env="e", seed=0, available_episodes=[5, 2, 9]
        )
        == 2
    )


# --------------------------------------------------------------------- #
# Mission Control: KPI strip, live leaderboard, anomalies, throttle      #
# --------------------------------------------------------------------- #


def test_compute_mission_kpis_from_synthetic_manifest() -> None:
    """KPI roll-up from a manifest: one done cell, one running, one failed.

    Three (policy, env) cells (one seed each for brevity): a completed
    cell, a running cell (started but pending), and a failed cell. The
    KPI strip must count 1 done / 1 failed / 1 running and -- because a
    failure is present -- raise the red health banner.
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="aloha_transfer_cube",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(
                policy="diffusion_policy",
                env="pusht",
                seed=0,
                status="pending",
                started_utc="2026-05-12T03:10:00+00:00",
            ),
            _manifest_entry(
                policy="random",
                env="pusht",
                seed=0,
                status="failed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:01:00+00:00",
                exit_code=4,
            ),
        ]
    )
    now = dt.datetime(2026, 5, 12, 3, 12, 0, tzinfo=dt.UTC)
    kpis = compute_mission_kpis(manifest, now_utc=now)

    assert kpis.cells_done == 1
    assert kpis.cells_failed == 1
    assert kpis.cells_running == 1
    assert kpis.cells_total == 3
    # A failure present -> red banner, regardless of done/running counts.
    assert kpis.health == HEALTH_RED
    assert "fail" in kpis.health_message.lower()
    # The running label names the most-recently-started pending seed.
    assert "diffusion_policy" in kpis.running_label
    # Elapsed is measured from the manifest started_utc (3:00 -> 3:12).
    assert "12m" in kpis.elapsed_label


def test_compute_mission_kpis_empty_manifest_is_idle_amber() -> None:
    """No manifest on disk -> IDLE state, amber banner, never a crash."""
    kpis = compute_mission_kpis({})
    assert kpis.cells_total == 0
    assert kpis.state == "IDLE"
    assert kpis.health == HEALTH_AMBER
    assert kpis.denom == V1_RUNNABLE_CELLS


def test_compute_mission_kpis_throttled_forces_red() -> None:
    """A frozen cgroup forces the THROTTLED state + red banner.

    Even a clean all-done manifest: if ``throttled=True`` the operator
    must see red, because a frozen sweep is not making progress.
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="aloha_transfer_cube",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            )
        ]
    )
    kpis = compute_mission_kpis(manifest, throttled=True)
    assert kpis.state == SWEEP_STATE_THROTTLED
    assert kpis.health == HEALTH_RED
    assert "FROZEN" in kpis.health_message or "froze" in kpis.health_message.lower()


def test_build_live_leaderboard_aggregates_per_policy_best_first() -> None:
    """Live leaderboard pools episodes per policy and sorts best-first.

    Two policies: ``act`` 30/50 across one env, ``random`` 5/50 across
    one env. The leaderboard must sort ``act`` first (higher rate),
    carry a Wilson CI bracketing each rate, and count cells done.
    """
    rows = _cell_rows(
        policy="act",
        env="aloha_transfer_cube",
        per_seed_successes=[2, 4, 6, 8, 10],
        n_episodes_per_seed=10,
    ) + _cell_rows(
        policy="random",
        env="pusht",
        per_seed_successes=[1, 1, 1, 1, 1],
        n_episodes_per_seed=10,
    )
    df = _results_frame(rows)
    board = build_live_leaderboard(df)
    assert [r.policy for r in board] == ["act", "random"]

    act = board[0]
    assert act.n_episodes == 50
    assert act.success_rate == pytest.approx(0.6)
    assert act.wilson_lo < act.success_rate < act.wilson_hi
    assert act.n_cells == 1

    # The dataframe projection carries the canonical columns.
    table = leaderboard_dataframe(board)
    assert list(table.columns) == list(LEADERBOARD_COLUMNS)
    assert len(table) == 2
    # Empty input -> canonical empty frame, no crash.
    empty = leaderboard_dataframe(build_live_leaderboard(None))
    assert list(empty.columns) == list(LEADERBOARD_COLUMNS)
    assert len(empty) == 0


def _review_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-episode frame with the full schema the anomaly review needs."""
    return pd.DataFrame(
        rows,
        columns=[
            "policy",
            "env",
            "seed",
            "episode_index",
            "success",
            "n_steps",
            "wallclock_s",
        ],
    )


def _review_rows(
    *,
    policy: str,
    env: str,
    n_seeds: int,
    n_episodes: int,
    success: bool,
    n_steps: int = 100,
) -> list[dict[str, Any]]:
    """Synthesise per-episode review rows with a fixed per-cell outcome."""
    rows: list[dict[str, Any]] = []
    for seed in range(n_seeds):
        for ep in range(n_episodes):
            rows.append(
                {
                    "policy": policy,
                    "env": env,
                    "seed": seed,
                    "episode_index": ep,
                    "success": success,
                    "n_steps": n_steps,
                    "wallclock_s": 60.0,
                }
            )
    return rows


def test_run_anomaly_review_clean_state(tmp_path: Path) -> None:
    """A healthy cell yields ok=True with no flagged lines."""
    clear_results_cache()
    # act x aloha_transfer_cube: a mix of outcomes so no degenerate /
    # never-succeeds / seed-disagreement flag fires.
    rows: list[dict[str, Any]] = []
    for seed in range(5):
        for ep in range(50):
            rows.append(
                {
                    "policy": "act",
                    "env": "aloha_transfer_cube",
                    "seed": seed,
                    "episode_index": ep,
                    # ~50% success, varied step counts -> healthy.
                    "success": ep % 2 == 0,
                    "n_steps": 100 + ep,
                    "wallclock_s": 60.0,
                }
            )
    parquet = tmp_path / "results.parquet"
    _review_frame(rows).to_parquet(parquet)

    report = run_anomaly_review(parquet)
    assert report.ok is True
    assert report.n_cells_flagged == 0
    assert report.lines == []
    assert report.error == ""


def test_run_anomaly_review_flags_baseline_above_floor(tmp_path: Path) -> None:
    """A no_op baseline scoring 100% trips the BASELINE-ABOVE-FLOOR check."""
    clear_results_cache()
    rows = _review_rows(
        policy="no_op",
        env="aloha_transfer_cube",
        n_seeds=5,
        n_episodes=50,
        success=True,
    )
    parquet = tmp_path / "results.parquet"
    _review_frame(rows).to_parquet(parquet)

    report = run_anomaly_review(parquet)
    assert report.ok is False
    assert report.n_cells_flagged >= 1
    assert any("BASELINE-ABOVE-FLOOR" in line for line in report.lines)
    # Each flagged line names the cell.
    assert any("no_op" in line for line in report.lines)


def test_run_anomaly_review_missing_parquet_degrades(tmp_path: Path) -> None:
    """No parquet on disk -> ok=True with a neutral error note, no crash."""
    clear_results_cache()
    report = run_anomaly_review(None)
    assert report.ok is True
    assert report.n_cells_reviewed == 0
    assert report.error  # a neutral "no results yet" note

    report2 = run_anomaly_review(tmp_path / "never-existed.parquet")
    assert report2.n_cells_reviewed == 0
    assert report2.error


def test_read_throttle_state_sweep_absent_degrades_gracefully(
    tmp_path: Path,
) -> None:
    """When no run_sweep.py PID is found, every field is None / not-running.

    Points the helper at an empty synthetic /proc with no matching
    process -- the Mission Control panel then renders "—" rather than
    crashing the 5 s tick.
    """
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    # A bystander process whose cmdline does not match run_sweep.py.
    pid_dir = fake_proc / "999"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(b"python\x00-m\x00http.server\x00")

    state = read_throttle_state(proc_root=fake_proc)
    assert state.running is False
    assert state.pid is None
    assert state.frozen is None
    assert state.memory_current is None
    assert state.memory_max is None
    assert state.state_label == "not running"


def test_read_throttle_state_resolves_frozen_cgroup(tmp_path: Path) -> None:
    """A synthetic /proc + cgroup layout resolves the freeze + memory state.

    Builds a fake ``run_sweep.py`` process whose cgroup files report a
    frozen state and a memory cap, then asserts the helper reads them.
    """
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    pid_dir = fake_proc / "4242"
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(
        b"python\x00scripts/run_sweep.py\x00--config\x00sweep_full.yaml\x00"
    )
    # cgroup v2 line: hierarchy 0, empty controller, path.
    (pid_dir / "cgroup").write_text("0::/sweep.scope\n")

    # The helper resolves the cgroup dir under /sys/fs/cgroup; we can't
    # write there in a test, so we only assert the PID + running parts
    # resolve and the unreadable cgroup files degrade to None.
    state = read_throttle_state(proc_root=fake_proc)
    assert state.running is True
    assert state.pid == 4242
    # /sys/fs/cgroup/sweep.scope does not exist -> freeze/memory are None.
    assert state.frozen is None
    assert state.memory_current is None
    assert state.state_label == "RUNNING"


def test_read_system_memory_parses_meminfo(tmp_path: Path) -> None:
    """read_system_memory parses a synthetic /proc/meminfo; bad file -> None."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32000000 kB\n"
        "MemFree:         8000000 kB\n"
        "MemAvailable:   20000000 kB\n"
        "Buffers:          200000 kB\n"
    )
    mem = read_system_memory(meminfo_path=meminfo)
    assert mem is not None
    assert mem.total_bytes == 32000000 * 1024
    assert mem.available_bytes == 20000000 * 1024
    assert mem.used_bytes == (32000000 - 20000000) * 1024
    assert mem.percent_used == pytest.approx(12000000 / 32000000 * 100.0)

    # Missing file -> None, never raises.
    assert read_system_memory(meminfo_path=tmp_path / "nope") is None


def test_format_bytes_gb_renders_and_handles_none() -> None:
    """format_bytes_gb renders GB; None -> the stat placeholder."""
    assert format_bytes_gb(None) == STAT_PLACEHOLDER
    assert format_bytes_gb(8 * 1024**3) == "8.0 GB"
    assert format_bytes_gb(0) == "0.0 GB"


# --------------------------------------------------------------------- #
# Status landing screen (rebuilt dashboard, 7 tabs -> 3)                 #
# --------------------------------------------------------------------- #
#
# The Status tab is the default landing screen and must pass the
# 5-second test: the instant it loads, a plain-English health line says
# whether the sweep needs the operator. The tests below pin the three
# health-banner states and the auto-load-newest-run behaviour that
# replaced the old (buggy) run-selector dropdown.


def test_status_health_message_is_plain_english_green_when_progressing() -> None:
    """A clean in-flight sweep -> green banner with the 5-second-test line.

    The message must lead with plain English ("Sweep healthy"), name the
    cells-done count, state 0 failed, and reassure ("nothing needs you")
    so the operator can close the laptop on a glance.
    """
    manifest = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="aloha_transfer_cube",
                seed=0,
                status="completed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:10:00+00:00",
                exit_code=0,
            ),
            _manifest_entry(
                policy="diffusion_policy",
                env="pusht",
                seed=0,
                status="pending",
                started_utc="2026-05-12T03:10:00+00:00",
            ),
        ]
    )
    now = dt.datetime(2026, 5, 12, 3, 12, 0, tzinfo=dt.UTC)
    kpis = compute_mission_kpis(manifest, now_utc=now)

    assert kpis.health == HEALTH_GREEN
    msg = kpis.health_message
    assert "Sweep healthy" in msg
    assert "0 failed" in msg
    assert "nothing needs you" in msg
    # Plain count, not a bare stats grid.
    assert f"{kpis.cells_done}/{kpis.denom}" in msg


def test_status_health_message_says_needs_you_on_failure() -> None:
    """Any failed cell -> red banner whose sentence says it needs the operator."""
    manifest = _manifest(
        [
            _manifest_entry(
                policy="random",
                env="pusht",
                seed=0,
                status="failed",
                started_utc="2026-05-12T03:00:00+00:00",
                finished_utc="2026-05-12T03:01:00+00:00",
                exit_code=4,
            ),
        ]
    )
    kpis = compute_mission_kpis(manifest)
    assert kpis.health == HEALTH_RED
    assert "needs you" in kpis.health_message.lower()
    assert "1 failed" in kpis.health_message


def test_status_empty_run_is_graceful_amber_no_crash() -> None:
    """No sweep on disk -> amber banner, plain hint, never a crash.

    This is the cold-start state: the dashboard opened before any sweep
    exists. The banner must still render a sentence (amber), not blank.
    """
    kpis = compute_mission_kpis({})
    assert kpis.health == HEALTH_AMBER
    assert "No sweep running" in kpis.health_message
    assert kpis.state == "IDLE"


def test_status_auto_loads_newest_run_no_selector(tmp_path: Path) -> None:
    """The Status grid auto-loads the newest run -- no manual selection.

    The old Sweep-progress tab rendered empty mid-sweep because its
    run-selector dropdown did not auto-select the live run. The rebuilt
    Status screen calls ``discover_sweep_runs`` and takes ``runs[0]``
    (newest by ``started_utc``) on every paint, so the grid populates
    on first paint with no operator action. This pins that contract.
    """
    older_dir = tmp_path / "sweep-old"
    older_dir.mkdir()
    older = _manifest(
        [
            _manifest_entry(
                policy="act",
                env="pusht",
                seed=0,
                status="completed",
                started_utc="2026-05-10T00:00:00+00:00",
                finished_utc="2026-05-10T00:10:00+00:00",
                exit_code=0,
            )
        ],
        finished="2026-05-10T00:10:00+00:00",
    )
    older["started_utc"] = "2026-05-10T00:00:00+00:00"
    (older_dir / "sweep_manifest.json").write_text(json.dumps(older))

    newer_dir = tmp_path / "sweep-new"
    newer_dir.mkdir()
    newer = _manifest(
        [
            _manifest_entry(
                policy="diffusion_policy",
                env="pusht",
                seed=0,
                status="pending",
                started_utc="2026-05-12T03:10:00+00:00",
            )
        ]
    )
    newer["started_utc"] = "2026-05-12T03:00:00+00:00"
    (newer_dir / "sweep_manifest.json").write_text(json.dumps(newer))

    runs = discover_sweep_runs(tmp_path)
    # runs[0] -- the path the Status screen auto-loads -- is the newest.
    assert runs[0].name == "sweep-new"
    grid = build_progress_table(load_manifest(runs[0].manifest_path))
    assert not grid.empty
    assert grid.iloc[0]["policy"] == "diffusion_policy"


def test_status_auto_load_empty_results_dir_returns_no_runs(tmp_path: Path) -> None:
    """An empty results dir -> no runs -> Status renders its empty state."""
    assert discover_sweep_runs(tmp_path) == []


# --------------------------------------------------------------------- #
# v1 policy filter (xvla_libero deferral, PR #76)                       #
# --------------------------------------------------------------------- #

V1_POLICIES = _dashboard_helpers.V1_POLICIES
filter_to_v1_policies = _dashboard_helpers.filter_to_v1_policies


def test_v1_policies_excludes_xvla() -> None:
    """xvla_libero is deferred to v1.1; the leaderboard tuple must not list it."""
    assert "xvla_libero" not in V1_POLICIES
    assert set(V1_POLICIES) == {
        "act",
        "diffusion_policy",
        "smolvla_libero",
        "no_op",
        "random",
    }


def test_filter_to_v1_policies_drops_xvla_rows() -> None:
    """The standalone helper drops non-v1 rows without touching the rest."""
    df = pd.DataFrame(
        {
            "policy": ["act", "xvla_libero", "diffusion_policy", "pi0", "random"],
            "env": ["pusht"] * 5,
            "success": [True, False, True, False, True],
        }
    )
    out = filter_to_v1_policies(df)
    assert set(out["policy"]) == {"act", "diffusion_policy", "random"}
    assert "xvla_libero" not in set(out["policy"])


def test_space_and_dashboard_share_one_v1_policy_gate() -> None:
    """Regression guard against future drift (task #142).

    The Gradio Space and this dashboard must apply the identical v1
    policy gate. Both now re-export from
    ``lerobot_bench.leaderboard_filter`` instead of redefining the tuple
    + filter, so the two surfaces and the canonical module are the *same
    object* -- a future edit to one cannot silently diverge from the
    other. Loaded via importlib because both helper files are named
    ``_helpers`` and would otherwise collide in ``sys.modules``.
    """
    from lerobot_bench import leaderboard_filter

    space_helpers_path = _DASHBOARD_DIR.parent / "space" / "_helpers.py"
    space_spec = importlib.util.spec_from_file_location("space_helpers", space_helpers_path)
    assert space_spec is not None and space_spec.loader is not None
    space_helpers = importlib.util.module_from_spec(space_spec)
    sys.modules["space_helpers"] = space_helpers
    space_spec.loader.exec_module(space_helpers)

    assert _dashboard_helpers.V1_POLICIES is leaderboard_filter.V1_POLICIES
    assert space_helpers.V1_POLICIES is leaderboard_filter.V1_POLICIES
    assert _dashboard_helpers.filter_to_v1_policies is leaderboard_filter.filter_to_v1_policies
    assert space_helpers.filter_to_v1_policies is leaderboard_filter.filter_to_v1_policies


def test_load_results_parquet_filters_xvla_but_parquet_preserves_it(
    tmp_path: Path,
) -> None:
    """Parquet on disk keeps xvla rows for reproducibility; the dashboard's
    loader drops them so build_live_leaderboard / run_anomaly_review /
    policy cards never see them.
    """
    rows = _cell_rows(
        policy="act", env="pusht", per_seed_successes=[3], n_episodes_per_seed=10
    ) + _cell_rows(
        policy="xvla_libero",
        env="libero_10",
        per_seed_successes=[5],
        n_episodes_per_seed=10,
    )
    df_in = _results_frame(rows)
    parquet = tmp_path / "results.parquet"
    df_in.to_parquet(parquet, index=False)
    clear_results_cache()

    # Raw file on disk still has xvla.
    raw = pd.read_parquet(parquet)
    assert "xvla_libero" in set(raw["policy"])

    # Dashboard loader filters.
    loaded = load_results_parquet(parquet)
    assert loaded is not None
    assert "xvla_libero" not in set(loaded["policy"])
    assert set(loaded["policy"]) == {"act"}

    # Live leaderboard built on the loaded df does not surface xvla.
    board = build_live_leaderboard(loaded)
    assert [r.policy for r in board] == ["act"]
