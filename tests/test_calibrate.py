"""Tests for ``scripts/calibrate.py``.

The script lives in ``scripts/`` (not ``src/``) because it is a CLI
entrypoint, not library code. ``scripts/__init__.py`` makes it
importable as ``from scripts.calibrate import ...`` -- no path
manipulation needed (pyproject already lists ``scripts`` under
``[tool.ruff].src``).

These tests cover the CI-runnable surface only:
    1-6.  Auto-downscope rule (the brain of the matrix shaping).
    7-10. Plan-cells: filters, skipped, incompat.
    11-12. CalibrationReport JSON roundtrip + filename derivation.
    13.   Dry-run path does NOT pull torch into ``sys.modules``.
    14.   "no runnable cells" exits 3.
    15.   ``measure_cell`` returns ``status="skipped"`` for unrunnable specs.
    16.   The script source has no top-level ``import torch``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from scripts import calibrate
from scripts.calibrate import (
    CalibrationReport,
    CellTiming,
    auto_downscope,
    measure_cell,
    plan_cells,
    run_calibration,
    write_report,
)

from lerobot_bench.envs import EnvRegistry
from lerobot_bench.policies import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"
CALIBRATE_SOURCE = REPO_ROOT / "scripts" / "calibrate.py"


# --------------------------------------------------------------------- #
# Fixtures + helpers                                                    #
# --------------------------------------------------------------------- #


def _ok(
    *,
    policy: str = "p",
    env: str = "e",
    mean_ms: float = 10.0,
    p95_ms: float = 12.0,
    vram_mb: float = 1000.0,
) -> CellTiming:
    return CellTiming(
        policy=policy,
        env=env,
        n_steps_measured=20,
        mean_ms_per_step=mean_ms,
        p95_ms_per_step=p95_ms,
        vram_peak_mb=vram_mb,
        status="ok",
        error=None,
        recommended=None,
    )


# --------------------------------------------------------------------- #
# 1-6. Auto-downscope rule                                              #
# --------------------------------------------------------------------- #


def test_auto_downscope_normal_keeps_base() -> None:
    timing = _ok(mean_ms=10.0, vram_mb=1000.0)
    assert auto_downscope(timing) == {"seeds": 5, "episodes": 50}


def test_auto_downscope_high_vram_cuts_episodes() -> None:
    timing = _ok(mean_ms=10.0, vram_mb=6000.0)
    assert auto_downscope(timing) == {"seeds": 5, "episodes": 25}


def test_auto_downscope_very_high_vram_cuts_seeds() -> None:
    timing = _ok(mean_ms=10.0, vram_mb=7500.0)
    assert auto_downscope(timing) == {"seeds": 2, "episodes": 50}


def test_auto_downscope_slow_cuts_episodes() -> None:
    timing = _ok(mean_ms=150.0, vram_mb=1000.0)
    assert auto_downscope(timing) == {"seeds": 5, "episodes": 25}


def test_auto_downscope_very_slow_cuts_seeds() -> None:
    timing = _ok(mean_ms=600.0, vram_mb=1000.0)
    assert auto_downscope(timing) == {"seeds": 2, "episodes": 50}


def test_auto_downscope_failed_cell_drops() -> None:
    timing = CellTiming(
        policy="p",
        env="e",
        n_steps_measured=0,
        mean_ms_per_step=0.0,
        p95_ms_per_step=0.0,
        vram_peak_mb=0.0,
        status="oom",
        error="cuda OOM",
    )
    assert auto_downscope(timing) == {"seeds": 0, "episodes": 0}


def test_auto_downscope_respects_custom_base() -> None:
    timing = _ok(mean_ms=10.0, vram_mb=1000.0)
    assert auto_downscope(timing, base_seeds=3, base_episodes=10) == {
        "seeds": 3,
        "episodes": 10,
    }


# --------------------------------------------------------------------- #
# 7-10. Plan cells                                                      #
# --------------------------------------------------------------------- #


def test_plan_cells_marks_unrunnable_skipped(tmp_path: Path) -> None:
    """A policy with revision_sha=None is marked skipped by the planner.

    Uses a tmp policies.yaml since the shipped diffusion_policy/act now have
    locked SHAs (Day 0a, 2026-05-03).
    """
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
        """
policies:
  - name: no_op
    is_baseline: true
    env_compat: [pusht]
  - name: not_yet_locked
    is_baseline: false
    env_compat: [pusht]
    repo_id: lerobot/some_future_policy
    revision_sha: null
    fp_precision: fp32
"""
    )
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)

    plan = plan_cells(policies, envs)

    not_locked_pusht = next(
        (status for (p, e, status) in plan if p.name == "not_yet_locked" and e.name == "pusht"),
        None,
    )
    assert not_locked_pusht == "skipped", (
        f"expected not_yet_locked x pusht skipped, got {not_locked_pusht}"
    )

    no_op_pusht = next(
        (status for (p, e, status) in plan if p.name == "no_op" and e.name == "pusht"),
        None,
    )
    assert no_op_pusht == "ready"


def test_plan_cells_marks_incompat(tmp_path: Path) -> None:
    """A policy with env_compat=[pusht] only -> aloha cell is incompat."""
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
        """
policies:
  - name: pusht_only_baseline
    is_baseline: true
    env_compat: [pusht]
"""
    )
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)

    plan = plan_cells(policies, envs)
    statuses = {(p.name, e.name): status for (p, e, status) in plan}

    assert statuses[("pusht_only_baseline", "pusht")] == "ready"
    assert statuses[("pusht_only_baseline", "aloha_transfer_cube")] == "incompat"


def test_plan_cells_filter_by_policy() -> None:
    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)

    plan = plan_cells(policies, envs, policy_filter="no_op")
    policy_names = {p.name for (p, _e, _s) in plan}
    assert policy_names == {"no_op"}


def test_plan_cells_filter_by_env() -> None:
    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)

    plan = plan_cells(policies, envs, env_filter="pusht")
    env_names = {e.name for (_p, e, _s) in plan}
    assert env_names == {"pusht"}


# --------------------------------------------------------------------- #
# 11-12. Report I/O                                                     #
# --------------------------------------------------------------------- #


def test_calibration_report_roundtrip() -> None:
    cell = _ok(policy="no_op", env="pusht")
    cell = CellTiming(**{**cell.__dict__, "recommended": {"seeds": 5, "episodes": 50}})
    report = CalibrationReport(
        timestamp_utc="2026-05-01T12:34:56+00:00",
        git_sha="deadbeef",
        lerobot_version="0.5.1",
        cells=(cell,),
    )
    data = report.to_json()
    # Roundtrip through json bytes to flush any non-JSON-safe types.
    parsed = json.loads(json.dumps(data))
    rebuilt = CalibrationReport.from_json(parsed)
    assert rebuilt == report


def test_calibration_report_roundtrip_no_lerobot() -> None:
    """``lerobot_version`` is None pre-Day-0a; must roundtrip cleanly."""
    report = CalibrationReport(
        timestamp_utc="2026-05-01T00:00:00+00:00",
        git_sha="abc123",
        lerobot_version=None,
        cells=(),
    )
    rebuilt = CalibrationReport.from_json(json.loads(json.dumps(report.to_json())))
    assert rebuilt == report


def test_write_report_uses_timestamp_date(tmp_path: Path) -> None:
    """Filename comes from the report's own timestamp, not now()."""
    report = CalibrationReport(
        timestamp_utc="2026-03-15T22:11:00+00:00",
        git_sha="abc",
        lerobot_version=None,
        cells=(),
    )
    out_path = write_report(report, tmp_path)
    assert out_path.name == "calibration-2026-03-15.json"
    assert out_path.exists()
    # And the contents are the JSON we expect.
    parsed = json.loads(out_path.read_text())
    assert parsed["timestamp_utc"] == "2026-03-15T22:11:00+00:00"


# --------------------------------------------------------------------- #
# 13. Lazy-import contract                                              #
# --------------------------------------------------------------------- #


def test_calibrate_module_source_has_no_top_level_torch_import() -> None:
    """Static guarantee: ``import torch`` MUST be inside a function body.

    Easier and faster than a subprocess + sys.modules check, and gives
    a clear error message if a future edit accidentally hoists the
    import to module scope.
    """
    tree = ast.parse(CALIBRATE_SOURCE.read_text())
    for node in tree.body:  # only top-level statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch", (
                    "scripts/calibrate.py must not import torch at module scope -- "
                    "lazy-import inside measure_cell instead."
                )
                assert not alias.name.startswith("torch."), (
                    f"top-level import of {alias.name} would pull torch in eagerly."
                )
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module != "torch" and not node.module.startswith("torch."), (
                f"top-level `from {node.module}` would pull torch in eagerly."
            )


def test_calibrate_module_source_has_no_top_level_lerobot_import() -> None:
    """Same contract for lerobot -- module must import without lerobot installed."""
    tree = ast.parse(CALIBRATE_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "lerobot", (
                    "scripts/calibrate.py must not import lerobot at module scope."
                )
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module != "lerobot" and not node.module.startswith("lerobot."), (
                f"top-level `from {node.module}` would require lerobot at import time."
            )


# --------------------------------------------------------------------- #
# 14. CLI exit codes                                                    #
# --------------------------------------------------------------------- #


def test_main_no_runnable_cells_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A --policy filter that matches nothing -> exit 3."""
    out_dir = tmp_path / "out"
    rc = calibrate.main(
        [
            "--policies-yaml",
            str(DEFAULT_POLICIES_YAML),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-dir",
            str(out_dir),
            "--policy",
            "this_policy_does_not_exist",
        ]
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "no runnable cells" in err


def test_main_dry_run_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run is a no-op orchestration: exit 0, no JSON written."""
    out_dir = tmp_path / "out"
    rc = calibrate.main(
        [
            "--policies-yaml",
            str(DEFAULT_POLICIES_YAML),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    # No JSON file should land on disk during dry-run.
    assert not out_dir.exists() or list(out_dir.glob("calibration-*.json")) == []


def test_main_missing_yaml_exits_4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = calibrate.main(
        [
            "--policies-yaml",
            str(tmp_path / "nope.yaml"),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 4
    err = capsys.readouterr().err
    assert "config not found" in err
    assert "resume" in err


# --------------------------------------------------------------------- #
# 15. measure_cell short-circuits non-runnable                          #
# --------------------------------------------------------------------- #


def test_measure_cell_skipped_when_not_runnable(tmp_path: Path) -> None:
    """A policy with revision_sha=None short-circuits in measure_cell.

    Uses a tmp policies.yaml since the shipped diffusion_policy/act now have
    locked SHAs (Day 0a, 2026-05-03).
    """
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
        """
policies:
  - name: not_yet_locked
    is_baseline: false
    env_compat: [pusht]
    repo_id: lerobot/some_future_policy
    revision_sha: null
    fp_precision: fp32
"""
    )
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    spec = policies.get("not_yet_locked")
    pusht = envs.get("pusht")

    timing = measure_cell(spec, pusht, n_steps=1, n_episodes=1)
    assert timing.status == "skipped"
    assert timing.policy == "not_yet_locked"
    assert timing.env == "pusht"
    assert timing.error is not None
    assert "Day 0a" in timing.error
    # Timing fields are zero placeholders for non-ok cells.
    assert timing.n_steps_measured == 0
    assert timing.mean_ms_per_step == 0.0


def test_measure_cell_missing_runtime_returns_error() -> None:
    """When any required runtime dep is missing (torch / lerobot /
    gymnasium / gym-pusht), measure_cell short-circuits to
    ``status="error"`` with a clear message instead of raising.
    """
    try:
        import gym_pusht  # noqa: F401
        import gymnasium  # noqa: F401
        import lerobot  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
        envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
        no_op = policies.get("no_op")
        pusht = envs.get("pusht")
        timing = measure_cell(no_op, pusht, n_steps=1, n_episodes=1)
        assert timing.status == "error"
        assert timing.error is not None
        # Either the top-level lazy-import guard ("missing runtime: ...")
        # or load_env's namespace-import ImportError fires; both fine.
        assert "missing" in timing.error.lower() or "ImportError" in timing.error
        return
    pytest.skip("all runtimes installed -- this test exercises the missing-runtime branch")


def test_measure_cell_baseline_runs_when_runtime_available() -> None:
    """On the dev box where torch + lerobot + gym are installed, the
    no_op baseline against PushT should yield ``status="ok"`` with real
    timing and VRAM numbers. Skipped in CI fast (no runtime).
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    pytest.importorskip("gymnasium")
    pytest.importorskip("gym_pusht")

    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    no_op = policies.get("no_op")
    pusht = envs.get("pusht")

    timing = measure_cell(no_op, pusht, n_steps=2, n_episodes=1, device="cpu")
    assert timing.status == "ok", f"got status={timing.status} error={timing.error}"
    assert timing.n_steps_measured == 2
    assert timing.mean_ms_per_step >= 0.0
    assert timing.recommended is not None  # auto_downscope ran


# --------------------------------------------------------------------- #
# Bonus: end-to-end orchestration (no torch / no lerobot path)          #
# --------------------------------------------------------------------- #


def test_run_calibration_writes_report_with_skipped_and_error_cells(
    tmp_path: Path,
) -> None:
    """End-to-end: should write a JSON with at least the baseline cells.

    Without lerobot installed, baselines come back as status="error"
    and pretrained come back as status="skipped". Either way, the JSON
    is written and exit code is 2 (partial).
    """
    out_dir = tmp_path / "out"
    report, exit_code = run_calibration(
        DEFAULT_POLICIES_YAML,
        DEFAULT_ENVS_YAML,
        out_dir=out_dir,
        steps=1,
        episodes=1,
        policy_filter=None,
        env_filter=None,
        dry_run=False,
    )
    # Exit code is 0 only if every cell is "ok" -- which requires the
    # measurement loop. In the scaffold state, expect 2 (partial).
    assert exit_code == 2
    json_files = list(out_dir.glob("calibration-*.json"))
    assert len(json_files) == 1
    parsed = json.loads(json_files[0].read_text())
    # The report must round-trip through CalibrationReport.from_json.
    rebuilt = CalibrationReport.from_json(parsed)
    assert rebuilt == report
    # And every cell carries a status from the documented vocabulary.
    valid_statuses = {"ok", "oom", "skipped", "error"}
    for cell in rebuilt.cells:
        assert cell.status in valid_statuses
