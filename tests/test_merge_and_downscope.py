"""Tests for ``scripts/merge_calibration.py`` and ``scripts/auto_downscope.py``.

These two scripts shape the Phase-4 sweep config from per-policy calibration
runs, so correctness matters: a bug here silently writes too-many or
too-few episodes into ``configs/sweep_full.yaml``.

Test scope:
    1. ``merge_calibration``: dedupe by (policy, env), last-wins.
    2. ``merge_calibration``: cells from a missing input file are skipped
       with a warning, not a hard failure.
    3. ``auto_downscope.build_overrides``: only ``ok`` cells contribute.
    4. ``auto_downscope.build_overrides``: ``n_episodes`` written only when
       ``recommended.episodes < base_episodes``.
    5. ``auto_downscope.build_overrides``: ``seeds_subset`` written only when
       ``recommended.seeds < base_seeds``, and warning is surfaced.
    6. ``auto_downscope.merge_into_sweep``: policies/envs come from the
       union of ``ok`` cells; non-overridable top-level fields are preserved.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


merge_calibration = _load_module(
    "merge_calibration", REPO_ROOT / "scripts" / "merge_calibration.py"
)
auto_downscope = _load_module("auto_downscope", REPO_ROOT / "scripts" / "auto_downscope.py")


def _cell(policy: str, env: str, *, status: str = "ok", episodes: int = 50, seeds: int = 5):
    return {
        "policy": policy,
        "env": env,
        "status": status,
        "mean_step_ms": 10.0,
        "vram_peak_mb": 100.0,
        "recommended": {"episodes": episodes, "seeds": seeds},
    }


def _report(cells, *, timestamp_utc: str, git_sha: str = "abc123"):
    return {
        "timestamp_utc": timestamp_utc,
        "git_sha": git_sha,
        "lerobot_version": "0.5.1",
        "cells": cells,
    }


def test_merge_calibration_dedupes_with_last_wins(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(
            _report(
                [_cell("p1", "e1", episodes=25), _cell("p1", "e2", episodes=50)],
                timestamp_utc="2026-05-01T00:00:00Z",
            )
        )
    )
    b.write_text(
        json.dumps(
            _report(
                [_cell("p1", "e1", episodes=10)],  # overrides a's (p1, e1)
                timestamp_utc="2026-05-02T00:00:00Z",
            )
        )
    )
    out = tmp_path / "merged.json"
    rc = subprocess.call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "merge_calibration.py"),
            str(a),
            str(b),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    merged = json.loads(out.read_text())
    cells = {(c["policy"], c["env"]): c for c in merged["cells"]}
    assert len(cells) == 2
    # b is later in argv → (p1, e1) episodes should be 10 (b's value)
    assert cells[("p1", "e1")]["recommended"]["episodes"] == 10
    assert cells[("p1", "e2")]["recommended"]["episodes"] == 50
    # Merged timestamp_utc = max
    assert merged["timestamp_utc"] == "2026-05-02T00:00:00Z"


def test_merge_calibration_skips_missing_file(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_report([_cell("p1", "e1")], timestamp_utc="2026-05-01T00:00:00Z")))
    missing = tmp_path / "does-not-exist.json"
    out = tmp_path / "merged.json"
    rc = subprocess.call(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "merge_calibration.py"),
            str(a),
            str(missing),
            "--out",
            str(out),
        ]
    )
    # Missing input is a warning, not a fatal error.
    assert rc == 0
    merged = json.loads(out.read_text())
    assert len(merged["cells"]) == 1


def test_build_overrides_skips_non_ok_cells():
    cells = [
        _cell("p1", "e1", status="ok", episodes=50, seeds=5),
        _cell("p1", "e2", status="skipped", episodes=25, seeds=3),
        _cell("p1", "e3", status="failed", episodes=10, seeds=2),
    ]
    overrides, warnings = auto_downscope.build_overrides(cells)
    # Only ok cells considered; (p1, e1) has no override because recommended == base.
    assert overrides == {}
    assert warnings == []


def test_build_overrides_emits_n_episodes_only_when_below_base():
    cells = [
        _cell("p1", "e1", episodes=50, seeds=5),  # at base — no entry
        _cell("p1", "e2", episodes=25, seeds=5),  # below base — n_episodes
        _cell("p1", "e3", episodes=10, seeds=5),  # well below
    ]
    overrides, warnings = auto_downscope.build_overrides(cells, base_episodes=50, base_seeds=5)
    assert "e1" not in overrides.get("p1", {})
    assert overrides["p1"]["e2"] == {"n_episodes": 25}
    assert overrides["p1"]["e3"] == {"n_episodes": 10}
    assert warnings == []


def test_build_overrides_seeds_below_base_warns():
    cells = [
        _cell("p1", "e1", episodes=50, seeds=2),  # seeds below base
    ]
    overrides, warnings = auto_downscope.build_overrides(cells, base_episodes=50, base_seeds=5)
    assert overrides["p1"]["e1"]["seeds_subset"] == [0, 1]
    assert warnings == [("p1", "e1", 2)]


def test_merge_into_sweep_preserves_other_fields():
    existing = {
        "seeds": [0, 1, 2, 3, 4],
        "episodes_per_seed": 10,
        "videos_dir": "results/videos",
        # These will be overwritten by merge_into_sweep:
        "policies": ["stale-policy"],
        "envs": ["stale-env"],
        "overrides": {"stale-policy": {"stale-env": {"n_episodes": 1}}},
    }
    cells = [
        _cell("p1", "e1", episodes=25),
        _cell("p2", "e2", episodes=50),
        _cell("p3", "e3", status="failed"),  # not in policies/envs union
    ]
    overrides, _ = auto_downscope.build_overrides(cells)
    merged = auto_downscope.merge_into_sweep(existing, overrides, cells)
    # Other top-level fields preserved.
    assert merged["seeds"] == [0, 1, 2, 3, 4]
    assert merged["episodes_per_seed"] == 10
    assert merged["videos_dir"] == "results/videos"
    # Policies/envs derived from ok cells only.
    assert merged["policies"] == ["p1", "p2"]
    assert merged["envs"] == ["e1", "e2"]
    # Overrides reflect new shape, not stale state.
    assert merged["overrides"] == {"p1": {"e1": {"n_episodes": 25}}}
