"""Tests for the #165 publish pre-flight REQUIRED policy/env coverage gate.

The gap this closes: before #165, ``scripts/publish_results._preflight``
validated parquet schema, manifest, video presence, and the act×aloha
stale-rows floor -- but never checked that every *intended* (policy, env)
cell was actually present. A v1 parquet silently missing an entire policy
(or an entire benchmark env) could publish "clean".

The REQUIRED set is the runnable v1 policies (``V1_POLICIES`` ∩ registry
runnable) crossed with each policy's ``env_compat``, intersected with the
env axis the v1 sweep runs (``configs/sweep_full.yaml``). Pairs present in
the data but NOT required are OPTIONAL and must never be an error.

Construction mirrors ``tests/test_publish_results.py``: synthetic parquet
built from ``RESULT_SCHEMA`` rows, written via ``_atomic_write_parquet``,
plus a minimal valid manifest. No network, no torch, no huggingface_hub.

NOTE on the legacy-suite shim: ``tests/conftest.py`` neutralizes the
coverage gate (empty REQUIRED set) ONLY for ``tests/test_publish_results``.
This module is *not* covered by that shim, so the gate runs for real here;
where we want a controlled REQUIRED set we monkeypatch the injection point
``publish_results._required_coverage_pairs_for_preflight`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts import publish_results as pr

from lerobot_bench.checkpointing import RESULT_SCHEMA, _atomic_write_parquet
from lerobot_bench.policies import PolicyRegistry, PolicySpec

# A small, self-contained REQUIRED set used by the gate tests below so they
# don't depend on the exact contents of configs/. Mirrors the real shape:
# a baseline on two envs, a single-env specialist, a multi-env VLA.
_TEST_REQUIRED: frozenset[tuple[str, str]] = frozenset(
    {
        ("random", "pusht"),
        ("random", "aloha_transfer_cube"),
        ("act", "aloha_transfer_cube"),
        ("smolvla_libero", "libero_10"),
    }
)


# --------------------------------------------------------------------- #
# Fixture helpers (mirror tests/test_publish_results.py)                 #
# --------------------------------------------------------------------- #


def _row(
    *,
    policy: str,
    env: str,
    seed: int = 0,
    episode_index: int = 0,
    success: bool = True,
) -> dict[str, Any]:
    """One RESULT_SCHEMA row. video_sha256 empty so the video gate is a no-op."""
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": episode_index,
        "success": success,
        "return_": 1.0 if success else 0.0,
        "n_steps": 10,
        "wallclock_s": 0.05,
        "video_sha256": "",
        "code_sha": "deadbeef",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-01T00:00:00+00:00",
        "errored": False,
        "eval_run_id": "test-run",
    }


def _build_sweep_dir(tmp_path: Path, pairs: list[tuple[str, str]]) -> dict[str, Path]:
    """Build a sweep dir whose parquet contains exactly one cell per (policy, env).

    Each cell gets a single episode -- enough to count as "present" for the
    coverage gate. Returns the canonical publish input paths.
    """
    sweep_dir = tmp_path / "sweep-test"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "videos").mkdir()

    rows = [_row(policy=p, env=e) for p, e in pairs]
    df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
    results_path = sweep_dir / "results.parquet"
    _atomic_write_parquet(results_path, df)

    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "started_utc": "2026-05-03T00:00:00+00:00",
                "finished_utc": "2026-05-03T01:00:00+00:00",
                "code_sha": "abc",
                "lerobot_version": "0.5.1",
                "config_path": "configs/sweep_full.yaml",
                "cells": [],
            }
        )
    )
    return {
        "results_path": results_path,
        "manifest_path": manifest_path,
        "videos_dir": sweep_dir / "videos",
    }


def _preflight(paths: dict[str, Path]) -> Any:
    return pr._preflight(
        results_path=paths["results_path"],
        manifest_path=paths["manifest_path"],
        videos_dir=paths["videos_dir"],
        skip_videos=True,
    )


@pytest.fixture
def _fixed_required(monkeypatch: pytest.MonkeyPatch) -> frozenset[tuple[str, str]]:
    """Pin the gate's REQUIRED set to :data:`_TEST_REQUIRED` for deterministic asserts."""
    monkeypatch.setattr(
        pr,
        "_required_coverage_pairs_for_preflight",
        lambda: _TEST_REQUIRED,
    )
    return _TEST_REQUIRED


# --------------------------------------------------------------------- #
# (a) complete parquet passes                                           #
# --------------------------------------------------------------------- #


def test_complete_coverage_passes(
    tmp_path: Path, _fixed_required: frozenset[tuple[str, str]]
) -> None:
    """Every REQUIRED (policy, env) pair present -> no coverage error."""
    paths = _build_sweep_dir(tmp_path, sorted(_fixed_required))
    result = _preflight(paths)
    assert result.error is None, result.error


# --------------------------------------------------------------------- #
# (b) missing a required pair fails, naming the pair                     #
# --------------------------------------------------------------------- #


def test_missing_required_pair_fails_with_name(
    tmp_path: Path, _fixed_required: frozenset[tuple[str, str]]
) -> None:
    """Drop one REQUIRED pair -> exit-3 error that names the missing pair."""
    dropped = ("act", "aloha_transfer_cube")
    present = [p for p in sorted(_fixed_required) if p != dropped]
    paths = _build_sweep_dir(tmp_path, present)

    result = _preflight(paths)
    assert result.error is not None
    assert "missing REQUIRED" in result.error
    assert "act" in result.error
    assert "aloha_transfer_cube" in result.error


def test_missing_whole_env_fails_via_cli_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entire env absent surfaces as exit 3 through the CLI dry-run path."""
    monkeypatch.setattr(
        pr,
        "_required_coverage_pairs_for_preflight",
        lambda: _TEST_REQUIRED,
    )
    # Present everything except the two pusht-axis... here drop both pusht cells.
    present = [p for p in sorted(_TEST_REQUIRED) if p[1] != "pusht"]
    paths = _build_sweep_dir(tmp_path, present)

    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--skip-videos",
            "--dry-run",
        ]
    )
    assert rc == 3


# --------------------------------------------------------------------- #
# (c) an EXTRA non-required pair still passes                            #
# --------------------------------------------------------------------- #


def test_extra_optional_pair_still_passes(
    tmp_path: Path, _fixed_required: frozenset[tuple[str, str]]
) -> None:
    """A pair present in the data but NOT required is OPTIONAL, never an error."""
    pairs = [*sorted(_fixed_required), ("diffusion_policy", "libero_goal")]
    paths = _build_sweep_dir(tmp_path, pairs)
    result = _preflight(paths)
    assert result.error is None, result.error


# --------------------------------------------------------------------- #
# (d) REQUIRED-set derivation excludes non-v1 + non-runnable policies    #
# --------------------------------------------------------------------- #


def _spec(
    name: str,
    *,
    env_compat: tuple[str, ...],
    is_baseline: bool = False,
    runnable: bool = True,
) -> PolicySpec:
    """Minimal PolicySpec; a non-baseline with no revision_sha is non-runnable."""
    if is_baseline:
        return PolicySpec(name=name, is_baseline=True, env_compat=env_compat)
    return PolicySpec(
        name=name,
        is_baseline=False,
        env_compat=env_compat,
        repo_id=f"org/{name}",
        revision_sha="abc123" if runnable else None,
        fp_precision="fp32",
    )


def test_required_set_excludes_non_v1_and_non_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derivation drops non-v1 (xvla) and non-runnable policies; honors sweep envs."""
    # Pin the sweep env axis so the test is independent of configs/sweep_full.yaml.
    sweep_envs = ("pusht", "aloha_transfer_cube", "libero_10")
    monkeypatch.setattr(pr, "_v1_sweep_envs", lambda: (sweep_envs, False))

    registry = PolicyRegistry(
        {
            # v1 + runnable -> included (libero_goal NOT in sweep envs -> dropped).
            "smolvla_libero": _spec("smolvla_libero", env_compat=("libero_10", "libero_goal")),
            # v1 baseline -> included on every sweep env it supports.
            "random": _spec(
                "random",
                env_compat=("pusht", "aloha_transfer_cube"),
                is_baseline=True,
            ),
            # v1 but NOT runnable (no revision_sha) -> excluded entirely.
            "act": _spec("act", env_compat=("aloha_transfer_cube",), runnable=False),
            # NOT a v1 policy (deferred to v1.1) -> excluded even though runnable.
            "xvla_libero": _spec("xvla_libero", env_compat=("libero_10",)),
        }
    )

    pairs = pr._required_coverage_pairs(registry)

    assert ("smolvla_libero", "libero_10") in pairs
    assert ("smolvla_libero", "libero_goal") not in pairs  # not a sweep env
    assert ("random", "pusht") in pairs
    assert ("random", "aloha_transfer_cube") in pairs
    # Non-runnable v1 policy excluded:
    assert not any(p == "act" for p, _ in pairs)
    # Non-v1 policy excluded:
    assert not any(p == "xvla_libero" for p, _ in pairs)


def test_required_set_falls_back_to_env_compat_union(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the sweep YAML yields no clean env list, fall back to env_compat union + log."""
    monkeypatch.setattr(pr, "_v1_sweep_envs", lambda: ((), True))
    registry = PolicyRegistry(
        {
            "smolvla_libero": _spec("smolvla_libero", env_compat=("libero_10", "libero_goal")),
        }
    )
    with caplog.at_level("WARNING"):
        pairs = pr._required_coverage_pairs(registry)
    # Both env_compat envs are required despite no sweep env intersection.
    assert ("smolvla_libero", "libero_10") in pairs
    assert ("smolvla_libero", "libero_goal") in pairs
    assert any("fall" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------- #
# Live-config sanity: the real derivation is non-empty and sane          #
# --------------------------------------------------------------------- #


def test_live_required_set_is_sane() -> None:
    """The real configs derive a non-empty REQUIRED set with the expected shape."""
    pairs = pr._default_required_coverage_pairs()
    assert pairs  # non-empty
    # act is aloha-only; never required on a libero env.
    assert ("act", "aloha_transfer_cube") in pairs
    assert not any(p == "act" and e != "aloha_transfer_cube" for p, e in pairs)
    # xvla is deferred to v1.1 -> never required.
    assert not any(p == "xvla_libero" for p, _ in pairs)
    # pi0 family deferred -> never required.
    assert not any(p.startswith("pi0") or p.startswith("pi05") for p, _ in pairs)
