"""Tests for ``scripts/publish_results.py``.

Same playbook as ``tests/test_run_one.py`` and ``tests/test_run_sweep.py``:
synthetic parquet built from ``RESULT_SCHEMA``-compatible rows, fake
MP4s on disk as raw byte blobs, and a mocked ``HfApi`` injected via
:data:`scripts.publish_results._get_hf_api`. No network, no torch, no
huggingface_hub at module import time.

Coverage:
    1.  Pre-flight: parquet missing -> exit 3.
    2.  Pre-flight: schema drift -> exit 3.
    3.  Pre-flight: manifest unreadable -> exit 3.
    4.  Pre-flight: video_sha256 references missing MP4 -> exit 3.
    5.  Auth path: whoami() raises -> exit 4.
    6.  Auth path: repo_info() raises -> exit 4.
    7.  Dry-run: no upload_folder call recorded; staging dir written.
    8.  Happy path: upload_folder called once with expected allow_patterns.
    9.  --skip-videos: allow_patterns excludes ``videos/*.mp4``.
    10. Oversize video: skipped from staging, exit 2, warning logged.
    11. AST guard: no top-level huggingface_hub import.
    12. Idempotent: running twice produces same provenance modulo published_utc.
    13. Hard upload failure -> exit 5.
    14. _video_filename naming matches scripts/run_one.render_episodes_to_videos.
    15. CLI main() returns the right exit codes end-to-end.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from scripts import publish_results as pr

from lerobot_bench.checkpointing import RESULT_SCHEMA, _atomic_write_parquet

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_SOURCE = REPO_ROOT / "scripts" / "publish_results.py"


# --------------------------------------------------------------------- #
# Fixture helpers                                                       #
# --------------------------------------------------------------------- #


def _fake_row(
    *,
    policy: str = "random",
    env: str = "pusht",
    seed: int = 0,
    episode_index: int = 0,
    success: bool = True,
    video_sha256: str = "deadbeef",
) -> dict[str, Any]:
    """Build one parquet row matching RESULT_SCHEMA exactly."""
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": episode_index,
        "success": success,
        "return_": 1.0 if success else 0.0,
        "n_steps": 10,
        "wallclock_s": 0.05,
        "video_sha256": video_sha256,
        "code_sha": "deadbeef",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-01T00:00:00+00:00",
    }


def _build_sweep_dir(
    tmp_path: Path,
    *,
    n_cells: int = 4,
    n_episodes_per_cell: int = 5,
    video_bytes: bytes = b"\x00\x01\x02FAKEMP4PAYLOAD",
    create_videos: bool = True,
    extra_oversize: int = 0,
    oversize_bytes: int = 0,
) -> dict[str, Path]:
    """Build a synthetic sweep directory: parquet + manifest + videos.

    Returns a dict of ``{"results_path", "manifest_path", "videos_dir"}``.
    Each cell uses a unique (policy, env, seed) tuple so there are no
    duplicate-key clashes. ``video_sha256`` is non-empty for every row,
    so pre-flight expects every MP4 on disk.
    """
    sweep_dir = tmp_path / "sweep-test"
    sweep_dir.mkdir(parents=True)
    videos_dir = sweep_dir / "videos"
    videos_dir.mkdir()

    rows: list[dict[str, Any]] = []
    for cell_idx in range(n_cells):
        # Differentiate cells by seed to keep policy/env constant
        # (tests assert against expected uploaded counts).
        for ep_idx in range(n_episodes_per_cell):
            row = _fake_row(
                policy="random",
                env="pusht",
                seed=cell_idx,
                episode_index=ep_idx,
                video_sha256=f"sha-{cell_idx}-{ep_idx}",
            )
            rows.append(row)
            if create_videos:
                fname = pr._video_filename(
                    policy=row["policy"],
                    env=row["env"],
                    seed=row["seed"],
                    episode_index=row["episode_index"],
                )
                (videos_dir / fname).write_bytes(video_bytes)

    # Optionally add extra oversize videos (not referenced in parquet,
    # so they show as orphan files; useful for size-cap tests).
    for i in range(extra_oversize):
        big = b"\x00" * oversize_bytes
        (videos_dir / f"random__pusht__seed99__ep{i:03d}.mp4").write_bytes(big)

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
                "config_path": "configs/sweep_mini.yaml",
                "cells": [],
            }
        )
    )

    return {
        "results_path": results_path,
        "manifest_path": manifest_path,
        "videos_dir": videos_dir,
        "sweep_dir": sweep_dir,
    }


def _make_fake_api() -> MagicMock:
    """Build a MagicMock standing in for huggingface_hub.HfApi.

    ``whoami`` and ``repo_info`` return truthy stubs by default;
    ``upload_folder`` records its kwargs for assertion. Tests that need
    failure modes override individual methods.
    """
    api = MagicMock(name="HfApi")
    api.whoami.return_value = {"name": "test-user"}
    api.repo_info.return_value = MagicMock(id="Theozinh0/lerobot-bench-results-v1")
    api.upload_folder.return_value = MagicMock(commit_url="https://hf.co/fake/commit")
    return api


# --------------------------------------------------------------------- #
# 1-4. Pre-flight                                                       #
# --------------------------------------------------------------------- #


def test_preflight_missing_parquet_exits_3(tmp_path: Path) -> None:
    """No parquet on disk -> exit 3 with explicit message."""
    rc = pr.main(
        [
            "--results-path",
            str(tmp_path / "missing.parquet"),
            "--manifest-path",
            str(tmp_path / "manifest.json"),
            "--videos-dir",
            str(tmp_path / "videos"),
            "--dry-run",
        ]
    )
    assert rc == 3


def test_preflight_schema_drift_exits_3(tmp_path: Path) -> None:
    """A parquet with the wrong columns -> exit 3."""
    bad_df = pd.DataFrame({"unrelated": [1, 2], "columns": [3, 4]})
    bad_path = tmp_path / "bad.parquet"
    bad_df.to_parquet(bad_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    rc = pr.main(
        [
            "--results-path",
            str(bad_path),
            "--manifest-path",
            str(manifest_path),
            "--videos-dir",
            str(tmp_path / "videos"),
            "--dry-run",
        ]
    )
    assert rc == 3


def test_preflight_unreadable_manifest_exits_3(tmp_path: Path) -> None:
    """Manifest is not valid JSON -> exit 3."""
    paths = _build_sweep_dir(tmp_path, n_cells=1, n_episodes_per_cell=1)
    paths["manifest_path"].write_text("{not: valid json,,,}")

    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--dry-run",
        ]
    )
    assert rc == 3


def test_preflight_missing_video_referenced_in_parquet_exits_3(
    tmp_path: Path,
) -> None:
    """Parquet references a video by sha256 but the MP4 file is missing -> exit 3."""
    paths = _build_sweep_dir(
        tmp_path,
        n_cells=1,
        n_episodes_per_cell=2,
        create_videos=False,  # videos referenced by sha but not on disk
    )
    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--dry-run",
        ]
    )
    assert rc == 3


# --------------------------------------------------------------------- #
# 5-6. Auth path                                                        #
# --------------------------------------------------------------------- #


def test_main_auth_whoami_failure_exits_4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HfApi.whoami() raises -> exit 4."""
    paths = _build_sweep_dir(tmp_path, n_cells=1, n_episodes_per_cell=1)
    api = _make_fake_api()
    api.whoami.side_effect = RuntimeError("401 Unauthorized")
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(tmp_path / "stage"),
        ]
    )
    assert rc == 4
    api.upload_folder.assert_not_called()


def test_main_auth_repo_info_failure_exits_4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HfApi.repo_info() raises (e.g. dataset doesn't exist) -> exit 4."""
    paths = _build_sweep_dir(tmp_path, n_cells=1, n_episodes_per_cell=1)
    api = _make_fake_api()
    api.repo_info.side_effect = RuntimeError("404 not found")
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(tmp_path / "stage"),
        ]
    )
    assert rc == 4
    api.upload_folder.assert_not_called()


# --------------------------------------------------------------------- #
# 7. Dry-run                                                            #
# --------------------------------------------------------------------- #


def test_dry_run_does_not_call_upload_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run: stage everything, write provenance, never touch the API."""
    paths = _build_sweep_dir(tmp_path, n_cells=2, n_episodes_per_cell=3)
    api = _make_fake_api()

    # The fake API factory should not even be called in dry-run mode.
    factory_calls = {"n": 0}

    def _factory() -> Any:
        factory_calls["n"] += 1
        return api

    monkeypatch.setattr(pr, "_get_hf_api", _factory)

    staging = tmp_path / "stage"
    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(staging),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert factory_calls["n"] == 0
    api.upload_folder.assert_not_called()
    # Provenance JSON written into staging dir.
    assert (staging / "_provenance.json").exists()
    assert (staging / "results.parquet").exists()
    assert (staging / "sweep_manifest.json").exists()
    assert (staging / "videos").is_dir()
    # 2 cells * 3 episodes = 6 MP4s staged.
    assert len(list((staging / "videos").glob("*.mp4"))) == 6


# --------------------------------------------------------------------- #
# 8. Happy path                                                         #
# --------------------------------------------------------------------- #


def test_happy_path_upload_folder_called_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4 cells * 5 episodes -> exit 0, upload_folder called once with full glob."""
    paths = _build_sweep_dir(tmp_path, n_cells=4, n_episodes_per_cell=5)
    api = _make_fake_api()
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    staging = tmp_path / "stage"
    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(staging),
            "--commit-message",
            "test commit",
        ]
    )
    assert rc == 0
    assert api.upload_folder.call_count == 1
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == pr.DEFAULT_HUB_REPO
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["commit_message"] == "test commit"
    assert kwargs["folder_path"] == str(staging)
    # The full-corpus allow_patterns glob is a stable contract.
    assert set(kwargs["allow_patterns"]) == set(pr.ALLOW_PATTERNS_FULL)
    # Provenance JSON written, with expected counts.
    provenance = json.loads((staging / "_provenance.json").read_text())
    assert provenance["n_cells"] == 4
    assert provenance["n_episodes"] == 20
    assert provenance["hub_repo"] == pr.DEFAULT_HUB_REPO


# --------------------------------------------------------------------- #
# 9. --skip-videos                                                      #
# --------------------------------------------------------------------- #


def test_skip_videos_excludes_mp4_pattern(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--skip-videos: allow_patterns must NOT include videos/*.mp4."""
    paths = _build_sweep_dir(tmp_path, n_cells=1, n_episodes_per_cell=2)
    api = _make_fake_api()
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    staging = tmp_path / "stage"
    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(staging),
            "--skip-videos",
        ]
    )
    assert rc == 0
    kwargs = api.upload_folder.call_args.kwargs
    assert "videos/*.mp4" not in kwargs["allow_patterns"]
    assert "results.parquet" in kwargs["allow_patterns"]
    assert "sweep_manifest.json" in kwargs["allow_patterns"]
    # And no videos copied into staging.
    assert not (staging / "videos").exists()


# --------------------------------------------------------------------- #
# 10. Oversize video skip                                               #
# --------------------------------------------------------------------- #


def test_oversize_video_is_skipped_with_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single MP4 above --max-video-mib -> excluded from staging, exit 2, warning."""
    # Build sweep with one normal cell + one orphan oversize video on
    # disk (not referenced in parquet, so pre-flight passes).
    paths = _build_sweep_dir(
        tmp_path,
        n_cells=1,
        n_episodes_per_cell=2,
        extra_oversize=1,
        oversize_bytes=3 * 1024 * 1024,  # 3 MiB
    )
    api = _make_fake_api()
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    staging = tmp_path / "stage"
    with caplog.at_level("WARNING"):
        rc = pr.main(
            [
                "--results-path",
                str(paths["results_path"]),
                "--manifest-path",
                str(paths["manifest_path"]),
                "--videos-dir",
                str(paths["videos_dir"]),
                "--staging-dir",
                str(staging),
                "--max-video-mib",
                "2",
            ]
        )
    assert rc == 2
    # The 2 normal MP4s made it; the oversize one did not.
    staged_mp4s = list((staging / "videos").glob("*.mp4"))
    assert len(staged_mp4s) == 2
    assert any("oversized" in r.message.lower() for r in caplog.records)
    # upload still happened, with the same allow_patterns.
    api.upload_folder.assert_called_once()
    # Provenance records the skipped file.
    provenance = json.loads((staging / "_provenance.json").read_text())
    assert len(provenance["skipped_videos"]) == 1


# --------------------------------------------------------------------- #
# 11. AST guard                                                         #
# --------------------------------------------------------------------- #


def _module_imports_huggingface_hub_at_top_level() -> bool:
    """True iff scripts/publish_results.py has any top-level huggingface_hub import.

    Mirrors the AST guards in tests/test_run_one.py and
    tests/test_calibrate.py so the publish step's lazy-import contract
    is statically enforced.
    """
    tree = ast.parse(PUBLISH_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "huggingface_hub" or alias.name.startswith("huggingface_hub."):
                    return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "huggingface_hub" or node.module.startswith("huggingface_hub."))
        ):
            return True
    return False


def test_module_does_not_import_huggingface_hub_at_top_level() -> None:
    """The lazy-import contract: no top-level huggingface_hub import."""
    assert not _module_imports_huggingface_hub_at_top_level(), (
        "scripts/publish_results.py must lazy-import huggingface_hub inside functions"
    )


# --------------------------------------------------------------------- #
# 12. Idempotence                                                       #
# --------------------------------------------------------------------- #


def test_idempotent_provenance_modulo_published_utc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two back-to-back runs produce identical provenance JSON except for published_utc."""
    paths = _build_sweep_dir(tmp_path, n_cells=2, n_episodes_per_cell=2)
    api = _make_fake_api()
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    staging = tmp_path / "stage"

    rc1 = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(staging),
        ]
    )
    assert rc1 == 0
    prov_1 = json.loads((staging / "_provenance.json").read_text())

    rc2 = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(staging),
        ]
    )
    assert rc2 == 0
    prov_2 = json.loads((staging / "_provenance.json").read_text())

    # Strip the only field that legitimately changes per-publish.
    prov_1.pop("published_utc")
    prov_2.pop("published_utc")
    assert prov_1 == prov_2
    # And both runs called upload_folder exactly once -- the dedup is
    # the Hub's job, not ours.
    assert api.upload_folder.call_count == 2


# --------------------------------------------------------------------- #
# 13. Hard upload failure                                               #
# --------------------------------------------------------------------- #


def test_upload_folder_raises_returns_exit_5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """upload_folder explodes mid-flight -> exit 5 with the error in the log line."""
    paths = _build_sweep_dir(tmp_path, n_cells=1, n_episodes_per_cell=2)
    api = _make_fake_api()
    api.upload_folder.side_effect = RuntimeError("network blew up")
    monkeypatch.setattr(pr, "_get_hf_api", lambda: api)

    rc = pr.main(
        [
            "--results-path",
            str(paths["results_path"]),
            "--manifest-path",
            str(paths["manifest_path"]),
            "--videos-dir",
            str(paths["videos_dir"]),
            "--staging-dir",
            str(tmp_path / "stage"),
        ]
    )
    assert rc == 5


# --------------------------------------------------------------------- #
# 14. Naming contract with run_one                                      #
# --------------------------------------------------------------------- #


def test_video_filename_matches_run_one_render_naming() -> None:
    """publish_results._video_filename must match the renderer's naming scheme.

    If this test fails, the publish step would silently 404 every video
    because the staged MP4s wouldn't match the parquet's referenced
    paths.
    """
    name = pr._video_filename(
        policy="diffusion_policy",
        env="pusht",
        seed=3,
        episode_index=42,
    )
    # Mirror the literal in scripts/run_one.render_episodes_to_videos.
    assert name == "diffusion_policy__pusht__seed3__ep042.mp4"


# --------------------------------------------------------------------- #
# 15. CLI surface                                                       #
# --------------------------------------------------------------------- #


def test_main_rejects_zero_max_video_mib(tmp_path: Path) -> None:
    """--max-video-mib must be positive -> exit 3 before any pre-flight work."""
    rc = pr.main(
        [
            "--results-path",
            str(tmp_path / "missing.parquet"),
            "--manifest-path",
            str(tmp_path / "manifest.json"),
            "--videos-dir",
            str(tmp_path / "videos"),
            "--max-video-mib",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 3
