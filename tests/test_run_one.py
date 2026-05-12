"""Tests for ``scripts/run_one.py``.

Same pattern as ``tests/test_calibrate.py``: drive the script's pure
orchestration with monkeypatched in-process fakes. No torch, no
lerobot, no gymnasium — every cell-execution test substitutes
:func:`lerobot_bench.eval.run_cell_from_specs` with a builder that
returns a synthetic :class:`CellResult`.

The "real registries" tests (#3, #4) deliberately depend on the
shipped ``configs/policies.yaml`` invariants:
  * baselines are runnable
  * pretrained policies (diffusion_policy) ship with ``revision_sha=null``

If Day 0a lands before this PR is merged and locks the SHA on
``diffusion_policy``, test #4 needs to switch to a different
not-yet-locked policy or use a tmp policies.yaml. Documented inline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scripts import run_one as ro

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.eval import CellResult, EpisodeResult

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"
RUN_ONE_SOURCE = REPO_ROOT / "scripts" / "run_one.py"


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _fake_episode(
    *,
    idx: int,
    success: bool,
    error: str | None = None,
    with_frame: bool = True,
    video_sha256: str = "",
    video_path: Path | None = None,
) -> EpisodeResult:
    """Build one EpisodeResult; success=False + error sets the crash row shape.

    After the streaming-encode refactor, ``EpisodeResult.frames`` is
    typically empty in production results -- the renderer runs inside
    :func:`run_cell` and drops frames before each episode returns. The
    ``video_sha256`` / ``video_path`` fields carry the publish-time
    references. Tests can populate them explicitly via the kwargs.
    """
    if error is not None:
        return EpisodeResult(
            episode_index=idx,
            success=False,
            return_=0.0,
            n_steps=0,
            wallclock_s=0.001,
            frames=(),
            final_reward=0.0,
            error=error,
        )
    frame = np.zeros((64, 64, 3), dtype=np.uint8) if with_frame else None
    frames = (frame,) if frame is not None else ()
    return EpisodeResult(
        episode_index=idx,
        success=success,
        return_=1.0 if success else 0.0,
        n_steps=10,
        wallclock_s=0.05,
        frames=frames,
        final_reward=1.0 if success else 0.0,
        error=None,
        video_path=video_path,
        video_sha256=video_sha256,
    )


def _fake_cell_result(
    *,
    policy: str = "random",
    env: str = "pusht",
    seed: int = 0,
    n: int = 3,
    n_success: int = 2,
    n_errors: int = 0,
) -> CellResult:
    """Build a CellResult with ``n_success`` successes, ``n_errors`` errored
    episodes, and the rest as failed-but-not-errored episodes. Frames are a
    single 64x64 frame each (errored episodes have zero frames).
    """
    if n_success + n_errors > n:
        raise ValueError("n_success + n_errors must be <= n")

    eps: list[EpisodeResult] = []
    idx = 0
    for _ in range(n_success):
        eps.append(_fake_episode(idx=idx, success=True))
        idx += 1
    for _ in range(n_errors):
        eps.append(_fake_episode(idx=idx, success=False, error="RuntimeError: synthetic"))
        idx += 1
    while idx < n:
        eps.append(_fake_episode(idx=idx, success=False))
        idx += 1
    return CellResult(
        policy=policy,
        env=env,
        seed=seed,
        episodes=tuple(eps),
        code_sha="deadbeef",
        lerobot_version="0.5.1",
        timestamp_utc="2026-05-01T00:00:00+00:00",
    )


def _patch_run_cell(monkeypatch: pytest.MonkeyPatch, cell: CellResult) -> None:
    """Replace ``eval.run_cell_from_specs`` with a one-shot stub that returns ``cell``.

    Uses ``monkeypatch.setattr`` against the imported eval module rather
    than the orchestrator's namespace because :func:`scripts.run_one.run_one`
    does ``from lerobot_bench import eval as eval_mod`` at call time.
    """

    def _stub(*_args: Any, **_kwargs: Any) -> CellResult:
        return cell

    from lerobot_bench import eval as eval_mod

    monkeypatch.setattr(eval_mod, "run_cell_from_specs", _stub)


def _patch_lerobot_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :func:`run_one._check_lerobot_available` succeed unconditionally."""
    monkeypatch.setattr(ro, "_check_lerobot_available", lambda: None)


# --------------------------------------------------------------------- #
# 1-2. resolve_specs                                                    #
# --------------------------------------------------------------------- #


def test_resolve_specs_unknown_policy_raises() -> None:
    """KeyError carries the registry's available-list message."""
    with pytest.raises(KeyError) as exc_info:
        ro.resolve_specs(
            "fakenet",
            "pusht",
            policies_yaml=DEFAULT_POLICIES_YAML,
            envs_yaml=DEFAULT_ENVS_YAML,
        )
    msg = str(exc_info.value)
    assert "fakenet" in msg
    assert "available" in msg
    # The shipped registry has at least these two baselines listed:
    assert "no_op" in msg
    assert "random" in msg


def test_resolve_specs_env_incompat_raises(tmp_path: Path) -> None:
    """A pusht-only baseline + the aloha env should ValueError with the compat list."""
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
        """
policies:
  - name: pusht_only_baseline
    is_baseline: true
    env_compat: [pusht]
"""
    )
    with pytest.raises(ValueError) as exc_info:
        ro.resolve_specs(
            "pusht_only_baseline",
            "aloha_transfer_cube",
            policies_yaml=policies_yaml,
            envs_yaml=DEFAULT_ENVS_YAML,
        )
    msg = str(exc_info.value)
    assert "pusht_only_baseline" in msg
    assert "aloha_transfer_cube" in msg
    assert "supports" in msg
    assert "pusht" in msg


# --------------------------------------------------------------------- #
# 3-4. main() exit codes against real registries                        #
# --------------------------------------------------------------------- #


def test_main_unknown_policy_exits_5(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown policy name -> pre-flight failure -> exit 5."""
    rc = ro.main(
        [
            "--policy",
            "fakenet",
            "--env",
            "pusht",
            "--seed",
            "0",
            "--policies-yaml",
            str(DEFAULT_POLICIES_YAML),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-parquet",
            str(tmp_path / "results.parquet"),
        ]
    )
    assert rc == 5
    err = capsys.readouterr().err
    assert "aborted" in err
    assert "fakenet" in err
    assert "resume" in err


def test_main_unrunnable_pretrained_exits_3(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pretrained policy with revision_sha=null -> exit 3.

    Uses a tmp policies.yaml since the shipped diffusion_policy/act now
    have locked revision SHAs (Day 0a, 2026-05-03).
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
    rc = ro.main(
        [
            "--policy",
            "not_yet_locked",
            "--env",
            "pusht",
            "--seed",
            "0",
            "--policies-yaml",
            str(policies_yaml),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-parquet",
            str(tmp_path / "results.parquet"),
        ]
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "not_yet_locked" in err
    assert "revision_sha" in err


# --------------------------------------------------------------------- #
# 5. AST: lazy-import contract                                          #
# --------------------------------------------------------------------- #


def _module_imports_torch_at_top_level() -> bool:
    """True iff ``scripts/run_one.py`` has any top-level torch/lerobot/render import."""
    tree = ast.parse(RUN_ONE_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return True
                if alias.name == "lerobot" or alias.name.startswith("lerobot."):
                    return True
                if alias.name == "lerobot_bench.render":
                    return True
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "torch" or node.module.startswith("torch."):
                return True
            if node.module == "lerobot" or node.module.startswith("lerobot."):
                return True
            if node.module == "lerobot_bench.render":
                return True
    return False


def test_main_dry_run_exits_0_no_torch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run is a no-op + AST guarantees no top-level torch/lerobot import."""
    rc = ro.main(
        [
            "--policy",
            "random",
            "--env",
            "pusht",
            "--seed",
            "0",
            "--policies-yaml",
            str(DEFAULT_POLICIES_YAML),
            "--envs-yaml",
            str(DEFAULT_ENVS_YAML),
            "--out-parquet",
            str(tmp_path / "results.parquet"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "random/pusht/seed0" in out
    # Static guarantee: no top-level import of torch/lerobot/render.
    assert not _module_imports_torch_at_top_level(), (
        "scripts/run_one.py must not import torch/lerobot/render at module scope"
    )


# --------------------------------------------------------------------- #
# 6-7. Parquet append behaviour                                         #
# --------------------------------------------------------------------- #


def test_run_one_writes_parquet_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: 3 fake episodes -> 3 rows in parquet, schema preserved."""
    cell = _fake_cell_result(n=3, n_success=2, n_errors=0)
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )

    assert outcome.exit_code == 0
    assert outcome.n_rows_appended == 3
    assert out_parquet.exists()

    df = pd.read_parquet(out_parquet)
    assert len(df) == 3
    assert set(df.columns) == set(RESULT_SCHEMA)
    # Roundtrip: episode indices preserved
    assert sorted(df["episode_index"].tolist()) == [0, 1, 2]


def test_run_one_appends_to_existing_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cell A (3 rows) then cell B (different seed, 3 rows) -> 6 total rows."""
    out_parquet = tmp_path / "results.parquet"
    _patch_lerobot_available(monkeypatch)

    # Cell A: seed 0
    cell_a = _fake_cell_result(seed=0, n=3, n_success=2)
    _patch_run_cell(monkeypatch, cell_a)
    outcome_a = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome_a.exit_code == 0

    # Cell B: seed 1 (different cell key -> no duplicate-key clash)
    cell_b = _fake_cell_result(seed=1, n=3, n_success=3)
    _patch_run_cell(monkeypatch, cell_b)
    outcome_b = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=1,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome_b.exit_code == 0

    df = pd.read_parquet(out_parquet)
    assert len(df) == 6
    assert sorted(df["seed"].unique().tolist()) == [0, 1]


# --------------------------------------------------------------------- #
# 8. Partial-error semantics                                            #
# --------------------------------------------------------------------- #


def test_run_one_partial_errors_returns_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1 errored episode out of 3 -> exit 2, all rows still appended."""
    cell = _fake_cell_result(n=3, n_success=2, n_errors=1)
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )

    assert outcome.exit_code == 2
    assert outcome.n_episodes_errored == 1
    assert outcome.n_rows_appended == 3
    df = pd.read_parquet(out_parquet)
    assert len(df) == 3
    assert "errors=1" in outcome.log_message


# --------------------------------------------------------------------- #
# 9-10. Video-render plumbing                                           #
# --------------------------------------------------------------------- #


def test_run_one_records_video_sha_when_record_video_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When record_video=True, the parquet's video_sha256 column is the
    list returned by render_episodes_to_videos (parallel order)."""
    cell = _fake_cell_result(n=3, n_success=2)
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    fake_shas = ["abc", "def", "ghi"]
    monkeypatch.setattr(
        ro,
        "render_episodes_to_videos",
        lambda cell_result, *, videos_dir: list(fake_shas),
    )

    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=True,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome.exit_code == 0

    df = pd.read_parquet(out_parquet)
    # Order is preserved by to_rows -> append_cell_rows.
    assert df["video_sha256"].tolist() == fake_shas


def test_run_one_streaming_encode_reads_sha_from_episode_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end with the post-refactor flow: the stub ``run_cell_from_specs``
    returns ``EpisodeResult``s that already carry ``video_sha256`` (as if
    the streaming encoder ran inside the cell loop), and ``run_one``
    folds those SHAs straight into the parquet without re-encoding.
    """
    # Pre-write fake MP4s on disk so the shim's existence sanity-check passes.
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    pre_paths = []
    for i in range(3):
        p = videos_dir / f"random__pusht__seed0__ep{i:03d}.mp4"
        p.write_bytes(b"FAKE")
        pre_paths.append(p)

    eps = tuple(
        _fake_episode(
            idx=i,
            success=(i < 2),
            with_frame=False,
            video_sha256=f"streamed-sha-{i}",
            video_path=pre_paths[i],
        )
        for i in range(3)
    )
    cell = CellResult(
        policy="random",
        env="pusht",
        seed=0,
        episodes=eps,
        code_sha="deadbeef",
        lerobot_version="0.5.1",
        timestamp_utc="2026-05-01T00:00:00+00:00",
    )
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=3,
        out_parquet=out_parquet,
        videos_dir=videos_dir,
        record_video=True,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome.exit_code == 0
    df = pd.read_parquet(out_parquet)
    assert df["video_sha256"].tolist() == [f"streamed-sha-{i}" for i in range(3)]


def test_run_one_no_record_video_skips_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """record_video=False -> render_episodes_to_videos must not be called."""
    cell = _fake_cell_result(n=2, n_success=1)
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    def _boom(*_args: Any, **_kwargs: Any) -> list[str]:
        raise AssertionError("render_episodes_to_videos was called despite record_video=False")

    monkeypatch.setattr(ro, "render_episodes_to_videos", _boom)

    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=2,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome.exit_code == 0
    assert outcome.videos_dir is None
    df = pd.read_parquet(out_parquet)
    # video_sha256 column should be all empty strings (the default).
    assert df["video_sha256"].tolist() == ["", ""]


# --------------------------------------------------------------------- #
# 11. Dry-run outcome surface                                           #
# --------------------------------------------------------------------- #


def test_run_one_dry_run_returns_outcome_with_zero_rows(tmp_path: Path) -> None:
    """dry_run=True: exit 0, zero rows appended, no parquet on disk."""
    out_parquet = tmp_path / "results.parquet"
    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=5,
        out_parquet=out_parquet,
        videos_dir=tmp_path / "videos",
        record_video=True,
        device="cuda",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=True,
    )
    assert outcome.exit_code == 0
    assert outcome.n_rows_appended == 0
    assert outcome.n_episodes_attempted == 0
    assert outcome.out_parquet is None
    assert not out_parquet.exists()
    assert "dry-run" in outcome.log_message


# --------------------------------------------------------------------- #
# 12. mkdir -p contract on out_parquet parent                           #
# --------------------------------------------------------------------- #


def test_run_one_creates_out_parquet_parent_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointing --out-parquet at a non-existent nested path -> dirs are created."""
    cell = _fake_cell_result(n=2, n_success=1)
    _patch_run_cell(monkeypatch, cell)
    _patch_lerobot_available(monkeypatch)

    nested = tmp_path / "a" / "b" / "c" / "results.parquet"
    assert not nested.parent.exists()

    outcome = ro.run_one(
        policy_name="random",
        env_name="pusht",
        seed=0,
        n_episodes=2,
        out_parquet=nested,
        videos_dir=tmp_path / "videos",
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        dry_run=False,
    )
    assert outcome.exit_code == 0
    assert nested.exists()
    assert nested.parent.is_dir()
