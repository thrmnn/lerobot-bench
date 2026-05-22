"""Tests for ``scripts/reproduce_cell.py``.

Same discipline as ``tests/test_run_one.py``: exercise the script's pure
logic with synthetic in-memory parquet frames. No torch, no lerobot, no
gymnasium, and -- critically -- no real cell is ever run. The one test that
drives the full ``reproduce()`` orchestration monkeypatches ``rerun_cell``
with a stub that just writes a synthetic parquet, so the ~15-minute
``run_one.py`` subprocess never fires.

Coverage:
  * ``parse_cell`` -- the helper the ``make reproduce CELL=...`` target leans on.
  * ``compare_cells`` -- exact-match -> REPRODUCED; a single flipped bit ->
    MISMATCH with the correct first-divergence report.
  * missing reference parquet / missing cell -> clear error + exit 2.
  * AST guard: ``scripts/reproduce_cell.py`` imports no torch/lerobot at
    module scope (mirrors test #5 of ``test_run_one.py``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from scripts import reproduce_cell as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRODUCE_SOURCE = REPO_ROOT / "scripts" / "reproduce_cell.py"


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _cell_frame(
    *,
    policy: str = "act",
    env: str = "pusht",
    seed: int = 0,
    successes: list[bool] | None = None,
    n_steps: list[int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic per-episode results frame for one cell.

    Mirrors the reference parquet's columns (``policy, env, seed,
    episode_index, success, n_steps, ...``) closely enough for the
    comparison code, which only reads the key columns + ``success`` +
    ``n_steps``.
    """
    successes = [True, False, True] if successes is None else successes
    n = len(successes)
    n_steps = [10] * n if n_steps is None else n_steps
    return pd.DataFrame(
        {
            "policy": [policy] * n,
            "env": [env] * n,
            "seed": [seed] * n,
            "episode_index": list(range(n)),
            "success": successes,
            "n_steps": n_steps,
        }
    )


# --------------------------------------------------------------------- #
# 1. parse_cell -- the Makefile CELL string parser                      #
# --------------------------------------------------------------------- #


def test_parse_cell_happy_path() -> None:
    assert rc.parse_cell("act/pusht/0") == ("act", "pusht", 0)
    assert rc.parse_cell("diffusion_policy/aloha_transfer_cube/4") == (
        "diffusion_policy",
        "aloha_transfer_cube",
        4,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "act/pusht",  # too few parts
        "act/pusht/0/extra",  # too many parts
        "act//0",  # empty env
        "/pusht/0",  # empty policy
        "act/pusht/notanint",  # non-integer seed
        "act/pusht/-1",  # negative seed
    ],
)
def test_parse_cell_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        rc.parse_cell(bad)


# --------------------------------------------------------------------- #
# 2. compare_cells -- exact-match REPRODUCED                             #
# --------------------------------------------------------------------- #


def test_compare_cells_identical_is_reproduced() -> None:
    ref = _cell_frame(successes=[True, False, True, True], n_steps=[10, 20, 30, 40])
    rep = _cell_frame(successes=[True, False, True, True], n_steps=[10, 20, 30, 40])
    result = rc.compare_cells(ref, rep)
    assert result.reproduced is True
    assert result.n_episodes == 4
    assert result.first_divergence is None
    assert result.n_divergent_episodes == 0


def test_compare_cells_robust_to_row_order() -> None:
    """Comparison sorts by episode_index, so shuffled reference rows still match."""
    ref = _cell_frame(successes=[True, False, True], n_steps=[10, 20, 30])
    rep = _cell_frame(successes=[True, False, True], n_steps=[10, 20, 30])
    shuffled = ref.iloc[[2, 0, 1]].reset_index(drop=True)
    assert rc.compare_cells(shuffled, rep).reproduced is True


# --------------------------------------------------------------------- #
# 3. compare_cells -- a single flipped bit -> MISMATCH                   #
# --------------------------------------------------------------------- #


def test_compare_cells_flipped_success_is_mismatch() -> None:
    ref = _cell_frame(successes=[True, False, True, True])
    rep = _cell_frame(successes=[True, True, True, True])  # episode 1 flipped
    result = rc.compare_cells(ref, rep)
    assert result.reproduced is False
    assert result.n_divergent_episodes == 1
    div = result.first_divergence
    assert div is not None
    assert div.episode_index == 1
    assert div.column == "success"
    assert div.reference_value is False
    assert div.reproduced_value is True


def test_compare_cells_first_divergence_is_earliest_episode() -> None:
    """With two divergences, the report names the lower episode index."""
    ref = _cell_frame(successes=[True, True, True, True])
    rep = _cell_frame(successes=[True, False, True, False])  # episodes 1 and 3
    result = rc.compare_cells(ref, rep)
    assert result.reproduced is False
    assert result.n_divergent_episodes == 2
    assert result.first_divergence is not None
    assert result.first_divergence.episode_index == 1


def test_compare_cells_flipped_n_steps_is_mismatch() -> None:
    ref = _cell_frame(successes=[True, True], n_steps=[10, 20])
    rep = _cell_frame(successes=[True, True], n_steps=[10, 21])  # n_steps drift
    result = rc.compare_cells(ref, rep)
    assert result.reproduced is False
    div = result.first_divergence
    assert div is not None
    assert div.episode_index == 1
    assert div.column == "n_steps"
    assert div.reference_value == 20
    assert div.reproduced_value == 21


def test_compare_cells_episode_count_mismatch_is_divergence() -> None:
    """A re-run that produced fewer episodes diverges at the missing index."""
    ref = _cell_frame(successes=[True, True, True])
    rep = _cell_frame(successes=[True, True])  # episode 2 absent
    result = rc.compare_cells(ref, rep)
    assert result.reproduced is False
    div = result.first_divergence
    assert div is not None
    assert div.episode_index == 2
    assert div.column == "episode_index"
    assert div.reference_value == "present"
    assert div.reproduced_value == "absent"


def test_compare_cells_empty_frame_raises() -> None:
    ref = _cell_frame()
    with pytest.raises(ValueError):
        rc.compare_cells(ref, ref.iloc[0:0])


# --------------------------------------------------------------------- #
# 4. format_verdict -- the operator-facing strings                      #
# --------------------------------------------------------------------- #


def test_format_verdict_reproduced_line() -> None:
    ref = _cell_frame(successes=[True] * 5)
    result = rc.compare_cells(ref, ref.copy())
    verdict = rc.format_verdict("act/pusht/seed0", result)
    assert "REPRODUCED" in verdict
    assert "5/5 episodes identical" in verdict
    assert "act/pusht/seed0" in verdict


def test_format_verdict_mismatch_has_first_divergence_and_hints() -> None:
    ref = _cell_frame(successes=[True, True, True])
    rep = _cell_frame(successes=[True, False, True])
    result = rc.compare_cells(ref, rep)
    verdict = rc.format_verdict("act/pusht/seed0", result)
    assert "MISMATCH" in verdict
    assert "episode 1" in verdict
    assert "lerobot version drift" in verdict
    assert "checkpoint SHA drift" in verdict
    assert "nondeterminism" in verdict


# --------------------------------------------------------------------- #
# 5. select_cell                                                        #
# --------------------------------------------------------------------- #


def test_select_cell_filters_to_one_cell() -> None:
    a = _cell_frame(policy="act", env="pusht", seed=0)
    b = _cell_frame(policy="act", env="pusht", seed=1)
    combined = pd.concat([a, b], ignore_index=True)
    picked = rc.select_cell(combined, policy="act", env="pusht", seed=1)
    assert len(picked) == len(b)
    assert set(picked["seed"]) == {1}


def test_select_cell_missing_column_raises_keyerror() -> None:
    df = pd.DataFrame({"policy": ["act"], "env": ["pusht"]})  # no seed/episode_index
    with pytest.raises(KeyError):
        rc.select_cell(df, policy="act", env="pusht", seed=0)


# --------------------------------------------------------------------- #
# 6. reproduce() -- missing reference / missing cell -> exit 2           #
# --------------------------------------------------------------------- #


def test_reproduce_missing_reference_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=0,
        reference=tmp_path / "does-not-exist.parquet",
        n_episodes=50,
        device="cpu",
        dry_run=False,
    )
    assert rc_code == 2
    assert "reference parquet not found" in capsys.readouterr().err


def test_reproduce_cell_absent_from_reference_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ref_path = tmp_path / "results.parquet"
    _cell_frame(policy="act", env="pusht", seed=0).to_parquet(ref_path)
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=9,  # not in the reference
        reference=ref_path,
        n_episodes=50,
        device="cpu",
        dry_run=False,
    )
    assert rc_code == 2
    assert "not present in the reference" in capsys.readouterr().err


def test_reproduce_dry_run_does_not_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run reports the cell exists and never calls rerun_cell."""
    ref_path = tmp_path / "results.parquet"
    _cell_frame(policy="act", env="pusht", seed=0, successes=[True] * 6).to_parquet(ref_path)

    def _explode(**_kwargs: object) -> int:
        raise AssertionError("rerun_cell must not run in --dry-run")

    monkeypatch.setattr(rc, "rerun_cell", _explode)
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=0,
        reference=ref_path,
        n_episodes=50,
        device="cpu",
        dry_run=True,
    )
    assert rc_code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "6 episodes" in out


# --------------------------------------------------------------------- #
# 7. reproduce() -- full orchestration with a stubbed re-run             #
# --------------------------------------------------------------------- #


def test_reproduce_matching_rerun_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A re-run that matches the reference -> REPRODUCED, exit 0."""
    ref_path = tmp_path / "results.parquet"
    cell = _cell_frame(policy="act", env="pusht", seed=0, successes=[True, False, True])
    cell.to_parquet(ref_path)

    def _fake_rerun(*, out_parquet: Path, **_kwargs: object) -> int:
        cell.to_parquet(out_parquet)
        return 0

    monkeypatch.setattr(rc, "rerun_cell", _fake_rerun)
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=0,
        reference=ref_path,
        n_episodes=50,
        device="cpu",
        dry_run=False,
    )
    assert rc_code == 0
    assert "REPRODUCED" in capsys.readouterr().out


def test_reproduce_mismatching_rerun_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A re-run with a flipped success bit -> MISMATCH, exit 1."""
    ref_path = tmp_path / "results.parquet"
    _cell_frame(policy="act", env="pusht", seed=0, successes=[True, False, True]).to_parquet(
        ref_path
    )

    def _fake_rerun(*, out_parquet: Path, **_kwargs: object) -> int:
        _cell_frame(policy="act", env="pusht", seed=0, successes=[True, True, True]).to_parquet(
            out_parquet
        )
        return 0

    monkeypatch.setattr(rc, "rerun_cell", _fake_rerun)
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=0,
        reference=ref_path,
        n_episodes=50,
        device="cpu",
        dry_run=False,
    )
    assert rc_code == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_reproduce_rerun_failure_exits_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_one.py exiting non-zero (and writing nothing) -> exit 3."""
    ref_path = tmp_path / "results.parquet"
    _cell_frame(policy="act", env="pusht", seed=0).to_parquet(ref_path)

    def _fake_rerun(**_kwargs: object) -> int:
        return 4  # run_one's "missing runtime" code; no parquet written

    monkeypatch.setattr(rc, "rerun_cell", _fake_rerun)
    rc_code = rc.reproduce(
        policy="act",
        env="pusht",
        seed=0,
        reference=ref_path,
        n_episodes=50,
        device="cpu",
        dry_run=False,
    )
    assert rc_code == 3
    assert "re-run failed" in capsys.readouterr().err


# --------------------------------------------------------------------- #
# 8. AST: lazy-import contract (mirrors test_run_one.py test #5)         #
# --------------------------------------------------------------------- #


def _module_imports_torch_at_top_level() -> bool:
    """True iff ``scripts/reproduce_cell.py`` has a top-level torch/lerobot import."""
    tree = ast.parse(REPRODUCE_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    return True
                if alias.name == "lerobot" or alias.name.startswith("lerobot."):
                    return True
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "torch" or node.module.startswith("torch."):
                return True
            if node.module == "lerobot" or node.module.startswith("lerobot."):
                return True
    return False


def test_no_top_level_torch_import() -> None:
    """--help / arg-parsing path must not drag in torch or lerobot."""
    assert not _module_imports_torch_at_top_level(), (
        "scripts/reproduce_cell.py must not import torch/lerobot at module scope"
    )
