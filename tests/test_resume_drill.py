"""Resume-drill scenarios for ``scripts/run_sweep.py``.

Premortem mitigation moved up from Day 4. These tests drive
:func:`scripts.run_sweep.run_sweep` against a fake subprocess injector
so the orchestrator's resume / OOM / dry-run / shuffle behaviour can be
verified without spawning python or installing lerobot.

Five scenarios live here (DESIGN.md § Methodology + the agent brief):

1. Cold start, all succeed -- 4 cells, all return 0, all 4 in parquet,
   manifest all ``completed``.
2. Mid-sweep KeyboardInterrupt -- cell 2 raises in the fake subprocess
   before writing rows. Manifest leaves cell 2 ``pending``. Re-invoke
   on the same parquet: cell 1 is skipped (resumed), cells 2-4 run
   fresh, all 4 land in parquet with no duplicate rows.
3. Partial cell on disk -- before sweep, cell A has only N/M episodes
   in the parquet. plan_resume classifies A as partial.
   drop_partial_cells removes the stale rows. Sweep re-runs A and
   writes M rows.
4. OOM cell -- cell B's fake subprocess returns exit code 4
   (lerobot missing) or 2 (per-episode error). Manifest shows B
   ``failed`` with exit_code; sweep continues; final exit code is 2.
5. ``--dry-run`` -- writes manifest with all cells ``pending``, no
   subprocess calls fire, parquet is untouched.

The fake subprocess takes the place of ``scripts/run_one.py`` and
writes synthetic rows directly to the parquet via
:func:`lerobot_bench.checkpointing.append_cell_rows`. This is the same
contract real ``run_one`` follows (atomic per-cell append), so the
sweep's view of the world is indistinguishable from a real run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts import run_sweep as rs

from lerobot_bench.checkpointing import (
    RESULT_SCHEMA,
    CellKey,
    append_cell_rows,
    load_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _build_config(
    *,
    results_path: Path,
    policies: tuple[str, ...] = ("no_op", "random"),
    envs: tuple[str, ...] = ("pusht",),
    seeds: tuple[int, ...] = (0, 1),
    n_episodes: int = 3,
    record_video: bool = False,
    cell_timeout_s: float | None = None,
) -> rs.SweepConfig:
    """Build a SweepConfig pointing at a tmp parquet. Baselines on PushT only.

    Default shape (2 policies x 1 env x 2 seeds = 4 cells) keeps the
    drill scenarios under "tens of milliseconds" each, matching CI's
    fast-test budget.
    """
    return rs.SweepConfig(
        policies=policies,
        envs=envs,
        seeds=seeds,
        episodes_per_seed=n_episodes,
        results_path=results_path,
        videos_dir=None,
        record_video=record_video,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=cell_timeout_s,
        max_parallel=1,
        overrides={},
    )


def _row(*, policy: str, env: str, seed: int, episode_index: int) -> dict[str, Any]:
    """One results-schema row. Synthetic values mirror what run_one would write."""
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": episode_index,
        "success": True,
        "return_": 1.0,
        "n_steps": 10,
        "wallclock_s": 0.05,
        "video_sha256": "",
        "code_sha": "deadbeef",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-03T00:00:00+00:00",
    }


def _write_cell_rows(parquet_path: Path, key: CellKey, n_episodes: int) -> None:
    """Append a full set of rows for one cell, mimicking a successful run_one call."""
    rows = [
        _row(
            policy=key.policy,
            env=key.env,
            seed=key.seed,
            episode_index=i,
        )
        for i in range(n_episodes)
    ]
    df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    append_cell_rows(parquet_path, df)


def _parse_argv_to_cell(argv: list[str]) -> tuple[str, str, int, int, Path]:
    """Pull (policy, env, seed, n_episodes, out_parquet) out of a run_one argv.

    The fake subprocess uses this to know which cell it is supposed to
    pretend to run and where to put the synthetic rows. Mirrors the
    flag layout in :func:`scripts.run_sweep._build_run_one_argv`.
    """
    flat = dict(zip(argv[2:][::2], argv[2:][1::2], strict=False))
    return (
        flat["--policy"],
        flat["--env"],
        int(flat["--seed"]),
        int(flat["--n-episodes"]),
        Path(flat["--out-parquet"]),
    )


def _make_fake_subprocess(
    *,
    behaviour: dict[tuple[str, str, int], int] | None = None,
    raise_on: tuple[str, str, int] | None = None,
    write_rows: bool = True,
    call_log: list[tuple[str, str, int]] | None = None,
) -> Callable[..., rs.SubprocessOutcome]:
    """Build a fake :data:`scripts.run_sweep._run_subprocess`.

    ``behaviour`` maps ``(policy, env, seed)`` to the exit code that
    fake should return. Default is 0 for any cell not in the dict.

    ``raise_on`` names a single cell whose dispatch must raise
    KeyboardInterrupt -- used by drill #2 to simulate a Ctrl-C between
    rows-already-written and the next cell starting.

    ``write_rows`` controls whether the fake "succeeds" (writes the
    cell's rows to the parquet) before exiting; only effective when
    the exit code is 0 or 2 (i.e. run_one's "rows appended" codes).

    ``call_log`` is appended to with each cell tuple the fake sees, so
    tests can assert dispatch order + count.
    """
    behaviour = behaviour or {}

    def fake(argv: list[str], *, timeout_s: float | None = None) -> rs.SubprocessOutcome:
        policy, env, seed, n_eps, out_parquet = _parse_argv_to_cell(argv)
        triple = (policy, env, seed)
        if call_log is not None:
            call_log.append(triple)

        if raise_on is not None and triple == raise_on:
            raise KeyboardInterrupt(f"fake interrupt at {triple}")

        rc = behaviour.get(triple, 0)
        if write_rows and rc in {0, 2}:
            _write_cell_rows(out_parquet, CellKey(policy=policy, env=env, seed=seed), n_eps)

        # Synthetic stderr only for failures; helps test the
        # stderr_tail manifest field.
        stderr = "" if rc in {0, 2} else f"FAKE FAIL exit={rc} cell={triple}\n"
        return rs.SubprocessOutcome(returncode=rc, stdout="", stderr=stderr)

    return fake


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


# --------------------------------------------------------------------- #
# 1. Cold start -- all succeed                                          #
# --------------------------------------------------------------------- #


def test_drill1_cold_start_all_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4 cells, all return 0 -- all 4 in parquet, manifest all 'completed'."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=3)

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome.exit_code == 0
    assert outcome.n_planned == 4
    assert outcome.n_completed == 4
    assert outcome.n_failed == 0
    assert outcome.n_already_done == 0
    assert outcome.n_skipped == 0

    # Every cell was dispatched exactly once, in sorted order.
    expected = [
        ("no_op", "pusht", 0),
        ("no_op", "pusht", 1),
        ("random", "pusht", 0),
        ("random", "pusht", 1),
    ]
    assert call_log == expected

    # Parquet has 4 cells x 3 episodes = 12 rows.
    df = load_results(parquet_path)
    assert len(df) == 12
    assert set(zip(df["policy"], df["env"], df["seed"], strict=True)) == {
        ("no_op", "pusht", 0),
        ("no_op", "pusht", 1),
        ("random", "pusht", 0),
        ("random", "pusht", 1),
    }

    # Manifest says all completed, with started_utc/finished_utc + exit_code 0.
    manifest = _read_manifest(outcome.manifest_path)
    assert manifest["finished_utc"] is not None
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses == ["completed"] * 4
    for cell in manifest["cells"]:
        assert cell["exit_code"] == 0
        assert cell["started_utc"] is not None
        assert cell["finished_utc"] is not None
        assert cell["stderr_tail"] == ""


# --------------------------------------------------------------------- #
# 2. Mid-sweep KeyboardInterrupt + clean resume                         #
# --------------------------------------------------------------------- #


def test_drill2_mid_sweep_kill_then_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cell 1 succeeds; cell 2 raises KeyboardInterrupt before writing rows.

    Manifest after the kill: cell 1 'completed', cells 2-4 'pending'
    (cell 2 because it never wrote rows; checkpointing classifies
    pending = no rows in parquet). Re-invoke run_sweep on the same
    parquet path: cell 1 is skipped (resumed), cells 2-4 dispatch.
    """
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=3)

    # First pass: kill on the second cell.
    second_cell = ("no_op", "pusht", 1)
    fake1 = _make_fake_subprocess(raise_on=second_cell)
    monkeypatch.setattr(rs, "_run_subprocess", fake1)

    outcome1 = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    # Interrupted = exit 2 in this orchestrator's convention (some
    # cells didn't run).
    assert outcome1.exit_code == 2
    assert outcome1.n_completed == 1  # cell 1 finished
    assert outcome1.n_failed == 0

    # Parquet has only cell 1's rows (3 of them).
    df = load_results(parquet_path)
    assert len(df) == 3
    assert set(zip(df["policy"], df["env"], df["seed"], strict=True)) == {
        ("no_op", "pusht", 0),
    }

    # Manifest reflects the interrupt: cell 1 completed, cell 2 still
    # at "pending" status (started_utc set; finished_utc still None).
    manifest1 = _read_manifest(outcome1.manifest_path)
    cells_by_triple = {(c["policy"], c["env"], c["seed_idx"]): c for c in manifest1["cells"]}
    assert cells_by_triple[("no_op", "pusht", 0)]["status"] == "completed"
    assert cells_by_triple[("no_op", "pusht", 1)]["status"] == "pending"
    assert cells_by_triple[("no_op", "pusht", 1)]["finished_utc"] is None

    # Second pass: resume. Same config + same parquet. All cells succeed.
    call_log: list[tuple[str, str, int]] = []
    fake2 = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake2)

    outcome2 = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome2.exit_code == 0
    # Cell 1 was skipped (resumed); cells 2-4 dispatched (3 cells).
    assert outcome2.n_already_done == 1
    assert outcome2.n_completed == 3
    assert outcome2.n_failed == 0
    assert call_log == [
        ("no_op", "pusht", 1),
        ("random", "pusht", 0),
        ("random", "pusht", 1),
    ]

    # Final parquet: 4 cells x 3 episodes = 12 rows, no duplicates.
    df_final = load_results(parquet_path)
    assert len(df_final) == 12
    assert df_final.duplicated(subset=["policy", "env", "seed", "episode_index"]).sum() == 0

    # Manifest after resume: every cell completed.
    manifest2 = _read_manifest(outcome2.manifest_path)
    assert all(c["status"] == "completed" for c in manifest2["cells"])
    assert manifest2["finished_utc"] is not None


# --------------------------------------------------------------------- #
# 3. Partial cell on disk -- dropped + re-run                           #
# --------------------------------------------------------------------- #


def test_drill3_partial_cell_dropped_and_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell A has 2/3 episodes on disk; sweep drops them and writes 3 fresh rows."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(
        results_path=parquet_path,
        policies=("no_op",),
        envs=("pusht",),
        seeds=(0, 1),
        n_episodes=3,
    )

    # Pre-seed: cell (no_op, pusht, 0) has only 2 episodes (indices 0, 1).
    partial_rows = pd.DataFrame(
        [
            _row(policy="no_op", env="pusht", seed=0, episode_index=0),
            _row(policy="no_op", env="pusht", seed=0, episode_index=1),
        ],
        columns=list(RESULT_SCHEMA),
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    append_cell_rows(parquet_path, partial_rows)

    # Sanity: 2 rows on disk before the sweep starts.
    assert len(load_results(parquet_path)) == 2

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome.exit_code == 0
    assert outcome.n_planned == 2
    assert outcome.n_completed == 2  # the partial cell + the fresh seed=1 cell
    assert outcome.n_already_done == 0  # partial != completed -- it gets re-dispatched

    # Both cells were dispatched (the partial got re-queued).
    assert ("no_op", "pusht", 0) in call_log
    assert ("no_op", "pusht", 1) in call_log

    # Final parquet: 2 cells x 3 episodes = 6 rows, the stale partial
    # rows are gone, the rewrite produced clean 0..2 indices.
    df = load_results(parquet_path)
    assert len(df) == 6
    seed0_indices = sorted(df.loc[df["seed"] == 0, "episode_index"].tolist())
    assert seed0_indices == [0, 1, 2]


# --------------------------------------------------------------------- #
# 4. Failed cell -- sweep continues, exit code 2                        #
# --------------------------------------------------------------------- #


def test_drill4_failed_cell_continues_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cell exits 4 (lerobot missing) -- sweep continues, final exit 2."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=3)

    # Cell B = (random, pusht, 0) exits 4 (lerobot missing).
    fake = _make_fake_subprocess(behaviour={("random", "pusht", 0): 4})
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome.exit_code == 2
    assert outcome.n_planned == 4
    assert outcome.n_completed == 3
    assert outcome.n_failed == 1

    # Manifest: failing cell is recorded with exit_code=4 and stderr_tail.
    manifest = _read_manifest(outcome.manifest_path)
    cells_by_triple = {(c["policy"], c["env"], c["seed_idx"]): c for c in manifest["cells"]}
    failed = cells_by_triple[("random", "pusht", 0)]
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 4
    assert "FAKE FAIL" in failed["stderr_tail"]
    assert failed["finished_utc"] is not None

    # All four cells have a manifest row; the three successes are completed.
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("completed") == 3
    assert statuses.count("failed") == 1

    # The successful cells did write rows (3 cells x 3 episodes = 9).
    df = load_results(parquet_path)
    assert len(df) == 9
    assert ("random", "pusht", 0) not in set(zip(df["policy"], df["env"], df["seed"], strict=True))


# --------------------------------------------------------------------- #
# 5. --dry-run -- manifest written, no subprocess fired                 #
# --------------------------------------------------------------------- #


def test_drill5_dry_run_writes_manifest_no_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run: manifest with all 'pending', no subprocess called, no parquet."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=3)

    # Hard guard: any subprocess call would fail this test.
    def boom(*_args: Any, **_kwargs: Any) -> rs.SubprocessOutcome:
        raise AssertionError("--dry-run must not call _run_subprocess")

    monkeypatch.setattr(rs, "_run_subprocess", boom)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"), dry_run=True)

    assert outcome.exit_code == 0
    assert outcome.n_completed == 0
    assert outcome.n_failed == 0
    assert outcome.n_planned == 4

    # Parquet untouched.
    assert not parquet_path.exists()

    # Manifest exists, all cells pending. finished_utc stays None on dry-run
    # (the sweep did not finish dispatch).
    manifest = _read_manifest(outcome.manifest_path)
    assert manifest["finished_utc"] is None
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses == ["pending"] * 4


# --------------------------------------------------------------------- #
# 6. Resume on already-complete sweep is a no-op (idempotence)          #
# --------------------------------------------------------------------- #


def test_drill6_resume_after_full_completion_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running an already-complete sweep dispatches zero cells, exits 0."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=3)

    # First pass: everything succeeds.
    fake1 = _make_fake_subprocess()
    monkeypatch.setattr(rs, "_run_subprocess", fake1)
    out1 = rs.run_sweep(config=config, config_path=Path("test.yaml"))
    assert out1.exit_code == 0
    assert out1.n_completed == 4

    # Second pass: every cell is "already_done"; subprocess must not fire.
    def boom(*_args: Any, **_kwargs: Any) -> rs.SubprocessOutcome:
        raise AssertionError("resume on completed sweep must not dispatch")

    monkeypatch.setattr(rs, "_run_subprocess", boom)
    out2 = rs.run_sweep(config=config, config_path=Path("test.yaml"))
    assert out2.exit_code == 0
    assert out2.n_completed == 0
    assert out2.n_already_done == 4

    # Parquet still has exactly 12 rows -- no duplicates were appended.
    assert len(load_results(parquet_path)) == 12


# --------------------------------------------------------------------- #
# 7. Manifest survives kill -9 between cells (atomic write check)       #
# --------------------------------------------------------------------- #


def test_drill7_manifest_atomic_no_tmp_leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful sweep, no .tmp.json sibling is left on disk."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=2)

    fake = _make_fake_subprocess()
    monkeypatch.setattr(rs, "_run_subprocess", fake)
    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))
    assert outcome.exit_code == 0

    # Manifest is on disk; no .tmp.json sibling lingering after the
    # final write (atomic-rename property of write_manifest).
    manifest_dir = outcome.manifest_path.parent
    leftovers = list(manifest_dir.glob("sweep_manifest.tmp*"))
    assert leftovers == [], f"unexpected tmp files: {leftovers}"


# --------------------------------------------------------------------- #
# 8. CLI surface: --max-cells caps dispatch                             #
# --------------------------------------------------------------------- #


def test_drill8_max_cells_caps_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--max-cells=2 dispatches the first 2 sorted cells, manifest matches."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=2)

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"), max_cells=2)

    assert outcome.exit_code == 0
    assert outcome.n_planned == 2  # capped before classify
    assert outcome.n_completed == 2
    # Sorted order means no_op/pusht seeds 0,1 first.
    assert call_log == [("no_op", "pusht", 0), ("no_op", "pusht", 1)]

    # Manifest only has the 2 capped cells, not all 4.
    manifest = _read_manifest(outcome.manifest_path)
    assert len(manifest["cells"]) == 2


# --------------------------------------------------------------------- #
# 9. CLI surface: --shuffle reorders dispatch                           #
# --------------------------------------------------------------------- #


def test_drill9_shuffle_reorders_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--shuffle SEED produces a deterministic non-sorted order; same set of cells."""
    parquet_path = tmp_path / "results.parquet"
    config = _build_config(results_path=parquet_path, n_episodes=2)

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"), shuffle_seed=42)
    assert outcome.exit_code == 0

    # Same set of cells, possibly different order.
    sorted_expected = {
        ("no_op", "pusht", 0),
        ("no_op", "pusht", 1),
        ("random", "pusht", 0),
        ("random", "pusht", 1),
    }
    assert set(call_log) == sorted_expected
    assert len(call_log) == 4

    # Determinism: re-running with same seed against a fresh parquet
    # produces an identical call order.
    parquet2 = tmp_path / "results2.parquet"
    config2 = replace(config, results_path=parquet2)
    call_log2: list[tuple[str, str, int]] = []
    monkeypatch.setattr(rs, "_run_subprocess", _make_fake_subprocess(call_log=call_log2))
    rs.run_sweep(config=config2, config_path=Path("test.yaml"), shuffle_seed=42)
    assert call_log == call_log2


# --------------------------------------------------------------------- #
# 10. Pre-flight skip: incompatible cell never dispatches               #
# --------------------------------------------------------------------- #


def test_drill10_incompat_cells_never_dispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A policy with env_compat=[pusht] only -> aloha cells appear as 'skipped'.

    Uses a tmp policies.yaml with one baseline that supports pusht only.
    Aloha cells must show up in the manifest as ``skipped`` with a
    reason, never reach the subprocess fake.
    """
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
        """
policies:
  - name: pusht_only
    is_baseline: true
    env_compat: [pusht]
"""
    )

    parquet_path = tmp_path / "results.parquet"
    config = rs.SweepConfig(
        policies=("pusht_only",),
        envs=("pusht", "aloha_transfer_cube"),
        seeds=(0, 1),
        episodes_per_seed=2,
        results_path=parquet_path,
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=policies_yaml,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome.exit_code == 0
    # 4 planned (1 policy x 2 envs x 2 seeds); 2 dispatched; 2 skipped.
    assert outcome.n_planned == 4
    assert outcome.n_completed == 2
    assert outcome.n_skipped == 2

    # Aloha cells were never dispatched.
    aloha_calls = [c for c in call_log if c[1] == "aloha_transfer_cube"]
    assert aloha_calls == []

    # Manifest records skipped reason.
    manifest = _read_manifest(outcome.manifest_path)
    aloha_entries = [c for c in manifest["cells"] if c["env"] == "aloha_transfer_cube"]
    assert len(aloha_entries) == 2
    for entry in aloha_entries:
        assert entry["status"] == "skipped"
        assert "env_compat" in entry["skip_reason"]


# --------------------------------------------------------------------- #
# 11. Pre-flight skip: not-runnable policy (revision_sha=null)          #
# --------------------------------------------------------------------- #


def test_drill11_unrunnable_policy_appears_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pretrained policy with revision_sha=null -> manifest shows skipped.

    Uses a tmp policies.yaml since the shipped diffusion_policy/act now have
    locked revision SHAs (Day 0a, 2026-05-03).
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
    parquet_path = tmp_path / "results.parquet"
    config = rs.SweepConfig(
        policies=("no_op", "not_yet_locked"),
        envs=("pusht",),
        seeds=(0,),
        episodes_per_seed=2,
        results_path=parquet_path,
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=policies_yaml,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )

    call_log: list[tuple[str, str, int]] = []
    fake = _make_fake_subprocess(call_log=call_log)
    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))

    assert outcome.exit_code == 0
    assert outcome.n_planned == 2
    assert outcome.n_completed == 1  # no_op runs
    assert outcome.n_skipped == 1  # not_yet_locked skipped

    # The unrunnable policy was never dispatched.
    assert ("not_yet_locked", "pusht", 0) not in call_log
    assert ("no_op", "pusht", 0) in call_log

    manifest = _read_manifest(outcome.manifest_path)
    diff = next(c for c in manifest["cells"] if c["policy"] == "not_yet_locked")
    assert diff["status"] == "skipped"
    assert "not runnable" in diff["skip_reason"]


# --------------------------------------------------------------------- #
# 12. Bonus: AST guard -- no torch / lerobot at module scope            #
# --------------------------------------------------------------------- #


def test_drill12_module_has_no_top_level_torch_or_lerobot_import() -> None:
    """Same lazy-import contract as run_one / calibrate."""
    import ast

    source = (REPO_ROOT / "scripts" / "run_sweep.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch" and not alias.name.startswith("torch."), (
                    f"top-level import {alias.name!r} would pull torch in eagerly"
                )
                assert alias.name != "lerobot" and not alias.name.startswith("lerobot."), (
                    f"top-level import {alias.name!r} would require lerobot at import time"
                )
                assert alias.name != "lerobot_bench.render", (
                    "top-level lerobot_bench.render import would pull imageio in eagerly"
                )
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            mod = node.module
            assert mod != "torch" and not mod.startswith("torch."), (
                f"top-level `from {mod}` would pull torch in eagerly"
            )
            assert mod != "lerobot" and not mod.startswith("lerobot."), (
                f"top-level `from {mod}` would require lerobot at import time"
            )
            assert mod != "lerobot_bench.render", (
                "top-level `from lerobot_bench.render` would pull imageio in eagerly"
            )
