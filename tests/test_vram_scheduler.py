"""Unit tests for the VRAM-budget admission gate (no GPU).

These tests prove the admission rule the overnight sweep relies on,
without ever allocating GPU memory: cells are simulated with a known
``vram_peak_mb`` and a fake subprocess that records, at every moment,
how many cells are running and the live reserved-VRAM sum. The
assertions are:

* (a) cells whose summed peaks fit the budget run CONCURRENTLY,
* (b) a cell that would exceed the budget WAITS until room frees,
* (c) an unknown-VRAM cell runs EXCLUSIVELY (alone),
* (d) missing calibration -> serial 1-at-a-time fallback (fail-safe),
* and across every scenario the live reserved sum NEVER exceeds the
  budget and concurrency never exceeds ``max_concurrent``.

Two layers are covered: the :class:`VramBudgetScheduler` directly (with
a barrier that pins reservations open so concurrency is observable), and
the end-to-end ``run_sweep`` concurrent dispatch path against a fake
``_run_subprocess``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts import run_sweep as rs

from embodimetry.checkpointing import RESULT_SCHEMA, append_cell_rows
from embodimetry.vram_scheduler import (
    BudgetExceededError,
    CalibrationVramTable,
    VramBudgetScheduler,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"

# Calibrated peaks from the audit (MB). diffusion ~1.1 GB, smolvla ~920,
# act ~266. Budget ~7000 MB of the 8 GB card.
ACT_MB = 266.0
SMOLVLA_MB = 920.0
DIFFUSION_MB = 1100.0
BUDGET_MB = 7000.0


# --------------------------------------------------------------------- #
# CalibrationVramTable                                                  #
# --------------------------------------------------------------------- #


def _report(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "timestamp_utc": "2026-06-08T00:00:00+00:00",
        "git_sha": "deadbeef",
        "lerobot_version": "0.5.1",
        "cells": cells,
    }


def _cell(policy: str, env: str, vram: float, status: str = "ok") -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "n_steps_measured": 20,
        "mean_ms_per_step": 10.0,
        "p95_ms_per_step": 12.0,
        "vram_peak_mb": vram,
        "status": status,
        "error": None,
        "recommended": {"seeds": 5, "episodes": 50},
    }


def test_table_exact_lookup() -> None:
    table = CalibrationVramTable.from_report(
        _report([_cell("act", "pusht", ACT_MB), _cell("smolvla", "pusht", SMOLVLA_MB)])
    )
    assert table.vram_for("act", "pusht") == ACT_MB
    assert table.vram_for("smolvla", "pusht") == SMOLVLA_MB


def test_table_policy_max_fallback() -> None:
    """An env with no exact entry falls back to the per-policy max (worst case)."""
    table = CalibrationVramTable.from_report(
        _report(
            [
                _cell("diffusion", "pusht", 1000.0),
                _cell("diffusion", "aloha", DIFFUSION_MB),  # larger
            ]
        )
    )
    # An un-measured (diffusion, libero) cell uses the larger peak, never optimistic.
    assert table.vram_for("diffusion", "libero") == DIFFUSION_MB


def test_table_only_ok_cells_count() -> None:
    """oom/error/skipped cells are unknown -> vram_for returns None (exclusive)."""
    table = CalibrationVramTable.from_report(
        _report(
            [
                _cell("pi0", "pusht", 0.0, status="oom"),
                _cell("act", "pusht", ACT_MB, status="ok"),
            ]
        )
    )
    assert table.vram_for("pi0", "pusht") is None
    assert table.vram_for("act", "pusht") == ACT_MB


def test_table_unknown_policy_is_none() -> None:
    table = CalibrationVramTable.from_report(_report([_cell("act", "pusht", ACT_MB)]))
    assert table.vram_for("mystery", "pusht") is None


def test_table_empty_report() -> None:
    table = CalibrationVramTable.from_report(_report([]))
    assert table.is_empty()
    assert table.vram_for("act", "pusht") is None


def test_table_from_json_path(tmp_path: Path) -> None:
    path = tmp_path / "calibration-20260608.json"
    path.write_text(json.dumps(_report([_cell("act", "pusht", ACT_MB)])))
    table = CalibrationVramTable.from_json_path(path)
    assert table.vram_for("act", "pusht") == ACT_MB


# --------------------------------------------------------------------- #
# VramBudgetScheduler -- direct, with a barrier to observe concurrency  #
# --------------------------------------------------------------------- #


def _hold_reservation(
    scheduler: VramBudgetScheduler,
    *,
    policy: str,
    vram_mb: float | None,
    release_event: threading.Event,
    admitted_event: threading.Event,
    seed: int = 0,
) -> threading.Thread:
    """Spawn a thread that acquires, signals admission, then holds until released."""

    def worker() -> None:
        with scheduler.admission(policy=policy, env="pusht", seed=seed, vram_mb=vram_mb):
            admitted_event.set()
            release_event.wait(timeout=5.0)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def test_scheduler_fitting_cells_run_concurrently() -> None:
    """(a) act + smolvla + diffusion (sum ~2286 MB < 7000) all admit at once."""
    sched = VramBudgetScheduler(budget_mb=BUDGET_MB, max_concurrent=3)
    release = threading.Event()
    a_in, s_in, d_in = (threading.Event() for _ in range(3))

    ta = _hold_reservation(
        sched, policy="act", vram_mb=ACT_MB, release_event=release, admitted_event=a_in
    )
    ts = _hold_reservation(
        sched,
        policy="smolvla",
        vram_mb=SMOLVLA_MB,
        release_event=release,
        admitted_event=s_in,
        seed=1,
    )
    td = _hold_reservation(
        sched,
        policy="diffusion",
        vram_mb=DIFFUSION_MB,
        release_event=release,
        admitted_event=d_in,
        seed=2,
    )

    assert a_in.wait(timeout=2.0)
    assert s_in.wait(timeout=2.0)
    assert d_in.wait(timeout=2.0)
    # All three are admitted simultaneously.
    assert sched.running == 3
    assert sched.reserved_mb == pytest.approx(ACT_MB + SMOLVLA_MB + DIFFUSION_MB)
    assert sched.reserved_mb <= sched.budget_mb

    release.set()
    for t in (ta, ts, td):
        t.join(timeout=2.0)
    assert sched.running == 0
    assert sched.reserved_mb == pytest.approx(0.0)


def test_scheduler_overbudget_cell_waits_until_room_frees() -> None:
    """(b) A 5000 MB cell can't join a 4000 MB resident; it WAITS, never overshoots."""
    sched = VramBudgetScheduler(budget_mb=7000.0, max_concurrent=3)
    release_first = threading.Event()
    first_in = threading.Event()

    # Resident cell reserves 4000 MB and holds.
    t_first = _hold_reservation(
        sched, policy="big_a", vram_mb=4000.0, release_event=release_first, admitted_event=first_in
    )
    assert first_in.wait(timeout=2.0)
    assert sched.running == 1

    # Second cell needs 5000 MB -> 4000+5000=9000 > 7000 -> must wait.
    second_admitted = threading.Event()
    release_second = threading.Event()
    t_second = _hold_reservation(
        sched,
        policy="big_b",
        vram_mb=5000.0,
        release_event=release_second,
        admitted_event=second_admitted,
        seed=1,
    )
    # It must NOT be admitted while the first is resident.
    assert not second_admitted.wait(timeout=0.3)
    assert sched.running == 1
    assert sched.reserved_mb == pytest.approx(4000.0)
    assert sched.reserved_mb <= sched.budget_mb

    # Free the first; now there is room.
    release_first.set()
    t_first.join(timeout=2.0)
    assert second_admitted.wait(timeout=2.0)
    assert sched.running == 1
    assert sched.reserved_mb == pytest.approx(5000.0)
    assert sched.reserved_mb <= sched.budget_mb

    release_second.set()
    t_second.join(timeout=2.0)


def test_scheduler_unknown_vram_runs_exclusively() -> None:
    """(c) An unknown-VRAM cell admits only when idle, and blocks others while resident."""
    sched = VramBudgetScheduler(budget_mb=BUDGET_MB, max_concurrent=3)
    release_unknown = threading.Event()
    unknown_in = threading.Event()

    # Unknown cell (vram_mb=None) acquires exclusively.
    t_unknown = _hold_reservation(
        sched, policy="pi0", vram_mb=None, release_event=release_unknown, admitted_event=unknown_in
    )
    assert unknown_in.wait(timeout=2.0)
    assert sched.running == 1

    # A tiny known cell must NOT join while the exclusive cell is resident.
    act_in = threading.Event()
    release_act = threading.Event()
    t_act = _hold_reservation(
        sched,
        policy="act",
        vram_mb=ACT_MB,
        release_event=release_act,
        admitted_event=act_in,
        seed=1,
    )
    assert not act_in.wait(timeout=0.3)
    assert sched.running == 1

    # Release the exclusive cell; now the known cell admits.
    release_unknown.set()
    t_unknown.join(timeout=2.0)
    assert act_in.wait(timeout=2.0)
    assert sched.running == 1

    release_act.set()
    t_act.join(timeout=2.0)


def test_scheduler_unknown_waits_for_running_known_cell() -> None:
    """An unknown cell may not start while ANY known cell is still running."""
    sched = VramBudgetScheduler(budget_mb=BUDGET_MB, max_concurrent=3)
    release_act = threading.Event()
    act_in = threading.Event()
    t_act = _hold_reservation(
        sched, policy="act", vram_mb=ACT_MB, release_event=release_act, admitted_event=act_in
    )
    assert act_in.wait(timeout=2.0)

    unknown_in = threading.Event()
    release_unknown = threading.Event()
    t_unknown = _hold_reservation(
        sched,
        policy="pi0",
        vram_mb=None,
        release_event=release_unknown,
        admitted_event=unknown_in,
        seed=1,
    )
    # Exclusive cell waits because a known cell is still resident.
    assert not unknown_in.wait(timeout=0.3)

    release_act.set()
    t_act.join(timeout=2.0)
    assert unknown_in.wait(timeout=2.0)
    release_unknown.set()
    t_unknown.join(timeout=2.0)


def test_scheduler_max_concurrent_backstop() -> None:
    """Even with tiny cells, never more than max_concurrent run at once."""
    sched = VramBudgetScheduler(budget_mb=BUDGET_MB, max_concurrent=2)
    release = threading.Event()
    ins = [threading.Event() for _ in range(3)]
    threads = [
        _hold_reservation(
            sched,
            policy=f"tiny{i}",
            vram_mb=10.0,
            release_event=release,
            admitted_event=ins[i],
            seed=i,
        )
        for i in range(3)
    ]
    # Only 2 of 3 tiny cells may be resident.
    assert ins[0].wait(timeout=2.0)
    assert ins[1].wait(timeout=2.0)
    assert not ins[2].wait(timeout=0.3)
    assert sched.running == 2

    release.set()
    for t in threads:
        t.join(timeout=2.0)


def test_scheduler_single_cell_over_budget_raises() -> None:
    """A known cell whose peak alone exceeds the budget can never be admitted."""
    sched = VramBudgetScheduler(budget_mb=1000.0, max_concurrent=3)
    with pytest.raises(BudgetExceededError):
        sched.acquire(policy="huge", env="pusht", seed=0, vram_mb=2000.0)


def test_scheduler_acquire_timeout() -> None:
    """acquire(timeout=...) raises TimeoutError rather than blocking forever."""
    sched = VramBudgetScheduler(budget_mb=1000.0, max_concurrent=1)
    res = sched.acquire(policy="a", env="pusht", seed=0, vram_mb=900.0)
    with pytest.raises(TimeoutError):
        sched.acquire(policy="b", env="pusht", seed=1, vram_mb=900.0, timeout=0.2)
    sched.release(res)


def test_scheduler_release_on_exception_frees_budget() -> None:
    """A crashing cell body must not leak its reservation."""
    sched = VramBudgetScheduler(budget_mb=1000.0, max_concurrent=1)
    with (
        pytest.raises(RuntimeError),
        sched.admission(policy="a", env="pusht", seed=0, vram_mb=500.0),
    ):
        raise RuntimeError("cell crashed")
    assert sched.running == 0
    assert sched.reserved_mb == pytest.approx(0.0)


def test_scheduler_rejects_bad_construction() -> None:
    with pytest.raises(ValueError):
        VramBudgetScheduler(budget_mb=0.0)
    with pytest.raises(ValueError):
        VramBudgetScheduler(budget_mb=100.0, max_concurrent=0)


# --------------------------------------------------------------------- #
# _build_scheduler -- serial fallback decisions                         #
# --------------------------------------------------------------------- #


def _config(tmp_path: Path) -> rs.SweepConfig:
    return rs.SweepConfig(
        policies=("act", "smolvla", "diffusion"),
        envs=("pusht",),
        seeds=(0,),
        episodes_per_seed=3,
        results_path=tmp_path / "results.parquet",
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )


def test_build_scheduler_serial_when_no_budget(tmp_path: Path) -> None:
    """(d) No budget requested -> serial dispatch (scheduler is None)."""
    sched, table = rs._build_scheduler(
        vram_budget_mb=None, max_concurrent=3, calibration_path=None, config=_config(tmp_path)
    )
    assert sched is None
    assert table.is_empty()


def test_build_scheduler_serial_when_no_calibration(tmp_path: Path) -> None:
    """(d) Budget requested but no calibration JSON -> fail-safe serial."""
    sched, _ = rs._build_scheduler(
        vram_budget_mb=BUDGET_MB,
        max_concurrent=3,
        calibration_path=tmp_path / "missing.json",
        config=_config(tmp_path),
    )
    assert sched is None


def test_build_scheduler_serial_when_calibration_has_no_ok_cells(tmp_path: Path) -> None:
    """Budget requested but every cell is oom/error -> fail-safe serial."""
    path = tmp_path / "calibration-20260608.json"
    path.write_text(json.dumps(_report([_cell("act", "pusht", 0.0, status="oom")])))
    sched, _ = rs._build_scheduler(
        vram_budget_mb=BUDGET_MB, max_concurrent=3, calibration_path=path, config=_config(tmp_path)
    )
    assert sched is None


def test_build_scheduler_concurrent_when_calibration_present(tmp_path: Path) -> None:
    path = tmp_path / "calibration-20260608.json"
    path.write_text(
        json.dumps(
            _report(
                [
                    _cell("act", "pusht", ACT_MB),
                    _cell("smolvla", "pusht", SMOLVLA_MB),
                    _cell("diffusion", "pusht", DIFFUSION_MB),
                ]
            )
        )
    )
    sched, table = rs._build_scheduler(
        vram_budget_mb=BUDGET_MB, max_concurrent=3, calibration_path=path, config=_config(tmp_path)
    )
    assert sched is not None
    assert sched.budget_mb == BUDGET_MB
    assert table.vram_for("smolvla", "pusht") == SMOLVLA_MB


# --------------------------------------------------------------------- #
# End-to-end run_sweep concurrent dispatch                              #
# --------------------------------------------------------------------- #


def _row(policy: str, env: str, seed: int, ep: int) -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": ep,
        "success": True,
        "return_": 1.0,
        "n_steps": 10,
        "wallclock_s": 0.05,
        "video_sha256": "",
        "code_sha": "deadbeef",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-06-08T00:00:00+00:00",
    }


def _parse_cell(argv: list[str]) -> tuple[str, str, int, int, Path]:
    """Pull (policy, env, seed, n_episodes, out_parquet) out of a run_one argv."""

    def opt(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    return (
        opt("--policy"),
        opt("--env"),
        int(opt("--seed")),
        int(opt("--n-episodes")),
        Path(opt("--out-parquet")),
    )


class _ConcurrencyProbe:
    """Records peak observed concurrency + peak reserved VRAM during dispatch.

    The fake subprocess increments a live counter on entry, sleeps briefly
    so overlapping cells actually overlap, then writes the cell's rows and
    decrements. The scheduler's own peak counters are checked separately;
    here we assert what the *dispatcher* actually achieved.
    """

    def __init__(self, vram: dict[str, float | None], hold_s: float = 0.05) -> None:
        self.vram = vram
        self.hold_s = hold_s
        self._lock = threading.Lock()
        self.live = 0
        self.peak_live = 0
        self.live_vram = 0.0
        self.peak_vram = 0.0
        self.calls: list[tuple[str, str, int]] = []

    def fake(self, argv: list[str], *, timeout_s: float | None = None) -> rs.SubprocessOutcome:
        policy, env, seed, n_eps, out_parquet = _parse_cell(argv)
        v = self.vram.get(policy)
        with self._lock:
            self.calls.append((policy, env, seed))
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)
            if v is not None:
                self.live_vram += v
            self.peak_vram = max(self.peak_vram, self.live_vram)
        time.sleep(self.hold_s)
        rows = [_row(policy, env, seed, i) for i in range(n_eps)]
        df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
        append_cell_rows(out_parquet, df)
        with self._lock:
            self.live -= 1
            if v is not None:
                self.live_vram -= v
        return rs.SubprocessOutcome(returncode=0, stdout="", stderr="")


def _write_calibration(tmp_path: Path, vram: dict[str, float | None]) -> Path:
    cells = [_cell(p, "pusht", v) for p, v in vram.items() if v is not None]
    path = tmp_path / "calibration-20260608.json"
    path.write_text(json.dumps(_report(cells)))
    return path


# Real, runnable, pusht-compatible registry policies. The scheduler keys
# its VRAM lookup on the policy *name string*, so the audit's act/smolvla/
# diffusion peaks are simply attached to these three concrete cells. (The
# end-to-end path validates policy names against the real registry, so the
# e2e tests must use registry names; the scheduler unit tests above use the
# audit names directly and need no registry.)
PUSHT_POLICIES = ("no_op", "random", "diffusion_policy")


def test_run_sweep_concurrent_runs_small_cells_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 small pusht cells run >1 at a time; budget is never exceeded."""
    vram = {"no_op": ACT_MB, "random": SMOLVLA_MB, "diffusion_policy": DIFFUSION_MB}
    calib = _write_calibration(tmp_path, vram)
    config = rs.SweepConfig(
        policies=PUSHT_POLICIES,
        envs=("pusht",),
        seeds=(0,),
        episodes_per_seed=2,
        results_path=tmp_path / "results.parquet",
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )
    probe = _ConcurrencyProbe(vram)
    monkeypatch.setattr(rs, "_run_subprocess", probe.fake)

    outcome = rs.run_sweep(
        config=config,
        config_path=Path("test.yaml"),
        vram_budget_mb=BUDGET_MB,
        max_concurrent=3,
        calibration_path=calib,
    )

    assert outcome.exit_code == 0
    assert outcome.n_completed == 3
    # The whole point: more than one cell ran at once.
    assert probe.peak_live > 1
    # Safety invariant: live reserved VRAM never exceeded the budget.
    assert probe.peak_vram <= BUDGET_MB


def test_run_sweep_concurrent_budget_serializes_when_tight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a budget that fits only one cell at a time, cells serialize."""
    vram = {"diffusion_policy": DIFFUSION_MB}
    calib = _write_calibration(tmp_path, vram)
    config = rs.SweepConfig(
        policies=("diffusion_policy",),
        envs=("pusht",),
        seeds=(0, 1, 2),
        episodes_per_seed=2,
        results_path=tmp_path / "results.parquet",
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )
    probe = _ConcurrencyProbe(vram)
    monkeypatch.setattr(rs, "_run_subprocess", probe.fake)

    # Budget only fits one diffusion cell (1100 MB) at a time.
    outcome = rs.run_sweep(
        config=config,
        config_path=Path("test.yaml"),
        vram_budget_mb=DIFFUSION_MB + 100.0,
        max_concurrent=3,
        calibration_path=calib,
    )
    assert outcome.exit_code == 0
    assert outcome.n_completed == 3
    assert probe.peak_live == 1  # tight budget => strictly serial
    assert probe.peak_vram <= DIFFUSION_MB + 100.0


def test_run_sweep_concurrent_unknown_policy_serializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An un-calibrated policy never overlaps any other cell.

    no_op + random are calibrated; diffusion_policy is deliberately left
    OUT of the calibration JSON, so its VRAM is unknown -> exclusive.
    """
    vram_known = {"no_op": ACT_MB, "random": SMOLVLA_MB}
    calib = _write_calibration(tmp_path, vram_known)
    config = rs.SweepConfig(
        policies=PUSHT_POLICIES,
        envs=("pusht",),
        seeds=(0,),
        episodes_per_seed=2,
        results_path=tmp_path / "results.parquet",
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )

    unknown_policy = "diffusion_policy"
    overlap_with_unknown = {"value": False}
    lock = threading.Lock()
    live_set: set[str] = set()

    def fake(argv: list[str], *, timeout_s: float | None = None) -> rs.SubprocessOutcome:
        policy, env, seed, n_eps, out_parquet = _parse_cell(argv)
        with lock:
            if policy == unknown_policy and live_set:
                overlap_with_unknown["value"] = True
            if unknown_policy in live_set and policy != unknown_policy:
                overlap_with_unknown["value"] = True
            live_set.add(policy)
        time.sleep(0.05)
        rows = [_row(policy, env, seed, i) for i in range(n_eps)]
        df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
        append_cell_rows(out_parquet, df)
        with lock:
            live_set.discard(policy)
        return rs.SubprocessOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rs, "_run_subprocess", fake)

    outcome = rs.run_sweep(
        config=config,
        config_path=Path("test.yaml"),
        vram_budget_mb=BUDGET_MB,
        max_concurrent=3,
        calibration_path=calib,
    )
    assert outcome.exit_code == 0
    assert outcome.n_completed == 3
    assert overlap_with_unknown["value"] is False, (
        "unknown-VRAM diffusion_policy must run exclusively"
    )


def test_run_sweep_serial_path_unchanged_without_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --vram-budget-mb, dispatch is strictly serial (peak_live == 1)."""
    vram = {"no_op": ACT_MB, "random": SMOLVLA_MB}
    config = rs.SweepConfig(
        policies=("no_op", "random"),
        envs=("pusht",),
        seeds=(0, 1),
        episodes_per_seed=2,
        results_path=tmp_path / "results.parquet",
        videos_dir=None,
        record_video=False,
        device="cpu",
        policies_yaml=DEFAULT_POLICIES_YAML,
        envs_yaml=DEFAULT_ENVS_YAML,
        cell_timeout_s=None,
        max_parallel=1,
        overrides={},
    )
    probe = _ConcurrencyProbe(vram)
    monkeypatch.setattr(rs, "_run_subprocess", probe.fake)

    outcome = rs.run_sweep(config=config, config_path=Path("test.yaml"))
    assert outcome.exit_code == 0
    assert probe.peak_live == 1  # serial loop => never overlaps
