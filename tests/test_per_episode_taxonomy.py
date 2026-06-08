"""Tests for the OPTIONAL per-episode failure-taxonomy emission (audit #171).

Strictly additive feature: a NEW per-episode sidecar
(:meth:`CellResult.to_per_episode_rows`, :func:`append_per_episode_rows`,
the ``per_episode_sink`` opt-in on :func:`run_cell`). These tests pin two
things:

1. The heuristic ``failure_label`` is derived correctly from the signals
   the rollout records (success / terminated / truncated / errored).
2. Enabling the sink does NOT perturb the canonical
   :meth:`CellResult.to_rows` output or ``success_rate`` — byte-identical
   to a run with the sink off.

Pure orchestration; no torch / lerobot / gymnasium.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from embodimetry.checkpointing import RESULT_SCHEMA
from embodimetry.envs import EnvSpec
from embodimetry.eval import (
    PER_EPISODE_SCHEMA,
    CellResult,
    EpisodeResult,
    append_per_episode_rows,
    run_cell,
)

# --------------------------------------------------------------------- #
# Synthetic env: full control over (reward, terminated, truncated)      #
# --------------------------------------------------------------------- #


class ScriptedEnv:
    """Env whose last step emits a caller-chosen (reward, term, trunc).

    Runs ``stop_at_step`` no-op steps (reward 0, neither flag), then on
    that step emits ``final_reward`` with ``terminated``/``truncated`` as
    configured. With ``stop_at_step=None`` it never sets a flag and the
    harness step cap is the limiter (the bench-side timeout path).
    """

    def __init__(
        self,
        *,
        max_steps: int,
        stop_at_step: int | None,
        final_reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        action_shape: tuple[int, ...] = (2,),
    ) -> None:
        self._max_steps = max_steps
        self._stop_at_step = stop_at_step
        self._final_reward = final_reward
        self._terminated = terminated
        self._truncated = truncated
        self._action_shape = action_shape
        self._step_count = 0

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        return {"obs": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = {"obs": np.zeros(4, dtype=np.float32)}
        if self._stop_at_step is not None and self._step_count == self._stop_at_step:
            return obs, self._final_reward, self._terminated, self._truncated, {}
        return obs, 0.0, False, False, {}

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _ZeroPolicy:
    def __init__(self, action_shape: tuple[int, ...] = (2,)) -> None:
        self._action_shape = action_shape

    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        return np.zeros(self._action_shape, dtype=np.float32)

    def reset(self) -> None:
        return None


@pytest.fixture
def env_spec() -> EnvSpec:
    return EnvSpec(
        name="mock_env",
        family="mock",
        gym_id="Mock-v0",
        max_steps=10,
        success_threshold=0.5,
        lerobot_module="mock",
    )


# --------------------------------------------------------------------- #
# failure_label heuristic (unit, on EpisodeResult directly)             #
# --------------------------------------------------------------------- #


def _ep(**kw: Any) -> EpisodeResult:
    base: dict[str, Any] = {
        "episode_index": 0,
        "success": False,
        "return_": 0.0,
        "n_steps": 5,
        "wallclock_s": 0.1,
        "frames": (),
        "final_reward": 0.0,
    }
    base.update(kw)
    return EpisodeResult(**base)


def test_failure_label_success_wins_over_flags() -> None:
    # success short-circuits regardless of terminated/truncated.
    assert _ep(success=True, terminated=True).failure_label == "success"
    assert _ep(success=True, truncated=True).failure_label == "success"


def test_failure_label_timeout() -> None:
    assert _ep(success=False, terminated=False, truncated=True).failure_label == "timeout"


def test_failure_label_early_termination() -> None:
    assert _ep(success=False, terminated=True, truncated=False).failure_label == "early_termination"


def test_failure_label_errored_short_circuits() -> None:
    # An errored episode is labelled errored even though success=False.
    assert _ep(error="RuntimeError: boom").failure_label == "errored"


def test_failure_label_unknown_when_no_flags() -> None:
    assert _ep(success=False, terminated=False, truncated=False).failure_label == "unknown"


# --------------------------------------------------------------------- #
# End-to-end through run_cell: labels reflect real rollout outcomes     #
# --------------------------------------------------------------------- #


def test_run_cell_labels_success(env_spec: EnvSpec) -> None:
    env = ScriptedEnv(max_steps=10, stop_at_step=3, final_reward=1.0, terminated=True)
    cell = run_cell(
        _ZeroPolicy(),
        env,
        policy_name="p",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=False,
    )
    labels = [ep.failure_label for ep in cell.episodes]
    assert labels == ["success", "success"]
    assert all(ep.terminated and not ep.truncated for ep in cell.episodes)


def test_run_cell_labels_early_termination(env_spec: EnvSpec) -> None:
    # terminated with reward below threshold (0.5) -> not success.
    env = ScriptedEnv(max_steps=10, stop_at_step=3, final_reward=0.0, terminated=True)
    cell = run_cell(
        _ZeroPolicy(),
        env,
        policy_name="p",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=False,
    )
    assert [ep.failure_label for ep in cell.episodes] == ["early_termination"] * 2
    assert all(not ep.success for ep in cell.episodes)


def test_run_cell_labels_timeout_via_env_truncation(env_spec: EnvSpec) -> None:
    env = ScriptedEnv(max_steps=10, stop_at_step=10, final_reward=0.0, truncated=True)
    cell = run_cell(
        _ZeroPolicy(),
        env,
        policy_name="p",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=False,
    )
    assert [ep.failure_label for ep in cell.episodes] == ["timeout"] * 2


def test_run_cell_labels_timeout_via_harness_cap(env_spec: EnvSpec) -> None:
    # Env never sets a flag; the bench step cap is the limiter -> timeout.
    env = ScriptedEnv(max_steps=10, stop_at_step=None)
    cell = run_cell(
        _ZeroPolicy(),
        env,
        policy_name="p",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=1,
        record_video=False,
    )
    ep = cell.episodes[0]
    assert ep.n_steps == 10
    assert ep.failure_label == "timeout"
    assert ep.truncated and not ep.terminated


# --------------------------------------------------------------------- #
# to_per_episode_rows schema + success agreement                        #
# --------------------------------------------------------------------- #


def _mixed_cell() -> CellResult:
    eps = (
        EpisodeResult(0, True, 1.0, 3, 0.1, (), 1.0, terminated=True),
        EpisodeResult(1, False, 0.0, 10, 0.1, (), 0.0, truncated=True),
        EpisodeResult(2, False, 0.0, 3, 0.1, (), 0.0, terminated=True),
        EpisodeResult(3, False, 0.0, 0, 0.1, (), 0.0, error="RuntimeError: boom"),
    )
    return CellResult(
        policy="p",
        env="e",
        seed=0,
        episodes=eps,
        code_sha="sha",
        lerobot_version="0.5.1",
        timestamp_utc="2026-06-08T00:00:00+00:00",
        eval_run_id="run-1",
    )


def test_to_per_episode_rows_schema_and_labels() -> None:
    df = _mixed_cell().to_per_episode_rows()
    assert tuple(df.columns) == PER_EPISODE_SCHEMA
    assert df["failure_label"].tolist() == [
        "success",
        "timeout",
        "early_termination",
        "errored",
    ]
    assert df["errored"].tolist() == [False, False, False, True]


def test_per_episode_success_column_matches_canonical() -> None:
    # The success boolean in the sidecar is the same set as the canonical
    # success column -> failure_label=="success" can never disagree.
    cell = _mixed_cell()
    canonical = cell.to_rows()["success"].tolist()
    sidecar = cell.to_per_episode_rows()["success"].tolist()
    assert canonical == sidecar
    label_success = [lbl == "success" for lbl in cell.to_per_episode_rows()["failure_label"]]
    assert label_success == canonical


# --------------------------------------------------------------------- #
# The hard constraint: sink ON vs OFF leaves canonical output identical #
# --------------------------------------------------------------------- #


def _run(env_spec: EnvSpec, *, sink: Path | None) -> CellResult:
    # Fresh env each call (stateful step counter); deterministic by seed.
    env = ScriptedEnv(max_steps=10, stop_at_step=4, final_reward=1.0, terminated=True)
    return run_cell(
        _ZeroPolicy(),
        env,
        policy_name="p",
        env_spec=env_spec,
        seed_idx=1,
        n_episodes=3,
        record_video=False,
        code_sha="fixed-sha",
        lerobot_version="0.5.1",
        per_episode_sink=sink,
    )


def test_canonical_output_byte_identical_with_sink_on(env_spec: EnvSpec, tmp_path: Path) -> None:
    off = _run(env_spec, sink=None)
    on = _run(env_spec, sink=tmp_path / "ladder" / "per_episode.parquet")

    # success_rate unchanged.
    assert off.success_rate == on.success_rate

    # Canonical parquet rows are identical except the two wall-clock-
    # derived columns (timestamp_utc + measured wallclock_s), which vary
    # between any two real runs regardless of this feature.
    nondeterministic = ["timestamp_utc", "wallclock_s"]
    df_off = off.to_rows().drop(columns=nondeterministic)
    df_on = on.to_rows().drop(columns=nondeterministic)
    pd.testing.assert_frame_equal(df_off, df_on)
    assert tuple(df_on.columns) == tuple(c for c in RESULT_SCHEMA if c not in nondeterministic)


def test_sink_written_at_cell_boundary(env_spec: EnvSpec, tmp_path: Path) -> None:
    sink = tmp_path / "ladder" / "per_episode.parquet"
    assert not sink.exists()
    _run(env_spec, sink=sink)
    assert sink.exists()
    df = pd.read_parquet(sink)
    assert tuple(df.columns) == PER_EPISODE_SCHEMA
    assert len(df) == 3
    assert df["failure_label"].tolist() == ["success"] * 3


def test_sink_appends_across_cells(env_spec: EnvSpec, tmp_path: Path) -> None:
    sink = tmp_path / "per_episode.parquet"
    _run(env_spec, sink=sink)
    _run(env_spec, sink=sink)
    assert len(pd.read_parquet(sink)) == 6


# --------------------------------------------------------------------- #
# append_per_episode_rows: schema gate + atomic append                  #
# --------------------------------------------------------------------- #


def test_append_per_episode_rows_rejects_wrong_columns(tmp_path: Path) -> None:
    bad = pd.DataFrame({"policy": ["p"], "env": ["e"]})
    with pytest.raises(ValueError, match="wrong columns"):
        append_per_episode_rows(tmp_path / "s.parquet", bad)


def test_append_per_episode_rows_empty_is_noop(tmp_path: Path) -> None:
    sink = tmp_path / "s.parquet"
    empty = pd.DataFrame({col: [] for col in PER_EPISODE_SCHEMA})
    assert append_per_episode_rows(sink, empty) == 0
    assert not sink.exists()
