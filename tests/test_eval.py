"""Tests for ``lerobot_bench.eval``.

Pure orchestration tests using ``MockEnv`` / ``MockPolicy`` -- no
torch, no lerobot, no gymnasium imports needed. The seeding contract
end-to-end test (#20) uses the global numpy RNG via the random
baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.envs import EnvSpec
from lerobot_bench.eval import (
    CellResult,
    EpisodeResult,
    _NoOpPolicy,
    _RandomPolicy,
    load_env,
    load_policy,
    run_cell,
    seed_everything,
)
from lerobot_bench.policies import PolicySpec

# --------------------------------------------------------------------- #
# Mocks                                                                 #
# --------------------------------------------------------------------- #


class MockEnv:
    """Minimal gymnasium-shaped env for orchestration tests.

    ``success_at_step`` (1-indexed) sets ``terminated=True`` with
    reward 1.0 on that step. If ``None``, episodes always run until
    ``truncated`` at ``max_steps`` with reward 0.0.
    """

    def __init__(
        self,
        *,
        max_steps: int = 10,
        success_at_step: int | None = None,
        action_shape: tuple[int, ...] = (2,),
        reset_raises: Exception | None = None,
        step_raises_on_episode: int | None = None,
    ) -> None:
        self._max_steps = max_steps
        self._success_at_step = success_at_step
        self._action_shape = action_shape
        self._reset_raises = reset_raises
        self._step_raises_on_episode = step_raises_on_episode

        self._step_count = 0
        self._episode_count = -1  # incremented on reset
        self.reset_seeds: list[int] = []

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_shape

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._reset_raises is not None:
            raise self._reset_raises
        self.reset_seeds.append(seed)
        self._step_count = 0
        self._episode_count += 1
        return {"obs": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if (
            self._step_raises_on_episode is not None
            and self._episode_count == self._step_raises_on_episode
        ):
            raise RuntimeError("synthetic step crash")
        self._step_count += 1
        terminated = self._success_at_step is not None and self._step_count == self._success_at_step
        truncated = not terminated and self._step_count >= self._max_steps
        reward = 1.0 if terminated else 0.0
        return {"obs": np.zeros(4, dtype=np.float32)}, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class MockPolicy:
    """Counter-instrumented zero-action policy."""

    def __init__(self, action_shape: tuple[int, ...] = (2,)) -> None:
        self._action_shape = action_shape
        self.call_count = 0
        self.reset_count = 0

    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        self.call_count += 1
        return np.zeros(self._action_shape, dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


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
# 1. seed_everything                                                    #
# --------------------------------------------------------------------- #


def test_seed_everything_seeds_numpy() -> None:
    seed_everything(3)
    a = np.random.rand(5)
    seed_everything(3)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_returns_base_seed() -> None:
    assert seed_everything(2) == 2000
    assert seed_everything(0) == 0
    assert seed_everything(4) == 4000


# --------------------------------------------------------------------- #
# 3-11. run_cell orchestration                                          #
# --------------------------------------------------------------------- #


def test_run_cell_calls_env_reset_with_correct_seeds(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3)
    policy = MockPolicy()
    run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=2,
        n_episodes=5,
        record_video=False,
    )
    assert env.reset_seeds == [2000, 2001, 2002, 2003, 2004]


def test_run_cell_calls_policy_reset_per_episode(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3)
    policy = MockPolicy()
    run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=4,
        record_video=False,
    )
    assert policy.reset_count == 4


def test_run_cell_records_correct_episode_count(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=7,
        record_video=False,
    )
    assert len(result.episodes) == 7
    assert [ep.episode_index for ep in result.episodes] == list(range(7))


def test_run_cell_success_when_terminated_with_high_reward(env_spec: EnvSpec) -> None:
    # success_at_step=5 -> terminated at step 5 with reward=1.0; threshold=0.5 -> success
    env = MockEnv(max_steps=10, success_at_step=5)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=3,
        record_video=False,
    )
    assert all(ep.success for ep in result.episodes)
    assert all(ep.n_steps == 5 for ep in result.episodes)
    assert result.success_rate == 1.0


def test_run_cell_failure_when_truncated(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=10, success_at_step=None)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=3,
        record_video=False,
    )
    assert all(not ep.success for ep in result.episodes)
    assert all(ep.n_steps == 10 for ep in result.episodes)
    assert result.success_rate == 0.0


def test_run_cell_collects_frames_when_record_video_true(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3, success_at_step=2)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=True,
    )
    for ep in result.episodes:
        assert len(ep.frames) > 0
        assert ep.frames[0].shape == (64, 64, 3)
        assert ep.frames[0].dtype == np.uint8
        # Order: 1 reset frame + 1 frame per step. n_steps=2 -> 3 frames.
        assert len(ep.frames) == ep.n_steps + 1


def test_run_cell_skips_frames_when_record_video_false(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3, success_at_step=2)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=False,
    )
    for ep in result.episodes:
        assert ep.frames == ()


def test_run_cell_per_episode_exception_recorded(env_spec: EnvSpec) -> None:
    # Episode 1 (0-indexed) crashes during step.
    env = MockEnv(max_steps=5, success_at_step=3, step_raises_on_episode=1)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=4,
        record_video=False,
    )
    assert len(result.episodes) == 4
    crashed = result.episodes[1]
    assert crashed.error is not None
    assert "synthetic step crash" in crashed.error
    assert crashed.success is False
    assert crashed.return_ == 0.0
    assert crashed.n_steps == 0
    # Other episodes still ran normally.
    for i, ep in enumerate(result.episodes):
        if i == 1:
            continue
        assert ep.error is None
        assert ep.success is True


def test_run_cell_whole_cell_exception_propagates(env_spec: EnvSpec) -> None:
    # First reset crashes -- this is recorded as a per-episode error
    # too (not a hard propagation), since the loop catches Exception.
    # Verify by using a reset that raises BaseException-ish: we instead
    # use an env whose reset raises only on episode 0; the per-episode
    # try/except catches it. To get a true propagation we use an env
    # whose reset raises a KeyboardInterrupt (BaseException).
    class BadEnv(MockEnv):
        def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
            raise KeyboardInterrupt("simulated SIGINT mid-cell")

    env = BadEnv(max_steps=3)
    policy = MockPolicy()
    with pytest.raises(KeyboardInterrupt):
        run_cell(
            policy,
            env,
            policy_name="mock",
            env_spec=env_spec,
            seed_idx=0,
            n_episodes=2,
            record_video=False,
        )


# --------------------------------------------------------------------- #
# 12-15. CellResult / to_rows                                           #
# --------------------------------------------------------------------- #


def _make_episode(idx: int, *, success: bool) -> EpisodeResult:
    return EpisodeResult(
        episode_index=idx,
        success=success,
        return_=1.0 if success else 0.0,
        n_steps=10,
        wallclock_s=0.5,
        frames=(),
        final_reward=1.0 if success else 0.0,
        error=None,
    )


def _make_cell(successes: int, total: int) -> CellResult:
    eps = tuple(_make_episode(i, success=i < successes) for i in range(total))
    return CellResult(
        policy="mock",
        env="mock_env",
        seed=0,
        episodes=eps,
        code_sha="abc123",
        lerobot_version="0.5.1",
        timestamp_utc="2026-05-01T00:00:00+00:00",
    )


def test_cell_result_success_rate() -> None:
    cell = _make_cell(successes=3, total=5)
    assert cell.success_rate == pytest.approx(0.6)


def test_cell_result_success_rate_empty() -> None:
    cell = CellResult(
        policy="mock",
        env="mock_env",
        seed=0,
        episodes=(),
        code_sha="",
        lerobot_version="",
        timestamp_utc="",
    )
    assert cell.success_rate == 0.0


def test_cell_result_to_rows_schema_matches_checkpointing() -> None:
    cell = _make_cell(successes=2, total=3)
    df = cell.to_rows()
    assert tuple(df.columns) == RESULT_SCHEMA
    assert len(df) == 3


def test_cell_result_to_rows_video_sha_filled() -> None:
    cell = _make_cell(successes=1, total=3)
    shas = ["sha-a", "sha-b", "sha-c"]
    df = cell.to_rows(video_sha256_per_episode=shas)
    assert df["video_sha256"].tolist() == shas


def test_cell_result_to_rows_video_sha_empty_default() -> None:
    cell = _make_cell(successes=1, total=3)
    df = cell.to_rows()
    assert df["video_sha256"].tolist() == ["", "", ""]


def test_cell_result_to_rows_video_sha_length_mismatch_raises() -> None:
    cell = _make_cell(successes=1, total=3)
    with pytest.raises(ValueError, match="length 2"):
        cell.to_rows(video_sha256_per_episode=["a", "b"])


def test_cell_result_to_rows_carries_metadata() -> None:
    cell = _make_cell(successes=1, total=2)
    df = cell.to_rows()
    assert (df["code_sha"] == "abc123").all()
    assert (df["lerobot_version"] == "0.5.1").all()
    assert (df["timestamp_utc"] == "2026-05-01T00:00:00+00:00").all()
    assert (df["policy"] == "mock").all()
    assert (df["env"] == "mock_env").all()
    assert (df["seed"] == 0).all()


# --------------------------------------------------------------------- #
# 16-17. Baselines                                                      #
# --------------------------------------------------------------------- #


def test_baseline_no_op_returns_zeros() -> None:
    pol = _NoOpPolicy((2,))
    out = pol({"obs": np.zeros(4)})
    np.testing.assert_array_equal(out, np.zeros((2,), dtype=np.float32))
    # Returned array is a fresh copy -- mutating it doesn't poison subsequent calls.
    out[0] = 99.0
    out2 = pol({"obs": np.zeros(4)})
    np.testing.assert_array_equal(out2, np.zeros((2,), dtype=np.float32))


def test_baseline_random_uses_seeded_rng() -> None:
    seed_everything(0)
    pol = _RandomPolicy((3,))
    a1 = pol({})
    a2 = pol({})

    seed_everything(0)
    pol2 = _RandomPolicy((3,))
    b1 = pol2({})
    b2 = pol2({})

    np.testing.assert_array_equal(a1, b1)
    np.testing.assert_array_equal(a2, b2)
    # And the two draws within a seed are NOT identical (smoke check on randomness).
    assert not np.array_equal(a1, a2)


# --------------------------------------------------------------------- #
# 18-19. load_policy                                                    #
# --------------------------------------------------------------------- #


def test_load_policy_unrunnable_raises() -> None:
    spec = PolicySpec(
        name="diffusion_policy",
        is_baseline=False,
        env_compat=("pusht",),
        repo_id="lerobot/diffusion_pusht",
        revision_sha=None,  # not yet locked -- not runnable
    )
    with pytest.raises(RuntimeError, match="revision_sha"):
        load_policy(spec, action_shape=(2,))


def test_load_policy_no_op_baseline_returns_callable() -> None:
    spec = PolicySpec(name="no_op", is_baseline=True, env_compat=("pusht",))
    pol = load_policy(spec, action_shape=(2,))
    out = pol({"obs": np.zeros(4)})
    np.testing.assert_array_equal(out, np.zeros((2,), dtype=np.float32))


def test_load_policy_random_baseline_returns_callable() -> None:
    spec = PolicySpec(name="random", is_baseline=True, env_compat=("pusht",))
    pol = load_policy(spec, action_shape=(2,))
    out = pol({"obs": np.zeros(4)})
    assert out.shape == (2,)
    assert out.dtype == np.float32


def test_load_policy_baseline_requires_action_shape() -> None:
    spec = PolicySpec(name="no_op", is_baseline=True, env_compat=("pusht",))
    with pytest.raises(ValueError, match="action_shape"):
        load_policy(spec)


def test_load_policy_pretrained_runnable_not_implemented_yet() -> None:
    # A "runnable" pretrained spec (revision_sha set) currently raises
    # NotImplementedError -- the lerobot factory call is Day 0b TODO.
    spec = PolicySpec(
        name="diffusion_policy",
        is_baseline=False,
        env_compat=("pusht",),
        repo_id="lerobot/diffusion_pusht",
        revision_sha="deadbeef",
        fp_precision="fp32",
    )
    with pytest.raises(NotImplementedError, match="Day 0b"):
        load_policy(spec, action_shape=(2,))


# --------------------------------------------------------------------- #
# load_env: namespace-import side-effect                                #
# --------------------------------------------------------------------- #


def test_load_env_raises_clear_error_for_missing_namespace_package() -> None:
    """gym_X/EnvName style ids require the gym_X package installed.

    Without the namespace import side-effect, gymnasium raises
    NamespaceNotFound which is opaque. ``load_env`` resolves the namespace
    eagerly so the user gets a one-line install hint.

    Skipped if gymnasium itself is not importable (CI fast job).
    """
    pytest.importorskip("gymnasium")
    spec = EnvSpec(
        name="fake",
        family="fake",
        gym_id="gym_does_not_exist_xyz/Fake-v0",
        max_steps=10,
        success_threshold=0.5,
        lerobot_module="lerobot.envs.fake",
    )
    with pytest.raises(ImportError, match=r"gym_does_not_exist_xyz.*pip install"):
        load_env(spec)


def test_load_env_skips_namespace_import_for_non_gym_prefixed_ids() -> None:
    """Built-in gym envs like CartPole-v1 have no gym_X/ namespace and
    should not trigger an import-resolve attempt. We do not actually
    instantiate CartPole here (avoid mujoco / classic-control dep
    surprise in CI); we only assert the function does not raise
    our namespace-resolve ImportError before getting to gym.make.

    Skipped if gymnasium itself is not importable.
    """
    pytest.importorskip("gymnasium")
    spec = EnvSpec(
        name="cartpole",
        family="classic",
        gym_id="CartPole-v1",
        max_steps=10,
        success_threshold=0.5,
        lerobot_module="n/a",
    )
    try:
        env = load_env(spec)
        env.close()  # type: ignore[attr-defined]
    except ImportError as exc:
        # Our namespace-resolve error message must NOT appear since the id
        # has no gym_ prefix; any other ImportError (missing classic-control,
        # etc.) is acceptable since CI fast doesn't install full sim extras.
        assert "namespace package" not in str(exc)


# --------------------------------------------------------------------- #
# 20. End-to-end seeding via random baseline                            #
# --------------------------------------------------------------------- #


def test_run_cell_seeded_random_policy_reproducible(env_spec: EnvSpec) -> None:
    """Two run_cell invocations with the same seed_idx + identical env produce
    byte-identical action sequences -- proves the seeding contract works
    end-to-end through the random baseline.
    """

    class RecordingEnv(MockEnv):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.actions_seen: list[np.ndarray] = []

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.actions_seen.append(action.copy())
            return super().step(action)

    spec = PolicySpec(name="random", is_baseline=True, env_compat=("pusht",))

    env_a = RecordingEnv(max_steps=5)
    pol_a = load_policy(spec, action_shape=(2,))
    run_cell(
        pol_a,
        env_a,
        policy_name="random",
        env_spec=env_spec,
        seed_idx=4,
        n_episodes=3,
        record_video=False,
    )

    env_b = RecordingEnv(max_steps=5)
    pol_b = load_policy(spec, action_shape=(2,))
    run_cell(
        pol_b,
        env_b,
        policy_name="random",
        env_spec=env_spec,
        seed_idx=4,
        n_episodes=3,
        record_video=False,
    )

    assert len(env_a.actions_seen) == len(env_b.actions_seen)
    for a, b in zip(env_a.actions_seen, env_b.actions_seen, strict=True):
        np.testing.assert_array_equal(a, b)


def test_run_cell_seeded_random_different_seeds_diverge(env_spec: EnvSpec) -> None:
    """Sanity: different seed_idx -> different action sequences."""

    class RecordingEnv(MockEnv):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.actions_seen: list[np.ndarray] = []

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.actions_seen.append(action.copy())
            return super().step(action)

    spec = PolicySpec(name="random", is_baseline=True, env_compat=("pusht",))

    env_a = RecordingEnv(max_steps=5)
    run_cell(
        load_policy(spec, action_shape=(2,)),
        env_a,
        policy_name="random",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=2,
        record_video=False,
    )

    env_b = RecordingEnv(max_steps=5)
    run_cell(
        load_policy(spec, action_shape=(2,)),
        env_b,
        policy_name="random",
        env_spec=env_spec,
        seed_idx=1,
        n_episodes=2,
        record_video=False,
    )

    # At least one action position should differ across the seeds.
    diffs = [
        not np.array_equal(a, b)
        for a, b in zip(env_a.actions_seen, env_b.actions_seen, strict=True)
    ]
    assert any(diffs)


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


def test_run_cell_rejects_zero_episodes(env_spec: EnvSpec) -> None:
    with pytest.raises(ValueError, match="n_episodes"):
        run_cell(
            MockPolicy(),
            MockEnv(),
            policy_name="mock",
            env_spec=env_spec,
            seed_idx=0,
            n_episodes=0,
        )


def test_run_cell_rejects_negative_seed(env_spec: EnvSpec) -> None:
    with pytest.raises(ValueError, match="seed_idx"):
        run_cell(
            MockPolicy(),
            MockEnv(),
            policy_name="mock",
            env_spec=env_spec,
            seed_idx=-1,
            n_episodes=1,
        )


def test_run_cell_passes_through_explicit_metadata(env_spec: EnvSpec) -> None:
    env = MockEnv(max_steps=3)
    result = run_cell(
        MockPolicy(),
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=1,
        record_video=False,
        code_sha="explicit-sha",
        lerobot_version="explicit-version",
    )
    assert result.code_sha == "explicit-sha"
    assert result.lerobot_version == "explicit-version"


def test_episode_result_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    ep = _make_episode(0, success=True)
    with pytest.raises(FrozenInstanceError):
        ep.episode_index = 1  # type: ignore[misc]


def test_to_rows_dtypes_compatible_with_parquet_roundtrip(tmp_path: Any, env_spec: EnvSpec) -> None:
    """Sanity: a real CellResult round-trips through parquet without column drift."""
    env = MockEnv(max_steps=3, success_at_step=2)
    result = run_cell(
        MockPolicy(),
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=1,
        n_episodes=2,
        record_video=False,
    )
    df = result.to_rows()
    path = tmp_path / "out.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    roundtripped = pd.read_parquet(path)
    assert tuple(roundtripped.columns) == RESULT_SCHEMA
    assert len(roundtripped) == 2
