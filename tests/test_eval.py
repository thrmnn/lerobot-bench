"""Tests for ``lerobot_bench.eval``.

Pure orchestration tests using ``MockEnv`` / ``MockPolicy`` -- no
torch, no lerobot, no gymnasium imports needed. The seeding contract
end-to-end test (#20) uses the global numpy RNG via the random
baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.envs import EnvSpec
from lerobot_bench.eval import (
    CellResult,
    EpisodeResult,
    _gym_obs_to_batch,
    _LerobotPolicyAdapter,
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


def test_load_policy_pretrained_calls_lerobot_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretrained branch dispatches to ``_load_pretrained_policy`` with
    ``repo_id`` + ``revision`` taken straight from the spec.

    The actual lerobot factory call requires torch + lerobot installed;
    those are sim/GPU-marked. Here we monkeypatch the inner function so
    the test runs in CI fast without either dependency. The test
    asserts the *contract* between :func:`load_policy` and the inner
    helper: positional args go through verbatim, the spec's locked SHA
    is forwarded as the Hub revision pin.
    """
    from lerobot_bench import eval as eval_mod

    captured: dict[str, Any] = {}

    class _SentinelAdapter:
        def __call__(self, obs: dict[str, Any]) -> NDArray[np.float32]:
            return np.zeros((2,), dtype=np.float32)

        def reset(self) -> None:
            return None

    def _fake_inner(*, repo_id: str, revision: str, action_shape: Any, device: str) -> Any:
        captured.update(
            {
                "repo_id": repo_id,
                "revision": revision,
                "action_shape": action_shape,
                "device": device,
            }
        )
        return _SentinelAdapter()

    monkeypatch.setattr(eval_mod, "_load_pretrained_policy", _fake_inner)

    spec = PolicySpec(
        name="diffusion_policy",
        is_baseline=False,
        env_compat=("pusht",),
        repo_id="lerobot/diffusion_pusht",
        revision_sha="84a7c23178445c6bbf7e1a884ff497017910f653",
        fp_precision="fp32",
    )
    pol = load_policy(spec, action_shape=(2,), device="cpu")
    assert captured == {
        "repo_id": "lerobot/diffusion_pusht",
        "revision": "84a7c23178445c6bbf7e1a884ff497017910f653",
        "action_shape": (2,),
        "device": "cpu",
    }
    # And the returned policy is callable + resets.
    pol.reset()
    out = pol({"pixels": np.zeros((96, 96, 3), dtype=np.uint8)})
    assert out.shape == (2,)
    assert out.dtype == np.float32


# --------------------------------------------------------------------- #
# _LerobotPolicyAdapter unit tests                                      #
#                                                                       #
# We can't call real make_policy in CI fast (no torch/lerobot). These   #
# tests target the adapter logic with a fake model + fake processors.   #
# `pytest.importorskip("torch")` keeps them runnable in any env that    #
# happens to have torch installed (lerobot dev env), but they skip      #
# cleanly in the base CI fast job.                                      #
# --------------------------------------------------------------------- #


class _FakeModel:
    """Stand-in for a lerobot ``PreTrainedPolicy``.

    Records the batch passed to ``select_action`` (so tests can assert
    shape/keys/dtype), counts ``reset`` calls, and optionally raises if
    invoked outside ``torch.no_grad()`` so the adapter's gradient
    discipline is testable end-to-end.
    """

    def __init__(
        self,
        *,
        action_value: Any = None,
        require_no_grad: bool = False,
    ) -> None:
        self._action_value = action_value
        self._require_no_grad = require_no_grad
        self.reset_calls = 0
        self.last_batch: dict[str, Any] | None = None

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch: dict[str, Any]) -> Any:
        import torch

        if self._require_no_grad and torch.is_grad_enabled():
            raise AssertionError("select_action invoked with grad enabled")
        self.last_batch = batch
        if self._action_value is None:
            return torch.zeros(1, 2, dtype=torch.float32)
        return self._action_value


def _identity(x: Any) -> Any:
    return x


def test_lerobot_adapter_calls_model_with_torch_no_grad() -> None:
    pytest.importorskip("torch")

    model = _FakeModel(require_no_grad=True)
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    out = adapter({"pixels": np.zeros((96, 96, 3), dtype=np.uint8)})
    assert out.shape == (2,)


def test_lerobot_adapter_returns_numpy_float32_with_correct_shape() -> None:
    torch = pytest.importorskip("torch")

    model = _FakeModel(action_value=torch.tensor([[1.5, 2.5]], dtype=torch.float32))
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    out = adapter({"pixels": np.zeros((96, 96, 3), dtype=np.uint8)})
    assert out.dtype == np.float32
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([1.5, 2.5], dtype=np.float32))


def test_lerobot_adapter_reset_delegates_to_model() -> None:
    pytest.importorskip("torch")

    model = _FakeModel()
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    assert model.reset_calls == 0
    adapter.reset()
    adapter.reset()
    assert model.reset_calls == 2


def test_lerobot_adapter_reset_noop_when_model_has_no_reset() -> None:
    pytest.importorskip("torch")

    class _ModelNoReset:
        def select_action(self, batch: dict[str, Any]) -> Any:
            import torch

            return torch.zeros(1, 2)

    adapter = _LerobotPolicyAdapter(
        _ModelNoReset(),
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    adapter.reset()  # must not raise


def test_lerobot_adapter_handles_pusht_obs_dict() -> None:
    """PushT obs_type=pixels_agent_pos returns {pixels, agent_pos}.

    The adapter translates this to {observation.image, observation.state}
    via ``_gym_obs_to_batch`` (HWC->CHW + scale by 255 for the image).
    """
    torch = pytest.importorskip("torch")

    model = _FakeModel(action_value=torch.zeros(1, 2))
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )

    pixels = np.full((96, 96, 3), 255, dtype=np.uint8)  # all-white
    agent_pos = np.array([100.0, 200.0], dtype=np.float64)
    adapter({"pixels": pixels, "agent_pos": agent_pos})

    assert model.last_batch is not None
    assert set(model.last_batch.keys()) == {"observation.image", "observation.state"}
    image_t = model.last_batch["observation.image"]
    state_t = model.last_batch["observation.state"]
    # CHW shape, float in [0, 1].
    assert tuple(image_t.shape) == (3, 96, 96)
    assert image_t.dtype == torch.float32
    np.testing.assert_allclose(image_t.numpy().max(), 1.0)
    # agent_pos kept as float32 with original values, no scaling.
    assert tuple(state_t.shape) == (2,)
    np.testing.assert_array_equal(state_t.numpy(), [100.0, 200.0])


def test_lerobot_adapter_handles_aloha_multiview_obs_dict() -> None:
    """Aloha-style ``pixels.<view>`` keys map to ``observation.images.<view>``."""
    torch = pytest.importorskip("torch")

    model = _FakeModel(action_value=torch.zeros(1, 14))
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(14,),
        device="cpu",
    )

    obs = {
        "pixels.top": np.zeros((480, 640, 3), dtype=np.uint8),
        "agent_pos": np.zeros(14, dtype=np.float64),
    }
    adapter(obs)

    assert model.last_batch is not None
    assert "observation.images.top" in model.last_batch
    assert "observation.state" in model.last_batch
    assert tuple(model.last_batch["observation.images.top"].shape) == (3, 480, 640)


def test_gym_obs_to_batch_unpacks_nested_pixels_dict_for_aloha() -> None:
    """gym-aloha emits ``pixels`` as a ``{view: HWC}`` dict (not flat keys).

    Routing must keep this in the PushT/Aloha branch (it has no
    ``robot_state``) and unpack each view into ``observation.images.<view>``.
    """
    torch = pytest.importorskip("torch")
    from lerobot_bench.eval import _gym_obs_to_batch

    obs = {
        "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
        "agent_pos": np.zeros(14, dtype=np.float64),
    }
    batch = _gym_obs_to_batch(obs)
    assert "observation.images.top" in batch
    assert "observation.state" in batch
    # Did NOT route to the libero branch.
    assert "observation.image" not in batch
    assert tuple(batch["observation.images.top"].shape) == (3, 480, 640)
    assert isinstance(batch["observation.state"], torch.Tensor)


def test_lerobot_adapter_rejects_non_dict_obs() -> None:
    pytest.importorskip("torch")

    adapter = _LerobotPolicyAdapter(
        _FakeModel(),
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="dict observation"):
        adapter(np.zeros(5))  # type: ignore[arg-type]


def test_lerobot_adapter_action_shape_mismatch_raises() -> None:
    torch = pytest.importorskip("torch")

    # Model returns a (1, 14) action; adapter expects shape (2,) -> mismatch.
    model = _FakeModel(action_value=torch.zeros(1, 14))
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match=r"size 14, expected 2"):
        adapter({"pixels": np.zeros((96, 96, 3), dtype=np.uint8)})


def test_gym_obs_to_batch_unknown_key_raises() -> None:
    pytest.importorskip("torch")

    with pytest.raises(RuntimeError, match="unknown gym observation key"):
        _gym_obs_to_batch({"weird_key": np.zeros(2)})


def test_gym_obs_to_batch_handles_environment_state() -> None:
    pytest.importorskip("torch")

    obs = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "environment_state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    batch = _gym_obs_to_batch(obs)
    assert "observation.environment_state" in batch
    assert tuple(batch["observation.environment_state"].shape) == (3,)


# --------------------------------------------------------------------- #
# load_env: gym_kwargs forwarding                                       #
# --------------------------------------------------------------------- #


def test_load_env_forwards_gym_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EnvSpec.gym_kwargs`` must be forwarded verbatim to ``gym.make``.

    Stubs ``importlib.import_module`` to skip the ``gym_X`` namespace
    side-effect import (we don't have gym-pusht installed in CI fast),
    then stubs ``gymnasium.make`` to capture its kwargs without
    actually instantiating an env.
    """
    gym = pytest.importorskip("gymnasium")

    spec = EnvSpec(
        name="pusht",
        family="pusht",
        gym_id="gym_pusht/PushT-v0",
        max_steps=300,
        success_threshold=0.95,
        lerobot_module="lerobot.envs.pusht",
        gym_kwargs=(("obs_type", "pixels_agent_pos"),),
    )

    captured: dict[str, Any] = {}

    def _fake_make(env_id: str, **kwargs: Any) -> Any:
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs

        class _Stub:
            def close(self) -> None:
                return None

        return _Stub()

    monkeypatch.setattr(gym, "make", _fake_make)

    # Skip the namespace import by no-op'ing importlib.import_module.
    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda _name: None)

    load_env(spec)
    assert captured["env_id"] == "gym_pusht/PushT-v0"
    assert captured["kwargs"] == {"max_episode_steps": 300, "obs_type": "pixels_agent_pos"}


def test_env_spec_gym_kwargs_dict_roundtrips() -> None:
    spec = EnvSpec(
        name="x",
        family="x",
        gym_id="x/v0",
        max_steps=10,
        success_threshold=0.5,
        lerobot_module="x",
        gym_kwargs=(("a", 1), ("b", "two")),
    )
    assert spec.gym_kwargs_dict() == {"a": 1, "b": "two"}


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
# load_env: factory dispatch path                                       #
# --------------------------------------------------------------------- #


def test_load_env_dispatches_to_factory_when_factory_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``EnvSpec.factory`` is set, ``load_env`` resolves through
    ``importlib.import_module(factory).make_env(cfg, n_envs=1)`` and
    skips the gym.make namespace-resolve path entirely.

    We mock the factory module via ``importlib.import_module`` so the
    test runs without lerobot/libero installed. The mock records that
    ``make_env_config(env_type="libero", ...)`` was called and that
    ``make_env(cfg, n_envs=1)`` returned the expected dict shape.
    """
    import sys
    import types

    captured: dict[str, Any] = {}

    fake_mod = types.ModuleType("fake_factory_mod")

    class _StubVecEnv:
        num_envs = 1

        def step(self, action: Any) -> Any:
            return {}, [0.0], [False], [False], {}

        def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
            return {}, {}

        def close(self) -> None:
            return None

    def _make_env_config(*, env_type: str, **kwargs: Any) -> Any:
        captured["cfg_call"] = {"env_type": env_type, **kwargs}
        return {"_cfg_marker": env_type}

    def _make_env(cfg: Any, *, n_envs: int) -> Any:
        captured["make_env_args"] = {"cfg": cfg, "n_envs": n_envs}
        return {"libero_spatial": {0: _StubVecEnv()}}

    fake_mod.make_env_config = _make_env_config  # type: ignore[attr-defined]
    fake_mod.make_env = _make_env  # type: ignore[attr-defined]
    sys.modules["fake_factory_mod"] = fake_mod

    try:
        spec = EnvSpec(
            name="libero_spatial",
            family="libero",
            max_steps=280,
            success_threshold=1.0,
            lerobot_module="lerobot.envs.libero",
            factory="fake_factory_mod",
            factory_kwargs=(
                ("env_type", "libero"),
                ("task", "libero_spatial"),
                ("task_ids", (0,)),  # tuple to keep the spec hashable
            ),
        )

        # Stub out _ensure_libero_setup so the test doesn't touch ~/.libero.
        from lerobot_bench import eval as eval_mod

        monkeypatch.setattr(eval_mod, "_ensure_libero_setup", lambda: None)

        env = load_env(spec)
    finally:
        del sys.modules["fake_factory_mod"]

    assert captured["cfg_call"] == {
        "env_type": "libero",
        "task": "libero_spatial",
        "task_ids": (0,),
    }
    assert captured["make_env_args"]["n_envs"] == 1
    # The returned env is wrapped in our debatching adapter.
    from lerobot_bench.eval import _DebatchedVecEnvAdapter

    assert isinstance(env, _DebatchedVecEnvAdapter)


def test_load_env_factory_skips_gym_namespace_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory specs must NOT trigger the ``gym_X/`` namespace import
    side-effect. Confirms the dispatch branches are mutually exclusive.
    """
    import sys
    import types

    namespace_imports: list[str] = []
    real_import = __import__

    def _record_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("gym_"):
            namespace_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _record_import)

    fake_mod = types.ModuleType("factory_no_gym")

    def _make_env(**_: Any) -> Any:
        # Bare factory, no env_type — exercises the no-config branch.
        class _SingleEnv:
            def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
                return {}, {}

            def step(self, action: Any) -> Any:
                return {}, 0.0, False, False, {}

            def close(self) -> None:
                return None

            def render(self) -> Any:
                return np.zeros((4, 4, 3), dtype=np.uint8)

        return _SingleEnv()

    fake_mod.make_env = _make_env  # type: ignore[attr-defined]
    sys.modules["factory_no_gym"] = fake_mod

    try:
        spec = EnvSpec(
            name="custom",
            family="custom",
            max_steps=10,
            success_threshold=1.0,
            lerobot_module="x",
            factory="factory_no_gym",
        )
        from lerobot_bench import eval as eval_mod

        monkeypatch.setattr(eval_mod, "_ensure_libero_setup", lambda: None)

        env = load_env(spec)
        env.close()  # type: ignore[attr-defined]
    finally:
        del sys.modules["factory_no_gym"]

    assert namespace_imports == [], (
        f"factory dispatch leaked into gym namespace import path: {namespace_imports}"
    )


def test_load_env_factory_rejects_n_envs_above_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bench runs single episodes; n_envs > 1 in factory_kwargs is rejected."""
    import sys
    import types

    fake_mod = types.ModuleType("factory_multi")
    fake_mod.make_env_config = lambda **_: object()  # type: ignore[attr-defined]
    fake_mod.make_env = lambda *_, **__: {}  # type: ignore[attr-defined]
    sys.modules["factory_multi"] = fake_mod

    try:
        spec = EnvSpec(
            name="multi",
            family="x",
            max_steps=10,
            success_threshold=1.0,
            lerobot_module="x",
            factory="factory_multi",
            factory_kwargs=(("env_type", "libero"), ("n_envs", 4)),
        )
        from lerobot_bench import eval as eval_mod

        monkeypatch.setattr(eval_mod, "_ensure_libero_setup", lambda: None)

        with pytest.raises(ValueError, match="n_envs must be 1"):
            load_env(spec)
    finally:
        del sys.modules["factory_multi"]


def test_load_env_factory_rejects_multi_suite_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a factory returns >1 suite or >1 task, load_env refuses (single-cell contract)."""
    import sys
    import types

    fake_mod = types.ModuleType("factory_multi_result")
    fake_mod.make_env_config = lambda **_: object()  # type: ignore[attr-defined]

    class _Stub:
        num_envs = 1

        def reset(self, *, seed: int) -> Any:
            return {}, {}

        def step(self, a: Any) -> Any:
            return {}, [0.0], [False], [False], {}

        def close(self) -> None:
            return None

    fake_mod.make_env = lambda *_, **__: {  # type: ignore[attr-defined]
        "suite_a": {0: _Stub()},
        "suite_b": {0: _Stub()},
    }
    sys.modules["factory_multi_result"] = fake_mod

    try:
        spec = EnvSpec(
            name="multi",
            family="x",
            max_steps=10,
            success_threshold=1.0,
            lerobot_module="x",
            factory="factory_multi_result",
            factory_kwargs=(("env_type", "libero"),),
        )
        from lerobot_bench import eval as eval_mod

        monkeypatch.setattr(eval_mod, "_ensure_libero_setup", lambda: None)

        with pytest.raises(RuntimeError, match="expected 1 suite, got 2"):
            load_env(spec)
    finally:
        del sys.modules["factory_multi_result"]


def test_debatched_vec_env_adapter_strips_and_adds_batch_dim() -> None:
    """The adapter strips axis-0 from obs/reward and adds it to actions."""
    pytest.importorskip("gymnasium")

    actions_seen: list[np.ndarray] = []

    class _FakeVecEnv:
        num_envs = 1
        single_action_space = type("S", (), {"shape": (7,)})()
        single_observation_space: Any = None

        def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
            return (
                {"pixels": {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}},
                {"task": ["x"]},
            )

        def step(
            self, action: np.ndarray
        ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
            actions_seen.append(action.copy())
            return (
                {"pixels": {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}},
                np.array([0.5]),
                np.array([True]),
                np.array([False]),
                {},
            )

        def close(self) -> None:
            return None

    from lerobot_bench.eval import _DebatchedVecEnvAdapter

    adapter = _DebatchedVecEnvAdapter(_FakeVecEnv())
    obs, _info = adapter.reset(seed=42)
    # Stripped: image is now (4, 4, 3) not (1, 4, 4, 3).
    assert obs["pixels"]["image"].shape == (4, 4, 3)

    obs, reward, terminated, truncated, _ = adapter.step(np.zeros(7, dtype=np.float32))
    assert reward == pytest.approx(0.5)
    assert terminated is True
    assert truncated is False
    # Action passed to vec env got a leading batch dim.
    assert actions_seen[-1].shape == (1, 7)


def test_strip_batch_dim_handles_nested_dicts() -> None:
    """LIBERO obs has 2-3 levels of dict nesting under robot_state."""
    from lerobot_bench.eval import _strip_batch_dim

    nested = {
        "pixels": {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)},
        "robot_state": {
            "eef": {
                "pos": np.zeros((1, 3)),
                "quat": np.zeros((1, 4)),
            },
            "gripper": {"qpos": np.zeros((1, 2))},
        },
    }
    out = _strip_batch_dim(nested)
    assert out["pixels"]["image"].shape == (4, 4, 3)
    assert out["robot_state"]["eef"]["pos"].shape == (3,)
    assert out["robot_state"]["eef"]["quat"].shape == (4,)
    assert out["robot_state"]["gripper"]["qpos"].shape == (2,)


def test_libero_obs_to_batch_translates_correctly() -> None:
    """LIBERO obs → lerobot batch: H+W flip on images, quat→axisangle on state."""
    torch = pytest.importorskip("torch")

    from lerobot_bench.eval import _libero_obs_to_batch

    # Identity quaternion (w=1) -> zero axis-angle.
    obs = {
        "pixels": {
            "image": np.full((4, 4, 3), 100, dtype=np.uint8),
            "image2": np.full((4, 4, 3), 200, dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([1.0, 2.0, 3.0]),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),  # x,y,z,w identity
                "mat": np.eye(3),
            },
            "gripper": {
                "qpos": np.array([0.5, -0.5]),
                "qvel": np.array([0.0, 0.0]),
            },
            "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
        },
    }
    batch = _libero_obs_to_batch(obs)
    assert "observation.images.image" in batch
    assert "observation.images.image2" in batch
    assert "observation.state" in batch
    # CHW float in [0, 1].
    img = batch["observation.images.image"]
    assert tuple(img.shape) == (3, 4, 4)
    assert img.dtype == torch.float32
    # State = pos(3) + axisangle(3) + qpos(2) = 8.
    state = batch["observation.state"]
    assert tuple(state.shape) == (8,)
    np.testing.assert_allclose(state.numpy()[:3], [1.0, 2.0, 3.0])
    # Identity quat -> zero rotation in axis-angle.
    np.testing.assert_allclose(state.numpy()[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state.numpy()[6:], [0.5, -0.5])


def test_gym_obs_to_batch_dispatches_to_libero_branch_on_dict_pixels() -> None:
    """The dispatcher recognizes nested LIBERO obs by `pixels: dict` or `robot_state` key."""
    pytest.importorskip("torch")
    from lerobot_bench.eval import _gym_obs_to_batch

    obs = {
        "pixels": {"image": np.zeros((4, 4, 3), dtype=np.uint8)},
        "robot_state": {
            "eef": {"pos": np.zeros(3), "quat": np.array([0, 0, 0, 1.0])},
            "gripper": {"qpos": np.zeros(2)},
        },
    }
    batch = _gym_obs_to_batch(obs)
    assert "observation.images.image" in batch
    assert "observation.state" in batch
    # Confirm we did NOT route through the flat-pixels path.
    assert "observation.image" not in batch


def test_ensure_libero_setup_idempotent_when_config_exists(tmp_path: Path) -> None:
    """If ~/.libero/config.yaml already exists, _ensure_libero_setup is a no-op."""
    from lerobot_bench import eval as eval_mod

    fake_libero_dir = tmp_path / "libero_cfg"
    fake_libero_dir.mkdir()
    cfg_file = fake_libero_dir / "config.yaml"
    cfg_file.write_text("preexisting: marker\n")

    # Reset module-level guard so the function actually runs its body.
    eval_mod._LIBERO_SETUP_DONE = False
    try:
        import os

        os.environ["LIBERO_CONFIG_PATH"] = str(fake_libero_dir)
        eval_mod._ensure_libero_setup()
    finally:
        os.environ.pop("LIBERO_CONFIG_PATH", None)

    # File contents unchanged — we did not overwrite a user config.
    assert cfg_file.read_text() == "preexisting: marker\n"


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
