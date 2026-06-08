"""Tests for ``embodimetry.eval``.

Pure orchestration tests using ``MockEnv`` / ``MockPolicy`` -- no
torch, no lerobot, no gymnasium imports needed. The seeding contract
end-to-end test (#20) uses the global numpy RNG via the random
baseline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from embodimetry.checkpointing import RESULT_SCHEMA
from embodimetry.envs import EnvSpec
from embodimetry.eval import (
    CellResult,
    EpisodeResult,
    _buffer_name_to_feature_key,
    _gym_obs_to_batch,
    _LerobotPolicyAdapter,
    _NoOpPolicy,
    _patch_processors_for_policy,
    _patch_xvla_postprocessor,
    _patch_xvla_preprocessor,
    _RandomPolicy,
    load_env,
    load_policy,
    run_cell,
    seed_everything,
)
from embodimetry.policies import PolicySpec

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


def test_seed_everything_sets_cublas_workspace_env() -> None:
    """CUBLAS_WORKSPACE_CONFIG is set even when torch is absent (audit C3)."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    seed_everything(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_seed_everything_respects_preset_cublas_workspace() -> None:
    """setdefault: an operator-exported value is left untouched."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        seed_everything(0)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_seed_everything_pins_torch_determinism() -> None:
    """cuDNN deterministic on / benchmark off after seeding (audit C3)."""
    torch = pytest.importorskip("torch")
    seed_everything(1)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


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


def test_run_cell_drops_frames_after_per_episode_encode(
    env_spec: EnvSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming MP4 encode path: with ``videos_dir`` set, frames must be
    dropped from each :class:`EpisodeResult` as soon as that episode's
    MP4 has been written -- the bug we fixed was 50 episodes worth of
    frames pinned in memory until the end of the cell. This test stubs
    the actual encoder so it runs imageio-free, then asserts every
    post-cell ``ep.frames`` tuple is empty and that each successful
    episode carries a ``video_path`` + ``video_sha256``.
    """
    from embodimetry import eval as eval_mod
    from embodimetry.render import RenderResult

    encoded_paths: list[Path] = []

    def _fake_render(stacked: np.ndarray, out_path: Path) -> RenderResult:
        # Touch the file so the on-disk sanity check in run_one's shim
        # passes (and so this test mirrors what the real encoder does).
        out_path.write_bytes(b"FAKE-MP4")
        encoded_paths.append(out_path)
        from embodimetry.render import EncoderSettings

        return RenderResult(
            path=out_path,
            bytes_written=len(b"FAKE-MP4"),
            frame_count=int(stacked.shape[0]),
            encoder_settings=EncoderSettings(
                fps=10, size=256, codec="libx264", pixel_format="yuv420p", crf=23, rung_index=0
            ),
            content_sha256=f"sha-{out_path.name}",
        )

    # Replace render_episode at the module the streaming encode imports
    # from, not at eval's namespace -- the import is inside the helper.
    import embodimetry.render as render_mod

    monkeypatch.setattr(render_mod, "render_episode", _fake_render)

    env = MockEnv(max_steps=5, success_at_step=3)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=7,
        n_episodes=4,
        record_video=True,
        videos_dir=tmp_path / "videos",
    )

    # Every episode encoded -> frames dropped, MP4 written, SHA set.
    assert len(encoded_paths) == 4
    for ep in result.episodes:
        assert ep.frames == ()
        assert ep.video_path is not None
        assert ep.video_path.exists()
        # Filename convention: {policy}__{env}__seed{N}__ep{K:03d}.mp4
        assert ep.video_path.name == f"mock__mock_env__seed7__ep{ep.episode_index:03d}.mp4"
        assert ep.video_sha256 == f"sha-{ep.video_path.name}"

    # to_rows() with no explicit override pulls the SHA from each
    # EpisodeResult -- the new streaming-path contract.
    df = result.to_rows()
    assert df["video_sha256"].tolist() == [
        f"sha-mock__mock_env__seed7__ep{i:03d}.mp4" for i in range(4)
    ]
    # And the videos directory the caller passed in really got created.
    assert (tmp_path / "videos").is_dir()
    # Reference unused parameter to silence linters.
    _ = eval_mod


def test_run_cell_streaming_skips_errored_and_zero_frame_episodes(
    env_spec: EnvSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errored episodes contribute no MP4 (matches the legacy behaviour)."""
    from embodimetry import render as render_mod

    calls: list[Path] = []

    def _fake_render(stacked: np.ndarray, out_path: Path) -> Any:
        out_path.write_bytes(b"x")
        calls.append(out_path)
        from embodimetry.render import EncoderSettings, RenderResult

        return RenderResult(
            path=out_path,
            bytes_written=1,
            frame_count=int(stacked.shape[0]),
            encoder_settings=EncoderSettings(
                fps=10, size=256, codec="libx264", pixel_format="yuv420p", crf=23, rung_index=0
            ),
            content_sha256="sha-ok",
        )

    monkeypatch.setattr(render_mod, "render_episode", _fake_render)

    # Episode 1 crashes mid-step.
    env = MockEnv(max_steps=5, success_at_step=3, step_raises_on_episode=1)
    policy = MockPolicy()
    result = run_cell(
        policy,
        env,
        policy_name="mock",
        env_spec=env_spec,
        seed_idx=0,
        n_episodes=3,
        record_video=True,
        videos_dir=tmp_path / "videos",
    )

    assert result.episodes[1].error is not None
    assert result.episodes[1].video_path is None
    assert result.episodes[1].video_sha256 == ""

    # Episodes 0 and 2 succeeded -> got MP4s.
    assert result.episodes[0].video_sha256 == "sha-ok"
    assert result.episodes[2].video_sha256 == "sha-ok"
    assert len(calls) == 2  # the crashed episode never reached the encoder


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


def test_cell_result_to_rows_errored_column_reflects_episode_error() -> None:
    """The optional ``errored`` column is True iff the episode crashed (audit H3)."""
    good = _make_episode(0, success=False)  # legit failure, error=None
    crashed = EpisodeResult(
        episode_index=1,
        success=False,
        return_=0.0,
        n_steps=0,
        wallclock_s=0.1,
        frames=(),
        final_reward=0.0,
        error="RuntimeError: CUDA out of memory",
    )
    cell = CellResult(
        policy="mock",
        env="mock_env",
        seed=0,
        episodes=(good, crashed),
        code_sha="abc123",
        lerobot_version="0.5.1",
        timestamp_utc="2026-05-01T00:00:00+00:00",
    )
    df = cell.to_rows()
    assert df["errored"].tolist() == [False, True]


def test_cell_result_to_rows_eval_run_id_defaults_empty() -> None:
    cell = _make_cell(successes=1, total=2)
    df = cell.to_rows()
    assert (df["eval_run_id"] == "").all()


def test_cell_result_to_rows_eval_run_id_threaded() -> None:
    eps = (_make_episode(0, success=True),)
    cell = CellResult(
        policy="mock",
        env="mock_env",
        seed=0,
        episodes=eps,
        code_sha="abc123",
        lerobot_version="0.5.1",
        timestamp_utc="2026-05-01T00:00:00+00:00",
        eval_run_id="2026-05-01T00:00:00+00:00-abc12345",
    )
    df = cell.to_rows()
    assert (df["eval_run_id"] == "2026-05-01T00:00:00+00:00-abc12345").all()
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
    from embodimetry import eval as eval_mod

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


@pytest.mark.parametrize(
    ("policy_name", "expected_repo", "expected_sha"),
    [
        (
            "act_insertion",
            "lerobot/act_aloha_sim_insertion_human",
            "33259aa86eb45fdf85350280044a33d9d50e40c3",
        ),
        (
            "pi05_libero",
            "lerobot/pi05-libero",
            "10522ae373a9ce84d263b808a4ecf5af8f1944fa",
        ),
    ],
)
def test_load_policy_expansion_specs_forward_locked_sha(
    policy_name: str,
    expected_repo: str,
    expected_sha: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each new expansion policy resolves from the shipped YAML and forwards
    its pinned Hub SHA to the loader verbatim (CI-safe: the lerobot
    ``from_pretrained`` call is monkeypatched out, so no network/GPU)."""
    from pathlib import Path

    from embodimetry import eval as eval_mod
    from embodimetry.policies import PolicyRegistry

    repo_root = Path(__file__).resolve().parents[1]
    spec = PolicyRegistry.from_yaml(repo_root / "configs" / "policies.yaml").get(policy_name)
    assert spec.repo_id == expected_repo
    assert spec.revision_sha == expected_sha

    captured: dict[str, Any] = {}

    class _SentinelAdapter:
        def __call__(self, obs: dict[str, Any]) -> NDArray[np.float32]:
            return np.zeros((2,), dtype=np.float32)

        def reset(self) -> None:
            return None

    def _fake_inner(*, repo_id: str, revision: str, action_shape: Any, device: str) -> Any:
        captured.update({"repo_id": repo_id, "revision": revision})
        return _SentinelAdapter()

    monkeypatch.setattr(eval_mod, "_load_pretrained_policy", _fake_inner)
    load_policy(spec, action_shape=(2,), device="cpu")
    assert captured == {"repo_id": expected_repo, "revision": expected_sha}


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
    from embodimetry.eval import _gym_obs_to_batch

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


def test_lerobot_adapter_oversized_action_slices_to_env_shape() -> None:
    """Multi-embodiment VLAs (xvla, pi0fast) emit padded action vectors.

    The convention is that the first ``env_action_dim`` entries are
    active for the current embodiment; trailing dims are zero-padded
    slots for other embodiments. The adapter slices to the env shape.
    """
    torch = pytest.importorskip("torch")

    # Model emits 14 dims; env wants 2. Take the first 2.
    full_action = torch.tensor([[1.0, 2.0, 99.0, 99.0] + [0.0] * 10])
    model = _FakeModel(action_value=full_action)
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(2,),
        device="cpu",
    )
    out = adapter({"pixels": np.zeros((96, 96, 3), dtype=np.uint8)})
    assert tuple(out.shape) == (2,)
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_lerobot_adapter_undersized_action_raises() -> None:
    """An action smaller than the env's expected shape is a real bug
    (not the padded multi-embodiment case) and must raise.
    """
    torch = pytest.importorskip("torch")

    model = _FakeModel(action_value=torch.zeros(1, 1))
    adapter = _LerobotPolicyAdapter(
        model,
        preprocessor=_identity,
        postprocessor=_identity,
        action_shape=(7,),
        device="cpu",
    )
    with pytest.raises(RuntimeError, match=r"size 1, expected 7"):
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
        from embodimetry import eval as eval_mod

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
    from embodimetry.eval import _DebatchedVecEnvAdapter

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
        from embodimetry import eval as eval_mod

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
        from embodimetry import eval as eval_mod

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
        from embodimetry import eval as eval_mod

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

    from embodimetry.eval import _DebatchedVecEnvAdapter

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
    from embodimetry.eval import _strip_batch_dim

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

    from embodimetry.eval import _libero_obs_to_batch

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
    from embodimetry.eval import _gym_obs_to_batch

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
    from embodimetry import eval as eval_mod

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


# --------------------------------------------------------------------- #
# Legacy-normalization recovery: buffer name -> feature key mapping      #
# --------------------------------------------------------------------- #


def test_buffer_name_maps_multidot_key_against_config() -> None:
    """A two-dot camera key must survive the buffer-name round trip.

    Regression for the act x aloha 0%-success bug: the legacy
    safetensors buffer ``buffer_observation_images_top`` flattens every
    dot to an underscore. Reversing only the FIRST underscore yields
    ``observation.images_top`` -- which never matches the policy
    config's ``observation.images.top``, so lerobot's
    NormalizerProcessorStep silently skips it and feeds raw pixels to
    the model. Disambiguating against the declared feature keys fixes
    it.
    """
    known = ("observation.images.top", "observation.state", "action")
    assert _buffer_name_to_feature_key("observation_images_top", known) == "observation.images.top"
    assert _buffer_name_to_feature_key("observation_state", known) == "observation.state"
    assert _buffer_name_to_feature_key("action", known) == "action"


def test_buffer_name_legacy_fallback_for_single_dot_key() -> None:
    """With no config keys, single-dot keys still recover (diffusion_pusht)."""
    assert _buffer_name_to_feature_key("observation_image", ()) == "observation.image"
    assert _buffer_name_to_feature_key("observation_state", ()) == "observation.state"
    assert _buffer_name_to_feature_key("action", ()) == "action"


def test_buffer_name_falls_back_when_no_known_key_matches() -> None:
    """An unmatched buffer name degrades to the legacy single-underscore reversal."""
    known = ("observation.state", "action")
    # No known key flattens to 'observation_images_top'; fall back.
    assert _buffer_name_to_feature_key("observation_images_top", known) == "observation.images_top"


# --------------------------------------------------------------------- #
# XVLA processor patching: input-side ImageNet + output-side rotation    #
#                                                                       #
# Regression for the v1 xvla_libero 0/875-success cell. The Hub's       #
# `lerobot/xvla-libero` ships a `policy_postprocessor.json` that omits  #
# `XVLARotation6DToAxisAngleProcessorStep` AND a `policy_preprocessor.  #
# json` whose `normalizer_processor` step has `VISUAL: IDENTITY` -- so  #
# images reach the Florence-2 backbone as raw [0, 1] floats instead of  #
# the ImageNet-normalized inputs it was pretrained on, and the policy's #
# 20-dim `[eef(3), rot6d(6), gripper(1), padding(10)]` action lands in  #
# the LIBERO env with the rot6d components in place of axis-angle. The  #
# loader patches both pipelines to insert the missing steps.            #
# --------------------------------------------------------------------- #


class _StubCfg:
    """Minimal config stand-in: only ``type`` is read by the patcher."""

    def __init__(self, policy_type: str) -> None:
        self.type = policy_type


def _build_hub_shaped_xvla_postprocessor():
    """Reconstruct the postprocessor pipeline the Hub JSON loads for xvla-libero.

    Matches the actual ``policy_postprocessor.json`` on
    ``lerobot/xvla-libero``@``12e8783...``: ``[UnnormalizerProcessorStep
    (ACTION=20, IDENTITY), DeviceProcessorStep(device='cpu')]``. Built
    against the live lerobot 0.5.1 classes so the structural assertion
    catches any upstream signature drift.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        DeviceProcessorStep,
        PolicyProcessorPipeline,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    unnorm = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        norm_map={
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
    )
    return PolicyProcessorPipeline(
        steps=[unnorm, DeviceProcessorStep(device="cpu")],
        name="policy_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def test_patch_postprocessor_inserts_rotation_step_for_xvla() -> None:
    """xvla cfg + Hub-shaped postprocessor -> rotation step inserted.

    The XVLA Hub repo ships a postprocessor of
    ``[Unnormalizer, DeviceProcessor]`` only. The patcher must inject
    :class:`XVLARotation6DToAxisAngleProcessorStep` before the trailing
    device hop so the LIBERO env receives ``[eef(3), axis_angle(3),
    gripper(1)] = 7`` rather than the raw 20-dim padded action.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep
    from lerobot.processor import DeviceProcessorStep, UnnormalizerProcessorStep

    pipeline = _build_hub_shaped_xvla_postprocessor()

    patched = _patch_xvla_postprocessor(pipeline)

    step_types = [type(s) for s in patched.steps]
    assert step_types == [
        UnnormalizerProcessorStep,
        XVLARotation6DToAxisAngleProcessorStep,
        DeviceProcessorStep,
    ], f"unexpected pipeline shape: {step_types}"


def test_patch_postprocessor_is_idempotent_for_xvla() -> None:
    """Calling the patcher twice must not stack two rotation steps."""
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep

    once = _patch_xvla_postprocessor(_build_hub_shaped_xvla_postprocessor())
    twice = _patch_xvla_postprocessor(once)

    rotation_steps = [
        s for s in twice.steps if isinstance(s, XVLARotation6DToAxisAngleProcessorStep)
    ]
    assert len(rotation_steps) == 1


def test_patch_processors_is_noop_for_non_xvla_policies() -> None:
    """Diffusion/ACT/SmolVLA processors must pass through unchanged."""
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")

    pre = _build_hub_shaped_xvla_preprocessor()
    post = _build_hub_shaped_xvla_postprocessor()
    for ptype in ("diffusion", "act", "smolvla", "pi0", "pi0_fast"):
        cfg = _StubCfg(policy_type=ptype)
        out_pre, out_post = _patch_processors_for_policy(cfg, pre, post)
        assert out_pre is pre, f"{ptype} preprocessor was unexpectedly rebuilt"
        assert out_post is post, f"{ptype} postprocessor was unexpectedly rebuilt"


def test_patched_xvla_postprocessor_emits_seven_dim_action() -> None:
    """End-to-end shape smoke: 20-dim model action -> 7-dim env action.

    Confirms the patched pipeline produces what LIBERO's 7-dim
    ``[eef(3), axis_angle(3), gripper(1)]`` action space expects, with
    the gripper coerced to {-1, +1} by the rotation step's trailing
    binarization. Uses a deterministic identity-rotation rot6d block
    ``[1,0,0, 0,1,0]`` so the axis-angle output is the zero rotation.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot")

    pipeline = _patch_xvla_postprocessor(_build_hub_shaped_xvla_postprocessor())

    # Build a batched 20-dim action: eef=[0.1,0.2,0.3], rot6d=identity-cols,
    # gripper=0.8 (>0.5 -> +1), 10 trailing padding zeros.
    action = torch.zeros(1, 20, dtype=torch.float32)
    action[0, :3] = torch.tensor([0.1, 0.2, 0.3])
    action[0, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # identity rotation in 6D
    action[0, 9] = 0.8

    out = pipeline(action)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    out = np.asarray(out)

    # Pipeline returns shape (B, 7) for batched input.
    assert out.shape[-1] == 7, f"expected 7-dim env action, got {out.shape}"
    flat = out.reshape(-1)
    np.testing.assert_allclose(flat[:3], [0.1, 0.2, 0.3], atol=1e-5)
    # Identity rotation -> axis-angle zero vector.
    np.testing.assert_allclose(flat[3:6], [0.0, 0.0, 0.0], atol=1e-5)
    # Gripper binarized to +1.
    assert float(flat[6]) == 1.0


# --------------------------------------------------------------------- #
# XVLA preprocessor patching: ImageNet normalization                     #
# --------------------------------------------------------------------- #


def _build_hub_shaped_xvla_preprocessor():
    """Reconstruct the preprocessor pipeline the Hub JSON loads for xvla-libero.

    Matches the actual ``policy_preprocessor.json`` on
    ``lerobot/xvla-libero``@``12e8783...``:
    ``[RenameObservationsProcessorStep, AddBatchDimensionProcessorStep,
    TokenizerProcessorStep, XVLAAddDomainIdProcessorStep,
    DeviceProcessorStep, NormalizerProcessorStep]``. Built against the
    live lerobot 0.5.1 classes so any upstream signature drift surfaces
    as a test failure.

    Notably absent: :class:`XVLAImageNetNormalizeProcessorStep`. The
    Hub JSON's :class:`NormalizerProcessorStep` has VISUAL=IDENTITY in
    its norm_map, so images pass through raw [0, 1] floats and the
    Florence-2 visual backbone sees the wrong input distribution
    (pretrained against ImageNet-normalized inputs).
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    # TokenizerProcessorStep wraps a HuggingFace tokenizer; instantiating it
    # requires transformers, which lives in lerobot's optional
    # transformers-dep extra and is not installed in the fast CI [dev] env.
    pytest.importorskip("transformers")
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.xvla.processor_xvla import XVLAAddDomainIdProcessorStep
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
    )

    rename = RenameObservationsProcessorStep(rename_map={})
    to_batch = AddBatchDimensionProcessorStep()
    tokenizer = TokenizerProcessorStep(
        tokenizer_name="facebook/bart-large",
        max_length=50,
        padding="max_length",
        padding_side="right",
    )
    add_domain = XVLAAddDomainIdProcessorStep(domain_id=3)
    device = DeviceProcessorStep(device="cpu")
    normalizer = NormalizerProcessorStep(
        features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(20,)),
        },
        norm_map={
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
    )
    return PolicyProcessorPipeline(
        steps=[rename, to_batch, tokenizer, add_domain, device, normalizer],
        name="policy_preprocessor",
    )


def test_patch_preprocessor_inserts_imagenet_step_for_xvla() -> None:
    """xvla cfg + Hub-shaped preprocessor -> ImageNet step inserted before device hop.

    The XVLA Hub repo's preprocessor lacks ImageNet normalization; the
    patcher inserts :class:`XVLAImageNetNormalizeProcessorStep` between
    :class:`XVLAAddDomainIdProcessorStep` and :class:`DeviceProcessorStep`
    so the visual backbone sees the same input distribution it was
    pretrained on.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.xvla.processor_xvla import (
        XVLAAddDomainIdProcessorStep,
        XVLAImageNetNormalizeProcessorStep,
    )
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
        RenameObservationsProcessorStep,
        TokenizerProcessorStep,
    )

    pipeline = _build_hub_shaped_xvla_preprocessor()
    patched = _patch_xvla_preprocessor(pipeline)

    step_types = [type(s) for s in patched.steps]
    assert step_types == [
        RenameObservationsProcessorStep,
        AddBatchDimensionProcessorStep,
        TokenizerProcessorStep,
        XVLAAddDomainIdProcessorStep,
        XVLAImageNetNormalizeProcessorStep,
        DeviceProcessorStep,
        NormalizerProcessorStep,
    ], f"unexpected pipeline shape: {step_types}"


def test_patch_preprocessor_is_idempotent_for_xvla() -> None:
    """Calling the patcher twice must not stack two ImageNet steps."""
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.policies.xvla.processor_xvla import XVLAImageNetNormalizeProcessorStep

    once = _patch_xvla_preprocessor(_build_hub_shaped_xvla_preprocessor())
    twice = _patch_xvla_preprocessor(once)

    normalize_steps = [s for s in twice.steps if isinstance(s, XVLAImageNetNormalizeProcessorStep)]
    assert len(normalize_steps) == 1


def test_patch_processors_loads_real_xvla_libero_pipeline_contains_both_steps() -> None:
    """Loader-shape smoke against the live ``lerobot/xvla-libero`` Hub repo.

    Pulls the actual pinned Hub preprocessor + postprocessor through
    :func:`lerobot.policies.factory.make_pre_post_processors`, runs the
    patcher, and asserts both XVLA-specific steps land in the right
    pipelines. This is the closest unit-test analogue of the user's GPU
    sanity run -- it would have caught both the v1 0/875 cell (missing
    rotation step) and the residual 0/10 sanity (missing ImageNet
    normalization) without a rollout.

    Marked ``@pytest.mark.sim`` because it touches the Hub cache, the
    lerobot processor factory, and a tokenizer download. CI fast skips
    it; the live sanity gate runs it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import factory as _factory
    from lerobot.policies.xvla.processor_xvla import (
        XVLAImageNetNormalizeProcessorStep,
        XVLARotation6DToAxisAngleProcessorStep,
    )

    from embodimetry.policies import PolicyRegistry

    registry = PolicyRegistry.from_yaml(Path("configs/policies.yaml"))
    spec = registry.get("xvla_libero")
    assert spec.repo_id is not None
    assert spec.revision_sha is not None

    cfg = PreTrainedConfig.from_pretrained(spec.repo_id, revision=spec.revision_sha)
    pre, post = _factory.make_pre_post_processors(
        cfg, pretrained_path=spec.repo_id, revision=spec.revision_sha
    )
    pre, post = _patch_processors_for_policy(cfg, pre, post)

    assert any(isinstance(s, XVLAImageNetNormalizeProcessorStep) for s in pre.steps), (
        "preprocessor is missing XVLAImageNetNormalizeProcessorStep after patch -- "
        "Florence-2 backbone would see raw [0, 1] images instead of ImageNet-normalized"
    )
    assert any(isinstance(s, XVLARotation6DToAxisAngleProcessorStep) for s in post.steps), (
        "postprocessor is missing XVLARotation6DToAxisAngleProcessorStep after patch -- "
        "LIBERO env would receive raw rot6d components in place of axis-angle"
    )


test_patch_processors_loads_real_xvla_libero_pipeline_contains_both_steps = pytest.mark.sim(
    test_patch_processors_loads_real_xvla_libero_pipeline_contains_both_steps
)
