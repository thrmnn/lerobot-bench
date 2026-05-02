"""Eval orchestration core: ``(policy, env, seed_idx, n_episodes) -> CellResult``.

This module is the keystone for the sweep. Everything in ``scripts/``
(``calibrate``, ``run_one``, ``run_sweep``) is a thin shell over
:func:`run_cell`. The seeding contract from ``docs/DESIGN.md``
§ Methodology is enforced here and nowhere else.

**Seeding contract (do not deviate).** Per cell ``(policy, env, seed_idx)``:

* At cell start (ONCE for the whole cell, before any episode runs)::

      numpy.random.seed(seed_idx * 1000)
      torch.manual_seed(seed_idx * 1000)
      torch.cuda.manual_seed_all(seed_idx * 1000)

* Per episode ``e in {0..n_episodes-1}``::

      env.reset(seed=seed_idx * 1000 + e)

  Policy stochasticity inherits the torch generator (NOT re-seeded
  per episode). This is what makes mid-cell resume non-bit-reproducible
  — the torch generator advances across episodes, so resuming at
  episode ``k`` would not produce the same ``k..n_episodes-1`` tail as
  a fresh run. ``checkpointing.py`` therefore resumes only at cell
  boundaries; if a process dies mid-cell, that cell restarts from
  episode 0.

**Termination & success.** Each episode runs until any of:
``terminated``, ``truncated``, or ``step_count == env_spec.max_steps``.
For v1 we use the conservative rule ``success = (final_reward >=
env_spec.success_threshold)`` — the reward at the final step (the last
step before exit). This matches Aloha/Libero task-complete semantics
and is a defensible (if slightly under-counting) PushT score; PushT's
reward is monotonically related to coverage so the final step usually
carries the peak. The choice is documented in the public methodology.

**Lazy imports.** ``torch`` and ``lerobot`` are imported lazily inside
:func:`seed_everything` and :func:`load_policy` respectively; importing
this module must not require either.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.envs import EnvSpec
from lerobot_bench.policies import PolicySpec

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Protocols                                                             #
# --------------------------------------------------------------------- #


class PolicyCallable(Protocol):
    """Anything that maps an obs dict to an action ndarray.

    ``reset`` is called once at the start of each episode (after
    ``env.reset``, before the first ``policy(obs)`` call). Stateless
    policies can implement it as a no-op.
    """

    def __call__(self, obs: dict[str, Any]) -> NDArray[np.floating[Any]]: ...

    def reset(self) -> None: ...


class GymLikeEnv(Protocol):
    """gymnasium-style env protocol -- the slice of the API we use."""

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def render(self) -> NDArray[np.uint8]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------- #
# Result types                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class EpisodeResult:
    """One episode within a cell.

    ``frames`` is a tuple (immutable / hashable) of ``(H, W, 3)`` uint8
    arrays collected via ``env.render()``. Empty tuple when
    ``record_video=False``. Order: the first frame is from the post-reset
    ``render()`` call; subsequent frames follow each ``step()``.

    ``error`` is ``None`` for successful (or cleanly-failed) episodes
    and a short stringified exception for crashes. When ``error`` is
    set, ``success=False``, ``return_=0.0``, ``n_steps=0``,
    ``final_reward=0.0`` — the cell continues, the row is preserved.
    """

    episode_index: int
    success: bool
    return_: float
    n_steps: int
    wallclock_s: float
    frames: tuple[NDArray[np.uint8], ...]
    final_reward: float
    error: str | None = None


@dataclass(frozen=True)
class CellResult:
    """Output of :func:`run_cell` -- ``n_episodes`` EpisodeResults plus metadata."""

    policy: str
    env: str
    seed: int
    episodes: tuple[EpisodeResult, ...]
    code_sha: str
    lerobot_version: str
    timestamp_utc: str

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def to_rows(self, *, video_sha256_per_episode: Sequence[str] | None = None) -> pd.DataFrame:
        """Convert to a DataFrame matching :data:`RESULT_SCHEMA`.

        ``video_sha256_per_episode`` is parallel to ``self.episodes``;
        if ``None``, the column is filled with empty strings (the same
        default that :mod:`render` uses when video rendering is off).
        """
        n = len(self.episodes)
        if video_sha256_per_episode is None:
            video_sha = [""] * n
        else:
            if len(video_sha256_per_episode) != n:
                raise ValueError(
                    f"video_sha256_per_episode has length {len(video_sha256_per_episode)}, "
                    f"expected {n} (one per episode)"
                )
            video_sha = list(video_sha256_per_episode)

        rows = [
            {
                "policy": self.policy,
                "env": self.env,
                "seed": self.seed,
                "episode_index": ep.episode_index,
                "success": ep.success,
                "return_": ep.return_,
                "n_steps": ep.n_steps,
                "wallclock_s": ep.wallclock_s,
                "video_sha256": video_sha[i],
                "code_sha": self.code_sha,
                "lerobot_version": self.lerobot_version,
                "timestamp_utc": self.timestamp_utc,
            }
            for i, ep in enumerate(self.episodes)
        ]
        return pd.DataFrame(rows, columns=list(RESULT_SCHEMA))


# --------------------------------------------------------------------- #
# Seeding helper                                                        #
# --------------------------------------------------------------------- #


def seed_everything(seed_idx: int) -> int:
    """Apply the per-cell seeding contract. Returns the base seed.

    Seeds numpy's global RNG immediately. Lazy-imports torch and seeds
    its CPU + CUDA generators if importable. Logs a warning (does not
    raise) if torch is unavailable -- that's a "no GPU work happens"
    condition, not necessarily fatal for tests using mocks.
    """
    base_seed = seed_idx * 1000
    np.random.seed(base_seed)

    try:
        import torch
    except ImportError:
        logger.warning(
            "torch not importable; only numpy was seeded. Policy stochasticity "
            "from torch will not be reproducible."
        )
        return base_seed

    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    return base_seed


# --------------------------------------------------------------------- #
# Baseline policies                                                     #
# --------------------------------------------------------------------- #


class _NoOpPolicy:
    """Always returns zeros of the env's action shape."""

    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self._action: NDArray[np.float32] = np.zeros(action_shape, dtype=np.float32)

    def __call__(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        # Return a copy so callers can mutate it without poisoning the source.
        return self._action.copy()

    def reset(self) -> None:
        return None


class _RandomPolicy:
    """Uniform random in [-1, 1].

    Uses the global ``numpy`` RNG (``np.random.uniform``) so the
    seeding contract in :func:`seed_everything` (which calls
    ``np.random.seed``) makes this baseline reproducible. Do NOT switch
    to ``np.random.default_rng`` -- that uses an independent generator
    that ``np.random.seed`` does not affect.
    """

    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self._action_shape = action_shape

    def __call__(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        return np.random.uniform(-1.0, 1.0, size=self._action_shape).astype(np.float32)

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------- #
# Loaders                                                               #
# --------------------------------------------------------------------- #


def load_policy(
    spec: PolicySpec,
    *,
    action_shape: tuple[int, ...] | None = None,
    device: str = "cuda",
) -> PolicyCallable:
    """Resolve a :class:`PolicySpec` to a callable policy.

    For baselines (``no_op``, ``random``) ``action_shape`` is required
    -- the action dim comes from the env, not the policy spec. For
    pretrained policies ``action_shape`` is ignored.

    Pretrained loading is a Day 0b TODO: it raises
    :class:`NotImplementedError` until ``revision_sha`` values land in
    ``configs/policies.yaml`` and the lerobot factory call is wired up.
    Non-runnable specs (pre-Day-0a entries) raise :class:`RuntimeError`
    with a Day 0a hint, regardless of whether pretrained loading is
    implemented yet.
    """
    if not spec.is_runnable():
        raise RuntimeError(
            f"policy '{spec.name}' is not runnable -- missing revision_sha. "
            "Lock SHAs at Day 0a per docs/NEXT_STEPS.md, then update "
            "configs/policies.yaml."
        )

    if spec.is_baseline:
        if action_shape is None:
            raise ValueError(
                f"baseline policy '{spec.name}' requires action_shape "
                "(comes from the env, not the policy spec)"
            )
        if spec.name == "no_op":
            return _NoOpPolicy(action_shape)
        if spec.name == "random":
            return _RandomPolicy(action_shape)
        raise ValueError(f"unknown baseline policy '{spec.name}'")

    # Pretrained loading: Day 0b TODO. The plan:
    #   from lerobot.common.policies.factory import make_policy
    #   model = make_policy(spec.repo_id, revision=spec.revision_sha,
    #                       fp_precision=spec.fp_precision).to(device).eval()
    #   return _LerobotPolicyAdapter(model)
    raise NotImplementedError(
        f"pretrained policy loading not wired up yet (policy '{spec.name}'); "
        "Day 0b TODO -- requires lerobot install + revision_sha lock."
    )


def load_env(spec: EnvSpec) -> GymLikeEnv:
    """Instantiate a gym env from the spec.

    Lazy-imports gymnasium. If the sim extra isn't installed this
    raises :class:`ImportError` at runtime -- intentional: Day 0b is a
    one-line install away from working without scaffolding more
    indirection here.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is not installed. Install sim extras: "
            "`pip install -e '.[sim]'` (and ensure gym-pusht / gym-aloha "
            "are pulled in for the env you are loading)."
        ) from exc

    env = gym.make(spec.gym_id, max_episode_steps=spec.max_steps)
    # gym.make returns a generic Env; we know our protocol is a slice
    # of the gymnasium API so this cast is sound at runtime.
    return cast(GymLikeEnv, env)


# --------------------------------------------------------------------- #
# Code/version detection                                                #
# --------------------------------------------------------------------- #


def _detect_code_sha() -> str:
    """Best-effort ``git rev-parse HEAD`` of the repo root. ``""`` on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _detect_lerobot_version() -> str:
    """Return ``lerobot.__version__`` or ``"unknown"`` if not importable."""
    try:
        import lerobot
    except ImportError:
        return "unknown"
    version = getattr(lerobot, "__version__", None)
    return str(version) if version is not None else "unknown"


# --------------------------------------------------------------------- #
# The main entry point                                                  #
# --------------------------------------------------------------------- #


def run_cell(
    policy: PolicyCallable,
    env: GymLikeEnv,
    *,
    policy_name: str,
    env_spec: EnvSpec,
    seed_idx: int,
    n_episodes: int,
    record_video: bool = True,
    code_sha: str | None = None,
    lerobot_version: str | None = None,
) -> CellResult:
    """Run ``n_episodes`` for one ``(policy, env, seed_idx)`` cell.

    The seeding contract is applied ONCE at cell start (see module
    docstring). For each episode:

    1. ``env.reset(seed=seed_idx*1000 + e)``.
    2. ``policy.reset()``.
    3. Optional first ``env.render()`` if ``record_video``.
    4. Loop: ``policy(obs) -> env.step -> [render] -> check termination``.
    5. Episode ends on ``terminated`` / ``truncated`` / step cap.
    6. Success = ``final_reward >= env_spec.success_threshold``.

    Per-episode exceptions are caught and recorded in
    ``EpisodeResult.error``; the cell continues. A whole-cell exception
    (e.g. ``env.reset`` itself crashes on the first attempt) propagates.

    ``code_sha`` and ``lerobot_version`` default to autodetection
    (``git rev-parse HEAD`` and ``lerobot.__version__``); pass them
    explicitly when the orchestrator already has them in hand.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if seed_idx < 0:
        raise ValueError(f"seed_idx must be non-negative, got {seed_idx}")

    base_seed = seed_everything(seed_idx)
    if code_sha is None:
        code_sha = _detect_code_sha()
    if lerobot_version is None:
        lerobot_version = _detect_lerobot_version()
    timestamp_utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    episodes: list[EpisodeResult] = []
    for e in range(n_episodes):
        episode_seed = base_seed + e
        episode = _run_one_episode(
            policy=policy,
            env=env,
            episode_index=e,
            episode_seed=episode_seed,
            max_steps=env_spec.max_steps,
            success_threshold=env_spec.success_threshold,
            record_video=record_video,
        )
        episodes.append(episode)

    return CellResult(
        policy=policy_name,
        env=env_spec.name,
        seed=seed_idx,
        episodes=tuple(episodes),
        code_sha=code_sha,
        lerobot_version=lerobot_version,
        timestamp_utc=timestamp_utc,
    )


def _run_one_episode(
    *,
    policy: PolicyCallable,
    env: GymLikeEnv,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    success_threshold: float,
    record_video: bool,
) -> EpisodeResult:
    """Inner loop. Catches per-episode exceptions and records them."""
    t0 = time.perf_counter()
    frames: list[NDArray[np.uint8]] = []
    cumulative_return = 0.0
    n_steps = 0
    final_reward = 0.0

    try:
        obs, _info = env.reset(seed=episode_seed)
        policy.reset()
        if record_video:
            frames.append(env.render())

        for _ in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            n_steps += 1
            cumulative_return += float(reward)
            final_reward = float(reward)
            if record_video:
                frames.append(env.render())
            if terminated or truncated:
                break
    except Exception as exc:
        wallclock = time.perf_counter() - t0
        logger.exception("episode %d crashed (seed=%d): %s", episode_index, episode_seed, exc)
        return EpisodeResult(
            episode_index=episode_index,
            success=False,
            return_=0.0,
            n_steps=0,
            wallclock_s=wallclock,
            frames=(),
            final_reward=0.0,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )

    wallclock = time.perf_counter() - t0
    success = final_reward >= success_threshold
    return EpisodeResult(
        episode_index=episode_index,
        success=success,
        return_=cumulative_return,
        n_steps=n_steps,
        wallclock_s=wallclock,
        frames=tuple(frames),
        final_reward=final_reward,
        error=None,
    )


def run_cell_from_specs(
    policy_spec: PolicySpec,
    env_spec: EnvSpec,
    *,
    seed_idx: int,
    n_episodes: int,
    device: str = "cuda",
    record_video: bool = True,
) -> CellResult:
    """Convenience: load policy + env from specs, then :func:`run_cell`.

    The env is created via :func:`load_env` (lazy gymnasium import) and
    the action shape is sniffed from ``env.action_space.shape`` for
    baseline policy construction. The caller is responsible for
    ``env.close()`` -- this function does not own the env's lifecycle
    once it returns.
    """
    env = load_env(env_spec)
    # gymnasium envs expose .action_space; we read its shape for baselines.
    # mypy can't see the gymnasium API through our protocol, so this is Any.
    action_space: Any = getattr(env, "action_space", None)
    action_shape: tuple[int, ...] | None
    if action_space is not None and hasattr(action_space, "shape"):
        action_shape = tuple(action_space.shape)
    else:
        action_shape = None

    policy = load_policy(policy_spec, action_shape=action_shape, device=device)
    return run_cell(
        policy,
        env,
        policy_name=policy_spec.name,
        env_spec=env_spec,
        seed_idx=seed_idx,
        n_episodes=n_episodes,
        record_video=record_video,
    )
