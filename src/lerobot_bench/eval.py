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
# Pretrained adapter                                                    #
# --------------------------------------------------------------------- #


# Gym observation dict keys we know how to translate to lerobot policy
# input keys. Keep this table tight -- if a new env shows up with a
# different shape, fail loudly in :func:`_gym_obs_to_batch` rather than
# guessing.
#
# Mapping rules (matches lerobot's internal feature naming convention):
#
# * ``pixels`` (HWC uint8) -> ``observation.image`` (CHW float in [0,1])
# * ``pixels.<view>`` (HWC uint8) -> ``observation.images.<view>``
#   (CHW float in [0,1])  -- e.g. Aloha's ``pixels.top`` becomes
#   ``observation.images.top`` for the ACT policy.
# * ``agent_pos`` (1D float) -> ``observation.state`` (1D float)
# * ``environment_state`` (1D float) -> ``observation.environment_state``
#
# A bare ``numpy.ndarray`` obs (the default PushT 5-vector with the
# implicit ``obs_type=state``) is NOT supported here -- pretrained
# PushT policies expect images. The configs/envs.yaml ships
# ``obs_type=pixels_agent_pos`` for this reason.


class _LerobotPolicyAdapter:
    """Wraps a lerobot ``PreTrainedPolicy`` to match :class:`PolicyCallable`.

    Translates the gym observation dict into the tensor batch the lerobot
    model expects, runs ``select_action`` under ``torch.no_grad`` (the
    decorator is already on most policies' ``select_action`` but we add
    a belt-and-braces ``no_grad`` context here so a future policy that
    drops it does not silently leak gradients across episodes), and
    casts the post-processed action back to ``numpy.float32`` of shape
    ``action_shape``.

    The pre/post processor pipelines from
    :func:`lerobot.policies.factory.make_pre_post_processors` are stored
    as ``Any`` because the ``PolicyProcessorPipeline`` type signature is
    parametric on the in/out types of each pipeline direction; we don't
    want to leak that into the adapter's surface.
    """

    def __init__(
        self,
        model: Any,
        *,
        preprocessor: Any,
        postprocessor: Any,
        action_shape: tuple[int, ...] | None,
        device: str,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._action_shape = action_shape
        self._device = device

    def reset(self) -> None:
        # Most PreTrainedPolicy subclasses have a reset() that clears the
        # internal action queue (diffusion/ACT both buffer a horizon of
        # actions per inference call). Required at episode boundaries to
        # avoid bleeding actions from the previous episode.
        if hasattr(self._model, "reset"):
            self._model.reset()

    def __call__(self, obs: dict[str, Any] | NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        import torch

        batch = _gym_obs_to_batch(obs)

        with torch.no_grad():
            batch = self._preprocessor(batch)
            action = self._model.select_action(batch)
            action = self._postprocessor(action)

        # action is a torch.Tensor; move to CPU and reshape to the env action.
        if hasattr(action, "detach"):
            action_np = action.detach().to("cpu").numpy()
        else:
            action_np = np.asarray(action)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
        if self._action_shape is not None:
            expected = int(np.prod(self._action_shape))
            if action_np.size != expected:
                raise RuntimeError(
                    f"policy emitted action of size {action_np.size}, "
                    f"expected {expected} for action_shape {self._action_shape}"
                )
            action_np = action_np.reshape(self._action_shape)
        return action_np


def _gym_obs_to_batch(obs: dict[str, Any] | NDArray[np.floating[Any]]) -> dict[str, Any]:
    """Translate a gym observation into a lerobot-style batch dict.

    See the comment block above :class:`_LerobotPolicyAdapter` for the
    full mapping table. Lazy-imports ``torch`` so this module can be
    imported in a torch-free environment.
    """
    import torch

    if not isinstance(obs, dict):
        raise RuntimeError(
            f"pretrained policies require a dict observation (got {type(obs).__name__}); "
            "ensure configs/envs.yaml sets gym_kwargs.obs_type to a pixels-based mode "
            "(e.g. 'pixels_agent_pos')"
        )

    batch: dict[str, Any] = {}
    for key, value in obs.items():
        # pixels[.view] -> observation.image[s.view] (HWC uint8 -> CHW float [0,1]).
        if key == "pixels":
            tensor = torch.from_numpy(np.asarray(value)).float() / 255.0
            batch["observation.image"] = tensor.permute(2, 0, 1)
        elif key.startswith("pixels."):
            view = key.split(".", 1)[1]
            tensor = torch.from_numpy(np.asarray(value)).float() / 255.0
            batch[f"observation.images.{view}"] = tensor.permute(2, 0, 1)
        elif key == "agent_pos":
            batch["observation.state"] = torch.from_numpy(np.asarray(value)).float()
        elif key == "environment_state":
            batch["observation.environment_state"] = torch.from_numpy(np.asarray(value)).float()
        else:
            raise RuntimeError(
                f"unknown gym observation key '{key}'; "
                "_gym_obs_to_batch supports {pixels, pixels.<view>, "
                "agent_pos, environment_state}"
            )
    return batch


def _recover_dataset_stats_from_safetensors(
    repo_id: str, revision: str
) -> dict[str, dict[str, NDArray[np.float32]]]:
    """Reconstruct ``dataset_stats`` from legacy safetensors normalize buffers.

    Pre-0.5.x lerobot checkpoints (e.g. ``lerobot/diffusion_pusht``)
    pre-date the processor-pipeline split: their normalization stats
    live as buffers inside ``model.safetensors`` rather than as
    ``policy_preprocessor.json`` on the Hub. lerobot 0.5.1 silently
    drops these buffers when loading the model (only a WARNING is
    emitted) so the policy's outputs would be in normalized action
    space, not pixel space — useless on the env.

    This helper reads the safetensors file, picks out the relevant
    ``normalize_inputs.buffer_*`` and ``normalize_targets.buffer_*``
    entries, and reshapes them into the dict-of-dicts format
    :func:`make_pre_post_processors` accepts as ``dataset_stats``. If
    the safetensors has no normalization buffers (newer checkpoint that
    ships proper processors), returns an empty dict — caller should
    fall back to loading the processors directly.

    Lazy-imports ``huggingface_hub`` and ``safetensors``; both are
    transitive deps of ``lerobot==0.5.1`` so they are always available
    when this function is called from the pretrained branch.

    The original buffer-name format converts the dot in feature keys
    to an underscore (``observation.image`` becomes
    ``buffer_observation_image``); we reverse the first underscore back
    to a dot to recover the canonical key.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sf_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
    state = load_file(sf_path)

    stats: dict[str, dict[str, NDArray[np.float32]]] = {}
    prefixes = ("normalize_inputs.buffer_", "normalize_targets.buffer_")
    for tensor_key, tensor in state.items():
        for prefix in prefixes:
            if not tensor_key.startswith(prefix):
                continue
            rest = tensor_key[len(prefix) :]
            # rest looks like 'observation_image.mean' or 'action.max'.
            feature_buf, _, stat_name = rest.rpartition(".")
            if not feature_buf or not stat_name:
                continue
            feature_key = feature_buf.replace("_", ".", 1)
            stats.setdefault(feature_key, {})[stat_name] = tensor.cpu().numpy().astype(np.float32)
            break
    return stats


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
    pretrained policies ``action_shape`` is also recommended (used as a
    final shape sanity check on the post-processed action) but optional.

    The pretrained branch lazy-imports ``torch`` and ``lerobot``; this
    function must remain importable in a torch-free CI job. Non-runnable
    specs (pre-Day-0a entries) raise :class:`RuntimeError` with a Day 0a
    hint before any heavy import is attempted.

    The factory call resolves to lerobot 0.5.1's
    ``lerobot.policies.pretrained.PreTrainedPolicy.from_pretrained``
    via :func:`lerobot.policies.factory.get_policy_class`. The
    historical one-shot ``make_policy(repo_id, ...)`` API in pre-0.5
    lerobot is GONE — 0.5.1's :func:`make_policy` takes a
    :class:`PreTrainedConfig` plus ``ds_meta``/``env_cfg`` for shape
    inference, neither of which we have at eval time. Normalization
    stats are recovered from the legacy safetensors buffers when the
    Hub repo predates the processor-pipeline split (the case for
    ``lerobot/diffusion_pusht``).
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

    # Pretrained branch (Day 0b). Lazy-import everything heavy so this
    # module stays importable in CI without torch/lerobot.
    assert spec.repo_id is not None  # is_runnable() guarantees both
    assert spec.revision_sha is not None
    return _load_pretrained_policy(
        repo_id=spec.repo_id,
        revision=spec.revision_sha,
        action_shape=action_shape,
        device=device,
    )


def _load_pretrained_policy(
    *,
    repo_id: str,
    revision: str,
    action_shape: tuple[int, ...] | None,
    device: str,
) -> _LerobotPolicyAdapter:
    """Inner helper: lazy-imports lerobot and instantiates the adapter.

    Split out of :func:`load_policy` so the lerobot imports are confined
    to a function that is only entered on the pretrained branch.
    """
    # Side-effect import: triggers registration of every PreTrainedConfig
    # subclass (act, diffusion, pi0, ...) in draccus's choice registry.
    # Without this, `PreTrainedConfig.from_pretrained` on an `act`
    # checkpoint raises `DecodingError: Couldn't find a choice class for
    # 'act'`. Importing the factory module is cheap; it does not load
    # any model weights.
    import lerobot.policies.factory as _lerobot_factory
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
    policy_cls = _lerobot_factory.get_policy_class(cfg.type)

    # Normalization stats: try the new processor-pipeline format on the
    # Hub first (newer checkpoints); on FileNotFoundError fall back to
    # recovering stats from legacy safetensors buffers.
    try:
        preprocessor, postprocessor = _lerobot_factory.make_pre_post_processors(
            cfg, pretrained_path=repo_id, revision=revision
        )
    except FileNotFoundError:
        logger.info(
            "no policy_preprocessor.json on '%s'@%s; recovering "
            "normalization stats from legacy safetensors buffers",
            repo_id,
            revision,
        )
        dataset_stats = _recover_dataset_stats_from_safetensors(repo_id, revision)
        if not dataset_stats:
            raise RuntimeError(
                f"could not recover normalization stats for '{repo_id}'@{revision}: "
                "no policy_preprocessor.json on the Hub AND no normalize_inputs/targets "
                "buffers in model.safetensors. Pretrained policy will not work without "
                "valid normalization."
            ) from None
        preprocessor, postprocessor = _lerobot_factory.make_pre_post_processors(
            cfg, dataset_stats=dataset_stats
        )

    model = policy_cls.from_pretrained(repo_id, revision=revision, config=cfg)
    model = model.to(device).eval()

    return _LerobotPolicyAdapter(
        model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_shape=action_shape,
        device=device,
    )


def load_env(spec: EnvSpec) -> GymLikeEnv:
    """Instantiate a gym env from the spec.

    Lazy-imports gymnasium. If the sim extra isn't installed this
    raises :class:`ImportError` at runtime -- intentional: Day 0b is a
    one-line install away from working without scaffolding more
    indirection here.

    Also import-resolves the gym_id namespace (e.g. ``gym_pusht`` from
    ``gym_pusht/PushT-v0``) so the env's registration side-effect fires
    before ``gym.make``. Without this, freshly-installed gym-pusht /
    gym-aloha packages register lazily and ``gym.make`` raises
    :class:`gymnasium.error.NamespaceNotFound`.

    ``spec.gym_kwargs`` (a tuple of ``(key, value)`` pairs from the
    YAML registry) is materialized into a dict and forwarded verbatim;
    ``obs_type='pixels_agent_pos'`` is the most common one (required by
    every pretrained policy).
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is not installed. Install sim extras: "
            "`pip install -e '.[sim]'` (and ensure gym-pusht / gym-aloha "
            "are pulled in for the env you are loading)."
        ) from exc

    # Trigger namespace registration (side-effect import).
    namespace = spec.gym_id.split("/", 1)[0] if "/" in spec.gym_id else None
    if namespace and namespace.startswith("gym_"):
        try:
            import importlib

            importlib.import_module(namespace)
        except ImportError as exc:
            raise ImportError(
                f"namespace package '{namespace}' is not installed; required to "
                f"register gym_id '{spec.gym_id}'. Install sim extras: "
                f"`pip install '{namespace.replace('_', '-')}'`."
            ) from exc

    env = gym.make(spec.gym_id, max_episode_steps=spec.max_steps, **spec.gym_kwargs_dict())
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
