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

    LIBERO envs emit a nested observation:
    ``{pixels: {image, image2}, robot_state: {eef: {pos, quat, mat},
    gripper: {qpos, qvel}, joints: {pos, vel}}}``. We translate this
    via :func:`_libero_obs_to_batch` (separate function so the simpler
    PushT/Aloha branch stays readable).
    """
    import torch

    if not isinstance(obs, dict):
        raise RuntimeError(
            f"pretrained policies require a dict observation (got {type(obs).__name__}); "
            "ensure configs/envs.yaml sets gym_kwargs.obs_type to a pixels-based mode "
            "(e.g. 'pixels_agent_pos')"
        )

    # LIBERO has a {pixels: dict, robot_state: dict, ...} structure;
    # the load-bearing signal is `robot_state`, not the nested pixels
    # dict (gym-aloha also uses nested pixels but emits agent_pos, not
    # robot_state). Routing on robot_state alone keeps Aloha in the
    # PushT/Aloha branch where its agent_pos is handled correctly.
    if "robot_state" in obs:
        return _libero_obs_to_batch(obs)

    # Pull the language-conditioning task string out before the
    # PushT/Aloha branch (it's not a tensor, so the loop below would
    # reject it). Single-env -> wrap as a length-1 list, matching how
    # lerobot's `add_envs_task` produces it for vector envs.
    task = obs.get("task")
    if task is not None and not isinstance(task, list):
        task = [str(task)]

    batch: dict[str, Any] = {}
    for key, value in obs.items():
        # Task language string is forwarded as-is into complementary_data
        # by lerobot's `_extract_complementary_data`; skip the per-key loop.
        if key == "task":
            continue
        # pixels[.view] -> observation.image[s.view] (HWC uint8 -> CHW float [0,1]).
        if key == "pixels":
            # gym-aloha returns pixels as a {view: HWC} dict (e.g. {top:
            # ...}); gym-pusht returns a flat HWC ndarray. Handle both.
            if isinstance(value, dict):
                for view, view_arr in value.items():
                    tensor = torch.from_numpy(np.asarray(view_arr)).float() / 255.0
                    batch[f"observation.images.{view}"] = tensor.permute(2, 0, 1)
            else:
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
                "agent_pos, environment_state, task}"
            )
    if task is not None:
        batch["task"] = task
    return batch


def _libero_obs_to_batch(obs: dict[str, Any]) -> dict[str, Any]:
    """Translate a (single-env, debatched) LIBERO observation to a lerobot batch.

    Mirrors lerobot's ``preprocess_observation`` + ``LiberoProcessorStep``
    pipeline (``lerobot.envs.utils`` + ``lerobot.processor.env_processor``),
    collapsed to the single-env case the bench runs:

    * ``pixels: {image: HWC uint8, image2: HWC uint8}`` →
      ``observation.images.image``, ``observation.images.image2`` as
      ``CHW float32 [0, 1]``, with both H+W flipped (the
      HuggingFaceVLA/libero camera-orientation convention applied by
      ``LiberoProcessorStep``).
    * ``robot_state: {eef: {pos: (3,), quat: (4,), mat: (3,3)},
      gripper: {qpos: (2,), qvel: (2,)}, joints: {pos: (7,), vel: (7,)}}`` →
      ``observation.state`` of shape ``(8,)`` = ``concat(eef.pos,
      quat→axisangle, gripper.qpos)``. The quat→axisangle conversion
      uses the standard ``2 * acos(w) * (x,y,z) / sqrt(1 - w*w)`` form;
      tiny denominators (well below the libero physics noise floor)
      collapse to the zero rotation.

    Keys not in the standard LIBERO layout are silently ignored — they
    are never inputs to the policies we evaluate (for instance,
    ``robot_state.eef.mat`` is the rotation matrix, redundant with the
    quaternion). This is intentional: the policies are trained with the
    8-dim flattened state and any extra obs slots would cause a
    normalization-key mismatch downstream.
    """
    import torch

    batch: dict[str, Any] = {}

    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for view, img in pixels.items():
            tensor = torch.from_numpy(np.asarray(img)).float() / 255.0
            # HWC -> CHW
            tensor = tensor.permute(2, 0, 1)
            # H+W flip — matches LiberoProcessorStep (torch.flip dims=[2,3] in batched form
            # is dims=[1,2] in CHW with no batch dim).
            tensor = torch.flip(tensor, dims=[1, 2])
            batch[f"observation.images.{view}"] = tensor

    robot_state = obs.get("robot_state")
    if isinstance(robot_state, dict):
        eef = robot_state.get("eef", {})
        gripper = robot_state.get("gripper", {})

        eef_pos = torch.from_numpy(np.asarray(eef["pos"])).float()  # (3,)
        eef_quat = torch.from_numpy(np.asarray(eef["quat"])).float()  # (4,) in (x,y,z,w)
        gripper_qpos = torch.from_numpy(np.asarray(gripper["qpos"])).float()  # (2,)

        # Quaternion -> axis-angle. Single sample, no batching.
        w = eef_quat[3].clamp(-1.0, 1.0)
        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
        if float(den) > 1e-10:
            angle = 2.0 * torch.acos(w)
            axis = eef_quat[:3] / den
            eef_axisangle = axis * angle
        else:
            eef_axisangle = torch.zeros(3, dtype=torch.float32)

        state = torch.cat([eef_pos, eef_axisangle, gripper_qpos], dim=-1).float()
        batch["observation.state"] = state

    # Task language string for VLA conditioning. lerobot expects a list
    # (one entry per env in the batch) under the "task" key; the
    # downstream pipeline picks it up via `_extract_complementary_data`.
    task = obs.get("task")
    if task is not None:
        batch["task"] = task if isinstance(task, list) else [str(task)]

    if not batch:
        raise RuntimeError(f"libero obs missing both pixels and robot_state; got keys={list(obs)}")
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
    """Instantiate a gym-like env from the spec.

    Two paths, mutually exclusive (enforced by :class:`EnvSpec`):

    * **gym path** (``spec.gym_id`` set) — lazy-imports ``gymnasium``,
      side-effect imports the ``gym_X`` namespace package so env
      registration fires (without this, freshly-installed gym-pusht /
      gym-aloha raise :class:`gymnasium.error.NamespaceNotFound`),
      then ``gymnasium.make(gym_id, max_episode_steps=max_steps,
      **gym_kwargs)``. ``obs_type='pixels_agent_pos'`` is the most
      common kwarg (every pretrained policy needs it).
    * **factory path** (``spec.factory`` set) — lazy-imports the
      dotted module path in ``spec.factory`` (typically
      ``lerobot.envs.factory``) and calls its ``make_env`` with
      ``factory_kwargs``. lerobot's factory returns a
      ``{suite: {task_id: vec_env}}`` mapping (because LIBERO can be
      multi-task); we extract a single ``(suite, task_id)`` and wrap
      the vector env in :class:`_DebatchedVecEnvAdapter` so the cell
      loop sees the same single-env API as the gym path.

    Sim extras must be installed for either path. The factory path
    additionally requires the LIBERO setup invariant — see
    :func:`_ensure_libero_setup` for the non-interactive
    ``~/.libero/config.yaml`` precondition.
    """
    if spec.uses_factory:
        return _load_factory_env(spec)
    return _load_gym_env(spec)


def _load_gym_env(spec: EnvSpec) -> GymLikeEnv:
    """gym.make path. Pre-condition: ``spec.gym_id`` is set."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is not installed. Install sim extras: "
            "`pip install -e '.[sim]'` (and ensure gym-pusht / gym-aloha "
            "are pulled in for the env you are loading)."
        ) from exc

    assert spec.gym_id is not None  # EnvSpec.__post_init__ guarantees one of {gym_id, factory}
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


def _load_factory_env(spec: EnvSpec) -> GymLikeEnv:
    """factory path. Pre-condition: ``spec.factory`` is set.

    The factory module must expose a ``make_env`` callable. lerobot's
    factory returns ``{suite: {task_id: vec_env}}``; we extract one
    vec_env (single-suite + single-task is the v1 contract) and wrap
    via :class:`_DebatchedVecEnvAdapter`. If the factory returns a
    bare gym env (some hub envs do this), it is returned as-is.
    """
    import importlib

    assert spec.factory is not None  # EnvSpec.__post_init__ guarantees one of {gym_id, factory}

    # Setup invariant: lerobot's libero loader will trigger
    # libero.libero.__init__'s interactive `input()` on first import
    # if ~/.libero/config.yaml is missing. Pre-create it before any
    # libero-touching factory call.
    if "libero" in spec.factory_kwargs_dict().get("env_type", "") or _factory_kwargs_mention_libero(
        spec
    ):
        _ensure_libero_setup()

    try:
        factory_mod = importlib.import_module(spec.factory)
    except ImportError as exc:
        raise ImportError(
            f"factory module '{spec.factory}' is not importable; required for env "
            f"'{spec.name}'. Install lerobot + the relevant sim package."
        ) from exc

    if not hasattr(factory_mod, "make_env"):
        raise RuntimeError(
            f"factory module '{spec.factory}' does not expose `make_env(...)` "
            f"(required for env '{spec.name}')"
        )

    factory_kwargs = spec.factory_kwargs_dict()

    # lerobot's `make_env(cfg, ...)` takes an EnvConfig as first positional;
    # we build it via `make_env_config(env_type=..., **kwargs)` (also exposed
    # by the same module). For non-lerobot factories with a different
    # signature, this branch is skipped.
    env_type = factory_kwargs.pop("env_type", None)
    n_envs = factory_kwargs.pop("n_envs", 1)
    if n_envs != 1:
        raise ValueError(
            f"factory env '{spec.name}': n_envs must be 1 (got {n_envs}); the bench "
            "runs one episode at a time"
        )
    if env_type is not None:
        if not hasattr(factory_mod, "make_env_config"):
            raise RuntimeError(
                f"factory module '{spec.factory}' has env_type='{env_type}' in "
                "factory_kwargs but no `make_env_config(...)` to construct the EnvConfig"
            )
        cfg = factory_mod.make_env_config(env_type=env_type, **factory_kwargs)
        result = factory_mod.make_env(cfg, n_envs=1)
    else:
        # Bare factory call — for hub envs or simpler integrations.
        result = factory_mod.make_env(**factory_kwargs)

    return _materialize_factory_result(result, spec=spec)


def _factory_kwargs_mention_libero(spec: EnvSpec) -> bool:
    """Heuristic: does this factory spec target libero (so we need the setup invariant)?

    True when the factory dotted path contains the substring 'libero',
    OR the factory_kwargs reference a libero suite name. Conservative
    by design — the setup is idempotent; running it for non-libero
    factories is a no-op cost.
    """
    if "libero" in spec.factory.lower() if spec.factory else False:
        return True
    kwargs = spec.factory_kwargs_dict()
    task = str(kwargs.get("task", ""))
    return task.startswith("libero_")


def _materialize_factory_result(result: Any, *, spec: EnvSpec) -> GymLikeEnv:
    """Pick the single env we expect from a factory result.

    Accepts:
    * a ``{suite: {task_id: vec_env}}`` mapping with exactly one suite
      and one task_id (lerobot's libero/metaworld factory return shape);
      the vec_env (size 1) is wrapped via :class:`_DebatchedVecEnvAdapter`.
    * a gymnasium ``VectorEnv`` (size 1) — wrapped via the same adapter.
    * a bare ``gym.Env`` — returned as-is.

    We avoid an explicit ``isinstance(leaf, gymnasium.vector.VectorEnv)``
    check (which would force a top-level gymnasium import) and instead
    duck-type on ``num_envs`` + ``step`` — both attrs are unique to
    vector envs in the gymnasium API surface we care about. This keeps
    the function importable in the CI fast job (no gymnasium installed)
    without leaking a sentinel ``Any``-typed gym module into the
    function body.
    """
    if isinstance(result, dict):
        if len(result) != 1:
            raise RuntimeError(
                f"factory env '{spec.name}': expected 1 suite, got {len(result)}: {sorted(result)}"
            )
        suite_name, task_map = next(iter(result.items()))
        if not isinstance(task_map, dict) or len(task_map) != 1:
            raise RuntimeError(
                f"factory env '{spec.name}': expected suite '{suite_name}' to have 1 task, "
                f"got {len(task_map) if isinstance(task_map, dict) else 'non-dict'}"
            )
        leaf = next(iter(task_map.values()))
        return _wrap_leaf(leaf)

    return _wrap_leaf(result)


def _wrap_leaf(leaf: Any) -> GymLikeEnv:
    """Wrap a single leaf env (vec or scalar) into a :class:`GymLikeEnv`.

    Vec envs are detected by the ``num_envs`` attribute; everything
    else is assumed to be a single :class:`gymnasium.Env` and returned
    as-is via the protocol cast.
    """
    if hasattr(leaf, "num_envs") and hasattr(leaf, "step"):
        return cast(GymLikeEnv, _DebatchedVecEnvAdapter(leaf))
    return cast(GymLikeEnv, leaf)


_LIBERO_SETUP_DONE = False


def _ensure_libero_setup() -> None:
    """Pre-create ``~/.libero/config.yaml`` so libero's first import is non-interactive.

    The libero package's top-level ``__init__.py`` calls ``input()`` to
    prompt for a custom dataset directory if ``~/.libero/config.yaml``
    does not exist (see ``libero/libero/__init__.py``). That prompt
    blocks any non-TTY context (CI, subprocess, headless eval). We
    write the default config (pointing at libero's own bundled
    ``bddl_files`` / ``init_files``) before any libero-touching import.

    Idempotent. If the file already exists we leave it alone — the
    user may have customized the dataset path. Setting the
    ``LIBERO_CONFIG_PATH`` env var is the alternative; we prefer the
    file write because it is what every downstream libero script
    expects.

    Module-level ``_LIBERO_SETUP_DONE`` flag avoids the repeated stat
    on every cell within a process.
    """
    global _LIBERO_SETUP_DONE
    if _LIBERO_SETUP_DONE:
        return

    import os

    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        _LIBERO_SETUP_DONE = True
        return

    cfg_dir.mkdir(parents=True, exist_ok=True)
    try:
        import libero
    except ImportError as exc:
        raise ImportError(
            "libero is not installed; required for LIBERO env factory specs. "
            "Install with `pip install hf-libero`."
        ) from exc

    base = Path(libero.__file__).resolve().parent / "libero"
    default_cfg = {
        "benchmark_root": str(base),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"),
        "datasets": str(base.parent / "datasets"),
        "assets": str(base / "assets"),
    }
    import yaml

    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(default_cfg, fh)
    logger.info("wrote default libero config to %s", cfg_file)
    _LIBERO_SETUP_DONE = True


class _DebatchedVecEnvAdapter:
    """Adapt a size-1 gymnasium ``VectorEnv`` into a single-env :class:`GymLikeEnv`.

    The cell loop in :func:`run_cell` is written against the single-env
    gymnasium API: ``reset(seed) -> (obs, info)``, ``step(action) ->
    (obs, reward, terminated, truncated, info)``, ``render() -> ndarray``,
    ``close()``. lerobot's factory returns vec envs even for a single
    task; this thin shim strips/adds the leading ``num_envs=1`` dim
    on the boundary so the cell loop does not need to know the env
    came through the factory path.

    Action handling: the cell loop emits a 1-D ``(action_dim,)`` array;
    the vec env expects ``(1, action_dim)``. We expand on the way in.
    Observation handling: the vec env returns batched arrays/dicts; we
    walk the (potentially nested) dict and squeeze axis 0. Rewards /
    termination flags come back as length-1 arrays; we extract item 0.
    Render delegates to the underlying single env via the vec env's
    ``call`` plumbing — vec envs do not expose render directly, but
    each sub-env does. We use ``env.envs[0].render()`` for the
    ``SyncVectorEnv`` (the default for our n_envs=1 case).

    ``action_space`` reports the single-env action space (vec env's
    ``single_action_space``) so callers reading ``env.action_space.shape``
    get ``(action_dim,)`` not ``(1, action_dim)``.

    **Task injection.** VLA policies need the task language string in
    the batch's ``complementary_data["task"]`` slot. lerobot's eval
    loop pulls this via :func:`add_envs_task` from
    ``env.envs[0].task_description``. We inject it directly into the
    returned obs dict as ``obs["task"] = "<task string>"`` (single
    string, not a list — :func:`_libero_obs_to_batch` will pass it
    through unchanged for the policy preprocessor's
    ``TokenizerProcessorStep`` to consume). For envs without a
    ``task_description`` attribute (PushT, Aloha when wrapped via the
    factory path), the key is omitted; the eval loop is unaffected.
    """

    def __init__(self, vec_env: Any) -> None:
        self._vec = vec_env
        self._task_description = self._discover_task_description()

    def _discover_task_description(self) -> str | None:
        """Best-effort: pull ``task_description`` (or ``task``) from the underlying env.

        SyncVectorEnv exposes ``envs`` as a list of underlying gym envs;
        the LIBERO env stores the language string on the instance as
        ``task_description``. Returns ``None`` if neither attribute
        exists — the obs will simply not carry a ``task`` key.
        """
        underlying: Any = None
        if hasattr(self._vec, "envs") and len(self._vec.envs) > 0:
            underlying = self._vec.envs[0]
        if underlying is None:
            return None
        # Walk Wrapper.unwrapped chain in case lerobot's factory adds a wrapper layer.
        if hasattr(underlying, "unwrapped"):
            underlying = underlying.unwrapped
        for attr in ("task_description", "task"):
            value = getattr(underlying, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @property
    def action_space(self) -> Any:
        return self._vec.single_action_space

    @property
    def observation_space(self) -> Any:
        return self._vec.single_observation_space

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self._vec.reset(seed=seed)
        # Re-discover task description: LIBERO sets it on env construction
        # (constant per task_id), but other factories may set it on reset.
        if self._task_description is None:
            self._task_description = self._discover_task_description()
        return self._inject_task(_strip_batch_dim(obs)), info

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Re-batch the 1-D action to (1, action_dim).
        action_arr = np.asarray(action)
        action_batched = action_arr[np.newaxis, ...] if action_arr.ndim == 1 else action_arr
        obs, reward, terminated, truncated, info = self._vec.step(action_batched)
        return (
            self._inject_task(_strip_batch_dim(obs)),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(terminated).reshape(-1)[0]),
            bool(np.asarray(truncated).reshape(-1)[0]),
            info,
        )

    def _inject_task(self, obs: Any) -> Any:
        """Add ``obs["task"] = <task string>`` if known and obs is a dict."""
        if isinstance(obs, dict) and self._task_description is not None:
            obs = dict(obs)  # shallow copy; do not mutate caller's reference
            obs.setdefault("task", self._task_description)
        return obs

    def render(self) -> NDArray[np.uint8]:
        # SyncVectorEnv exposes envs[0]; AsyncVectorEnv would need .call("render").
        if hasattr(self._vec, "envs"):
            frame = self._vec.envs[0].render()
        else:
            results = self._vec.call("render")
            frame = results[0] if isinstance(results, (list, tuple)) else results
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        self._vec.close()


def _strip_batch_dim(obs: Any) -> Any:
    """Recursively strip the leading axis-0 dim from a (possibly nested) obs.

    Vec env observations are batched: e.g. ``(1, H, W, C)`` images and
    ``(1, 7)`` proprioception. The cell loop is single-env, so we
    squeeze axis 0. Dict-of-dicts are walked recursively (LIBERO's
    obs has 2-3 levels of nesting under ``robot_state``).

    Non-array leaves are returned unchanged (keeps strings, ints, etc.
    intact in the info dict).
    """
    if isinstance(obs, dict):
        return {k: _strip_batch_dim(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return obs[0] if obs.shape and obs.shape[0] == 1 else obs
    return obs


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
