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
The success rule is driven by ``env_spec.success_metric`` (see
:mod:`embodimetry.envs`):

* ``final_reward_threshold`` (v1_legacy default for every env) —
  ``success := final_reward >= env_spec.success_threshold``. Defensible
  for binary-reward envs (Aloha, LIBERO via terminating-step semantics);
  *over-counts* on PushT because the lax 0.9025 coverage window admits
  near-converged truncations that the paper / Hub-card sticky rule does
  not (``docs/SUCCESS_CRITERION_AUDIT.md`` §5.2).
* ``sticky_is_success`` — ``success := any(info['is_success'])`` across
  the rollout. Matches the lerobot canonical eval. Used by the PushT
  ``canonical`` overlay.
* ``sticky_reward_eq`` — ``success := any(reward == strict_reward_value)``.
  Used by the Aloha ``canonical`` overlay so reward in {1, 2, 3} (touched
  / lifted / attempted) is no longer counted as a transfer.

The criterion is fixed on the :class:`EnvSpec` the eval loop receives;
flipping between v1_legacy and canonical happens at config-load time via
:meth:`embodimetry.envs.EnvSpec.with_criterion`.

**Lazy imports.** ``torch`` and ``lerobot`` are imported lazily inside
:func:`seed_everything` and :func:`load_policy` respectively; importing
this module must not require either.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from embodimetry.checkpointing import RESULT_SCHEMA
from embodimetry.envs import EnvSpec
from embodimetry.policies import PolicySpec

logger = logging.getLogger(__name__)


# Column order of the OPTIONAL per-episode taxonomy sidecar (audit #171).
# This is a SEPARATE artifact from the canonical results.parquet
# (``checkpointing.RESULT_SCHEMA``) and never feeds the leaderboard /
# success-rate path. It exists so a future sweep can reconstruct the real
# per-policy failure taxonomy, which the cell-aggregated canonical parquet
# cannot express.
PER_EPISODE_SCHEMA: tuple[str, ...] = (
    "policy",
    "env",
    "seed",
    "episode_index",
    "success",
    "failure_label",
    "n_steps",
    "terminated",
    "truncated",
    "final_reward",
    "return_",
    "errored",
    "code_sha",
    "lerobot_version",
    "timestamp_utc",
    "eval_run_id",
)


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
    ``record_video=False`` OR when ``videos_dir`` was set on
    :func:`run_cell` (frames are encoded inline and dropped from memory
    before the next episode starts -- the on-disk MP4 lives at
    ``video_path`` and its SHA in ``video_sha256``). Order, when
    populated: the first frame is from the post-reset ``render()`` call;
    subsequent frames follow each ``step()``.

    ``video_path`` is the absolute path to the encoded MP4 when the
    streaming-encode path produced one; ``None`` otherwise (record_video
    off, no videos_dir provided, episode errored, or zero frames
    collected).

    ``video_sha256`` is the hex SHA-256 of the encoded MP4 bytes when
    available; ``""`` otherwise. The parquet's ``video_sha256`` column
    is filled from this field in the streaming-encode flow.

    ``error`` is ``None`` for successful (or cleanly-failed) episodes
    and a short stringified exception for crashes. When ``error`` is
    set, ``success=False``, ``return_=0.0``, ``n_steps=0``,
    ``final_reward=0.0`` — the cell continues, the row is preserved.

    ``terminated`` / ``truncated`` are the gymnasium step flags from the
    episode's *last* step (False/False for a crashed episode). They are
    captured purely to drive :attr:`failure_label`; nothing in the
    success-rate / canonical-parquet path reads them. ``terminated`` is
    True when the env signalled an MDP-terminal transition (goal reached
    OR an absorbing failure state, env-dependent); ``truncated`` is True
    when the episode hit the step cap without terminating.
    """

    episode_index: int
    success: bool
    return_: float
    n_steps: int
    wallclock_s: float
    frames: tuple[NDArray[np.uint8], ...]
    final_reward: float
    error: str | None = None
    video_path: Path | None = None
    video_sha256: str = ""
    terminated: bool = False
    truncated: bool = False

    @property
    def failure_label(self) -> str:
        """Heuristic per-episode outcome label (audit #171).

        Derived ONLY from signals the rollout already records — no new
        env probing, no invented categories. The taxonomy is deliberately
        minimal and honest:

        * ``"errored"`` — the episode crashed (OOM, env death). Not a
          task outcome; flagged so a follow-up taxonomy can exclude it.
        * ``"success"`` — the success rule (whatever ``success_metric``
          was in force) fired. We do NOT re-derive success here; we read
          the already-computed :attr:`success` flag so this label can
          never disagree with the headline success-rate.
        * ``"timeout"`` — not success, and the episode ended by hitting
          the step cap (``truncated`` and not ``terminated``). The
          policy never drove the env into a terminal state.
        * ``"early_termination"`` — not success, but the env signalled
          ``terminated`` (an absorbing/terminal transition that was not a
          success). For binary-reward envs this is the closest honest
          proxy for a "task-failure terminal state".
        * ``"unknown"`` — not success and neither flag set. Only reachable
          if an env breaks out of the rollout loop without setting either
          gym flag; surfaced rather than silently bucketed elsewhere.

        These are coarse on purpose: separating e.g. "grasp slip" from
        "wrong object" requires per-step trajectory signals this loop does
        not collect (see the module docstring for what a fuller taxonomy
        would need).
        """
        if self.error is not None:
            return "errored"
        if self.success:
            return "success"
        if self.truncated and not self.terminated:
            return "timeout"
        if self.terminated:
            return "early_termination"
        return "unknown"


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
    eval_run_id: str = ""

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def to_rows(self, *, video_sha256_per_episode: Sequence[str] | None = None) -> pd.DataFrame:
        """Convert to a DataFrame matching :data:`RESULT_SCHEMA`.

        ``video_sha256_per_episode`` is parallel to ``self.episodes``;
        if ``None``, the column is filled from each
        :attr:`EpisodeResult.video_sha256` (populated by the streaming
        per-episode encoder in :func:`run_cell`). When the streaming
        encoder was not used the per-episode SHAs are empty strings,
        matching the legacy "no video" default.
        """
        n = len(self.episodes)
        if video_sha256_per_episode is None:
            video_sha = [ep.video_sha256 for ep in self.episodes]
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
                # OPTIONAL columns (audit H3 / M5). ``errored`` flags a
                # crashed episode (OOM, env death) so plan_resume re-runs
                # the cell instead of treating the crash as a legit
                # failure. ``eval_run_id`` is the per-sweep provenance
                # handle (empty for single-cell/back-compat callers).
                "errored": ep.error is not None,
                "eval_run_id": self.eval_run_id,
            }
            for i, ep in enumerate(self.episodes)
        ]
        return pd.DataFrame(rows, columns=list(RESULT_SCHEMA))

    def to_per_episode_rows(self) -> pd.DataFrame:
        """Build the OPTIONAL per-episode taxonomy sidecar (audit #171).

        Strictly additive: this is a SEPARATE DataFrame from
        :meth:`to_rows`, written (if at all) to its own sidecar parquet —
        it does NOT touch the canonical :data:`RESULT_SCHEMA` columns or
        any shipped number. The ``success`` column here is the identical
        boolean :meth:`to_rows` emits, so ``failure_label == "success"``
        rows are byte-for-byte the same set as the canonical successes.

        Columns are :data:`PER_EPISODE_SCHEMA`.
        """
        rows = [
            {
                "policy": self.policy,
                "env": self.env,
                "seed": self.seed,
                "episode_index": ep.episode_index,
                "success": ep.success,
                "failure_label": ep.failure_label,
                "n_steps": ep.n_steps,
                "terminated": ep.terminated,
                "truncated": ep.truncated,
                "final_reward": ep.final_reward,
                "return_": ep.return_,
                "errored": ep.error is not None,
                "code_sha": self.code_sha,
                "lerobot_version": self.lerobot_version,
                "timestamp_utc": self.timestamp_utc,
                "eval_run_id": self.eval_run_id,
            }
            for ep in self.episodes
        ]
        return pd.DataFrame(rows, columns=list(PER_EPISODE_SCHEMA))


# --------------------------------------------------------------------- #
# Per-episode taxonomy sidecar sink (audit #171)                        #
# --------------------------------------------------------------------- #


def append_per_episode_rows(sink_path: Path, new_rows: pd.DataFrame) -> int:
    """Atomically append per-episode taxonomy rows to a sidecar parquet.

    OPT-IN companion to :func:`embodimetry.checkpointing.append_cell_rows`.
    Deliberately kept here (not in ``checkpointing``) so this NEW sidecar
    cannot be confused with the canonical results.parquet append path —
    it has its own :data:`PER_EPISODE_SCHEMA` and never participates in
    resume planning.

    Strategy mirrors the canonical writer: load existing rows, concat,
    write to a ``.tmp.parquet`` sibling, then ``os.replace`` into place
    (atomic on POSIX). Empty ``new_rows`` is a no-op. ``new_rows`` must
    carry exactly :data:`PER_EPISODE_SCHEMA`. Unlike the canonical writer
    there is no duplicate-key guard: this sidecar is regenerable from a
    re-run and is not load-bearing for any shipped number.
    """
    actual = set(new_rows.columns)
    expected = set(PER_EPISODE_SCHEMA)
    if actual != expected:
        raise ValueError(
            f"per-episode rows have wrong columns: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if len(new_rows) == 0:
        return _count_parquet_rows(sink_path)

    new_rows_ordered = new_rows[list(PER_EPISODE_SCHEMA)]
    if sink_path.exists():
        existing = pd.read_parquet(sink_path)
        combined = pd.concat([existing, new_rows_ordered], ignore_index=True)
    else:
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_rows_ordered

    tmp_path = sink_path.with_suffix(".tmp.parquet")
    try:
        combined.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, sink_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("failed to clean up tmp per-episode parquet at %s", tmp_path)
        raise
    return len(combined)


def _count_parquet_rows(path: Path) -> int:
    return len(pd.read_parquet(path)) if path.exists() else 0


# --------------------------------------------------------------------- #
# Seeding helper                                                        #
# --------------------------------------------------------------------- #


def seed_everything(seed_idx: int) -> int:
    """Apply the per-cell seeding contract. Returns the base seed.

    Seeds numpy's global RNG immediately. Lazy-imports torch and seeds
    its CPU + CUDA generators if importable. Logs a warning (does not
    raise) if torch is unavailable -- that's a "no GPU work happens"
    condition, not necessarily fatal for tests using mocks.

    Beyond seeding (audit C3), this also pins torch into deterministic
    mode: cuDNN deterministic + benchmark-off, the cuBLAS workspace
    env var required for deterministic GEMMs, and
    ``use_deterministic_algorithms(..., warn_only=True)``. ``warn_only``
    is deliberate -- an upstream kernel without a deterministic
    implementation should log a warning, not crash a multi-hour sweep
    mid-cell.
    """
    # Must be set before the first cuBLAS handle is created; setdefault so
    # an operator who exported a different value keeps theirs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
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
            if action_np.size > expected:
                # Multi-embodiment VLAs (xvla, pi0fast) emit a padded
                # action vector (e.g. 20-dim) so a single policy can
                # cover several robots. The convention is that the
                # first `env_action_dim` entries are the active dims;
                # the trailing dims are zero-padded slots for other
                # embodiments. Slice to the env's action shape.
                action_np = np.asarray(action_np[:expected], dtype=np.float32)
            elif action_np.size < expected:
                raise RuntimeError(
                    f"policy emitted action of size {action_np.size}, "
                    f"expected {expected} for action_shape {self._action_shape} "
                    "(too few dims; not a padded multi-embodiment case)"
                )
            action_np = action_np.reshape(self._action_shape)
        return np.asarray(action_np, dtype=np.float32)


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


def _buffer_name_to_feature_key(feature_buf: str, known_keys: Sequence[str]) -> str:
    """Map a legacy safetensors buffer name back to a canonical feature key.

    The legacy buffer format flattens every dot in a feature key to an
    underscore: ``observation.images.top`` is stored as
    ``buffer_observation_images_top``. Naively reversing only the first
    underscore (``observation.images_top``) is WRONG for any key with
    more than one dot — and Aloha/Libero camera keys
    (``observation.images.<view>``) always have two. A feature key
    recovered under the wrong name never reaches lerobot's
    ``NormalizerProcessorStep`` (it silently skips unknown keys, see
    ``normalize_processor.py`` ``key not in self._tensor_stats``), so
    the input is fed to the model un-normalized — garbage features,
    near-zero success.

    The buffer name itself is lossy (underscore vs dot is ambiguous),
    so we disambiguate against the policy config's declared feature
    keys: a buffer matches the unique known key whose dots-as-
    underscores form equals ``feature_buf``. If no known key is given
    or none matches, fall back to the legacy single-underscore reversal
    (correct for single-dot keys such as ``observation.image`` on
    ``lerobot/diffusion_pusht``).
    """
    matches = [k for k in known_keys if k.replace(".", "_") == feature_buf]
    if len(matches) == 1:
        return matches[0]
    return feature_buf.replace("_", ".", 1)


def _recover_dataset_stats_from_safetensors(
    repo_id: str,
    revision: str,
    feature_keys: Sequence[str] = (),
) -> dict[str, dict[str, NDArray[np.float32]]]:
    """Reconstruct ``dataset_stats`` from legacy safetensors normalize buffers.

    Pre-0.5.x lerobot checkpoints (e.g. ``lerobot/diffusion_pusht``,
    ``lerobot/act_aloha_sim_transfer_cube_human``) pre-date the
    processor-pipeline split: their normalization stats live as buffers
    inside ``model.safetensors`` rather than as
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

    ``feature_keys`` is the set of canonical feature keys the policy
    config declares (``cfg.input_features`` + ``cfg.output_features``).
    It is used to disambiguate multi-dot buffer names — see
    :func:`_buffer_name_to_feature_key`. Passing it empty preserves the
    legacy single-underscore reversal.

    Lazy-imports ``huggingface_hub`` and ``safetensors``; both are
    transitive deps of ``lerobot==0.5.1`` so they are always available
    when this function is called from the pretrained branch.
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
            # rest looks like 'observation_images_top.mean' or 'action.max'.
            feature_buf, _, stat_name = rest.rpartition(".")
            if not feature_buf or not stat_name:
                continue
            feature_key = _buffer_name_to_feature_key(feature_buf, feature_keys)
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
            "Lock checkpoint SHAs per docs/REPRODUCE.md, then update "
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
        # Classical (hand-coded) controllers are weightless CPU baselines —
        # the L2 rung of the capability ladder. They carry is_baseline=true
        # (no repo_id/SHA, skip the run_one.py GPU precheck) and dispatch to
        # a scripted controller. Routed by name prefix so the registry + this
        # branch cannot silently drift; the strategy lives in
        # embodimetry.policies_classical.
        if spec.name.startswith("classical_"):
            from embodimetry.policies_classical import load_classical_policy

            return load_classical_policy(spec.name, action_shape=action_shape)
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
    revision: str | None,
    action_shape: tuple[int, ...] | None,
    device: str,
) -> _LerobotPolicyAdapter:
    """Inner helper: lazy-imports lerobot and instantiates the adapter.

    ``revision`` is a pinned Hub SHA for registry policies; it is ``None``
    when ``repo_id`` is a local checkpoint directory (the L1 fine-tuning
    rung loads its freshly-trained checkpoint by path). ``from_pretrained``
    accepts ``revision=None`` for local paths.

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
        if revision is None:
            # Local checkpoints (the L1 fine-tuned dir) always ship the
            # processor JSONs, so the try above succeeds and we never reach
            # here with revision=None. A None here means a local path with no
            # processor JSON AND no Hub revision to recover buffers from —
            # unrecoverable, so fail loud rather than pass None downstream.
            raise RuntimeError(
                f"no processor JSONs at local checkpoint '{repo_id}' and no Hub "
                "revision to recover legacy normalization buffers from"
            ) from None
        feature_keys = (*cfg.input_features.keys(), *cfg.output_features.keys())
        dataset_stats = _recover_dataset_stats_from_safetensors(
            repo_id, revision, feature_keys=feature_keys
        )
        if not dataset_stats:
            raise RuntimeError(
                f"could not recover normalization stats for '{repo_id}'@{revision}: "
                "no policy_preprocessor.json on the Hub AND no normalize_inputs/targets "
                "buffers in model.safetensors. Pretrained policy will not work without "
                "valid normalization."
            ) from None
        # A recovered key that is not a declared feature key means the buffer
        # name -> feature key mapping failed: the stats would silently never
        # reach NormalizerProcessorStep and the input would be fed raw to the
        # model. Fail loud rather than ship a 0%-success cell.
        unmapped = sorted(set(dataset_stats) - set(feature_keys))
        if unmapped:
            raise RuntimeError(
                f"recovered normalization stats for '{repo_id}'@{revision} contain "
                f"feature keys not declared by the policy config: {unmapped}. "
                f"Config declares: {sorted(feature_keys)}. The legacy safetensors "
                "buffer name could not be mapped to a canonical feature key, so "
                "normalization would be silently skipped for these inputs."
            ) from None
        preprocessor, postprocessor = _lerobot_factory.make_pre_post_processors(
            cfg, dataset_stats=dataset_stats
        )

    preprocessor, postprocessor = _patch_processors_for_policy(cfg, preprocessor, postprocessor)

    model = policy_cls.from_pretrained(repo_id, revision=revision, config=cfg)
    model = model.to(device).eval()

    return _LerobotPolicyAdapter(
        model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        action_shape=action_shape,
        device=device,
    )


def _patch_processors_for_policy(
    cfg: Any, preprocessor: Any, postprocessor: Any
) -> tuple[Any, Any]:
    """Insert policy-specific processor steps the Hub JSON omits.

    Some lerobot policies on the Hub ship ``policy_preprocessor.json`` /
    ``policy_postprocessor.json`` files that do NOT contain every step
    the policy's dedicated factory
    (``make_<policy>_<env>_pre_post_processors``) would have built.
    Loading via the generic
    :func:`lerobot.policies.factory.make_pre_post_processors` reads
    exactly what's on the Hub and stops there, so the missing step is
    silently absent and inputs/outputs land in the wrong space.

    Concrete cases for v1 (``lerobot/xvla-libero``):

    * **Output side (PR #71).** The Hub postprocessor is
      ``[UnnormalizerProcessorStep, DeviceProcessorStep]`` but XVLA
      emits a 20-dim action whose first 10 dims are
      ``[eef(3), rot6d(6), gripper(1)]``. LIBERO needs
      ``[eef(3), axis_angle(3), gripper(1)] = 7``. Insert
      :class:`XVLARotation6DToAxisAngleProcessorStep` before the
      trailing :class:`DeviceProcessorStep` (move-to-cpu).
    * **Input side (this fix).** The Hub preprocessor's
      ``normalizer_processor`` step declares VISUAL features but with
      ``norm_map = {VISUAL: IDENTITY}`` — so images are passed through
      raw [0, 1] floats. The training-time
      :class:`XVLAImageNetNormalizeProcessorStep` is omitted, but
      XVLA's Florence-2 visual backbone was pretrained against
      ImageNet-normalized inputs, so raw [0, 1] images produce
      garbage visual features and ~0% rollout success. Insert
      :class:`XVLAImageNetNormalizeProcessorStep` before the
      :class:`DeviceProcessorStep` (matches the ordering in
      :func:`lerobot.policies.xvla.processor_xvla.make_xvla_pre_post_processors`:
      tokenizer -> ImageNet -> add_domain_id -> device -> normalizer).

    **Why surgical insertion, not full pipeline replacement.** The
    obvious alternative — calling ``make_xvla_libero_pre_post_processors()``
    and swapping in its pipelines wholesale — does NOT work for
    inference: that factory ships only
    ``[LiberoProcessorStep, XVLAImageNetNormalize, XVLAAddDomainId]``
    on the preprocessor side. It drops the
    :class:`TokenizerProcessorStep` the inference path requires
    (XVLA's ``select_action`` reads ``input_ids`` from the batch), and
    its :class:`LiberoProcessorStep` produces a 20-dim state and
    flipped images — but the Hub checkpoint's config declares
    ``observation.state`` as 8-dim and our :func:`_libero_obs_to_batch`
    already produces the 8-dim axis-angle state plus pre-flipped
    images. Inserting only the missing ImageNet step preserves the
    Hub pipeline's tokenizer + normalizer (both load-bearing) and
    keeps our pre-translation contract intact.

    Both XVLA-specific steps are idempotent: re-running this patcher
    on an already-patched pipeline is a no-op. Non-xvla policies pass
    through unchanged.
    """
    if getattr(cfg, "type", None) != "xvla":
        return preprocessor, postprocessor

    preprocessor = _patch_xvla_preprocessor(preprocessor)
    postprocessor = _patch_xvla_postprocessor(postprocessor)
    return preprocessor, postprocessor


def _patch_xvla_preprocessor(preprocessor: Any) -> Any:
    """Insert :class:`XVLAImageNetNormalizeProcessorStep` into a Hub-loaded xvla preprocessor.

    Idempotent. Inserts before the trailing :class:`DeviceProcessorStep`
    so normalization runs on CPU and the device hop stays last (mirrors
    :func:`lerobot.policies.xvla.processor_xvla.make_xvla_pre_post_processors`).
    If no device step is present the new step is appended.
    """
    from lerobot.policies.xvla.processor_xvla import XVLAImageNetNormalizeProcessorStep
    from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline

    existing_steps = list(preprocessor.steps)
    if any(isinstance(s, XVLAImageNetNormalizeProcessorStep) for s in existing_steps):
        return preprocessor

    new_steps: list[Any] = []
    inserted = False
    for step in existing_steps:
        if not inserted and isinstance(step, DeviceProcessorStep):
            new_steps.append(XVLAImageNetNormalizeProcessorStep())
            inserted = True
        new_steps.append(step)
    if not inserted:
        new_steps.append(XVLAImageNetNormalizeProcessorStep())

    logger.info(
        "patched xvla preprocessor: inserted XVLAImageNetNormalizeProcessorStep "
        "before the device hop -- Hub policy_preprocessor.json sets VISUAL=IDENTITY "
        "in its normalizer step, so without ImageNet normalization the Florence-2 "
        "visual backbone sees raw [0, 1] images and emits garbage features (~0% success)"
    )
    return PolicyProcessorPipeline(
        steps=new_steps,
        name=preprocessor.name,
    )


def _patch_xvla_postprocessor(postprocessor: Any) -> Any:
    """Insert :class:`XVLARotation6DToAxisAngleProcessorStep` into a Hub-loaded xvla postprocessor.

    See :func:`_patch_processors_for_policy` for the rationale. Idempotent;
    no-op for already-patched pipelines.
    """
    from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep
    from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    existing_steps = list(postprocessor.steps)
    if any(isinstance(s, XVLARotation6DToAxisAngleProcessorStep) for s in existing_steps):
        return postprocessor

    # Insert the rotation conversion BEFORE the trailing DeviceProcessorStep
    # (move-to-cpu) when present; otherwise append at the end. The conversion
    # itself is device-agnostic but it is tidier to keep the device hop last.
    new_steps: list[Any] = []
    inserted = False
    for step in existing_steps:
        if not inserted and isinstance(step, DeviceProcessorStep):
            new_steps.append(XVLARotation6DToAxisAngleProcessorStep())
            inserted = True
        new_steps.append(step)
    if not inserted:
        new_steps.append(XVLARotation6DToAxisAngleProcessorStep())

    logger.info(
        "patched xvla postprocessor: inserted XVLARotation6DToAxisAngleProcessorStep "
        "to convert 6D rotation -> axis-angle for LIBERO env consumption"
    )
    return PolicyProcessorPipeline(
        steps=new_steps,
        name=postprocessor.name,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
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
    videos_dir: Path | None = None,
    code_sha: str | None = None,
    lerobot_version: str | None = None,
    eval_run_id: str = "",
    per_episode_sink: Path | None = None,
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

    **Streaming MP4 encode** (memory-safety contract). If ``record_video``
    and ``videos_dir`` are both set, each episode's frames are encoded
    to an MP4 at ``videos_dir / "{policy}__{env}__seed{seed}__ep{K:03d}.mp4"``
    immediately after the episode's last step, and the in-memory frame
    tuple is then dropped (``EpisodeResult.frames`` becomes ``()`` for
    that episode). This bounds the peak working set to one episode's
    frames at a time; previous behaviour buffered every episode's frames
    for the whole cell and OOMed on long-horizon envs (aloha, libero).
    The encoded path + SHA land in ``EpisodeResult.video_path`` and
    ``EpisodeResult.video_sha256`` respectively. ``videos_dir`` is
    created if missing.

    When ``record_video=True`` but ``videos_dir is None`` the old
    behaviour is preserved: frames are kept on ``EpisodeResult`` and
    the caller is responsible for encoding them later. The streaming
    path is the new default for any callers passing ``videos_dir``.

    ``code_sha`` and ``lerobot_version`` default to autodetection
    (``git rev-parse HEAD`` and ``lerobot.__version__``); pass them
    explicitly when the orchestrator already has them in hand.

    ``per_episode_sink`` (audit #171) is OPT-IN and OFF by default. When
    a path is given, the cell's per-episode taxonomy rows
    (:meth:`CellResult.to_per_episode_rows`, schema
    :data:`PER_EPISODE_SCHEMA`) are appended to that sidecar parquet at
    the cell boundary — same flush granularity as the canonical results,
    so a mid-cell crash leaves no partial taxonomy rows. This writes ONLY
    to the sidecar; the canonical results.parquet and every shipped
    number are untouched whether or not this is set. The path is created
    (parents included) on first write.
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

    stream_encode = bool(record_video and videos_dir is not None)
    if stream_encode:
        # Caller-provided dir; mkdir parents for ergonomics (same shape as
        # the legacy render_episodes_to_videos used to do).
        assert videos_dir is not None  # narrowing for mypy
        videos_dir.mkdir(parents=True, exist_ok=True)

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
            success_metric=env_spec.success_metric,
            strict_reward_value=env_spec.strict_reward_value,
            record_video=record_video,
        )
        if stream_encode and episode.error is None and len(episode.frames) > 0:
            assert videos_dir is not None
            episode = _encode_and_drop_frames(
                episode,
                videos_dir=videos_dir,
                policy_name=policy_name,
                env_name=env_spec.name,
                seed_idx=seed_idx,
            )
        episodes.append(episode)

    cell_result = CellResult(
        policy=policy_name,
        env=env_spec.name,
        seed=seed_idx,
        episodes=tuple(episodes),
        code_sha=code_sha,
        lerobot_version=lerobot_version,
        timestamp_utc=timestamp_utc,
        eval_run_id=eval_run_id,
    )

    if per_episode_sink is not None:
        append_per_episode_rows(per_episode_sink, cell_result.to_per_episode_rows())

    return cell_result


def _encode_and_drop_frames(
    episode: EpisodeResult,
    *,
    videos_dir: Path,
    policy_name: str,
    env_name: str,
    seed_idx: int,
) -> EpisodeResult:
    """Encode this episode's frames to MP4, return a new EpisodeResult with frames dropped.

    Filename convention matches the legacy
    ``render_episodes_to_videos`` helper: ``{policy}__{env}__seed{N}__ep{K:03d}.mp4``.

    Lazy-imports :mod:`embodimetry.render` so the eval module stays
    importable in CI fast (no imageio/ffmpeg required). The frame stack
    is materialized as a single ``(T, H, W, 3)`` numpy array, handed to
    :func:`render_episode`, and then immediately released by the
    function-local ``stacked`` going out of scope. The returned dataclass
    has ``frames=()`` so the cell-result tuple does not hold the source
    frames for the rest of the cell.

    If the encode itself raises (e.g. :class:`RenderSizeError` from the
    long-episode fallback path), we propagate — that is a real bug to
    investigate, not something to swallow. The partial MP4 (if any) is
    already cleaned up by :func:`render_episode` itself.
    """
    from embodimetry.render import render_episode

    out_path = (
        videos_dir / f"{policy_name}__{env_name}__seed{seed_idx}__ep{episode.episode_index:03d}.mp4"
    )
    stacked = np.stack(episode.frames, axis=0)
    result = render_episode(stacked, out_path)
    # Drop the intermediate numpy stack as soon as the encoder returns.
    del stacked
    return EpisodeResult(
        episode_index=episode.episode_index,
        success=episode.success,
        return_=episode.return_,
        n_steps=episode.n_steps,
        wallclock_s=episode.wallclock_s,
        frames=(),
        final_reward=episode.final_reward,
        error=episode.error,
        video_path=out_path,
        video_sha256=result.content_sha256,
        terminated=episode.terminated,
        truncated=episode.truncated,
    )


def _run_one_episode(
    *,
    policy: PolicyCallable,
    env: GymLikeEnv,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
    success_threshold: float,
    success_metric: str = "final_reward_threshold",
    strict_reward_value: float | None = None,
    record_video: bool,
) -> EpisodeResult:
    """Inner loop. Catches per-episode exceptions and records them.

    ``success_metric`` is the rule used to flip a rollout into a boolean
    -- see the module docstring for the three accepted values. The
    sticky variants OR-accumulate a per-step signal across the rollout
    so a flag that fires once-and-decays still counts; the
    ``final_reward_threshold`` default reads only the terminating step's
    reward (v1_legacy behaviour).

    ``strict_reward_value`` is required when ``success_metric ==
    'sticky_reward_eq'`` (e.g. ``4.0`` for ACT's Aloha-Transfer rule);
    a ``None`` here with that metric raises before the rollout starts.
    """
    if success_metric == "sticky_reward_eq" and strict_reward_value is None:
        raise ValueError(
            "_run_one_episode: success_metric='sticky_reward_eq' requires strict_reward_value"
        )

    t0 = time.perf_counter()
    frames: list[NDArray[np.uint8]] = []
    cumulative_return = 0.0
    n_steps = 0
    final_reward = 0.0
    sticky_is_success = False
    sticky_reward_match = False
    last_terminated = False
    last_truncated = False

    try:
        obs, _info = env.reset(seed=episode_seed)
        policy.reset()
        if record_video:
            frames.append(env.render())

        for _ in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            n_steps += 1
            cumulative_return += float(reward)
            final_reward = float(reward)
            # Sticky accumulators: OR across the rollout. The env may
            # terminate immediately when is_success fires (PushT, Aloha,
            # LIBERO all do), but accumulating lets the metric survive
            # a hypothetical decay-after-success and matches the
            # lerobot canonical eval's ``any(is_success)`` reduction.
            if isinstance(info, dict):
                is_success_flag = info.get("is_success")
                if isinstance(is_success_flag, bool | int | np.bool_) and bool(is_success_flag):
                    sticky_is_success = True
            if strict_reward_value is not None and float(reward) == float(strict_reward_value):
                sticky_reward_match = True
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
    if success_metric == "sticky_is_success":
        success = sticky_is_success
    elif success_metric == "sticky_reward_eq":
        success = sticky_reward_match
    else:
        # final_reward_threshold (v1_legacy default).
        success = final_reward >= success_threshold
    # If the rollout exhausted the step budget without the env setting
    # either gym flag, the bench's own step cap was the limiter -> a
    # timeout. Recording it as truncated keeps `failure_label` honest
    # for envs that defer truncation to the harness instead of the
    # gym TimeLimit wrapper. (No effect on `success`.)
    if not last_terminated and not last_truncated and n_steps >= max_steps:
        last_truncated = True
    return EpisodeResult(
        episode_index=episode_index,
        success=success,
        return_=cumulative_return,
        n_steps=n_steps,
        wallclock_s=wallclock,
        frames=tuple(frames),
        final_reward=final_reward,
        error=None,
        terminated=last_terminated,
        truncated=last_truncated,
    )


def run_cell_from_specs(
    policy_spec: PolicySpec,
    env_spec: EnvSpec,
    *,
    seed_idx: int,
    n_episodes: int,
    device: str = "cuda",
    record_video: bool = True,
    videos_dir: Path | None = None,
    eval_run_id: str = "",
    per_episode_sink: Path | None = None,
) -> CellResult:
    """Convenience: load policy + env from specs, then :func:`run_cell`.

    The env is created via :func:`load_env` (lazy gymnasium import) and
    the action shape is sniffed from ``env.action_space.shape`` for
    baseline policy construction. The caller is responsible for
    ``env.close()`` -- this function does not own the env's lifecycle
    once it returns.

    ``videos_dir`` is forwarded to :func:`run_cell`; when set together
    with ``record_video=True`` the streaming per-episode MP4 encode
    happens inside the cell loop and ``EpisodeResult.frames`` is dropped
    after each episode's encode. See :func:`run_cell` for the full
    contract.

    ``per_episode_sink`` is forwarded to :func:`run_cell` (OFF by
    default). It is the opt-in handle scripts use to emit the failure
    taxonomy sidecar without altering the canonical results path.
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
        videos_dir=videos_dir,
        eval_run_id=eval_run_id,
        per_episode_sink=per_episode_sink,
    )
