#!/usr/bin/env python3
"""Matrix sweep orchestrator. Dispatches one ``scripts/run_one.py`` subprocess per cell.

Sister script to :mod:`scripts.run_one` -- where ``run_one`` runs a
single ``(policy, env, seed)`` cell, ``run_sweep`` plans the cartesian
product, persists a ``results/sweep_manifest.json`` with a row per
planned cell, then dispatches each cell as its own subprocess so
torch / CUDA state is reset between cells (premortem #2: VRAM leaks
across long-running processes).

**Why subprocess.** Running every cell in the same process risks the
sweep dying on the second pretrained policy because the first one's
allocator never released the slab. ``subprocess.run`` per cell is the
cheapest way to make ``torch.cuda.empty_cache`` actually mean
something. Tests inject a fake via the module-level
:data:`_run_subprocess` Callable so the dispatch loop is exercised
without ever touching ``run_one`` for real.

**Resume contract.** Cells are atomic at the parquet boundary (see
``src/lerobot_bench/checkpointing.py``). On startup:

1. Build the planned cell list from the YAML config.
2. Call :func:`plan_resume` against ``results_path``: cells already at
   ``n_episodes`` rows are ``completed`` (skipped); cells with partial
   rows are ``partial`` (dropped via :func:`drop_partial_cells`, then
   re-queued); cells with no rows are ``pending``.
3. Write the manifest with every planned cell as ``pending`` /
   ``completed`` / ``skipped`` *before* any subprocess fires.
4. Dispatch ``pending`` cells in order (sorted, or shuffled with
   ``--shuffle SEED``). On each cell completion the manifest's row is
   atomically updated to ``completed`` / ``failed`` (same tmp+rename
   pattern as checkpointing) so a kill -9 between cells is recoverable.

**OOM rescue.** A non-zero exit from ``run_one`` (codes 2/3/4/5) is
recorded as ``failed`` with the exit code and the last 200 lines of
stderr. The sweep continues to the next cell. v1 does NOT retry with
downscoped episode counts -- calibration is the prevention layer; once
auto-downscope has shaped ``configs/sweep_full.yaml``, mid-sweep
surprises should be rare and re-runnable manually with ``run_one``.

**Lazy imports.** Same AST contract as :mod:`scripts.run_one`: no
top-level torch / lerobot / lerobot_bench.render imports. The script
must import cleanly in CI without sim/GPU extras.

Usage::

    python scripts/run_sweep.py --config configs/sweep_full.yaml
    python scripts/run_sweep.py --config configs/sweep_mini.yaml --max-cells 4
    python scripts/run_sweep.py --config configs/sweep_full.yaml --shuffle 42
    python scripts/run_sweep.py --config configs/sweep_mini.yaml --dry-run

Exit codes:
    0  every dispatched cell succeeded (or sweep was a no-op)
    2  some cells failed -- see manifest for per-cell exit codes
    3  config invalid (missing keys, bad types, unknown policy/env)
    4  no cells planned (every cell was incompatible or already done)
    5  results path conflict (existing parquet has wrong schema)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from lerobot_bench.checkpointing import (
    CellKey,
    drop_partial_cells,
    load_results,
    plan_resume,
)
from lerobot_bench.envs import EnvRegistry
from lerobot_bench.policies import PolicyRegistry

logger = logging.getLogger("run-sweep")

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_POLICIES_YAML = Path("configs/policies.yaml")
DEFAULT_ENVS_YAML = Path("configs/envs.yaml")

# Cell statuses written into the manifest. Kept narrow on purpose --
# the manifest is the operator's first stop after a kill -9, so each
# value should answer "what happened to this cell" in one word.
STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"  # incompatible (env not in policy.env_compat) or pre-resumed completed

# How much stderr to keep per failed cell. 200 lines is enough to read
# the traceback + a few helpful preceding log lines without blowing up
# the manifest's size when many cells fail.
STDERR_TAIL_LINES = 200


# --------------------------------------------------------------------- #
# Subprocess injection point                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubprocessOutcome:
    """What the dispatch loop needs to know after a cell finishes."""

    returncode: int
    stdout: str
    stderr: str


def _default_run_subprocess(
    argv: list[str],
    *,
    timeout_s: float | None,
) -> SubprocessOutcome:
    """Real subprocess invocation. Tests replace this via :data:`_run_subprocess`.

    ``timeout_s`` is intentionally **soft**: we pass it to
    ``subprocess.run`` only for surfacing a TimeoutExpired warning in
    the dispatch loop -- never SIGTERM. CUDA cleanup on signal is
    fragile; if a cell overruns, we log and let it finish.

    Notes:
      * Uses ``capture_output=True`` so stdout/stderr can be tailed
        into the manifest.
      * ``check=False`` because non-zero exits are normal here.
    """
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Soft timeout: log and treat as a failed cell with a clear
        # marker. Do NOT terminate -- the operator can decide whether
        # to kill the lingering process out-of-band.
        logger.warning(
            "cell exceeded soft timeout %.0fs (still running in background?): %s",
            timeout_s or -1,
            " ".join(argv),
        )
        stderr_blob = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else ""
        stdout_blob = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        return SubprocessOutcome(returncode=124, stdout=stdout_blob, stderr=stderr_blob)

    return SubprocessOutcome(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# Module-level injection point. Tests do
# ``monkeypatch.setattr(run_sweep, "_run_subprocess", fake)`` to drive
# the dispatch loop without spawning python.
_run_subprocess: Callable[..., SubprocessOutcome] = _default_run_subprocess


# --------------------------------------------------------------------- #
# Sweep config (YAML schema)                                            #
# --------------------------------------------------------------------- #


_REQUIRED_SWEEP_KEYS = frozenset({"policies", "envs", "seeds", "episodes_per_seed", "results_path"})
_OPTIONAL_SWEEP_KEYS = frozenset(
    {
        "videos_dir",
        "record_video",
        "device",
        "policies_yaml",
        "envs_yaml",
        "cell_timeout_s",
        "max_parallel",
        "overrides",
    }
)
_ALL_SWEEP_KEYS = _REQUIRED_SWEEP_KEYS | _OPTIONAL_SWEEP_KEYS


@dataclass(frozen=True)
class SweepConfig:
    """Validated sweep YAML.

    Built via :meth:`from_dict` so :func:`load_sweep_config` can read
    YAML and surface schema problems with one ``ValueError`` rather
    than a TypeError deep inside the dispatch loop.

    ``overrides`` is a two-level mapping
    ``{policy: {env: {"n_episodes": int}}}``. Any (policy, env) not
    present uses ``episodes_per_seed``. Unknown policies / envs in
    overrides raise during config validation -- a typo in the override
    table is the kind of surprise that wastes a whole night.
    """

    policies: tuple[str, ...]
    envs: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes_per_seed: int
    results_path: Path
    videos_dir: Path | None
    record_video: bool
    device: str
    policies_yaml: Path
    envs_yaml: Path
    cell_timeout_s: float | None
    max_parallel: int
    overrides: dict[str, dict[str, dict[str, int]]]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<dict>") -> SweepConfig:
        if not isinstance(data, dict):
            raise ValueError(f"{source}: top-level YAML must be a mapping")
        keys = set(data)
        missing = _REQUIRED_SWEEP_KEYS - keys
        if missing:
            raise ValueError(f"{source}: missing required fields: {sorted(missing)}")
        extras = keys - _ALL_SWEEP_KEYS
        if extras:
            raise ValueError(f"{source}: unknown fields: {sorted(extras)}")

        policies = _validate_str_list(data["policies"], source=f"{source}: policies")
        envs = _validate_str_list(data["envs"], source=f"{source}: envs")
        seeds = _validate_int_list(data["seeds"], source=f"{source}: seeds")

        episodes_per_seed = data["episodes_per_seed"]
        if not isinstance(episodes_per_seed, int) or episodes_per_seed <= 0:
            raise ValueError(
                f"{source}: episodes_per_seed must be a positive int, got {episodes_per_seed!r}"
            )

        results_path = Path(str(data["results_path"]))

        videos_dir_raw = data.get("videos_dir")
        videos_dir = Path(str(videos_dir_raw)) if videos_dir_raw else None

        record_video = bool(data.get("record_video", True))
        device = str(data.get("device", "cuda"))

        policies_yaml = Path(str(data.get("policies_yaml") or DEFAULT_POLICIES_YAML))
        envs_yaml = Path(str(data.get("envs_yaml") or DEFAULT_ENVS_YAML))

        cell_timeout_s_raw = data.get("cell_timeout_s")
        cell_timeout_s: float | None
        if cell_timeout_s_raw is None:
            cell_timeout_s = None
        else:
            if not isinstance(cell_timeout_s_raw, int | float) or cell_timeout_s_raw <= 0:
                raise ValueError(
                    f"{source}: cell_timeout_s must be a positive number or null, "
                    f"got {cell_timeout_s_raw!r}"
                )
            cell_timeout_s = float(cell_timeout_s_raw)

        max_parallel = int(data.get("max_parallel", 1))
        if max_parallel != 1:
            # v1 contract: serial only. The arg exists so the YAML
            # schema is forward-compatible, but anything > 1 is a
            # config bug today.
            raise ValueError(
                f"{source}: max_parallel must be 1 in v1 (got {max_parallel}); "
                "parallel sweep is a future enhancement."
            )

        overrides_raw = data.get("overrides") or {}
        overrides = _validate_overrides(
            overrides_raw,
            policies=policies,
            envs=envs,
            source=f"{source}: overrides",
        )

        return cls(
            policies=policies,
            envs=envs,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            results_path=results_path,
            videos_dir=videos_dir,
            record_video=record_video,
            device=device,
            policies_yaml=policies_yaml,
            envs_yaml=envs_yaml,
            cell_timeout_s=cell_timeout_s,
            max_parallel=max_parallel,
            overrides=overrides,
        )


def _validate_str_list(value: Any, *, source: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{source}: must be a non-empty list of strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{source}: duplicates not allowed: {value!r}")
    return tuple(value)


def _validate_int_list(value: Any, *, source: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: must be a non-empty list of ints, got {value!r}")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in value):
        raise ValueError(f"{source}: must be a list of non-negative ints, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{source}: duplicates not allowed: {value!r}")
    return tuple(value)


def _validate_overrides(
    raw: Any,
    *,
    policies: Sequence[str],
    envs: Sequence[str],
    source: str,
) -> dict[str, dict[str, dict[str, int]]]:
    """Validate the ``overrides`` table and return a plain dict.

    Unknown policy / env names raise immediately -- a typo here would
    silently fall back to the base ``episodes_per_seed`` and waste a
    night re-running cells at the wrong shape.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: must be a mapping, got {type(raw).__name__}")

    policy_set = set(policies)
    env_set = set(envs)
    out: dict[str, dict[str, dict[str, int]]] = {}
    for policy_name, env_map in raw.items():
        if not isinstance(policy_name, str):
            raise ValueError(f"{source}: keys must be policy name strings, got {policy_name!r}")
        if policy_name not in policy_set:
            raise ValueError(
                f"{source}: policy '{policy_name}' not in sweep config "
                f"(known: {sorted(policy_set)})"
            )
        if not isinstance(env_map, dict):
            raise ValueError(
                f"{source}: '{policy_name}' must map to a dict of env overrides, "
                f"got {type(env_map).__name__}"
            )
        out_inner: dict[str, dict[str, int]] = {}
        for env_name, fields in env_map.items():
            if not isinstance(env_name, str):
                raise ValueError(f"{source}: '{policy_name}' has non-string env key {env_name!r}")
            if env_name not in env_set:
                raise ValueError(
                    f"{source}: '{policy_name}' overrides unknown env '{env_name}' "
                    f"(known: {sorted(env_set)})"
                )
            if not isinstance(fields, dict):
                raise ValueError(
                    f"{source}: '{policy_name}.{env_name}' must be a mapping with "
                    f"'n_episodes', got {type(fields).__name__}"
                )
            extras = set(fields) - {"n_episodes"}
            if extras:
                raise ValueError(
                    f"{source}: '{policy_name}.{env_name}' has unknown override fields: "
                    f"{sorted(extras)}"
                )
            n_episodes = fields.get("n_episodes")
            if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
                raise ValueError(
                    f"{source}: '{policy_name}.{env_name}.n_episodes' must be a positive int, "
                    f"got {n_episodes!r}"
                )
            out_inner[env_name] = {"n_episodes": int(n_episodes)}
        out[policy_name] = out_inner
    return out


def load_sweep_config(path: Path) -> SweepConfig:
    """Read + validate a sweep YAML. Raises :class:`ValueError` on schema problems."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return SweepConfig.from_dict(data, source=str(path))


# --------------------------------------------------------------------- #
# Cell planning                                                         #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlannedCell:
    """One ``(policy, env, seed)`` triple plus its resolved episode count.

    ``compatible`` is False when ``env`` is not in
    ``PolicySpec.env_compat`` for ``policy``. Incompatible cells are
    written into the manifest as ``skipped`` rather than dropped --
    transparency about why a cell did not run is part of the resume
    contract. ``runnable`` mirrors ``PolicySpec.is_runnable()`` so
    pre-Day-0a entries (revision_sha=null) are also visibly skipped
    instead of attempted-and-failed.
    """

    policy: str
    env: str
    seed_idx: int
    n_episodes: int
    compatible: bool
    runnable: bool

    @property
    def cell_key(self) -> CellKey:
        return CellKey(policy=self.policy, env=self.env, seed=self.seed_idx)

    @property
    def display(self) -> str:
        return f"{self.policy}/{self.env}/seed{self.seed_idx}"


def expand_cells(
    config: SweepConfig,
    *,
    policy_registry: PolicyRegistry,
    env_registry: EnvRegistry,
) -> list[PlannedCell]:
    """Cartesian product (policy x env x seed) -> sorted list of PlannedCell.

    Sort order: ``(policy_name, env_name, seed_idx)`` -- deterministic,
    reproducible, and groups all of one policy's cells together so the
    operator tailing the log sees a coherent narrative.

    Per-cell episode count comes from
    ``config.overrides[policy][env]["n_episodes"]`` if present, else
    ``config.episodes_per_seed``.

    Compatibility / runnability are decided here so the manifest can
    record ``skipped`` cells without the dispatch loop having to learn
    about registries.
    """
    cells: list[PlannedCell] = []
    for policy_name in sorted(config.policies):
        try:
            policy_spec = policy_registry.get(policy_name)
        except KeyError as exc:
            raise ValueError(
                f"sweep config references unknown policy '{policy_name}'; "
                f"available: {policy_registry.names()}"
            ) from exc
        compat = set(policy_spec.env_compat)
        runnable = policy_spec.is_runnable()

        for env_name in sorted(config.envs):
            try:
                env_registry.get(env_name)
            except KeyError as exc:
                raise ValueError(
                    f"sweep config references unknown env '{env_name}'; "
                    f"available: {env_registry.names()}"
                ) from exc

            n_eps = config.episodes_per_seed
            override = config.overrides.get(policy_name, {}).get(env_name)
            if override is not None:
                n_eps = override["n_episodes"]

            is_compat = env_name in compat

            for seed_idx in sorted(config.seeds):
                cells.append(
                    PlannedCell(
                        policy=policy_name,
                        env=env_name,
                        seed_idx=seed_idx,
                        n_episodes=n_eps,
                        compatible=is_compat,
                        runnable=runnable,
                    )
                )
    return cells


# --------------------------------------------------------------------- #
# Manifest                                                              #
# --------------------------------------------------------------------- #


@dataclass
class CellManifestEntry:
    """One row in the sweep manifest. Mutable -- updated as the cell finishes.

    ``status`` transitions:

      * ``pending``    -> initial state for fresh cells.
      * ``completed``  -> dispatch returned 0 (or 2 with rows still appended).
      * ``failed``     -> dispatch returned non-zero pre-flight code.
      * ``skipped``    -> incompatible cell, OR already completed before
                          the sweep started (pre-resumed), OR explicitly
                          dropped (e.g. policy not runnable).

    ``stderr_tail`` is the last :data:`STDERR_TAIL_LINES` lines of
    stderr from the failed subprocess, joined with ``\\n``. Empty
    string when no failure has occurred. The full stderr is NOT
    persisted -- the manifest is meant to fit on a screen, and the
    operator can re-run the resume command for full output.
    """

    policy: str
    env: str
    seed_idx: int
    n_episodes: int
    status: str
    exit_code: int | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    stderr_tail: str = ""
    skip_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepManifest:
    """Top-level manifest written to ``<results_path>.parent / sweep_manifest.json``."""

    started_utc: str
    code_sha: str
    lerobot_version: str
    config_path: str
    cells: list[CellManifestEntry] = field(default_factory=list)
    finished_utc: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "code_sha": self.code_sha,
            "lerobot_version": self.lerobot_version,
            "config_path": self.config_path,
            "cells": [c.to_json() for c in self.cells],
        }


def manifest_path_for(results_path: Path) -> Path:
    """Where to put the manifest given the sweep's parquet path."""
    return results_path.parent / "sweep_manifest.json"


def write_manifest(manifest: SweepManifest, path: Path) -> None:
    """Atomically write the manifest. Tmp-sibling + os.replace, like checkpointing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    try:
        tmp_path.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("failed to clean up tmp manifest at %s", tmp_path)
        raise


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _git_sha() -> str:
    """Best-effort git SHA. Mirrors the pattern in :mod:`scripts.run_one`."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _lerobot_version() -> str:
    """Lazy: ``lerobot.__version__`` or ``"unknown"`` if not importable.

    Imported inside the function so importing this module does not
    require lerobot. Kept consistent with ``calibrate._lerobot_version``
    -- "unknown" sentinel rather than None because the manifest field
    is typed as ``str``.
    """
    try:
        import lerobot
    except ImportError:
        return "unknown"
    version = getattr(lerobot, "__version__", None)
    return str(version) if version is not None else "unknown"


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _tail_lines(text: str, n: int) -> str:
    """Return the last ``n`` lines of ``text``, joined with newlines.

    Used to keep the manifest's ``stderr_tail`` bounded -- 200 lines is
    enough to read the traceback, not enough to bloat the manifest if
    every cell crashes.
    """
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def _build_run_one_argv(
    cell: PlannedCell,
    *,
    config: SweepConfig,
    python_executable: str | None = None,
) -> list[str]:
    """Build the argv that dispatches one cell via ``scripts/run_one.py``.

    Mirrors the CLI surface in :mod:`scripts.run_one`. The script path
    is computed relative to this file so the sweep works from any cwd.

    ``python_executable`` defaults to ``sys.executable`` so the
    subprocess inherits the current venv (important on the dev box
    where lerobot is in a conda env, not the system python).
    """
    py = python_executable or sys.executable
    repo_root = Path(__file__).resolve().parent.parent
    run_one = repo_root / "scripts" / "run_one.py"
    argv: list[str] = [
        py,
        str(run_one),
        "--policy",
        cell.policy,
        "--env",
        cell.env,
        "--seed",
        str(cell.seed_idx),
        "--n-episodes",
        str(cell.n_episodes),
        "--out-parquet",
        str(config.results_path),
        "--policies-yaml",
        str(config.policies_yaml),
        "--envs-yaml",
        str(config.envs_yaml),
        "--device",
        config.device,
    ]
    if config.videos_dir is not None:
        argv += ["--videos-dir", str(config.videos_dir)]
    if not config.record_video:
        argv.append("--no-record-video")
    return argv


# --------------------------------------------------------------------- #
# Outcome dataclass                                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepOutcome:
    """What :func:`run_sweep` returns. Mirrors :class:`scripts.run_one.RunOneOutcome`."""

    exit_code: int
    n_planned: int
    n_completed: int
    n_failed: int
    n_skipped: int
    n_already_done: int  # cells skipped because resume found them complete
    manifest_path: Path
    log_message: str


# --------------------------------------------------------------------- #
# Orchestration                                                         #
# --------------------------------------------------------------------- #


def _classify_for_dispatch(
    cells: list[PlannedCell],
    *,
    results_path: Path,
) -> tuple[
    list[PlannedCell],  # to_dispatch (pending + partial)
    set[CellKey],  # completed_cells from prior run
    set[CellKey],  # partial_cells (will be dropped from parquet first)
    list[PlannedCell],  # skipped_at_plan (incompat / not runnable)
]:
    """Combine config-level skips with checkpointing-level resume info.

    Cells flagged ``compatible=False`` or ``runnable=False`` at plan
    time are skipped *before* asking checkpointing about them -- they
    must never end up in the parquet and shouldn't pollute the resume
    classification. Of the remaining cells, ``plan_resume`` decides
    completed / partial / pending; partials are added to ``to_dispatch``
    after the parquet is cleaned.
    """
    skipped_at_plan: list[PlannedCell] = []
    runnable_cells: list[PlannedCell] = []
    for cell in cells:
        if cell.compatible and cell.runnable:
            runnable_cells.append(cell)
        else:
            skipped_at_plan.append(cell)

    if not runnable_cells:
        return [], set(), set(), skipped_at_plan

    # All runnable cells share the same n_episodes per (policy, env)
    # pair (overrides are applied at expansion time, not per seed), so
    # we batch the resume call per-(policy, env) to honour the
    # n_episodes contract on plan_resume.
    by_n: dict[int, list[PlannedCell]] = {}
    for cell in runnable_cells:
        by_n.setdefault(cell.n_episodes, []).append(cell)

    completed_keys: set[CellKey] = set()
    partial_keys: set[CellKey] = set()
    for n_eps, batch in by_n.items():
        plan = plan_resume(
            results_path,
            requested_cells=[c.cell_key for c in batch],
            n_episodes=n_eps,
        )
        completed_keys |= plan.completed_cells
        partial_keys |= plan.partial_cells

    to_dispatch = [c for c in runnable_cells if c.cell_key not in completed_keys]
    return to_dispatch, completed_keys, partial_keys, skipped_at_plan


def _build_initial_manifest(
    *,
    config_path: Path,
    cells: list[PlannedCell],
    completed_keys: set[CellKey],
    skipped_at_plan: list[PlannedCell],
) -> SweepManifest:
    """Build the manifest with every planned cell stamped at its initial status.

    Ordering: same as ``cells`` (sorted at expand time, possibly
    shuffled by the caller). The manifest order is the dispatch order
    -- skimming the JSON shows what's done, what's next, what failed,
    in chronological dispatch order.
    """
    skipped_set = {(c.policy, c.env, c.seed_idx) for c in skipped_at_plan}
    started_utc = _now_utc()
    manifest = SweepManifest(
        started_utc=started_utc,
        code_sha=_git_sha(),
        lerobot_version=_lerobot_version(),
        config_path=str(config_path),
    )
    for cell in cells:
        triple = (cell.policy, cell.env, cell.seed_idx)
        if triple in skipped_set:
            # Plan-time skip: log why so the operator doesn't think
            # the cell silently failed.
            if not cell.runnable:
                reason = "policy not runnable (revision_sha not locked)"
            else:
                reason = f"env '{cell.env}' not in policy '{cell.policy}'.env_compat"
            entry = CellManifestEntry(
                policy=cell.policy,
                env=cell.env,
                seed_idx=cell.seed_idx,
                n_episodes=cell.n_episodes,
                status=STATUS_SKIPPED,
                skip_reason=reason,
            )
        elif cell.cell_key in completed_keys:
            # Resume-time skip: cell already has full row set.
            entry = CellManifestEntry(
                policy=cell.policy,
                env=cell.env,
                seed_idx=cell.seed_idx,
                n_episodes=cell.n_episodes,
                status=STATUS_COMPLETED,
                exit_code=0,
                skip_reason="already in results.parquet (resumed)",
            )
        else:
            entry = CellManifestEntry(
                policy=cell.policy,
                env=cell.env,
                seed_idx=cell.seed_idx,
                n_episodes=cell.n_episodes,
                status=STATUS_PENDING,
            )
        manifest.cells.append(entry)
    return manifest


def _find_entry(manifest: SweepManifest, cell: PlannedCell) -> CellManifestEntry:
    """Return the first manifest entry matching this cell's triple. Raises if missing."""
    for entry in manifest.cells:
        if (
            entry.policy == cell.policy
            and entry.env == cell.env
            and entry.seed_idx == cell.seed_idx
        ):
            return entry
    raise KeyError(f"no manifest entry for {cell.display}")


def run_sweep(
    *,
    config: SweepConfig,
    config_path: Path,
    policies_yaml: Path | None = None,
    envs_yaml: Path | None = None,
    max_cells: int | None = None,
    shuffle_seed: int | None = None,
    cell_timeout_s: float | None = None,
    dry_run: bool = False,
) -> SweepOutcome:
    """End-to-end orchestration. Mirrors :func:`scripts.run_one.run_one` in shape.

    Steps:

    1. Load both registries (uses YAML paths from ``config`` unless
       overridden by the kwargs -- the kwargs let the CLI swap configs
       without forcing a YAML edit).
    2. Expand the cartesian product (policy x env x seed) into a sorted
       :class:`PlannedCell` list. Apply ``--shuffle SEED`` and
       ``--max-cells`` here so resume and manifest see the same view.
    3. Decide skip/dispatch via :func:`plan_resume`. Drop partial cells
       from the parquet (idempotent).
    4. Build + write the initial manifest.
    5. (Unless dry-run) Dispatch each pending cell via
       :data:`_run_subprocess`, updating the manifest atomically after
       each.

    The ``cell_timeout_s`` kwarg overrides the YAML's value -- CLI
    wins. Both are soft; see :func:`_default_run_subprocess`.
    """
    p_yaml = policies_yaml if policies_yaml is not None else config.policies_yaml
    e_yaml = envs_yaml if envs_yaml is not None else config.envs_yaml
    policy_registry = PolicyRegistry.from_yaml(p_yaml)
    env_registry = EnvRegistry.from_yaml(e_yaml)

    # 2. expand
    cells = expand_cells(config, policy_registry=policy_registry, env_registry=env_registry)
    if not cells:
        return SweepOutcome(
            exit_code=4,
            n_planned=0,
            n_completed=0,
            n_failed=0,
            n_skipped=0,
            n_already_done=0,
            manifest_path=manifest_path_for(config.results_path),
            log_message="[run-sweep] aborted: cartesian product produced zero cells",
        )

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(cells)

    if max_cells is not None:
        if max_cells <= 0:
            raise ValueError(f"--max-cells must be positive, got {max_cells}")
        cells = cells[:max_cells]

    # 3. classify (results-path conflict surfaces here)
    try:
        to_dispatch, completed_keys, partial_keys, skipped_at_plan = _classify_for_dispatch(
            cells, results_path=config.results_path
        )
    except ValueError as exc:
        # load_results raises ValueError on schema mismatch -- treat as exit 5.
        return SweepOutcome(
            exit_code=5,
            n_planned=len(cells),
            n_completed=0,
            n_failed=0,
            n_skipped=0,
            n_already_done=0,
            manifest_path=manifest_path_for(config.results_path),
            log_message=f"[run-sweep] aborted: results path conflict: {exc}",
        )

    # 3b. drop any partial cells from the parquet so the dispatched
    # restart writes clean rows.
    if partial_keys:
        n_dropped = drop_partial_cells(config.results_path, partial_keys)
        logger.info(
            "dropped %d row(s) from %d partial cell(s) before resume",
            n_dropped,
            len(partial_keys),
        )

    # 4. write the manifest with every cell at its initial status.
    manifest = _build_initial_manifest(
        config_path=config_path,
        cells=cells,
        completed_keys=completed_keys,
        skipped_at_plan=skipped_at_plan,
    )
    manifest_path = manifest_path_for(config.results_path)
    write_manifest(manifest, manifest_path)

    n_already_done = len(completed_keys)
    n_skipped_plan = len(skipped_at_plan)

    if dry_run:
        # No subprocess fires. The manifest sits on disk so the
        # operator can grep through what *would* run.
        log = (
            f"[run-sweep] dry-run: planned {len(cells)} cells "
            f"(dispatch={len(to_dispatch)}, already_done={n_already_done}, "
            f"skipped={n_skipped_plan}); manifest at {manifest_path}"
        )
        return SweepOutcome(
            exit_code=0,
            n_planned=len(cells),
            n_completed=0,
            n_failed=0,
            n_skipped=n_skipped_plan,
            n_already_done=n_already_done,
            manifest_path=manifest_path,
            log_message=log,
        )

    if not to_dispatch:
        # Everything was already done or skipped -- valid no-op resume.
        manifest = replace(manifest, finished_utc=_now_utc())
        write_manifest(manifest, manifest_path)
        log = (
            f"[run-sweep] no-op: {n_already_done} cell(s) already complete, "
            f"{n_skipped_plan} skipped; nothing to dispatch"
        )
        return SweepOutcome(
            exit_code=0,
            n_planned=len(cells),
            n_completed=0,
            n_failed=0,
            n_skipped=n_skipped_plan,
            n_already_done=n_already_done,
            manifest_path=manifest_path,
            log_message=log,
        )

    # 5. dispatch loop
    timeout = cell_timeout_s if cell_timeout_s is not None else config.cell_timeout_s
    n_completed = 0
    n_failed = 0

    for i, cell in enumerate(to_dispatch, start=1):
        argv = _build_run_one_argv(cell, config=config)
        entry = _find_entry(manifest, cell)
        entry.started_utc = _now_utc()
        entry.status = STATUS_PENDING  # explicit -- in case the entry was reused
        write_manifest(manifest, manifest_path)

        logger.info(
            "[%d/%d] dispatch %s (n_episodes=%d, timeout_s=%s)",
            i,
            len(to_dispatch),
            cell.display,
            cell.n_episodes,
            timeout,
        )

        try:
            outcome = _run_subprocess(argv, timeout_s=timeout)
        except KeyboardInterrupt:
            # Mid-cell SIGINT: leave this cell at "pending" so the next
            # resume picks it up cleanly. Earlier cells stay at
            # "completed". Stamp finished_utc on the manifest so the
            # operator can see when the sweep was interrupted.
            manifest = replace(manifest, finished_utc=_now_utc())
            write_manifest(manifest, manifest_path)
            log = (
                f"[run-sweep] interrupted at cell {i}/{len(to_dispatch)} ({cell.display}); "
                f"resume: python scripts/run_sweep.py --config {config_path} --resume"
            )
            return SweepOutcome(
                exit_code=2,
                n_planned=len(cells),
                n_completed=n_completed,
                n_failed=n_failed,
                n_skipped=n_skipped_plan,
                n_already_done=n_already_done,
                manifest_path=manifest_path,
                log_message=log,
            )

        entry.exit_code = outcome.returncode
        entry.finished_utc = _now_utc()
        # exit 0 = full success; exit 2 = some episode errors but rows
        # still appended -- both are "completed" from the sweep's POV.
        if outcome.returncode in {0, 2}:
            entry.status = STATUS_COMPLETED
            entry.stderr_tail = ""
            n_completed += 1
            logger.info("    -> ok (exit=%d)", outcome.returncode)
        else:
            entry.status = STATUS_FAILED
            entry.stderr_tail = _tail_lines(outcome.stderr, STDERR_TAIL_LINES)
            n_failed += 1
            logger.warning(
                "    -> FAILED (exit=%d). Continuing to next cell. "
                "Resume this cell with: python scripts/run_one.py "
                "--policy %s --env %s --seed %d --n-episodes %d",
                outcome.returncode,
                cell.policy,
                cell.env,
                cell.seed_idx,
                cell.n_episodes,
            )

        write_manifest(manifest, manifest_path)

    # All cells dispatched.
    manifest = replace(manifest, finished_utc=_now_utc())
    write_manifest(manifest, manifest_path)

    exit_code = 2 if n_failed > 0 else 0
    log = (
        f"[run-sweep] done: planned={len(cells)} dispatched={len(to_dispatch)} "
        f"completed={n_completed} failed={n_failed} skipped={n_skipped_plan} "
        f"already_done={n_already_done} manifest={manifest_path}"
    )
    return SweepOutcome(
        exit_code=exit_code,
        n_planned=len(cells),
        n_completed=n_completed,
        n_failed=n_failed,
        n_skipped=n_skipped_plan,
        n_already_done=n_already_done,
        manifest_path=manifest_path,
        log_message=log,
    )


# --------------------------------------------------------------------- #
# Pre-flight: validate against existing parquet                         #
# --------------------------------------------------------------------- #


def _preflight_results_path(results_path: Path) -> str | None:
    """Cheap schema check against ``results_path`` before planning anything.

    Returns an error message if the file exists but its columns don't
    match :data:`RESULT_SCHEMA`. ``None`` means safe to proceed
    (file may or may not exist).
    """
    if not results_path.exists():
        return None
    try:
        load_results(results_path)
    except ValueError as exc:
        return str(exc)
    return None


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-sweep",
        description=(
            "Run the full benchmark matrix. Dispatches one scripts/run_one.py "
            "subprocess per (policy, env, seed) cell, writes results.parquet "
            "incrementally and a sweep_manifest.json that survives kill -9."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to sweep YAML (e.g. configs/sweep_full.yaml).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Override the YAML's results_path.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Premortem #3: cap total dispatched cells (debug / partial sweep).",
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="Shuffle the cell list with this seed (round-robin balance).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan + write manifest, do NOT dispatch any subprocesses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Explicit resume flag. Default behavior is already resume-aware "
            "via plan_resume(); this flag exists for operator clarity."
        ),
    )
    parser.add_argument(
        "--cell-timeout-s",
        type=float,
        default=None,
        help="Soft per-cell timeout in seconds (overrides YAML; warn-only, no kill).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="v1: must be 1 (serial dispatch). Param exists for forward compat.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.max_parallel != 1:
        print(
            f"[run-sweep] aborted: --max-parallel must be 1 in v1 (got {args.max_parallel})",
            file=sys.stderr,
        )
        return 3

    # Load + validate config first -- any schema problem exits 3.
    try:
        config = load_sweep_config(args.config)
    except FileNotFoundError as exc:
        print(f"[run-sweep] aborted: config not found: {exc}", file=sys.stderr)
        print(
            "[run-sweep] resume: python scripts/run_sweep.py --config <path>",
            file=sys.stderr,
        )
        return 3
    except (ValueError, yaml.YAMLError) as exc:
        print(f"[run-sweep] aborted: invalid config {args.config}: {exc}", file=sys.stderr)
        return 3

    if args.results_path is not None:
        config = replace(config, results_path=args.results_path)

    # Schema check on existing parquet (fast, before expanding cells).
    err = _preflight_results_path(config.results_path)
    if err is not None:
        print(f"[run-sweep] aborted: results path conflict: {err}", file=sys.stderr)
        print(
            f"[run-sweep] resume: move/delete {config.results_path} or fix its columns, then re-run",
            file=sys.stderr,
        )
        return 5

    try:
        outcome = run_sweep(
            config=config,
            config_path=args.config,
            max_cells=args.max_cells,
            shuffle_seed=args.shuffle,
            cell_timeout_s=args.cell_timeout_s,
            dry_run=args.dry_run,
        )
    except (KeyError, ValueError) as exc:
        # expand_cells raises ValueError for unknown policy/env names.
        print(f"[run-sweep] aborted: {exc}", file=sys.stderr)
        return 3

    stream = sys.stdout if outcome.exit_code in {0, 2} else sys.stderr
    print(outcome.log_message, file=stream)
    if outcome.exit_code == 2 and outcome.n_failed > 0:
        # One-line resume hint for the operator.
        print(
            f"[run-sweep] resume: python scripts/run_sweep.py --config {args.config} --resume "
            f"# manifest: {outcome.manifest_path}",
            file=sys.stderr,
        )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
