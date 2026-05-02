#!/usr/bin/env python3
"""Run a single (policy, env, seed) cell and append result rows to a parquet file.

This is the building block ``scripts/run_sweep.py`` dispatches to. It also
doubles as a spot-check tool — useful for re-running a single OOMed cell
after the sweep, or for poking at one row before the matrix runs overnight.

The orchestration mirrors :mod:`scripts.calibrate` style for consistency:
lazy-imports for ``torch`` / ``lerobot`` / ``lerobot_bench.render``, an
exit-code-driven CLI, and a pure-orchestration :func:`run_one` function
that the tests can drive without touching real lerobot/torch.

**On error capture**: rows produced by :meth:`CellResult.to_rows` follow
``RESULT_SCHEMA`` exactly — there is no ``error`` column. Per-episode
exceptions captured by :func:`lerobot_bench.eval.run_cell` show up as
``success=False`` rows with zeroed timing fields; the human-readable
error string is written to the operator log line only, not to the
parquet. This matches the leaderboard schema (one row per episode,
boolean success) and is documented in DESIGN.md § Methodology.

**On mid-cell resume**: this script runs ONE cell. It is atomic at the
cell granularity — either every episode's row lands in the parquet, or
nothing does. If the process dies mid-cell, ``run_sweep.py`` is
responsible for noticing the cell is partial and re-running it from
episode 0 (the seeding contract makes mid-cell resume non-bit-identical).

Usage::

    python scripts/run_one.py --policy diffusion_policy --env pusht --seed 0
    python scripts/run_one.py --policy random --env pusht --seed 2 --n-episodes 10
    python scripts/run_one.py --policy no_op --env pusht --seed 0 \\
        --out-parquet results/sweep-test/results.parquet \\
        --videos-dir results/sweep-test/videos \\
        --no-record-video
    python scripts/run_one.py --policy random --env pusht --seed 0 --dry-run

Exit codes:
    0  success — rows appended (or would be, in --dry-run)
    2  cell ran but with errors in some episodes — rows still appended
    3  policy not runnable (Day 0a TODO: lock revision_sha)
    4  missing runtime (lerobot or sim extras not installed)
    5  policy/env not in registry, or env not in policy.env_compat
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot_bench.checkpointing import append_cell_rows
from lerobot_bench.envs import EnvRegistry, EnvSpec
from lerobot_bench.policies import PolicyRegistry, PolicySpec

if TYPE_CHECKING:
    from lerobot_bench.eval import CellResult

logger = logging.getLogger("run-one")

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_N_EPISODES = 50
DEFAULT_DEVICE = "cuda"
DEFAULT_OUT_PARQUET = Path("results/results.parquet")
DEFAULT_VIDEOS_DIR = Path("results/videos")
DEFAULT_POLICIES_YAML = Path("configs/policies.yaml")
DEFAULT_ENVS_YAML = Path("configs/envs.yaml")


# --------------------------------------------------------------------- #
# Outcome dataclass                                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunOneOutcome:
    """What :func:`main` observed before exiting.

    Returned by :func:`run_one` so tests can assert on the orchestration
    decisions (rows appended, exit code, log line) without re-parsing
    stdout/stderr. ``out_parquet`` and ``videos_dir`` are ``None`` for
    pre-flight failures where no work was done.
    """

    exit_code: int
    cell_key: str  # f"{policy}/{env}/seed{seed}"
    n_episodes_attempted: int
    n_episodes_succeeded: int
    n_episodes_errored: int
    n_rows_appended: int
    out_parquet: Path | None
    videos_dir: Path | None
    log_message: str  # human-readable one-liner for stdout/stderr


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _git_sha() -> str:
    """Best-effort git SHA of the working tree. Returns ``"unknown"`` on failure.

    Copy-pasted from :mod:`scripts.calibrate` (one-line helper) to keep
    this script independent of any cross-script utility module.
    """
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


def _cell_key(policy: str, env: str, seed: int) -> str:
    return f"{policy}/{env}/seed{seed}"


# --------------------------------------------------------------------- #
# Spec resolution                                                       #
# --------------------------------------------------------------------- #


def resolve_specs(
    policy_name: str,
    env_name: str,
    *,
    policies_yaml: Path,
    envs_yaml: Path,
) -> tuple[PolicySpec, EnvSpec]:
    """Load both registries, look up by name, validate compat.

    Raises:
        KeyError: if either name is not registered. The registry's own
            ``KeyError`` carries the ``available: ...`` list, which is
            re-raised verbatim.
        ValueError: if ``env_name`` is not in ``policy.env_compat``.
            Message lists the policy's full compat tuple so the operator
            can pick a valid env without grep-ing the YAML.
    """
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(envs_yaml)

    policy = policies.get(policy_name)  # KeyError on miss
    env = envs.get(env_name)  # KeyError on miss

    if env_name not in policy.env_compat:
        compat = ", ".join(policy.env_compat) if policy.env_compat else "<empty>"
        raise ValueError(
            f"policy '{policy_name}' does not support env '{env_name}'; supports: {compat}"
        )

    return policy, env


# --------------------------------------------------------------------- #
# Video rendering                                                       #
# --------------------------------------------------------------------- #


def render_episodes_to_videos(
    cell_result: CellResult,
    *,
    videos_dir: Path,
) -> list[str]:
    """Render each episode's frames to MP4 and return the per-episode SHA list.

    Naming: ``f"{policy}__{env}__seed{seed}__ep{idx:03d}.mp4"``.

    Returns a list of sha256 hex strings parallel to ``cell_result.episodes``.
    Episodes that errored OR have zero frames produce an empty string —
    the ``video_sha256`` column is left blank for those rows so the
    leaderboard knows there is no clip to link to.

    The render module is imported lazily inside this function so the
    dry-run path stays imageio-free (and importing :mod:`scripts.run_one`
    in CI does not require imageio's ffmpeg binary).
    """
    import numpy as np

    from lerobot_bench.render import render_episode

    videos_dir.mkdir(parents=True, exist_ok=True)

    sha_list: list[str] = []
    for ep in cell_result.episodes:
        if ep.error is not None or len(ep.frames) == 0:
            sha_list.append("")
            continue

        out_path = videos_dir / (
            f"{cell_result.policy}__{cell_result.env}__seed{cell_result.seed}"
            f"__ep{ep.episode_index:03d}.mp4"
        )
        # Stack tuple-of-frames into a (T, H, W, 3) array for the encoder.
        stacked = np.stack(ep.frames, axis=0)
        result = render_episode(stacked, out_path)
        sha_list.append(result.content_sha256)

    return sha_list


# --------------------------------------------------------------------- #
# Pre-flight checks                                                     #
# --------------------------------------------------------------------- #


def _check_lerobot_available() -> str | None:
    """Return None if lerobot imports cleanly, else a one-line error string.

    Lazy-imported inside the function so importing :mod:`scripts.run_one`
    never requires lerobot.
    """
    try:
        import lerobot  # noqa: F401
    except ImportError as exc:
        return f"missing runtime: {exc}"
    return None


# --------------------------------------------------------------------- #
# Orchestration                                                         #
# --------------------------------------------------------------------- #


def run_one(
    *,
    policy_name: str,
    env_name: str,
    seed: int,
    n_episodes: int,
    out_parquet: Path,
    videos_dir: Path,
    record_video: bool,
    device: str,
    policies_yaml: Path,
    envs_yaml: Path,
    dry_run: bool,
) -> RunOneOutcome:
    """Run one cell end-to-end. Pure orchestration; lazy-imports torch/lerobot.

    Pre-flight order (fail-fast, distinct exit codes):

    1. Spec resolution (KeyError → exit 5).
    2. Env compat (ValueError → exit 5).
    3. Policy runnability (RuntimeError → exit 3).
    4. Lerobot importable (ImportError → exit 4).

    Then, if ``dry_run`` is True, return immediately with exit 0 — no
    torch import, no eval call, no parquet write.

    Otherwise: call :func:`lerobot_bench.eval.run_cell_from_specs`,
    optionally render videos, append rows atomically. If any episode
    errored, the exit code is 2 (rows still appended).

    The ``device`` argument is forwarded to ``run_cell_from_specs`` and
    is NOT pre-validated here — let CUDA fail with its own runtime
    error if it is not available; pre-checking would force a torch
    import outside the dry-run guard.
    """
    cell_key = _cell_key(policy_name, env_name, seed)

    # 1 + 2: spec resolution + compat. Both surface as exit 5.
    try:
        policy_spec, env_spec = resolve_specs(
            policy_name,
            env_name,
            policies_yaml=policies_yaml,
            envs_yaml=envs_yaml,
        )
    except (KeyError, ValueError) as exc:
        msg = str(exc).strip("'\"")
        return RunOneOutcome(
            exit_code=5,
            cell_key=cell_key,
            n_episodes_attempted=0,
            n_episodes_succeeded=0,
            n_episodes_errored=0,
            n_rows_appended=0,
            out_parquet=None,
            videos_dir=None,
            log_message=f"[run-one] aborted: {msg}",
        )

    # 3: runnability (revision_sha lock). Skip in dry-run so the operator
    # can still exercise the planner without Day 0a in place.
    if not policy_spec.is_runnable() and not dry_run:
        return RunOneOutcome(
            exit_code=3,
            cell_key=cell_key,
            n_episodes_attempted=0,
            n_episodes_succeeded=0,
            n_episodes_errored=0,
            n_rows_appended=0,
            out_parquet=None,
            videos_dir=None,
            log_message=(
                f"[run-one] aborted: policy '{policy_name}' is not runnable -- "
                "missing revision_sha (Day 0a TODO: lock SHA in configs/policies.yaml)"
            ),
        )

    # Dry-run short-circuit: BEFORE any torch / lerobot import so the
    # AST contract holds and the planner can run without GPU work.
    if dry_run:
        log = (
            f"[run-one] dry-run: would run {cell_key} "
            f"({n_episodes} episodes, record_video={record_video}, device={device})"
        )
        return RunOneOutcome(
            exit_code=0,
            cell_key=cell_key,
            n_episodes_attempted=0,
            n_episodes_succeeded=0,
            n_episodes_errored=0,
            n_rows_appended=0,
            out_parquet=None,
            videos_dir=None,
            log_message=log,
        )

    # 4: lerobot importable.
    err = _check_lerobot_available()
    if err is not None:
        return RunOneOutcome(
            exit_code=4,
            cell_key=cell_key,
            n_episodes_attempted=0,
            n_episodes_succeeded=0,
            n_episodes_errored=0,
            n_rows_appended=0,
            out_parquet=None,
            videos_dir=None,
            log_message=(
                f"[run-one] aborted: {err}. Install: pip install -e /home/theo/projects/lerobot"
            ),
        )

    # Cell execution. Lazy-import eval here so the dry-run path stays
    # torch-free (eval imports torch lazily but its module-level
    # `import pandas` is already cheap).
    from lerobot_bench import eval as eval_mod

    cell_result = eval_mod.run_cell_from_specs(
        policy_spec,
        env_spec,
        seed_idx=seed,
        n_episodes=n_episodes,
        device=device,
        record_video=record_video,
    )

    # Optional video render BEFORE building rows so video_sha256 is filled in.
    video_sha: list[str] | None = None
    if record_video:
        video_sha = render_episodes_to_videos(cell_result, videos_dir=videos_dir)

    df = cell_result.to_rows(video_sha256_per_episode=video_sha)

    # Atomic append. mkdir parents so a fresh output path Just Works.
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    n_total_rows_after = append_cell_rows(out_parquet, df)

    n_attempted = len(cell_result.episodes)
    n_errored = sum(1 for ep in cell_result.episodes if ep.error is not None)
    n_succeeded = sum(1 for ep in cell_result.episodes if ep.success)

    if n_errored > 0:
        exit_code = 2
        log = (
            f"[run-one] policy={policy_name} env={env_name} seed={seed} "
            f"eps={n_attempted} success={n_succeeded}/{n_attempted} "
            f"errors={n_errored} rows_appended={n_attempted} "
            f"(cell completed with errors; total_rows={n_total_rows_after})"
        )
    else:
        exit_code = 0
        log = (
            f"[run-one] policy={policy_name} env={env_name} seed={seed} "
            f"eps={n_attempted} success={n_succeeded}/{n_attempted} "
            f"rows_appended={n_attempted} out={out_parquet}"
        )

    return RunOneOutcome(
        exit_code=exit_code,
        cell_key=cell_key,
        n_episodes_attempted=n_attempted,
        n_episodes_succeeded=n_succeeded,
        n_episodes_errored=n_errored,
        n_rows_appended=n_attempted,
        out_parquet=out_parquet,
        videos_dir=videos_dir if record_video else None,
        log_message=log,
    )


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-one",
        description=(
            "Run a single (policy, env, seed) cell and append its rows to "
            "results.parquet. The building block scripts/run_sweep.py "
            "dispatches to."
        ),
    )
    parser.add_argument("--policy", type=str, required=True, help="Policy name from the registry.")
    parser.add_argument("--env", type=str, required=True, help="Env name from the registry.")
    parser.add_argument("--seed", type=int, required=True, help="Seed index (>= 0).")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Episodes to run for this cell (default: {DEFAULT_N_EPISODES}).",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=DEFAULT_OUT_PARQUET,
        help=f"Parquet file to append rows to (default: {DEFAULT_OUT_PARQUET}).",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory for rendered MP4s (default: {DEFAULT_VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Skip the per-episode MP4 render entirely (video_sha256 stays empty).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device for the policy (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--policies-yaml",
        type=Path,
        default=DEFAULT_POLICIES_YAML,
        help="Path to policy registry YAML (default: configs/policies.yaml).",
    )
    parser.add_argument(
        "--envs-yaml",
        type=Path,
        default=DEFAULT_ENVS_YAML,
        help="Path to env registry YAML (default: configs/envs.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve specs + log the plan; do NOT import torch/lerobot or write rows.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        outcome = run_one(
            policy_name=args.policy,
            env_name=args.env,
            seed=args.seed,
            n_episodes=args.n_episodes,
            out_parquet=args.out_parquet,
            videos_dir=args.videos_dir,
            record_video=not args.no_record_video,
            device=args.device,
            policies_yaml=args.policies_yaml,
            envs_yaml=args.envs_yaml,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        # Missing yaml is operator error -- treat as runtime-missing.
        print(f"[run-one] aborted: config not found: {exc}", file=sys.stderr)
        print(
            "[run-one] resume: python scripts/run_one.py "
            "--policies-yaml <path> --envs-yaml <path> ...",
            file=sys.stderr,
        )
        return 4

    # Route the log line to stdout (ok / partial) or stderr (pre-flight fail).
    stream = sys.stdout if outcome.exit_code in {0, 2} else sys.stderr
    print(outcome.log_message, file=stream)

    if outcome.exit_code not in {0, 2}:
        # Resume hint: the exact command that re-runs this cell.
        print(
            f"[run-one] resume: python scripts/run_one.py "
            f"--policy {args.policy} --env {args.env} --seed {args.seed} "
            f"--n-episodes {args.n_episodes}",
            file=sys.stderr,
        )

    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
