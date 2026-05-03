#!/usr/bin/env python3
"""Calibration spike: 20 steps x 1 episode per (policy, env) cell.

Outputs ``results/calibration-YYYYMMDD.json`` with per-cell timing +
VRAM stats. The :func:`auto_downscope` rule in this module produces a
recommended matrix shape (seeds, episodes) that gets pasted into
``configs/sweep_full.yaml`` after Day 0b.

Usage::

    python scripts/calibrate.py                          # every runnable (policy, env) cell
    python scripts/calibrate.py --policy diffusion_policy --env pusht
    python scripts/calibrate.py --steps 20 --episodes 1  # defaults shown
    python scripts/calibrate.py --dry-run                # plan + print, no torch import

Exit codes:
    0  success, output written
    2  partial success (some cells OOMed or errored) -- output still written
    3  nothing to calibrate (no runnable cells) -- output not written
    4  missing runtime (lerobot not installed, no GPU when required) -- output not written

Day 0b status: this is the **scaffold**. Plumbing (CLI, lazy import,
JSON shape, OOM catch, auto-downscope) is real and unit-tested. The
inner ~30-line measurement loop is marked TODO and currently returns
``status="error"`` so the scaffold can be CI-tested without lerobot
installed. Filled in on Day 0b after lerobot lock.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from lerobot_bench.envs import EnvRegistry, EnvSpec
from lerobot_bench.policies import PolicyRegistry, PolicySpec

logger = logging.getLogger("calibrate")

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_STEPS = 20
DEFAULT_EPISODES = 1
DEFAULT_OUT_DIR = Path("results")
DEFAULT_POLICIES_YAML = Path("configs/policies.yaml")
DEFAULT_ENVS_YAML = Path("configs/envs.yaml")

# Auto-downscope thresholds. Tuned for an 8 GB RTX 4060 (single laptop
# GPU) and the 8-day overnight budget in docs/CEO-PLAN.md. The rule
# trades episode/seed counts for fitting the full matrix inside the
# wall-clock window without OOMing. Re-tune after the first real
# calibration run produces concrete numbers.
SLOW_MS_PER_STEP_THRESHOLD = 100.0  # > 100 ms/step -> reduce episodes
VERY_SLOW_MS_PER_STEP_THRESHOLD = 500.0  # > 500 ms/step -> reduce seeds
HIGH_VRAM_THRESHOLD_MB = 5500.0  # > 5.5 GB -> reduce episodes (leave headroom)
VERY_HIGH_VRAM_THRESHOLD_MB = 7000.0  # > 7 GB -> reduce seeds (high OOM risk)

# Plan-level statuses returned by plan_cells (NOT the same vocabulary as
# CellTiming.status, which is measurement-level).
PLAN_READY = "ready"
PLAN_SKIPPED = "skipped"
PLAN_INCOMPAT = "incompat"


# --------------------------------------------------------------------- #
# Data classes                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellTiming:
    """One (policy, env) cell after calibration measurement.

    ``status`` is one of:
        ``"ok"``      -- measurement loop completed; timing fields valid.
        ``"oom"``     -- ``torch.cuda.OutOfMemoryError`` during measurement.
        ``"skipped"`` -- policy not runnable (pre-Day-0a or baseline gap);
                         timing fields are zero placeholders.
        ``"error"``   -- any other exception; ``error`` field has a short
                         truncated string. Includes "lerobot not installed"
                         until Day 0a.

    ``recommended`` is the auto-downscoped ``{"seeds": int, "episodes": int}``
    suggestion -- only populated when ``status == "ok"``. For non-ok cells
    callers should drop the cell from the matrix entirely.
    """

    policy: str
    env: str
    n_steps_measured: int
    mean_ms_per_step: float
    p95_ms_per_step: float
    vram_peak_mb: float
    status: str
    error: str | None = None
    recommended: dict[str, int] | None = None


@dataclass(frozen=True)
class CalibrationReport:
    """Top-level JSON shape for ``results/calibration-YYYYMMDD.json``."""

    timestamp_utc: str
    git_sha: str
    lerobot_version: str | None  # None if lerobot not importable
    cells: tuple[CellTiming, ...]

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict. Stable key order across cells."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "git_sha": self.git_sha,
            "lerobot_version": self.lerobot_version,
            "cells": [asdict(cell) for cell in self.cells],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CalibrationReport:
        cells = tuple(CellTiming(**cell) for cell in data.get("cells", []))
        return cls(
            timestamp_utc=str(data["timestamp_utc"]),
            git_sha=str(data["git_sha"]),
            lerobot_version=data.get("lerobot_version"),
            cells=cells,
        )


# --------------------------------------------------------------------- #
# Auto-downscope rule                                                   #
# --------------------------------------------------------------------- #


def auto_downscope(
    timing: CellTiming, *, base_seeds: int = 5, base_episodes: int = 50
) -> dict[str, int]:
    """Apply the downscope rule from DESIGN.md Methodology.

    Decision table (first match wins, both axes considered together):

    +------------------------------------------------+--------------------+
    | Condition                                      | Recommendation     |
    +------------------------------------------------+--------------------+
    | status != "ok"                                 | drop -> 0/0        |
    | vram > VERY_HIGH OR ms > VERY_SLOW             | seeds=2, episodes  |
    | vram > HIGH OR ms > SLOW                       | seeds, episodes=25 |
    | else                                           | base seeds/episodes|
    +------------------------------------------------+--------------------+
    """
    if timing.status != "ok":
        return {"seeds": 0, "episodes": 0}

    if (
        timing.vram_peak_mb > VERY_HIGH_VRAM_THRESHOLD_MB
        or timing.mean_ms_per_step > VERY_SLOW_MS_PER_STEP_THRESHOLD
    ):
        return {"seeds": 2, "episodes": base_episodes}

    if (
        timing.vram_peak_mb > HIGH_VRAM_THRESHOLD_MB
        or timing.mean_ms_per_step > SLOW_MS_PER_STEP_THRESHOLD
    ):
        return {"seeds": base_seeds, "episodes": 25}

    return {"seeds": base_seeds, "episodes": base_episodes}


# --------------------------------------------------------------------- #
# Planning                                                              #
# --------------------------------------------------------------------- #


def plan_cells(
    policies: PolicyRegistry,
    envs: EnvRegistry,
    *,
    policy_filter: str | None = None,
    env_filter: str | None = None,
) -> list[tuple[PolicySpec, EnvSpec, str]]:
    """Cross-product (policy, env) with per-cell status.

    Returned status is one of:
        ``"ready"``    -- runnable, env is in policy.env_compat -- will be measured.
        ``"skipped"``  -- policy.is_runnable() is False (pre-Day-0a entry).
        ``"incompat"`` -- env not declared in policy.env_compat tuple.

    Order is stable, sorted by (policy_name, env_name).
    """
    plan: list[tuple[PolicySpec, EnvSpec, str]] = []
    policy_names = sorted(policies.names())
    env_names = sorted(envs.names())

    for p_name in policy_names:
        if policy_filter is not None and p_name != policy_filter:
            continue
        policy = policies.get(p_name)
        for e_name in env_names:
            if env_filter is not None and e_name != env_filter:
                continue
            env = envs.get(e_name)
            if e_name not in policy.env_compat:
                plan.append((policy, env, PLAN_INCOMPAT))
                continue
            if not policy.is_runnable():
                plan.append((policy, env, PLAN_SKIPPED))
                continue
            plan.append((policy, env, PLAN_READY))

    return plan


# --------------------------------------------------------------------- #
# Measurement                                                           #
# --------------------------------------------------------------------- #


def _zero_timing(policy: str, env: str, status: str, error: str | None) -> CellTiming:
    """Helper for non-ok exits where timing fields are placeholders."""
    return CellTiming(
        policy=policy,
        env=env,
        n_steps_measured=0,
        mean_ms_per_step=0.0,
        p95_ms_per_step=0.0,
        vram_peak_mb=0.0,
        status=status,
        error=error,
        recommended=None,
    )


def measure_cell(
    policy: PolicySpec,
    env: EnvSpec,
    *,
    n_steps: int,
    n_episodes: int,
    device: str = "cuda",
) -> CellTiming:
    """Measure latency + VRAM for one (policy, env) cell.

    Lazy-imports torch + lerobot inside the function -- importing this
    module must not require either. Returns a CellTiming with status
    one of ``"ok" | "oom" | "skipped" | "error"``. Never raises.

    The measurement loop times **policy inference only** (not the env
    step), which is what the auto_downscope thresholds care about.
    ``torch.cuda.synchronize()`` is called after each policy call so
    GPU async work doesn't hide in subsequent steps.
    """
    if not policy.is_runnable():
        return _zero_timing(
            policy.name,
            env.name,
            status="skipped",
            error=f"policy '{policy.name}' is not runnable -- Day 0a TODO",
        )

    # Lazy import: any ImportError here means the runtime is missing
    # required deps. Surface as status="error" with a clear message
    # (NOT exit 4 -- that's only when nothing can run).
    try:
        import time

        import numpy as np
        import torch

        from lerobot_bench.eval import load_env, load_policy
    except ImportError as exc:
        return _zero_timing(
            policy.name,
            env.name,
            status="error",
            error=f"missing runtime: {exc}",
        )

    sim = None
    try:
        sim = load_env(env)
        action_shape = tuple(sim.action_space.shape)  # type: ignore[attr-defined]

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pol = load_policy(policy, action_shape=action_shape, device=device)

        step_times_ms: list[float] = []
        for ep_idx in range(n_episodes):
            obs, _ = sim.reset(seed=ep_idx)
            pol.reset()
            for _ in range(n_steps):
                t0 = time.perf_counter()
                action = pol(obs)
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                step_times_ms.append(elapsed_ms)
                obs, _, terminated, truncated, _ = sim.step(action)
                if terminated or truncated:
                    obs, _ = sim.reset(seed=ep_idx)
                    pol.reset()

        vram_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device == "cuda" and torch.cuda.is_available()
            else 0.0
        )
        times_arr = np.asarray(step_times_ms, dtype=np.float64)
        timing = CellTiming(
            policy=policy.name,
            env=env.name,
            n_steps_measured=int(times_arr.size),
            mean_ms_per_step=float(times_arr.mean()),
            p95_ms_per_step=float(np.percentile(times_arr, 95)),
            vram_peak_mb=float(vram_peak_mb),
            status="ok",
            error=None,
            recommended=None,
        )
        return replace(timing, recommended=auto_downscope(timing))
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _zero_timing(
            policy.name,
            env.name,
            status="oom",
            error=str(exc)[:200],
        )
    except Exception as exc:
        return _zero_timing(
            policy.name,
            env.name,
            status="error",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
    finally:
        if sim is not None:
            with contextlib.suppress(Exception):
                sim.close()  # type: ignore[attr-defined]


# --------------------------------------------------------------------- #
# Report I/O                                                            #
# --------------------------------------------------------------------- #


def _git_sha() -> str:
    """Best-effort git SHA of the working tree. Returns ``"unknown"`` on failure."""
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


def _lerobot_version() -> str | None:
    """Return lerobot.__version__ or None if not installed.

    Imported lazily inside the function so module-level import of this
    file does not pull lerobot in.
    """
    try:
        import lerobot

        version = getattr(lerobot, "__version__", None)
        return str(version) if version is not None else None
    except ImportError:
        return None


def write_report(report: CalibrationReport, out_dir: Path) -> Path:
    """Write the report JSON. Filename uses the report's timestamp date.

    Returns the path written. Creates ``out_dir`` if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parse the date out of the report's own timestamp so re-emitting the
    # same report onto disk is byte-stable (don't read wall-clock here).
    date_str = report.timestamp_utc[:10]  # ISO 8601 YYYY-MM-DD prefix
    out_path = out_dir / f"calibration-{date_str}.json"
    out_path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=False) + "\n")
    return out_path


# --------------------------------------------------------------------- #
# Orchestration                                                         #
# --------------------------------------------------------------------- #


def run_calibration(
    policies_yaml: Path,
    envs_yaml: Path,
    *,
    out_dir: Path,
    steps: int,
    episodes: int,
    policy_filter: str | None,
    env_filter: str | None,
    dry_run: bool,
) -> tuple[CalibrationReport, int]:
    """Plan, measure, write -- pure plumbing.

    Returns (report, exit_code). Exit code is 0/2/3/4 per module
    docstring. The caller is responsible for printing user-facing
    messages and the resume-command hint on failure.
    """
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(envs_yaml)

    plan = plan_cells(policies, envs, policy_filter=policy_filter, env_filter=env_filter)

    ready = [(p, e) for (p, e, status) in plan if status == PLAN_READY]
    skipped = [(p, e) for (p, e, status) in plan if status == PLAN_SKIPPED]

    timestamp_utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    report_skeleton = CalibrationReport(
        timestamp_utc=timestamp_utc,
        git_sha=_git_sha(),
        lerobot_version=_lerobot_version(),
        cells=(),
    )

    if not ready and not skipped:
        # Nothing in the matrix at all -- bad filter or empty config.
        logger.error("no cells matched the requested filters")
        return report_skeleton, 3

    if not ready:
        # Everything filtered out as skipped/incompat -- nothing to measure.
        # Still record the skipped cells so the operator sees why.
        skipped_timings = tuple(
            _zero_timing(p.name, e.name, status="skipped", error="not runnable")
            for (p, e) in skipped
        )
        return replace(report_skeleton, cells=skipped_timings), 3

    if dry_run:
        # Dry-run: print plan and exit 0 without importing torch. Attach
        # one placeholder CellTiming per planned cell so the user-facing
        # summary in main() can report an accurate count without writing
        # JSON to disk.
        dry_cells: list[CellTiming] = []
        for policy, env, status in plan:
            logger.info("plan %-20s %-25s %s", policy.name, env.name, status)
            # All plan rows collapse to status="skipped" in the dry-run
            # report -- nothing is actually measured, but each planned
            # cell is represented so main() can report an accurate count.
            dry_cells.append(
                _zero_timing(
                    policy.name,
                    env.name,
                    status="skipped",
                    error=f"dry-run: plan={status}",
                )
            )
        return replace(report_skeleton, cells=tuple(dry_cells)), 0

    cells: list[CellTiming] = []
    for policy, env in ready:
        logger.info("measuring %s x %s ...", policy.name, env.name)
        timing = measure_cell(policy, env, n_steps=steps, n_episodes=episodes)
        if timing.status == "ok" and timing.recommended is None:
            # measure_cell is responsible for filling recommended on ok,
            # but defend against a future caller that forgets.
            timing = replace(timing, recommended=auto_downscope(timing))
        logger.info(
            "  -> status=%s mean_ms=%.2f p95_ms=%.2f vram_mb=%.1f",
            timing.status,
            timing.mean_ms_per_step,
            timing.p95_ms_per_step,
            timing.vram_peak_mb,
        )
        cells.append(timing)

    # Always record the skipped cells too so the report shows the full picture.
    for policy, env in skipped:
        cells.append(_zero_timing(policy.name, env.name, status="skipped", error="not runnable"))

    report = replace(report_skeleton, cells=tuple(cells))
    write_report(report, out_dir)

    n_ok = sum(1 for c in cells if c.status == "ok")
    n_total = len(cells)
    exit_code = 0 if n_ok == n_total else 2
    return report, exit_code


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description=(
            "Per-cell latency + VRAM calibration spike. "
            "Outputs results/calibration-YYYYMMDD.json driving the auto-downscope rule."
        ),
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
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for the calibration JSON (default: results/).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Restrict to a single policy name.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Restrict to a single env name.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Steps to measure per episode (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episodes per cell (default: {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cells and exit 0 without importing torch.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _format_resume_hint(failed: list[CellTiming]) -> str:
    lines = [
        f"[calibrate] partial run -- {len(failed)} cells failed. To re-run only the failures:",
    ]
    for cell in failed:
        lines.append(f"    python scripts/calibrate.py --policy {cell.policy} --env {cell.env}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        report, exit_code = run_calibration(
            policies_yaml=args.policies_yaml,
            envs_yaml=args.envs_yaml,
            out_dir=args.out_dir,
            steps=args.steps,
            episodes=args.episodes,
            policy_filter=args.policy,
            env_filter=args.env,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        # A missing yaml file is operator error, not a runtime crash.
        print(f"[calibrate] config not found: {exc}", file=sys.stderr)
        print(
            "[calibrate] resume: python scripts/calibrate.py "
            "--policies-yaml <path> --envs-yaml <path>",
            file=sys.stderr,
        )
        return 4

    if exit_code == 3:
        print(
            "[calibrate] no runnable cells matched -- nothing to calibrate. "
            "Check --policy / --env filters or wait for Day 0a revision_sha lock.",
            file=sys.stderr,
        )
        return exit_code

    failed = [c for c in report.cells if c.status in {"oom", "error"}]
    n_ok = sum(1 for c in report.cells if c.status == "ok")
    n_oom = sum(1 for c in report.cells if c.status == "oom")
    n_err = sum(1 for c in report.cells if c.status == "error")
    n_skip = sum(1 for c in report.cells if c.status == "skipped")

    if args.dry_run:
        print(
            f"[calibrate] dry-run: planned {len(report.cells)} cells "
            "(no JSON written, no torch imported)."
        )
        return 0

    out_path = args.out_dir / f"calibration-{report.timestamp_utc[:10]}.json"
    summary = f"{len(report.cells)} cells: {n_ok} ok, {n_oom} oom, {n_err} error, {n_skip} skipped"
    if exit_code == 0:
        print(f"[calibrate] wrote {out_path} ({summary})")
    else:
        # exit_code == 2: partial -- still wrote the report.
        print(f"[calibrate] wrote {out_path} ({summary})")
        print(_format_resume_hint(failed), file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
