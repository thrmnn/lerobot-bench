#!/usr/bin/env python3
"""Incremental sweep sanity-checker.

Loads the *partial* ``results.parquet`` an overnight sweep writes one
cell at a time, joins it against the policy / env configs, and prints a
per-cell review table plus a flagged-anomalies section. The point is to
catch a misbehaving ``(policy, env, seed)`` cell within minutes of it
completing instead of after a 20-hour wait.

Strictly read-only: it never writes to ``results/`` and never touches
the running sweep. Safe to run (or cron) against a live sweep dir.

Usage::

    python scripts/review_results.py                       # default sweep-full
    python scripts/review_results.py --results path/to/results.parquet
    python scripts/review_results.py --manifest path/to/sweep_manifest.json

Exit codes:
    0  no anomalies — every completed cell looks healthy
    1  at least one cell flagged — inspect the ANOMALIES section
    2  could not run (missing parquet, empty results, bad config)

Anomaly checks (see DESIGN.md § Methodology):
    1. Far from paper — non-baseline cell more than 0.25 off its
       ``paper_reported_success`` AND outside the Wilson 95% CI.
    2. Baseline above floor — ``no_op`` / ``random`` scoring suspiciously
       high (reward / success-detection bug). PushT random has a known
       small non-zero floor, so the threshold is env-aware.
    3. Never-succeeds — every episode hit ``max_steps`` with 0 successes;
       a non-baseline policy that is effectively inert.
    4. Seed disagreement — a policy×env whose per-seed success rates span
       more than 0.30 (flaky seeding or an unstable checkpoint).
    5. Degenerate — every episode in a cell is byte-identical in both
       ``success`` and ``n_steps`` — a likely determinism / seeding bug.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lerobot_bench.envs import EnvRegistry, EnvSpec
from lerobot_bench.policies import PolicyRegistry, PolicySpec
from lerobot_bench.stats import wilson_ci

logger = logging.getLogger("review_results")

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_RESULTS = Path("results/sweep-full/results.parquet")
DEFAULT_MANIFEST = Path("results/sweep-full/sweep_manifest.json")
DEFAULT_POLICIES_YAML = Path("configs/policies.yaml")
DEFAULT_ENVS_YAML = Path("configs/envs.yaml")

# Anomaly thresholds. See module docstring + DESIGN.md § Methodology.
PAPER_GAP_THRESHOLD = 0.25
SEED_SPREAD_THRESHOLD = 0.30
BASELINE_FLOOR_THRESHOLD = 0.15
# PushT scores partial coverage even under a random policy, so a random
# agent clears the generic 0.15 floor by luck more often than elsewhere.
# Bump the bar for PushT specifically rather than false-flagging it.
BASELINE_FLOOR_THRESHOLD_PUSHT = 0.35

EXIT_OK = 0
EXIT_ANOMALIES = 1
EXIT_CANNOT_RUN = 2


# --------------------------------------------------------------------- #
# Data model                                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellReview:
    """Per-(policy, env, seed) summary plus any anomaly flags."""

    policy: str
    env: str
    seed: int
    n_episodes: int
    n_success: int
    success_rate: float
    mean_n_steps: float
    mean_wallclock_s: float
    ci_lo: float
    ci_hi: float
    is_baseline: bool
    paper_reported: float | None
    flags: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


# --------------------------------------------------------------------- #
# Loading                                                               #
# --------------------------------------------------------------------- #


def load_results(results_path: Path) -> pd.DataFrame:
    """Load the partial sweep parquet, raising a clear error if unusable."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"results parquet not found: {results_path}\n"
            "  The sweep writes this after its first cell completes -- wait a "
            "few minutes, or pass --results <path> if the sweep dir differs."
        )
    try:
        df = pd.read_parquet(results_path)
    except Exception as exc:
        # A live sweep may be mid-write when this read lands; pyarrow then
        # raises on the truncated footer. That is transient, not fatal.
        raise ValueError(
            f"could not read results parquet: {results_path} ({exc})\n"
            "  The sweep may be mid-write -- this is transient, just re-run "
            "review_results.py in a few seconds.\n"
            "  See docs/TROUBLESHOOTING.md -> Parquet mid-write read errors."
        ) from exc
    if df.empty:
        raise ValueError(
            f"results parquet is empty: {results_path}\n"
            "  No cells have been persisted yet -- re-run once the sweep has "
            "completed at least one cell."
        )
    required = {"policy", "env", "seed", "episode_index", "success", "n_steps", "wallclock_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"results parquet missing columns {sorted(missing)}: {results_path}\n"
            "  This does not look like a sweep results.parquet -- check the "
            "--results path points at the sweep output, not another parquet."
        )
    return df


# --------------------------------------------------------------------- #
# Review                                                                #
# --------------------------------------------------------------------- #


def _baseline_floor(env: str) -> float:
    """Success-rate ceiling a baseline cell may reach before it's a flag."""
    return BASELINE_FLOOR_THRESHOLD_PUSHT if env == "pusht" else BASELINE_FLOOR_THRESHOLD


def review_cells(
    df: pd.DataFrame,
    policies: PolicyRegistry,
    envs: EnvRegistry,
) -> list[CellReview]:
    """Build a :class:`CellReview` per completed cell, with anomaly flags.

    A "completed" cell is any ``(policy, env, seed)`` triple present in
    ``df`` — the sweep persists cells atomically, so presence means done.
    Per-cell checks (far-from-paper, baseline floor, never-succeeds,
    degenerate) run here; the cross-seed check is layered on after.
    """
    reviews: list[CellReview] = []

    grouped = df.groupby(["policy", "env", "seed"], sort=True)
    for group_key, cell in grouped:
        policy_name, env_name, seed = group_key
        policy = _safe_policy(policies, str(policy_name))
        env = _safe_env(envs, str(env_name))

        n_episodes = len(cell)
        successes = cell["success"].astype(bool)
        n_success = int(successes.sum())
        success_rate = n_success / n_episodes
        mean_n_steps = float(cell["n_steps"].mean())
        mean_wallclock = float(cell["wallclock_s"].mean())
        ci_lo, ci_hi = wilson_ci(n_success, n_episodes)

        is_baseline = bool(getattr(policy, "is_baseline", False)) if policy else False
        paper_map = getattr(policy, "paper_reported_success", None) or {}
        paper_reported = paper_map.get(str(env_name)) if paper_map else None

        flags: list[str] = []

        # Check 1 — far from paper-reported success.
        if not is_baseline and paper_reported is not None:
            gap = success_rate - paper_reported
            outside_ci = paper_reported < ci_lo or paper_reported > ci_hi
            if abs(gap) > PAPER_GAP_THRESHOLD and outside_ci:
                direction = "well-below" if gap < 0 else "well-above"
                flags.append(
                    f"FAR-FROM-PAPER ({direction}): our {success_rate:.1%} vs "
                    f"paper {paper_reported:.1%} (gap {gap:+.1%}, "
                    f"95% CI [{ci_lo:.1%}, {ci_hi:.1%}] excludes paper)"
                )

        # Check 2 — baseline scoring above its floor.
        if is_baseline:
            floor = _baseline_floor(str(env_name))
            if success_rate > floor:
                note = (
                    " (PushT random has a known small non-zero floor; "
                    "this is still above the env-specific bar)"
                    if env_name == "pusht"
                    else ""
                )
                flags.append(
                    f"BASELINE-ABOVE-FLOOR: {policy_name} scored "
                    f"{success_rate:.1%} > {floor:.0%} threshold — likely a "
                    f"reward/success-detection bug{note}"
                )

        # Check 3 — never succeeds, every episode burned max_steps.
        max_steps = getattr(env, "max_steps", None) if env else None
        if not is_baseline and n_success == 0 and max_steps is not None:
            all_max = bool((cell["n_steps"] == max_steps).all())
            if all_max:
                flags.append(
                    f"NEVER-SUCCEEDS: 0/{n_episodes} successes, every episode "
                    f"ran the full {max_steps} max_steps — policy may be inert"
                )

        # Check 5 — degenerate cell (every episode byte-identical).
        # Baselines are exempt: a no_op policy *is* deterministic by
        # construction (zero action every step), so identical episodes
        # are the expected floor, not a seeding bug.
        if not is_baseline and n_episodes >= 2:
            identical_success = successes.nunique() == 1
            identical_steps = cell["n_steps"].nunique() == 1
            if identical_success and identical_steps:
                flags.append(
                    f"DEGENERATE: all {n_episodes} episodes identical "
                    f"(success={bool(successes.iloc[0])}, "
                    f"n_steps={int(cell['n_steps'].iloc[0])}) — "
                    "possible seeding/determinism bug"
                )

        reviews.append(
            CellReview(
                policy=str(policy_name),
                env=str(env_name),
                seed=int(str(seed)),
                n_episodes=n_episodes,
                n_success=n_success,
                success_rate=success_rate,
                mean_n_steps=mean_n_steps,
                mean_wallclock_s=mean_wallclock,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                is_baseline=is_baseline,
                paper_reported=paper_reported,
                flags=flags,
            )
        )

    _flag_seed_disagreement(reviews)
    return reviews


def _flag_seed_disagreement(reviews: list[CellReview]) -> None:
    """Check 4 — flag every cell of a policy×env with a wide seed spread.

    Mutates the ``flags`` lists in place: when ≥2 seeds of the same
    policy×env span more than :data:`SEED_SPREAD_THRESHOLD` in success
    rate, every contributing cell is flagged so the table reads clearly.
    """
    by_pair: dict[tuple[str, str], list[CellReview]] = {}
    for r in reviews:
        by_pair.setdefault((r.policy, r.env), []).append(r)

    for (policy, env), cells in by_pair.items():
        if len(cells) < 2:
            continue
        rates = [c.success_rate for c in cells]
        spread = max(rates) - min(rates)
        if spread > SEED_SPREAD_THRESHOLD:
            lo_seed = min(cells, key=lambda c: c.success_rate)
            hi_seed = max(cells, key=lambda c: c.success_rate)
            for c in cells:
                c.flags.append(
                    f"SEED-DISAGREEMENT: {policy}×{env} seed rates span "
                    f"{spread:.1%} (seed {lo_seed.seed}={min(rates):.1%} .. "
                    f"seed {hi_seed.seed}={max(rates):.1%}) — exceeds "
                    f"{SEED_SPREAD_THRESHOLD:.0%}"
                )


def _safe_policy(registry: PolicyRegistry, name: str) -> PolicySpec | None:
    """Return the :class:`PolicySpec` for ``name`` or ``None`` if unknown.

    A partial sweep can contain a policy that was renamed or dropped
    from the configs; the review tool should degrade gracefully rather
    than crash, so an unknown name simply skips config-dependent checks.
    """
    try:
        return registry.get(name)
    except (KeyError, ValueError):
        return None


def _safe_env(registry: EnvRegistry, name: str) -> EnvSpec | None:
    """Return the :class:`EnvSpec` for ``name`` or ``None`` if unknown."""
    try:
        return registry.get(name)
    except (KeyError, ValueError):
        return None


# --------------------------------------------------------------------- #
# Rendering                                                             #
# --------------------------------------------------------------------- #


def render_report(reviews: list[CellReview]) -> str:
    """Render the full text report: summary header, table, ANOMALIES."""
    flagged = [r for r in reviews if r.flagged]
    lines: list[str] = []

    lines.append(
        f"{len(reviews)} cells reviewed, {len(flagged)} flagged"
        + (" — all healthy" if not flagged else "")
    )
    lines.append("")
    lines.extend(_render_table(reviews))

    lines.append("")
    lines.append("=" * 78)
    if not flagged:
        lines.append("ANOMALIES: none")
    else:
        lines.append(f"ANOMALIES ({len(flagged)} cell(s) flagged)")
        lines.append("=" * 78)
        for r in flagged:
            lines.append(f"  {r.policy} × {r.env} × seed {r.seed}")
            for f in r.flags:
                lines.append(f"      - {f}")
    return "\n".join(lines)


def _render_table(reviews: list[CellReview]) -> list[str]:
    """Format the per-cell review table as fixed-width text rows."""
    header = (
        f"{'policy':<22} {'env':<22} {'seed':>4} {'eps':>4} "
        f"{'succ':>5} {'rate':>7} {'wilson 95% ci':>17} "
        f"{'steps':>8} {'wall_s':>8}  flag"
    )
    rows = [header, "-" * len(header)]
    for r in sorted(reviews, key=lambda c: (c.policy, c.env, c.seed)):
        ci = f"[{r.ci_lo:.0%},{r.ci_hi:.0%}]"
        mark = "FLAG" if r.flagged else ""
        rows.append(
            f"{r.policy:<22} {r.env:<22} {r.seed:>4} {r.n_episodes:>4} "
            f"{r.n_success:>5} {r.success_rate:>7.1%} {ci:>17} "
            f"{r.mean_n_steps:>8.1f} {r.mean_wallclock_s:>8.1f}  {mark}"
        )
    return rows


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


class _RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show per-argument defaults, but keep the epilog's literal layout."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="review-results",
        description=(
            "Incremental sweep sanity-checker. Loads the partial\n"
            "results.parquet a running sweep writes one cell at a time, joins\n"
            "it against the policy/env configs, and prints a per-cell review\n"
            "table plus a flagged-anomalies section. Strictly read-only --\n"
            "safe to run (or cron) against a live sweep dir."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  # review the default sweep-full results\n"
            "  python scripts/review_results.py\n\n"
            "  # review a results.parquet at a custom path\n"
            "  python scripts/review_results.py --results path/to/results.parquet\n\n"
            "exit codes:\n"
            "  0  no anomalies -- every completed cell looks healthy\n"
            "  1  at least one cell flagged -- inspect the ANOMALIES section\n"
            "  2  could not run (missing/empty parquet, bad config)"
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        metavar="PARQUET",
        help="Path to the (partial) sweep results.parquet to review.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        metavar="JSON",
        help="Path to sweep_manifest.json (read for context only; "
        "review proceeds from the parquet alone if absent).",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=DEFAULT_POLICIES_YAML,
        metavar="YAML",
        help="Path to the policy registry YAML (joined for paper-gap and baseline-floor checks).",
    )
    parser.add_argument(
        "--envs",
        type=Path,
        default=DEFAULT_ENVS_YAML,
        metavar="YAML",
        help="Path to the env registry YAML (joined for the never-succeeds max_steps check).",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for bad input."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    try:
        df = load_results(args.results)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cannot review: %s", exc)
        return EXIT_CANNOT_RUN

    try:
        policies = PolicyRegistry.from_yaml(args.policies)
        envs = EnvRegistry.from_yaml(args.envs)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("cannot review: failed to load configs: %s", exc)
        return EXIT_CANNOT_RUN

    if args.manifest.exists():
        logger.info("manifest: %s", args.manifest)
    else:
        logger.warning("manifest not found (%s) — proceeding from parquet only", args.manifest)

    reviews = review_cells(df, policies, envs)
    report = render_report(reviews)
    logger.info("%s", report)

    return EXIT_ANOMALIES if any(r.flagged for r in reviews) else EXIT_OK


if __name__ == "__main__":
    print()
    sys.exit(run())
