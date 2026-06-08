"""Read a real leaderboard number — zero GPU, zero download, on a fresh clone.

This is the smallest "clone → see a real number" example. With **no
arguments** it reads the committed, pre-aggregated
``examples/results-mini.parquet`` — an honest, tiny view of the v1
leaderboard headline cells — and prints the whole table with each cell's
success rate and Wilson 95% confidence interval::

    python examples/read_results.py

No GPU, no Hugging Face download, no full per-episode parquet required.
The mini-parquet is regenerated deterministically by
``scripts/make_results_mini.py`` from numbers already committed to this
repo (see that script's docstring for provenance); the ``act × aloha``
cell carries the corrected canonical **0.824 [0.772, 0.866]**, and
``xvla`` / pi0 are excluded via ``embodimetry.leaderboard_filter``.

To read a *real per-episode* parquet (the full sweep on the Hub, or a
local ``run_one.py`` output) and drill into a single cell with the live
stats API::

    python examples/read_results.py --results results/sweep-full/results.parquet
    python examples/read_results.py --results results/results.parquet \\
        --policy act --env aloha_transfer_cube

Either way the numbers come from the *real* public stats API —
``embodimetry.stats.wilson_ci`` and ``wilson_halfwidth_at_p`` — so they
match what the leaderboard and ``docs/MDE_TABLE.md`` show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from embodimetry.stats import wilson_ci, wilson_halfwidth_at_p

# The committed zero-GPU view. Resolved relative to this file so it works
# from any working directory on a fresh clone.
_MINI_PARQUET = Path(__file__).resolve().parent / "results-mini.parquet"

# Aggregated columns the mini-parquet carries (one row per published cell).
_MINI_COLUMNS = (
    "policy",
    "env",
    "n_episodes",
    "n_success",
    "success_rate",
    "wilson_lo",
    "wilson_hi",
)


def cell_outcomes(df: pd.DataFrame, policy: str, env: str) -> NDArray[np.bool_]:
    """Flat bool array of per-episode outcomes for one (policy, env) pair.

    The leaderboard cell pools *all* seeds: the unit of evidence is the
    episode, not the seed. See ``stats.py`` module docstring on why
    bootstrapping over 5 seeds is the wrong granularity.
    """
    mask = (df["policy"] == policy) & (df["env"] == env)
    return df.loc[mask, "success"].to_numpy(dtype=bool)


def print_leaderboard(df: pd.DataFrame) -> None:
    """Print the aggregated mini-leaderboard table (policy/env/rate/CI)."""
    print("v1 leaderboard (headline cells) — success rate with Wilson 95% CI")
    print(f"source: {_MINI_PARQUET.name}  (committed; no GPU / no download)\n")
    header = f"{'policy':16}  {'env':20}  {'N':>4}  {'rate':>6}  {'Wilson 95% CI':>18}"
    print(header)
    print("-" * len(header))
    for row in df.itertuples(index=False):
        ci = f"[{row.wilson_lo:.3f}, {row.wilson_hi:.3f}]"
        print(
            f"{row.policy:16}  {row.env:20}  {row.n_episodes:>4}  "
            f"{row.success_rate:>6.3f}  {ci:>18}"
        )
    print()
    print("Read one cell deeper from a real per-episode parquet:")
    print("  python examples/read_results.py --results results/sweep-full/results.parquet \\")
    print("      --policy act --env aloha_transfer_cube")


def _print_single_cell(df: pd.DataFrame, policy: str, env: str) -> None:
    """Single-cell deep dive from a per-episode parquet (the --results path)."""
    outcomes = cell_outcomes(df, policy, env)
    if outcomes.size == 0:
        pairs = sorted(set(zip(df["policy"], df["env"], strict=True)))
        print(f"No rows for policy={policy!r} env={env!r}.")
        print("Available (policy, env) pairs in this parquet:")
        for p, e in pairs:
            print(f"  {p} / {e}")
        return

    n = int(outcomes.size)
    successes = int(outcomes.sum())
    p_hat = successes / n

    # Wilson 95% score interval — closed form, the leaderboard's CI.
    lo, hi = wilson_ci(successes, n, ci=0.95)
    half_width = (hi - lo) / 2.0

    # Half-width the MDE table would quote at this p̂ and N (sanity ref).
    hw_table = wilson_halfwidth_at_p(p_hat, n, alpha=0.05)

    print(f"cell:          {policy} / {env}")
    print(f"episodes (N):  {n}  ({successes} successes)")
    print(f"success rate:  {p_hat:.3f}")
    print(f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]   half-width ±{half_width:.3f}")
    print(f"  (MDE-table half-width at p̂={p_hat:.2f}, N={n}: ±{hw_table:.3f})")

    band = 2 * half_width
    print()
    print(f"Interpretation: any rival cell within ±{band:.3f} success rate of this")
    print("one is INCONCLUSIVE at this N — the gap is inside sampling noise.")
    print("See docs/tutorials/interpreting-the-leaderboard.md for the full reading.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to a real per-episode results.parquet. When omitted, read "
            "the committed examples/results-mini.parquet and print the whole "
            "leaderboard table (no GPU, no download)."
        ),
    )
    parser.add_argument("--policy", default="act", help="Policy name (default: act).")
    parser.add_argument(
        "--env", default="aloha_transfer_cube", help="Env name (default: aloha_transfer_cube)."
    )
    args = parser.parse_args()

    # Default path: the committed mini-parquet, printed as a full table.
    if args.results is None:
        if not _MINI_PARQUET.exists():
            raise SystemExit(
                f"[read-results] {_MINI_PARQUET} is missing. Regenerate it with "
                "`python scripts/make_results_mini.py`."
            )
        df = pd.read_parquet(_MINI_PARQUET)
        print_leaderboard(df)
        return

    # --results path: a real per-episode parquet → single-cell deep dive.
    if not args.results.exists():
        raise SystemExit(
            f"[read-results] {args.results} not found. The full per-episode parquet "
            "lives on the Hub (dataset thrmnn/embodimetry-v1) and is .gitignore-d. "
            "Run with no --results to read the committed examples/results-mini.parquet "
            "instead (no GPU, no download)."
        )
    df = pd.read_parquet(args.results)
    if "success" not in df.columns:
        raise SystemExit(
            f"[read-results] {args.results} has no per-episode 'success' column "
            f"(columns: {list(df.columns)}). This looks like an *aggregated* "
            "leaderboard parquet (e.g. examples/results-mini.parquet); the "
            "--results deep-dive needs a per-episode parquet. Run with no "
            "--results to print the aggregated table instead."
        )
    _print_single_cell(df, args.policy, args.env)


if __name__ == "__main__":
    main()
