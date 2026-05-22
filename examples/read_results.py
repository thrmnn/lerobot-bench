"""Load results.parquet and compute a success rate + Wilson CI for one cell.

This is the smallest end-to-end "read a leaderboard number" example. It
loads the per-episode parquet, picks one ``(policy, env)`` pair, pools
its seed cells into a flat list of binary outcomes, and reports the
success rate with a Wilson 95% confidence interval.

It uses the *real* public stats API — ``lerobot_bench.stats.wilson_ci``
and ``wilson_halfwidth_at_p`` — so the number it prints matches what the
leaderboard and ``docs/MDE_TABLE.md`` would show.

Run it::

    python examples/read_results.py
    python examples/read_results.py --results results/sweep-full/results.parquet
    python examples/read_results.py --policy act --env aloha_transfer_cube

If no parquet is found the script falls back to a small **synthetic**
sample so it still runs and demonstrates the API (clearly labelled in
the output). Point ``--results`` at a real parquet for real numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lerobot_bench.stats import wilson_ci, wilson_halfwidth_at_p

# Candidate parquet locations, in priority order. The full sweep writes
# results/sweep-full/results.parquet; a local run_one.py writes
# results/results.parquet.
_DEFAULT_PARQUET_CANDIDATES = (
    Path("results/sweep-full/results.parquet"),
    Path("results/results.parquet"),
)


def _synthetic_sample() -> pd.DataFrame:
    """A documented stand-in when no parquet is on disk.

    Schema matches ``lerobot_bench.checkpointing.RESULT_SCHEMA`` for the
    columns this example reads. The outcomes are an illustrative
    Bernoulli draw (success rate ~0.7), NOT a real benchmark result.
    """
    rng = np.random.default_rng(0)
    n_seeds, n_eps = 5, 50
    rows = []
    for seed in range(n_seeds):
        for ep in range(n_eps):
            rows.append(
                {
                    "policy": "act",
                    "env": "aloha_transfer_cube",
                    "seed": seed,
                    "episode_index": ep,
                    "success": bool(rng.random() < 0.7),
                }
            )
    return pd.DataFrame(rows)


def load_results(results_path: Path | None) -> tuple[pd.DataFrame, bool]:
    """Load the parquet, or fall back to the synthetic sample.

    Returns ``(df, is_real)``: ``is_real`` is ``False`` when the
    synthetic fallback was used.
    """
    candidates = [results_path] if results_path else list(_DEFAULT_PARQUET_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            return pd.read_parquet(candidate), True
    return _synthetic_sample(), False


def cell_outcomes(df: pd.DataFrame, policy: str, env: str) -> NDArray[np.bool_]:
    """Flat bool array of per-episode outcomes for one (policy, env) pair.

    The leaderboard cell pools *all* seeds: the unit of evidence is the
    episode, not the seed. See ``stats.py`` module docstring on why
    bootstrapping over 5 seeds is the wrong granularity.
    """
    mask = (df["policy"] == policy) & (df["env"] == env)
    return df.loc[mask, "success"].to_numpy(dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results.parquet (default: search results/sweep-full then results/).",
    )
    parser.add_argument("--policy", default="act", help="Policy name (default: act).")
    parser.add_argument(
        "--env", default="aloha_transfer_cube", help="Env name (default: aloha_transfer_cube)."
    )
    args = parser.parse_args()

    df, is_real = load_results(args.results)
    if not is_real:
        print("[!] No parquet found — using a SYNTHETIC sample (not real results).")
        print("    Pass --results <path> to read a real sweep parquet.\n")

    outcomes = cell_outcomes(df, args.policy, args.env)
    if outcomes.size == 0:
        pairs = sorted(set(zip(df["policy"], df["env"], strict=True)))
        print(f"No rows for policy={args.policy!r} env={args.env!r}.")
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

    print(f"cell:          {args.policy} / {args.env}")
    print(f"episodes (N):  {n}  ({successes} successes)")
    print(f"success rate:  {p_hat:.3f}")
    print(f"Wilson 95% CI: [{lo:.3f}, {hi:.3f}]   half-width ±{half_width:.3f}")
    print(f"  (MDE-table half-width at p̂={p_hat:.2f}, N={n}: ±{hw_table:.3f})")

    band = 2 * half_width
    print()
    print(f"Interpretation: any rival cell within ±{band:.3f} success rate of this")
    print("one is INCONCLUSIVE at this N — the gap is inside sampling noise.")
    print("See docs/tutorials/interpreting-the-leaderboard.md for the full reading.")


if __name__ == "__main__":
    main()
