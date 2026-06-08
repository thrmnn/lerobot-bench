"""Paired comparison of two policies on a shared env, with the MDE check.

Ranking two leaderboard cells is *not* "whichever success rate is
bigger". You need (a) the paired Δsuccess with a confidence interval and
(b) a minimum-detectable-effect check: if the observed |Δ| is inside the
MDE band, the ordering is **inconclusive at this N** and must not be
reported as a ranking. This is the precise mistake ``docs/MDE_TABLE.md``
exists to prevent.

This example uses the real public stats API:

* ``paired_diff_ci``       — pivotal paired-bootstrap CI on mean(a) - mean(b)
* ``paired_wilcoxon``      — paired signed-rank test on the per-episode outcomes
* ``cohens_h``             — effect size for the difference of two proportions
* ``wilson_halfwidth_at_p``— per-cell Wilson half-width, for the MDE band

Run it::

    python examples/compare_two_policies.py
    python examples/compare_two_policies.py --results results/sweep-full/results.parquet \\
        --env libero_spatial --policy-a smolvla_libero --policy-b random

With no parquet on disk it falls back to a **synthetic** sample (clearly
labelled) so the API call path still runs. Point ``--results`` at a real
parquet for real numbers.

Pairing requirement: ``paired_diff_ci`` and ``paired_wilcoxon`` pair
episodes *by index*, so ``a[i]`` and ``b[i]`` must come from the same
``(seed, episode_index)``. This example aligns the two cells on that key
and drops any episode not present in both before comparing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from embodimetry.stats import cohens_h, paired_diff_ci, paired_wilcoxon, wilson_halfwidth_at_p

_DEFAULT_PARQUET_CANDIDATES = (
    Path("results/sweep-full/results.parquet"),
    Path("results/results.parquet"),
)


def _synthetic_sample() -> pd.DataFrame:
    """Documented stand-in: two policies on one env, ~10pp apart.

    Illustrative Bernoulli draws (policy A ~0.62, policy B ~0.52) over
    the contracted 5 seeds × 50 episodes. NOT a real benchmark result.
    Both policies are in the v1 leaderboard set so the default run stays
    consistent with the published public surface.
    """
    rng = np.random.default_rng(7)
    rows = []
    for policy, p in (("smolvla_libero", 0.62), ("random", 0.52)):
        for seed in range(5):
            for ep in range(50):
                rows.append(
                    {
                        "policy": policy,
                        "env": "libero_spatial",
                        "seed": seed,
                        "episode_index": ep,
                        "success": bool(rng.random() < p),
                    }
                )
    return pd.DataFrame(rows)


def load_results(results_path: Path | None) -> tuple[pd.DataFrame, bool]:
    """Load the parquet, or fall back to the synthetic sample."""
    candidates = [results_path] if results_path else list(_DEFAULT_PARQUET_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            df = pd.read_parquet(candidate)
            missing = {"seed", "episode_index", "success"} - set(df.columns)
            if missing:
                raise SystemExit(
                    f"[compare] {candidate} is missing per-episode columns {sorted(missing)} "
                    f"(has: {list(df.columns)}). Paired comparison needs a per-episode "
                    "parquet; an aggregated leaderboard view (e.g. examples/results-mini.parquet) "
                    "won't work. Point --results at the full per-episode parquet."
                )
            return df, True
    return _synthetic_sample(), False


def aligned_outcomes(
    df: pd.DataFrame, env: str, policy_a: str, policy_b: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return (a, b) bool arrays aligned on the shared (seed, episode_index).

    Episodes present in only one cell are dropped — paired statistics
    require a one-to-one episode correspondence. If the two cells used
    an identical seeding contract and equal episode counts (the sweep's
    normal case) nothing is dropped.
    """
    key = ["seed", "episode_index"]
    cell_a = df[(df["env"] == env) & (df["policy"] == policy_a)][[*key, "success"]]
    cell_b = df[(df["env"] == env) & (df["policy"] == policy_b)][[*key, "success"]]
    merged = cell_a.merge(cell_b, on=key, suffixes=("_a", "_b"), how="inner").sort_values(key)
    return (
        merged["success_a"].to_numpy(dtype=bool),
        merged["success_b"].to_numpy(dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None, help="Path to results.parquet.")
    parser.add_argument("--env", default="libero_spatial", help="Shared env name.")
    parser.add_argument("--policy-a", default="smolvla_libero", help="First policy.")
    parser.add_argument("--policy-b", default="random", help="Second policy.")
    args = parser.parse_args()

    df, is_real = load_results(args.results)
    if not is_real:
        print("[!] No parquet found — using a SYNTHETIC sample (not real results).")
        print("    Pass --results <path> to read a real sweep parquet.\n")

    a, b = aligned_outcomes(df, args.env, args.policy_a, args.policy_b)
    if a.size == 0:
        print(
            f"No shared episodes for {args.policy_a} vs {args.policy_b} on {args.env}.\n"
            "Check the names against the parquet, and that both cells were swept."
        )
        return

    n = int(a.size)
    p_a, p_b = float(a.mean()), float(b.mean())
    delta = p_a - p_b

    # Paired pivotal-bootstrap CI on the delta (respects the pairing).
    lo, hi = paired_diff_ci(a, b, alpha=0.05, n_resamples=10_000, seed=0)
    # Paired signed-rank test and effect size.
    wil = paired_wilcoxon(a, b)
    h = cohens_h(p_a, p_b)

    # MDE band: per-cell Wilson half-width at the higher of the two p̂.
    # A delta inside 2·HW could come from sampling noise alone.
    hw = wilson_halfwidth_at_p(max(p_a, p_b), n, alpha=0.05)
    mde_band = 2.0 * hw
    inconclusive = abs(delta) < mde_band

    print(f"env:              {args.env}   (N = {n} paired episodes per cell)")
    print(f"{args.policy_a:<22} success rate: {p_a:.3f}")
    print(f"{args.policy_b:<22} success rate: {p_b:.3f}")
    print(f"Δsuccess (A − B): {delta:+.3f}")
    print(f"  paired 95% CI:  [{lo:+.3f}, {hi:+.3f}]")
    print(
        f"  Wilcoxon:       p = {wil.pvalue:.4f}  (n_pairs={wil.n_pairs}, ties={wil.n_zero_diffs})"
    )
    print(f"  Cohen's h:      {h:+.3f}  ({_h_label(h)})")
    print(f"  MDE band (2·HW at p̂={max(p_a, p_b):.2f}, N={n}): ±{mde_band:.3f}")
    print()

    if inconclusive:
        print(f"VERDICT: INCONCLUSIVE at N={n}.")
        print(f"  |Δ|={abs(delta):.3f} is inside the MDE band ±{mde_band:.3f} — the observed")
        print("  gap is within what sampling noise alone could produce. Do NOT rank")
        print("  these two cells. Report them as tied at this N.")
    else:
        ci_excludes_zero = lo > 0 or hi < 0
        verdict = "resolved" if ci_excludes_zero else "borderline (CI still spans 0)"
        print(f"VERDICT: difference is {verdict}.")
        print(f"  |Δ|={abs(delta):.3f} exceeds the MDE band ±{mde_band:.3f}. Quote the delta")
        print("  WITH its CI and Cohen's h — a 'significant' small effect is still small.")
    print()
    print("Full reading: docs/tutorials/interpreting-the-leaderboard.md")


def _h_label(h: float) -> str:
    """Conventional Cohen's h magnitude label."""
    a = abs(h)
    if a < 0.2:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


if __name__ == "__main__":
    main()
