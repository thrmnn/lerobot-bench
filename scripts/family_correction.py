#!/usr/bin/env python3
"""Family-wise Holm-Bonferroni correction across the cell matrix.

For every cell in ``results/sweep-full/results.parquet`` whose policy
has a ``paper_reported_success[env]`` entry, compute a two-sided exact
binomial p-value testing ``H_0: success_rate == paper_rate``, then
apply Holm-Bonferroni step-down correction across the resulting family
of cells (v1.1 DESIGN.md § Methodology: any individual α=0.05 claim
across the matrix must clear the family-wise threshold first).

Cells without a ``paper_reported_success`` entry are reported but
skipped from the correction (printed at the bottom of the table with
``-`` in the p-value columns).

Usage::

    python scripts/family_correction.py
    python scripts/family_correction.py --results path/to/results.parquet
    python scripts/family_correction.py --alpha 0.01

Exit codes:
    0  ran cleanly; table emitted to stdout
    2  inputs missing (no parquet, no config) — error logged to stderr
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

from lerobot_bench.policies import PolicyRegistry
from lerobot_bench.stats import holm_bonferroni

logger = logging.getLogger("family-correction")

DEFAULT_RESULTS = Path("results/sweep-full/results.parquet")
DEFAULT_POLICIES_YAML = Path("configs/policies.yaml")

EXIT_OK = 0
EXIT_CANNOT_RUN = 2


def _exact_two_sided_binom_p(successes: int, n: int, paper_rate: float) -> float:
    """Two-sided exact binomial p for ``H_0: p == paper_rate``.

    Wraps ``scipy.stats.binomtest`` (preferred over the deprecated
    ``binom_test``). Two-sided p is the standard "method='two-sided'"
    minlike convention: sum of binomial PMF over all k whose PMF is
    no greater than the observed k's PMF. Matches R's ``binom.test``.
    """
    res = scipy_stats.binomtest(successes, n, paper_rate, alternative="two-sided")
    return float(res.pvalue)


def _aggregate_cells(df: pd.DataFrame, registry: PolicyRegistry) -> pd.DataFrame:
    """Per-cell roll-up with paper-rate lookup; one row per (policy, env)."""
    grouped = (
        df.groupby(["policy", "env"], sort=True)
        .agg(n=("success", "count"), successes=("success", "sum"))
        .reset_index()
    )
    grouped["success_rate"] = grouped["successes"] / grouped["n"]

    paper_rates: list[float | None] = []
    for _, row in grouped.iterrows():
        try:
            spec = registry.get(str(row["policy"]))
        except (KeyError, ValueError):
            paper_rates.append(None)
            continue
        paper_map = getattr(spec, "paper_reported_success", None) or {}
        paper_rates.append(paper_map.get(str(row["env"])))
    grouped["paper_rate"] = paper_rates
    return grouped


def build_family_table(df: pd.DataFrame, registry: PolicyRegistry, alpha: float) -> pd.DataFrame:
    """Return a DataFrame with raw + Holm-adjusted p-values per cell.

    Rows without a ``paper_rate`` are kept (for full visibility) but
    excluded from the family used for correction. Their p_raw / p_adj /
    reject columns are ``NaN`` / ``False``.
    """
    table = _aggregate_cells(df, registry)
    in_family = table["paper_rate"].notna()
    family = table[in_family].copy()

    p_raw_values: list[float] = []
    for _, row in family.iterrows():
        p_raw_values.append(
            _exact_two_sided_binom_p(
                int(row["successes"]),
                int(row["n"]),
                float(row["paper_rate"]),
            )
        )
    family["p_raw"] = p_raw_values

    if p_raw_values:
        adjusted, reject = holm_bonferroni(p_raw_values, alpha=alpha)
        family["p_adj"] = adjusted
        family["reject"] = reject
    else:
        family["p_adj"] = []
        family["reject"] = []

    # Re-attach to the full table so cells without paper_rate are still listed.
    table["p_raw"] = pd.Series(dtype="float64")
    table["p_adj"] = pd.Series(dtype="float64")
    table["reject"] = pd.Series(dtype="bool")
    table.loc[family.index, ["p_raw", "p_adj", "reject"]] = family[
        ["p_raw", "p_adj", "reject"]
    ].values
    return table


def render_markdown(table: pd.DataFrame, alpha: float) -> str:
    """Markdown table; cells without paper_rate get ``-`` placeholders."""
    in_family = table["paper_rate"].notna()
    m = int(in_family.sum())
    n_rejected = int(table.loc[in_family, "reject"].sum()) if m > 0 else 0

    lines: list[str] = []
    lines.append(f"# Holm-Bonferroni family correction (m={m} cells in family, alpha={alpha})")
    lines.append("")
    lines.append(
        f"**{n_rejected}/{m} cells reject H_0 (rate == paper_rate) after "
        f"family-wise correction at alpha={alpha}.**"
    )
    lines.append("")
    lines.append(
        "| policy | env | n | success_rate | paper_rate | p_raw | p_adj | reject@" + f"{alpha} |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")

    # In-family rows first, sorted by adjusted p (most significant first),
    # then non-family rows alphabetically.
    family_rows = table[in_family].sort_values("p_adj", kind="stable")
    other_rows = table[~in_family].sort_values(["policy", "env"], kind="stable")

    for _, row in family_rows.iterrows():
        lines.append(
            f"| `{row['policy']}` | `{row['env']}` | {int(row['n'])} | "
            f"{row['success_rate']:.3f} | {row['paper_rate']:.3f} | "
            f"{row['p_raw']:.3g} | {row['p_adj']:.3g} | "
            f"{'YES' if bool(row['reject']) else 'no'} |"
        )

    if len(other_rows):
        lines.append("")
        lines.append("Cells without a `paper_reported_success` entry (excluded from family):")
        lines.append("")
        lines.append("| policy | env | n | success_rate |")
        lines.append("|---|---|---:|---:|")
        for _, row in other_rows.iterrows():
            lines.append(
                f"| `{row['policy']}` | `{row['env']}` | {int(row['n'])} | "
                f"{row['success_rate']:.3f} |"
            )

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="family-correction",
        description=(
            "Apply Holm-Bonferroni FWER correction across the cell matrix.\n"
            "Computes one two-sided exact binomial p-value per (policy, env)\n"
            "vs the policy's paper_reported_success[env]; cells without a\n"
            "paper rate are reported separately and not corrected."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        metavar="PARQUET",
        help="Path to results.parquet from the sweep.",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=DEFAULT_POLICIES_YAML,
        metavar="YAML",
        help="Path to the policy registry YAML (joined to look up paper_rate).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args(argv)

    if not args.results.exists():
        logger.error("results parquet not found: %s", args.results)
        return EXIT_CANNOT_RUN
    try:
        df = pd.read_parquet(args.results)
    except Exception as exc:
        logger.error("could not read results parquet: %s (%s)", args.results, exc)
        return EXIT_CANNOT_RUN

    try:
        registry = PolicyRegistry.from_yaml(args.policies)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("could not load policy registry: %s", exc)
        return EXIT_CANNOT_RUN

    table = build_family_table(df, registry, alpha=args.alpha)
    sys.stdout.write(render_markdown(table, alpha=args.alpha) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(run())
