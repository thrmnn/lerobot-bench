#!/usr/bin/env python3
"""Splice the corrected act×aloha_transfer_cube rows into the canonical parquet.

Provenance / why this exists
============================
A normalization bug (#51) made ``act`` evaluate at a pooled
``aloha_transfer_cube`` success of ~0.016 instead of the corrected
~0.824 [0.772, 0.866]. The fix landed at code_sha ``7361d96`` and the
re-run was persisted to a SEPARATE file
``results/sweep-full/results-act-rerun.parquet`` (pooled 0.824, N=250 at
``7361d96`` + a 50-row follow-on at ``fa6f9ac``) — never merged back into
the canonical ``results/sweep-full/results.parquet``.

Task #83 ("merge corrected act rows") was marked done but the merge was
never actually persisted: both the canonical parquet AND the
``_publish_staging`` copy still carry the STALE pre-#51 rows, while every
doc surface now asserts the corrected 0.824. This script closes that gap
reproducibly.

It REPLACES the act×aloha rows in the canonical parquet with the
corrected rows from the act-rerun parquet, leaving every other cell and
the total row count untouched, and writes the result atomically.

IMPORTANT — runs on gitignored data
===================================
``results/`` is gitignored. This script is the tracked, reviewable
*tooling*; the actual data mutation must be run by an operator on the
main-tree gitignored parquets BEFORE publish::

    python scripts/merge_corrected_act_rows.py --dry-run   # inspect deltas
    python scripts/merge_corrected_act_rows.py --staging   # canonical + staging

The publish preflight (:func:`scripts.publish_results._preflight`) refuses
to ship while the stale rows are present, so forgetting this step is now
un-shippable rather than silently wrong.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from lerobot_bench.checkpointing import (
    _ROW_KEY,
    RESULT_SCHEMA,
    _atomic_write_parquet,
    load_results,
)

logger = logging.getLogger("merge-corrected-act-rows")

# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #

DEFAULT_SWEEP_DIR = Path("results/sweep-full")
CANONICAL_NAME = "results.parquet"
ACT_RERUN_NAME = "results-act-rerun.parquet"
STAGING_SUBDIR = "_publish_staging"

# The corrected act×aloha re-run is expected to carry exactly this many
# episode rows (5 seeds × 50 episodes), pooling to ~0.824.
EXPECTED_RERUN_ROWS = 250
EXPECTED_RERUN_RATE = 0.824
RERUN_RATE_TOL = 0.05  # accept [0.774, 0.874]

# Post-merge the canonical act×aloha pooled rate must land in this band;
# anything outside means we spliced the wrong rows.
POST_MERGE_RATE_LO = 0.77
POST_MERGE_RATE_HI = 0.87


# --------------------------------------------------------------------- #
# Row selection                                                         #
# --------------------------------------------------------------------- #


def _act_aloha_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of act×aloha_transfer_cube rows.

    ``env`` is matched on substring ``aloha`` so the mask is robust to a
    ``aloha_transfer_cube`` vs ``aloha-transfer-cube`` rename; ``policy``
    is matched exactly.
    """
    return (df["policy"] == "act") & df["env"].astype(str).str.contains("aloha")


def _pooled_rate(df: pd.DataFrame) -> float:
    """Pooled success rate over ``df`` (mean of the boolean ``success``)."""
    if len(df) == 0:
        return float("nan")
    return float(df["success"].astype(float).mean())


def _cell_count(df: pd.DataFrame) -> int:
    """Number of distinct (policy, env, seed) cells in ``df``."""
    return df[["policy", "env", "seed"]].drop_duplicates().shape[0]


# --------------------------------------------------------------------- #
# Core merge                                                            #
# --------------------------------------------------------------------- #


def merge_corrected_act_rows(
    canonical: pd.DataFrame,
    rerun: pd.DataFrame,
) -> pd.DataFrame:
    """Return a new DataFrame with the canonical's act×aloha rows replaced.

    The rerun's act×aloha rows are spliced in for the canonical's; every
    other row is carried through unchanged and the total row count is
    preserved. Raises :class:`ValueError` if the rerun does not look like
    the expected #51 re-run, or if the splice would change the row count,
    cell count, or land the pooled rate outside the corrected band.
    """
    rerun_act = rerun[_act_aloha_mask(rerun)].copy()
    canon_act = canonical[_act_aloha_mask(canonical)]

    # --- guard the rerun is the expected #51 re-run ------------------- #
    if len(rerun_act) != EXPECTED_RERUN_ROWS:
        raise ValueError(
            f"act-rerun has {len(rerun_act)} act×aloha rows, expected {EXPECTED_RERUN_ROWS}"
        )
    rerun_rate = _pooled_rate(rerun_act)
    if abs(rerun_rate - EXPECTED_RERUN_RATE) > RERUN_RATE_TOL:
        raise ValueError(
            f"act-rerun act×aloha pooled rate {rerun_rate:.3f} is outside "
            f"{EXPECTED_RERUN_RATE} ± {RERUN_RATE_TOL}; refusing to splice"
        )

    if len(canon_act) != EXPECTED_RERUN_ROWS:
        # Not fatal on its own (a partial canonical could in principle be
        # shorter), but the count-preservation assertion below would then
        # fail anyway. Surface it loudly here for a readable error.
        logger.warning(
            "canonical has %d act×aloha rows (rerun has %d); "
            "row-count preservation will only hold if these match",
            len(canon_act),
            len(rerun_act),
        )

    n_before = len(canonical)
    cells_before = _cell_count(canonical)

    # --- splice: drop canonical's act×aloha, append rerun's ----------- #
    kept = canonical[~_act_aloha_mask(canonical)]
    merged = pd.concat([kept, rerun_act[list(RESULT_SCHEMA)]], ignore_index=True)
    merged = merged.sort_values(list(_ROW_KEY), kind="stable").reset_index(drop=True)
    merged = merged[list(RESULT_SCHEMA)]

    # --- post-conditions --------------------------------------------- #
    if len(merged) != n_before:
        raise ValueError(f"row count changed during merge: {n_before} -> {len(merged)}")
    if _cell_count(merged) != cells_before:
        raise ValueError(
            f"cell count changed during merge: {cells_before} -> {_cell_count(merged)}"
        )
    post_rate = _pooled_rate(merged[_act_aloha_mask(merged)])
    if not (POST_MERGE_RATE_LO <= post_rate <= POST_MERGE_RATE_HI):
        raise ValueError(
            f"post-merge act×aloha pooled rate {post_rate:.3f} outside "
            f"[{POST_MERGE_RATE_LO}, {POST_MERGE_RATE_HI}]"
        )
    return merged


def process_file(
    canonical_path: Path,
    rerun_path: Path,
    *,
    dry_run: bool,
) -> tuple[float, float, int]:
    """Merge one canonical file in place (or describe it under ``--dry-run``).

    Returns ``(before_rate, after_rate, n_rows)`` for the act×aloha cell.
    """
    canonical = load_results(canonical_path)
    rerun = load_results(rerun_path)

    before_rate = _pooled_rate(canonical[_act_aloha_mask(canonical)])
    merged = merge_corrected_act_rows(canonical, rerun)
    after_rate = _pooled_rate(merged[_act_aloha_mask(merged)])

    logger.info(
        "%s: act×aloha pooled %.3f -> %.3f (total rows %d, unchanged)",
        canonical_path,
        before_rate,
        after_rate,
        len(merged),
    )

    if dry_run:
        logger.info("--dry-run: not writing %s", canonical_path)
    else:
        _atomic_write_parquet(canonical_path, merged)
        logger.info("wrote %s", canonical_path)

    return before_rate, after_rate, len(merged)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Splice corrected act×aloha_transfer_cube rows (#51 fix, "
            "code_sha 7361d96) into the canonical results parquet."
        ),
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help="Directory holding results.parquet + results-act-rerun.parquet.",
    )
    parser.add_argument(
        "--canonical-path",
        type=Path,
        default=None,
        help="Override path to the canonical parquet (default: <sweep-dir>/results.parquet).",
    )
    parser.add_argument(
        "--rerun-path",
        type=Path,
        default=None,
        help="Override path to the act-rerun parquet (default: <sweep-dir>/results-act-rerun.parquet).",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Also process the _publish_staging/results.parquet copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print before/after pooled rate + row deltas without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    rerun_path = args.rerun_path or (args.sweep_dir / ACT_RERUN_NAME)
    canonical_path = args.canonical_path or (args.sweep_dir / CANONICAL_NAME)

    targets = [canonical_path]
    if args.staging:
        targets.append(args.sweep_dir / STAGING_SUBDIR / CANONICAL_NAME)

    for target in targets:
        if not target.exists():
            logger.error("canonical parquet not found: %s", target)
            return 3
    if not rerun_path.exists():
        logger.error("act-rerun parquet not found: %s", rerun_path)
        return 3

    try:
        for target in targets:
            process_file(target, rerun_path, dry_run=args.dry_run)
    except ValueError as exc:
        logger.error("merge failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
