#!/usr/bin/env python3
"""Regenerate ``examples/results-mini.parquet`` — the zero-GPU leaderboard view.

WHY THIS EXISTS
---------------
The canonical per-episode parquet (``results/sweep-full/results.parquet``)
lives on the Hub and is ``.gitignore``-d, so a fresh ``git clone`` has *no*
benchmark data on disk. That meant ``examples/read_results.py`` could not show
a real number without first running an eval (GPU) or downloading the Hub
dataset. ``examples/results-mini.parquet`` closes that gap: a tiny, committed,
pre-aggregated "honest shippable view" of the v1 leaderboard headline cells
that a reviewer can read instantly with ``python examples/read_results.py`` —
no GPU, no Hub download, no full parquet.

PROVENANCE (read, don't invent)
-------------------------------
Every row here is reconstructed deterministically from numbers already
committed to this repo as the published v1 leaderboard — NOT hand-typed
estimates:

  * The cell set + success rates + Wilson 95% CIs are the six headline cells
    rendered in ``docs/assets/fig-v1-leaderboard-bars.svg`` and registered as
    the cross-artifact source of truth in
    ``tests/test_headline_value_consistency.py`` (REGISTRY).
  * ``n_success`` is recovered as ``round(success_rate * n_episodes)``, and
    ``wilson_lo`` / ``wilson_hi`` are recomputed from
    ``embodimetry.stats.wilson_ci(n_success, n_episodes)``. Each recomputed
    interval reproduces the published CI to three decimals (asserted below),
    so this file cannot silently drift from the headline figure.

CONSISTENCY WITH THE 0.824 HEADLINE
-----------------------------------
The ``act × aloha_transfer_cube`` cell carries the CORRECTED canonical
**0.824 [0.772, 0.866]** (N=250), the post-#51 norm-fix reading that the
rerun parquet (``results/sweep-full/results-act-rerun.parquet``) and the
cross-artifact guard enforce — NOT the stale 0.016 still sitting in the
un-merged canonical parquet (#177). This mirrors how
``src/embodimetry/figures.py`` (``_act_aloha_rerun_cell``) sources the cell.

V1 POLICY GATE
--------------
Only the v1 policy set (``embodimetry.leaderboard_filter.V1_POLICIES``) appears
here; ``xvla_libero`` and the pi0 family are EXCLUDED by construction — the
headline cells contain none of them, and the final frame is run through
``filter_to_v1_policies`` as a belt-and-braces gate so a future edit that adds
a non-v1 cell fails loudly rather than shipping an ``xvla`` 0.000 row.

Run it::

    python scripts/make_results_mini.py            # regenerate in place
    python scripts/make_results_mini.py --check     # verify on-disk matches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from embodimetry.leaderboard_filter import filter_to_v1_policies
from embodimetry.stats import wilson_ci

OUT_PATH = Path(__file__).resolve().parents[1] / "examples" / "results-mini.parquet"

# The six published v1 leaderboard headline cells, transcribed from
# docs/assets/fig-v1-leaderboard-bars.svg (and pinned in
# tests/test_headline_value_consistency.py REGISTRY). Each tuple is
# (policy, env, success_rate, n_episodes, published_wilson_lo_hi). The
# act cell is the CORRECTED 0.824 (post-#51), NOT the stale 0.016.
_HEADLINE_CELLS: tuple[tuple[str, str, float, int, tuple[float, float]], ...] = (
    ("act", "aloha_transfer_cube", 0.824, 250, (0.772, 0.866)),
    ("diffusion_policy", "pusht", 0.816, 125, (0.739, 0.874)),
    ("smolvla_libero", "libero_spatial", 0.776, 250, (0.720, 0.823)),
    ("smolvla_libero", "libero_object", 0.528, 250, (0.466, 0.589)),
    ("smolvla_libero", "libero_goal", 0.928, 250, (0.889, 0.954)),
    ("smolvla_libero", "libero_10", 0.252, 250, (0.202, 0.309)),
)


def build_mini() -> pd.DataFrame:
    """Reconstruct the aggregated mini-leaderboard frame, deterministically.

    For each headline cell: recover ``n_success`` from the published rate and
    N, recompute the Wilson interval, and assert it reproduces the committed
    CI to three decimals (so the file can never drift from the figure). The
    assembled frame is passed through the v1 policy gate as a final guard.
    """
    rows: list[dict[str, object]] = []
    for policy, env, rate, n, (pub_lo, pub_hi) in _HEADLINE_CELLS:
        n_success = round(rate * n)
        lo, hi = wilson_ci(n_success, n)
        assert round(lo, 3) == pub_lo and round(hi, 3) == pub_hi, (
            f"{policy} x {env}: recomputed Wilson CI [{lo:.3f}, {hi:.3f}] does not "
            f"reproduce the published [{pub_lo}, {pub_hi}] — the headline figure and "
            f"this generator have drifted; fix the source before regenerating."
        )
        rows.append(
            {
                "policy": policy,
                "env": env,
                "n_episodes": int(n),
                "n_success": int(n_success),
                "success_rate": n_success / n,
                "wilson_lo": round(lo, 6),
                "wilson_hi": round(hi, 6),
            }
        )
    df = pd.DataFrame(rows)
    # Belt-and-braces: the cells above contain no non-v1 policy, but run the
    # canonical public-surface gate so a future edit that adds an xvla/pi0 cell
    # is dropped here rather than shipping in the committed file.
    df = filter_to_v1_policies(df)
    return df.sort_values(["policy", "env"], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the on-disk parquet matches a fresh build; do not write.",
    )
    args = parser.parse_args(argv)

    fresh = build_mini()
    if args.check:
        if not OUT_PATH.exists():
            print(f"[make-results-mini] MISSING: {OUT_PATH}", file=sys.stderr)
            return 1
        on_disk = pd.read_parquet(OUT_PATH)
        if not fresh.equals(on_disk):
            print(
                "[make-results-mini] DRIFT: on-disk parquet differs from a fresh build. "
                "Re-run `python scripts/make_results_mini.py` and commit.",
                file=sys.stderr,
            )
            return 1
        print(f"[make-results-mini] OK: {OUT_PATH} matches a fresh build ({len(fresh)} rows).")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(OUT_PATH, index=False)
    print(f"[make-results-mini] wrote {len(fresh)} rows -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
