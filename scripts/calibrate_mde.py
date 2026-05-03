#!/usr/bin/env python3
"""Compute the paired-MDE simulation grid that backs ``docs/MDE_TABLE.md``.

For each ``(p, ρ, Δ)`` cell on the spec grid, simulate ``N_OUTER``
synthetic paired cells of N=250 paired Bernoulli draws with marginals
``(p+Δ, p)`` and Pearson correlation ρ. For each simulated cell, run
``paired_delta_bootstrap`` and record whether the 95% CI on Δ excludes
zero. The empirical rejection rate is the **power** at that ``(p, ρ, Δ)``;
the smallest Δ with power ≥ 0.80 is the **MDE**.

The grid lives in this script as a top-level constant so the doc and the
unit tests both have a single source of truth. Re-running the script with
the locked seed (``numpy.random.default_rng(seed=42)``) reproduces the
numbers in ``docs/MDE_TABLE.md`` § 2b to the bit.

This script imports nothing from torch / lerobot / gymnasium. It is safe
to run on the CI box without GPU. Stats live entirely in
``lerobot_bench.stats``.

Usage::

    python scripts/calibrate_mde.py                       # full grid, ~5 min
    python scripts/calibrate_mde.py --pilot               # 100 outer iters, ~30 s
    python scripts/calibrate_mde.py --json-out FILE       # dump raw power values

Exit codes:
    0  success, table printed (and JSON written if requested)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Make ``lerobot_bench`` importable when the script is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lerobot_bench.stats import paired_delta_bootstrap  # noqa: E402

# Locked grid — these are the columns / rows in docs/MDE_TABLE.md § 2.
PS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98)
DELTAS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
RHOS: tuple[float, ...] = (0.0, 0.3, 0.5)
N_PAIRED: int = 250  # the contracted per-cell sample size
N_OUTER_DEFAULT: int = 1000  # outer power-simulation iterations
N_BOOT_DEFAULT: int = 2000  # within-resample bootstrap draws (production = 10000)
POWER_TARGET: float = 0.80
MASTER_SEED: int = 42  # locked — quoted in docs/MDE_TABLE.md


def paired_bernoulli(
    n: int,
    p_a: float,
    p_b: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Sample ``n`` paired Bernoulli draws with target marginals and ρ.

    Joint distribution is parameterised directly via the four-cell
    contingency table, which is exact for binary outcomes (no Gaussian-
    copula thresholding artefact). For binary X, Y with means p_a, p_b
    and Pearson correlation ρ:

        Cov(X, Y) = ρ · sqrt(p_a (1-p_a) p_b (1-p_b))
        P(X=1, Y=1) = p_a p_b + Cov(X, Y)
        P(X=1, Y=0) = p_a − P(X=1, Y=1)
        P(X=0, Y=1) = p_b − P(X=1, Y=1)
        P(X=0, Y=0) = 1 − p_a − p_b + P(X=1, Y=1)

    Probabilities are clipped to [0, 1] and renormalised to handle the
    edge case where the requested ρ violates the Fréchet-Hoeffding bound
    (e.g. ρ=1 with p_a ≠ p_b). The clipping is silent because the grid
    in this script never gets close to those edges.
    """
    cov = rho * float(np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b)))
    p11 = p_a * p_b + cov
    p10 = p_a - p11
    p01 = p_b - p11
    p00 = 1.0 - p_a - p_b + p11
    probs = np.clip(np.array([p00, p01, p10, p11], dtype=np.float64), 0.0, 1.0)
    probs = probs / probs.sum()
    cat = rng.choice(4, size=n, p=probs)
    a = (cat == 2) | (cat == 3)
    b = (cat == 1) | (cat == 3)
    return a, b


def power_at(
    n: int,
    p: float,
    delta: float,
    rho: float,
    n_outer: int,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """Empirical probability that the bootstrap CI on Δ excludes zero."""
    rejections = 0
    for _ in range(n_outer):
        a, b = paired_bernoulli(n, p + delta, p, rho, rng)
        result = paired_delta_bootstrap(a, b, rng=rng, n_resamples=n_boot)
        if result.lo > 0.0 or result.hi < 0.0:
            rejections += 1
    return rejections / n_outer


def compute_mde_grid(
    n: int = N_PAIRED,
    n_outer: int = N_OUTER_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = MASTER_SEED,
) -> dict[tuple[float, float, float], float]:
    """Run the full ``(p × ρ × Δ)`` simulation grid.

    Returns a dict keyed by ``(p, rho, delta)`` with the empirical power.
    Cells where ``p + delta > 1`` are stored as ``float('nan')`` (skipped).
    The master RNG is constructed once; per-cell calls share it so the
    full grid is deterministic under a fixed ``seed``.
    """
    rng = np.random.default_rng(seed)
    out: dict[tuple[float, float, float], float] = {}
    for p in PS:
        for rho in RHOS:
            for delta in DELTAS:
                if p + delta > 1.0:
                    out[(p, rho, delta)] = float("nan")
                    continue
                out[(p, rho, delta)] = power_at(n, p, delta, rho, n_outer, n_boot, rng)
    return out


def mde_summary(
    grid: dict[tuple[float, float, float], float],
) -> list[dict[str, float | str]]:
    """For each (p, ρ), pick the smallest Δ achieving ≥ POWER_TARGET."""
    rows: list[dict[str, float | str]] = []
    for p in PS:
        for rho in RHOS:
            mde: float | None = None
            pw: float | None = None
            for d in DELTAS:
                v = grid[(p, rho, d)]
                if not np.isnan(v) and v >= POWER_TARGET:
                    mde = d
                    pw = v
                    break
            rows.append(
                {
                    "p": p,
                    "rho": rho,
                    "mde": mde if mde is not None else float("nan"),
                    "power_at_mde": pw if pw is not None else float("nan"),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Smaller (n_outer=100) run for a quick sanity check (~30 s).",
    )
    parser.add_argument(
        "--n-outer",
        type=int,
        default=None,
        help=f"Outer iters per cell (default {N_OUTER_DEFAULT}; --pilot sets 100).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT_DEFAULT,
        help=f"Bootstrap resamples per outer iter (default {N_BOOT_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Master RNG seed (default {MASTER_SEED} — matches docs/MDE_TABLE.md).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump the raw power grid as JSON.",
    )
    args = parser.parse_args()

    n_outer = args.n_outer if args.n_outer is not None else (100 if args.pilot else N_OUTER_DEFAULT)

    print(
        f"Running paired-MDE grid: N={N_PAIRED}, n_outer={n_outer}, "
        f"n_boot={args.n_boot}, seed={args.seed}"
    )
    t0 = time.time()
    grid = compute_mde_grid(N_PAIRED, n_outer, args.n_boot, args.seed)
    print(f"Grid computed in {time.time() - t0:.1f} s")

    print()
    print(f"=== MDE table (smallest Δ achieving ≥ {POWER_TARGET:.0%} power) ===")
    print(f"{'p':>6} {'rho':>6} {'MDE':>10} {'power@MDE':>12}")
    for row in mde_summary(grid):
        mde_str = f"{row['mde']:.2f}" if not np.isnan(float(row["mde"])) else ">0.20"
        pw_str = (
            f"{row['power_at_mde']:.3f}" if not np.isnan(float(row["power_at_mde"])) else "<0.80"
        )
        print(f"{float(row['p']):>6.2f} {float(row['rho']):>6.1f} {mde_str:>10} {pw_str:>12}")

    if args.json_out is not None:
        # JSON keys must be strings; flatten the (p, rho, delta) tuple.
        as_json = {f"{k[0]:.2f}_{k[1]:.1f}_{k[2]:.2f}": v for k, v in grid.items()}
        args.json_out.write_text(json.dumps(as_json, indent=2))
        print(f"\nRaw grid written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
