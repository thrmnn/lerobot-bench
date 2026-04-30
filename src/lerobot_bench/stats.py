"""Statistical helpers for lerobot-bench leaderboard claims.

See ``docs/DESIGN.md`` § Methodology for the protocol these functions
implement: bootstrap CI over the flat list of (seed, episode) outcomes
per cell; paired bootstrap on Δsuccess for cell-vs-cell comparisons;
paired Wilcoxon signed-rank for the same when episodes are pairable;
Cohen's h for effect size on proportions.

Every stochastic function takes an explicit ``rng`` (no hidden global
state). Functions are pure: same inputs + same RNG state → same output.

The unit of resampling is the *episode*, not the seed. Bootstrapping
over the 5 seeds (or any 5 anything) gives huge CIs and is wrong; the
flat list of ``5 × n_episodes_per_seed`` binary outcomes per cell is
what these functions consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap point estimate and percentile CI."""

    mean: float
    lo: float
    hi: float
    n_resamples: int
    ci: float


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired Wilcoxon signed-rank test result.

    ``n_zero_diffs`` is reported separately because Wilcoxon drops
    zero-difference pairs by convention, and that drop materially
    changes the effective sample size for binary outcomes.
    """

    statistic: float
    pvalue: float
    n_pairs: int
    n_zero_diffs: int


def bootstrap_ci(
    outcomes: NDArray[np.bool_],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator,
) -> BootstrapResult:
    """Percentile-bootstrap CI over a flat list of binary episode outcomes.

    For each of ``n_resamples`` iterations, draw ``len(outcomes)`` indices
    with replacement and compute the resampled mean. The ``ci``-percentile
    interval over those means is returned. For a fixed Bernoulli at large
    n this converges to the Wilson interval; ``test_bootstrap_ci.py``
    asserts that.

    Args:
        outcomes: 1-D bool array, ``True`` = success.
        n_resamples: number of bootstrap iterations (default 10,000).
        ci: confidence level in (0, 1) (default 0.95).
        rng: numpy Generator. Required; no hidden global state.
    """
    if outcomes.ndim != 1:
        raise ValueError(f"outcomes must be 1-D, got shape {outcomes.shape}")
    if outcomes.size == 0:
        raise ValueError("outcomes must be non-empty")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    n = outcomes.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = outcomes[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo, hi = np.quantile(resampled_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        mean=float(outcomes.mean()),
        lo=float(lo),
        hi=float(hi),
        n_resamples=n_resamples,
        ci=ci,
    )


def paired_delta_bootstrap(
    a: NDArray[np.bool_],
    b: NDArray[np.bool_],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator,
) -> BootstrapResult:
    """Paired bootstrap of the success-rate delta ``mean(a) − mean(b)``.

    Episodes are paired by index: ``a[i]`` and ``b[i]`` must come from
    the same ``(seed_idx, episode_idx)``. Each resample draws shared
    indices into both arrays and computes ``mean(a[idx]) − mean(b[idx])``,
    yielding a CI on the delta that respects the pairing.

    Use this for cross-cell comparisons when the seeding contract was
    identical AND ``n_episodes_per_seed`` matches across cells. If those
    conditions don't hold, fall back to two independent
    :func:`bootstrap_ci` calls and document the loss of pairing power.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must share shape; got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("inputs must be non-empty")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    n = a.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo, hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        mean=float(a.mean() - b.mean()),
        lo=float(lo),
        hi=float(hi),
        n_resamples=n_resamples,
        ci=ci,
    )


def paired_wilcoxon(
    a: NDArray[np.bool_],
    b: NDArray[np.bool_],
) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test on per-episode outcomes.

    Episodes are paired by index. Pairs where ``a[i] == b[i]`` are
    dropped per the standard Wilcoxon convention (zero differences carry
    no signed-rank information). For binary paired data this reduces to
    McNemar in the limit; we use Wilcoxon for consistency with how it
    generalises to non-binary rewards in v2.

    Returns a :class:`WilcoxonResult` with ``pvalue=1.0`` and
    ``statistic=NaN`` when every pair is a tie (no information).
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must share shape; got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got shape {a.shape}")

    diffs = a.astype(np.int8) - b.astype(np.int8)
    n_zero_diffs = int((diffs == 0).sum())
    nonzero = diffs[diffs != 0]
    n_pairs = int(nonzero.size)

    if n_pairs == 0:
        return WilcoxonResult(
            statistic=float("nan"),
            pvalue=1.0,
            n_pairs=0,
            n_zero_diffs=n_zero_diffs,
        )

    result = scipy_stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        n_pairs=n_pairs,
        n_zero_diffs=n_zero_diffs,
    )


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h: effect size for the difference between two proportions.

    .. math::

        h = 2 \\cdot \\arcsin(\\sqrt{p_1}) - 2 \\cdot \\arcsin(\\sqrt{p_2})

    Conventional interpretation: ``|h| < 0.2`` small, ``< 0.5`` medium,
    ``≥ 0.8`` large. Always report alongside Δsuccess — a "significant"
    2-pp delta is a small effect and should not be framed as a
    meaningful improvement.
    """
    if not 0.0 <= p1 <= 1.0:
        raise ValueError(f"p1 must be in [0, 1], got {p1}")
    if not 0.0 <= p2 <= 1.0:
        raise ValueError(f"p2 must be in [0, 1], got {p2}")
    return float(2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2)))


def wilson_ci(successes: int, n: int, *, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion.

    Closed-form binomial CI used as a sanity reference for
    :func:`bootstrap_ci`: at large n the bootstrap interval converges to
    this. From Wilson, "Probable Inference, the Law of Succession, and
    Statistical Inference", JASA 1927.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} not in [0, {n}]")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    z = float(scipy_stats.norm.ppf(0.5 + ci / 2.0))
    p_hat = successes / n
    z2_n = z * z / n
    center = (p_hat + z2_n / 2.0) / (1.0 + z2_n)
    half = (z * np.sqrt((p_hat * (1.0 - p_hat) + z2_n / 4.0) / n)) / (1.0 + z2_n)
    return float(center - half), float(center + half)
