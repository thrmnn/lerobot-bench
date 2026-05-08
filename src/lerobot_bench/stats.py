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
    Statistical Inference", JASA 1927; cross-checked against
    Agresti & Coull (1998), "Approximate Is Better than 'Exact' for
    Interval Estimation of Binomial Proportions", The American
    Statistician 52:119-126.

    Note on signature: the spec for the audit referenced an ``alpha``
    keyword. We keep the existing ``ci`` keyword (used by ``space/``,
    ``docs/MDE_TABLE.md``, and the leaderboard helpers) and expose
    :func:`wilson_halfwidth_at_p` for the ``alpha``-style call.

    Args:
        successes: number of successes; must satisfy ``0 <= successes <= n``.
        n: number of trials; must be ``> 0``.
        ci: confidence level in ``(0, 1)`` (default 0.95).

    Returns:
        ``(lo, hi)`` Wilson interval bounds, both in ``[0, 1]``.
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


def wilson_halfwidth_at_p(p: float, n: int, *, alpha: float = 0.05) -> float:
    """Wilson score interval half-width at proportion ``p`` and ``n`` trials.

    Used by ``docs/MDE_TABLE.md`` and the leaderboard "inconclusive at
    this N" gate (DESIGN.md § Methodology). Computes the half-width as
    ``(hi - lo) / 2`` where ``(lo, hi) = wilson_ci(round(p · n), n)``.

    The ``round(p · n)`` integer discretisation matches the doc's
    convention; ``HW(p)`` and ``HW(1 - p)`` differ by at most one unit
    of integer rounding when ``p · n`` is non-integer.

    Args:
        p: target proportion in ``[0, 1]``.
        n: trial count; must be positive.
        alpha: significance level; the CI is at ``1 - alpha``
            (default ``0.05`` ⇒ 95% CI).

    Returns:
        The Wilson 95% half-width at ``(p, n)``, a non-negative float.

    References:
        Wilson (1927); Agresti & Coull (1998). Closed-form, no RNG.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    successes = round(p * n)
    successes = max(0, min(n, successes))
    lo, hi = wilson_ci(successes, n, ci=1.0 - alpha)
    return float((hi - lo) / 2.0)


def mcnemar_paired(b: int, c: int, *, exact: bool = True) -> tuple[float, float]:
    """Paired McNemar test on a 2×2 table of paired binary outcomes.

    For paired binary data the only informative cells of the 2×2 table
    are the *discordant* pairs:

    * ``b`` = pairs where method A succeeded and method B failed.
    * ``c`` = pairs where method A failed and method B succeeded.

    The null hypothesis is ``P(b) = P(c) = 0.5`` given a discordant
    pair. With ``n = b + c``:

    * ``exact=True`` and ``n <= 25``: exact two-sided binomial test on
      ``min(b, c) ~ Binomial(n, 0.5)``. The returned ``statistic`` is
      ``min(b, c)`` (so the caller can reproduce the exact p-value
      against ``scipy.stats.binom``).
    * Otherwise: chi-square with continuity correction (Edwards, 1948),

      .. math::

          \\chi^2 = (|b - c| - 1)^2 / (b + c)

      with ``df = 1``. ``statistic`` is the χ² value.

    Threshold rationale: at ``n > 25`` the χ²-with-continuity-correction
    p-value matches the exact binomial to ~3 decimal places, so the
    speedup is essentially free; below 25 the χ² approximation is
    materially anticonservative on the tails. See Fagerland, Lydersen &
    Laake (2013), "The McNemar test for binary matched-pairs data:
    mid-p and asymptotic are better than exact conditional".

    Edge cases:

    * ``b = c = 0``: no discordant pairs, no information; returns
      ``(0.0, 1.0)``.
    * ``b = c``: maximally non-significant; returns ``(0.0, 1.0)`` for
      the χ² branch and the appropriate exact-binomial p for the exact
      branch.

    Args:
        b: count of (A succeeds, B fails) pairs.
        c: count of (A fails, B succeeds) pairs.
        exact: if ``True``, switch to the exact binomial branch when
            ``b + c <= 25``.

    Returns:
        ``(statistic, pvalue)``. Two-sided p-value.

    References:
        McNemar (1947), "Note on the sampling error of the difference
        between correlated proportions or percentages", Psychometrika
        12:153-157.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative, got b={b}, c={c}")

    n = b + c
    if n == 0:
        return 0.0, 1.0

    if exact and n <= 25:
        # Exact two-sided binomial on min(b, c) ~ Binomial(n, 0.5). Under
        # the null the distribution is symmetric, so two-sided p reduces
        # to 2 · P(X <= min(b, c)) clipped to 1.0.
        k = min(b, c)
        p_one_sided = float(scipy_stats.binom.cdf(k, n, 0.5))
        pvalue = min(1.0, 2.0 * p_one_sided)
        return float(k), pvalue

    # Chi-square with continuity correction.
    chi2 = float((abs(b - c) - 1) ** 2) / float(n)
    pvalue = float(scipy_stats.chi2.sf(chi2, df=1))
    return chi2, pvalue


def bootstrap_pivotal_ci(
    values: NDArray[np.floating] | NDArray[np.bool_],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Pivotal (basic) bootstrap CI on the mean of ``values``.

    The percentile bootstrap CI ``(q_{alpha/2}, q_{1 - alpha/2})`` of
    the resampled means has a known bias when the sampling distribution
    of the statistic is asymmetric. The pivotal (a.k.a. "basic") form
    corrects this by reflecting the percentile interval through the
    point estimate:

    .. math::

        \\text{CI}_{\\text{pivotal}} = \\bigl(2\\hat\\theta - q_{1-\\alpha/2},
        \\; 2\\hat\\theta - q_{\\alpha/2}\\bigr)

    See Efron & Tibshirani (1993), "An Introduction to the Bootstrap",
    Chapter 13 §13.3 (Eq. 13.5). For Bernoulli proportions at the
    sample sizes this benchmark uses (N=250) the percentile and
    pivotal intervals agree to ~1 pp; the pivotal form is preferred for
    transparency in the paper since it has weaker symmetry assumptions.

    The sampling unit is whatever ``values`` represents. For
    leaderboard cells: pass the flat array of ``5 × n_episodes_per_seed``
    binary outcomes. Do **not** pass per-seed means — that hides the
    episode-level variance (DESIGN.md § Methodology, "wrong granularity").

    RNG contract: this function takes a deterministic ``seed: int``
    (per the spec) and constructs ``numpy.random.Generator(PCG64(seed))``
    internally. No hidden global state. Same seed → bitwise-identical CI.

    Args:
        values: 1-D array of per-unit measurements (bool or float).
        alpha: significance level; CI level is ``1 - alpha`` (default
            0.05 ⇒ 95% CI).
        n_resamples: bootstrap iterations; default 10,000 per the
            DESIGN.md methodology.
        seed: integer seed for ``numpy.random.PCG64``.

    Returns:
        ``(lo, hi)``: pivotal CI bounds for the mean of ``values``.

    Raises:
        ValueError: empty / non-1-D input, invalid ``alpha`` or
            non-positive ``n_resamples``.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    rng = np.random.Generator(np.random.PCG64(seed))
    n = values.size
    # Cast to float64 for accurate mean (bool would mean-as-bool errors
    # on numpy >= 2 in some axis configurations).
    values_f = values.astype(np.float64, copy=False)
    theta_hat = float(values_f.mean())

    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = values_f[idx].mean(axis=1)

    q_lo, q_hi = np.quantile(resampled_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Pivotal reflection: see E&T Eq. 13.5.
    lo = 2.0 * theta_hat - float(q_hi)
    hi = 2.0 * theta_hat - float(q_lo)
    return lo, hi


def paired_diff_ci(
    a: NDArray[np.floating] | NDArray[np.bool_],
    b: NDArray[np.floating] | NDArray[np.bool_],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Pivotal paired-bootstrap CI on the mean of ``a - b``.

    Episodes are paired by index — ``a[i]`` and ``b[i]`` must come from
    the same ``(seed_idx, episode_idx)``. The bootstrap resamples
    *index pairs*, preserving the within-pair correlation, then takes
    the mean of the per-pair difference. The pivotal CI is computed as
    in :func:`bootstrap_pivotal_ci` (Efron & Tibshirani 1993, Eq. 13.5).

    Use this when the seeding contract was identical AND
    ``n_episodes_per_seed`` matches across cells. If those don't hold,
    fall back to two independent :func:`bootstrap_pivotal_ci` calls
    and document the loss of pairing power (DESIGN.md "asymmetric
    n_episodes_per_seed" guard).

    Args:
        a: 1-D array of per-pair outcomes for cell A.
        b: same for cell B; must share shape with ``a``.
        alpha: significance level (default 0.05 ⇒ 95% CI).
        n_resamples: bootstrap iterations (default 10,000).
        seed: integer seed for ``numpy.random.PCG64``.

    Returns:
        ``(lo, hi)``: pivotal CI for ``mean(a) - mean(b)``.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must share shape; got {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("inputs must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")

    rng = np.random.Generator(np.random.PCG64(seed))
    n = a.size
    a_f = a.astype(np.float64, copy=False)
    b_f = b.astype(np.float64, copy=False)
    theta_hat = float(a_f.mean() - b_f.mean())

    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = a_f[idx].mean(axis=1) - b_f[idx].mean(axis=1)

    q_lo, q_hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    lo = 2.0 * theta_hat - float(q_hi)
    hi = 2.0 * theta_hat - float(q_lo)
    return lo, hi
