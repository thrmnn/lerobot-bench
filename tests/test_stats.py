"""Tests for ``lerobot_bench.stats``.

Each function is checked against an analytical or known-reference truth
where one exists (Wilson CI for the bootstrap, identity-pairs Wilcoxon,
closed-form Cohen's h). Where no closed form exists, properties that
must hold are asserted (sign correctness, monotonicity in n,
determinism under fixed RNG seed).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from lerobot_bench.stats import (
    bootstrap_ci,
    bootstrap_pivotal_ci,
    cohens_h,
    mcnemar_paired,
    paired_delta_bootstrap,
    paired_diff_ci,
    paired_wilcoxon,
    wilson_ci,
    wilson_halfwidth_at_p,
)

# --------------------------------------------------------------------- #
# bootstrap_ci                                                          #
# --------------------------------------------------------------------- #


def test_bootstrap_ci_deterministic_under_fixed_seed() -> None:
    outcomes = np.array([True, False, True, True, False, True, True, False])
    a = bootstrap_ci(outcomes, rng=np.random.default_rng(42), n_resamples=1_000)
    b = bootstrap_ci(outcomes, rng=np.random.default_rng(42), n_resamples=1_000)
    assert a == b


def test_bootstrap_ci_mean_matches_empirical() -> None:
    outcomes = np.array([True] * 6 + [False] * 4)
    result = bootstrap_ci(outcomes, rng=np.random.default_rng(0), n_resamples=200)
    assert result.mean == pytest.approx(0.6)


def test_bootstrap_ci_converges_to_wilson_at_large_n() -> None:
    rng = np.random.default_rng(7)
    p_true = 0.65
    n = 5_000
    outcomes = rng.uniform(size=n) < p_true
    boot = bootstrap_ci(outcomes, rng=np.random.default_rng(11), n_resamples=2_000)
    wlo, whi = wilson_ci(int(outcomes.sum()), n)
    # Bootstrap and Wilson should agree to within ~1 pp at n = 5000.
    assert abs(boot.lo - wlo) < 0.01
    assert abs(boot.hi - whi) < 0.01


def test_bootstrap_ci_width_shrinks_with_n() -> None:
    sample_rng = np.random.default_rng(99)
    p_true = 0.5
    small = bootstrap_ci(
        sample_rng.uniform(size=50) < p_true,
        rng=np.random.default_rng(1),
        n_resamples=500,
    )
    large = bootstrap_ci(
        sample_rng.uniform(size=5_000) < p_true,
        rng=np.random.default_rng(1),
        n_resamples=500,
    )
    # CI width scales ~1/√n; 100× more data gives ~10× tighter CI. Assert
    # at least 5× tighter to leave slack for resample noise.
    assert (large.hi - large.lo) * 5 < (small.hi - small.lo)


def test_bootstrap_ci_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_ci(np.zeros((3, 4), dtype=bool), rng=np.random.default_rng(0))


def test_bootstrap_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci(np.array([], dtype=bool), rng=np.random.default_rng(0))


def test_bootstrap_ci_rejects_invalid_ci_level() -> None:
    outcomes = np.array([True, False])
    with pytest.raises(ValueError, match=r"ci must be in"):
        bootstrap_ci(outcomes, rng=np.random.default_rng(0), ci=1.5)
    with pytest.raises(ValueError, match=r"ci must be in"):
        bootstrap_ci(outcomes, rng=np.random.default_rng(0), ci=0.0)


# --------------------------------------------------------------------- #
# paired_delta_bootstrap                                                #
# --------------------------------------------------------------------- #


def test_paired_delta_zero_when_arrays_identical() -> None:
    a = np.array([True, False, True, True, False])
    result = paired_delta_bootstrap(a, a, rng=np.random.default_rng(0), n_resamples=500)
    assert result.mean == 0.0
    # Resampled deltas are all zero by construction.
    assert result.lo == 0.0
    assert result.hi == 0.0


def test_paired_delta_detects_full_separation() -> None:
    a = np.array([True] * 50)
    b = np.array([False] * 50)
    result = paired_delta_bootstrap(a, b, rng=np.random.default_rng(0), n_resamples=500)
    assert result.mean == 1.0
    # Every resample gives delta = 1.0 — zero variance.
    assert result.lo == 1.0
    assert result.hi == 1.0


def test_paired_delta_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share shape"):
        paired_delta_bootstrap(
            np.array([True, False]),
            np.array([True, False, True]),
            rng=np.random.default_rng(0),
        )


# --------------------------------------------------------------------- #
# paired_wilcoxon                                                       #
# --------------------------------------------------------------------- #


def test_wilcoxon_pvalue_one_when_arrays_identical() -> None:
    a = np.array([True, False, True, False, True, True, False])
    result = paired_wilcoxon(a, a)
    assert result.pvalue == 1.0
    assert result.n_pairs == 0
    assert result.n_zero_diffs == len(a)


def test_wilcoxon_low_pvalue_under_consistent_difference() -> None:
    rng = np.random.default_rng(0)
    n = 80
    # a wins ~85% of episodes; b wins ~20%. With pairing this is a huge effect.
    a = rng.uniform(size=n) < 0.85
    b = rng.uniform(size=n) < 0.20
    result = paired_wilcoxon(a, b)
    assert result.pvalue < 1e-3
    assert result.n_pairs > 0


def test_wilcoxon_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share shape"):
        paired_wilcoxon(np.array([True, False]), np.array([True]))


# --------------------------------------------------------------------- #
# cohens_h                                                              #
# --------------------------------------------------------------------- #


def test_cohens_h_zero_when_proportions_equal() -> None:
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(0.0, 0.0) == 0.0
    assert cohens_h(1.0, 1.0) == 0.0


def test_cohens_h_sign_matches_direction() -> None:
    assert cohens_h(0.7, 0.3) > 0
    assert cohens_h(0.3, 0.7) < 0


def test_cohens_h_matches_closed_form() -> None:
    expected = float(2.0 * np.arcsin(np.sqrt(0.5)) - 2.0 * np.arcsin(np.sqrt(0.4)))
    assert cohens_h(0.5, 0.4) == pytest.approx(expected)


def test_cohens_h_rejects_out_of_range_proportions() -> None:
    with pytest.raises(ValueError):
        cohens_h(1.5, 0.5)
    with pytest.raises(ValueError):
        cohens_h(0.5, -0.1)


# --------------------------------------------------------------------- #
# wilson_ci                                                             #
# --------------------------------------------------------------------- #


def test_wilson_ci_matches_textbook_value() -> None:
    # 6/10 successes, 95% CI ≈ [0.3128, 0.8324] — Wikipedia's worked example.
    lo, hi = wilson_ci(6, 10)
    assert lo == pytest.approx(0.3128, abs=1e-3)
    assert hi == pytest.approx(0.8324, abs=1e-3)


def test_wilson_ci_centered_on_p_for_balanced_data() -> None:
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    # Symmetry: the interval is roughly centered on 0.5.
    assert abs((lo + hi) / 2.0 - 0.5) < 1e-9


def test_wilson_ci_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"successes="):
        wilson_ci(11, 10)
    with pytest.raises(ValueError, match=r"successes="):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError, match=r"n must be positive"):
        wilson_ci(5, 0)


# --------------------------------------------------------------------- #
# Additional Wilson reference values (Agresti & Coull 1998 cross-check) #
# --------------------------------------------------------------------- #


def test_wilson_ci_zero_successes_lower_bound_is_zero() -> None:
    """0/N: the Wilson lower bound is exactly 0 (closed-form: numerator 0)."""
    lo, hi = wilson_ci(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-9)
    # Upper bound at 0/100, 95% CI ≈ 0.0370 (Agresti & Coull table).
    assert hi == pytest.approx(0.0370, abs=1e-3)


def test_wilson_ci_full_successes_upper_bound_is_one() -> None:
    """N/N: the Wilson upper bound is exactly 1 (mirror of 0/N)."""
    lo, hi = wilson_ci(100, 100)
    assert hi == pytest.approx(1.0, abs=1e-9)
    # Lower bound at 100/100, 95% CI ≈ 0.9630.
    assert lo == pytest.approx(0.9630, abs=1e-3)


def test_wilson_ci_known_reference_table() -> None:
    """Cross-check half a dozen ``(successes, n)`` against statsmodels-equivalent values.

    Reference values produced by ``statsmodels.stats.proportion.proportion_confint(method="wilson")``
    at ``alpha=0.05`` and recorded inline (we don't take a statsmodels
    dep — see the project methodology rule). Each tuple is
    ``(successes, n, lo, hi)`` to 4 decimal places.
    """
    cases = [
        (1, 10, 0.0179, 0.4042),  # extreme low
        (3, 10, 0.1078, 0.6032),
        (5, 10, 0.2366, 0.7634),  # exactly 0.5 → symmetric
        (7, 10, 0.3968, 0.8922),
        (9, 10, 0.5958, 0.9821),  # extreme high
        (50, 250, 0.1551, 0.2540),  # production-relevant N
        (0, 100, 0.0000, 0.0370),  # boundary: zero successes
        (100, 100, 0.9630, 1.0000),  # boundary: all successes
    ]
    for s, n, lo_ref, hi_ref in cases:
        lo, hi = wilson_ci(s, n)
        assert lo == pytest.approx(lo_ref, abs=1e-3), f"({s}/{n}) lo: {lo} vs {lo_ref}"
        assert hi == pytest.approx(hi_ref, abs=1e-3), f"({s}/{n}) hi: {hi} vs {hi_ref}"


def test_wilson_ci_99_percent_wider_than_95() -> None:
    """Sanity: a 99% CI must be strictly wider than the 95% CI."""
    lo95, hi95 = wilson_ci(125, 250, ci=0.95)
    lo99, hi99 = wilson_ci(125, 250, ci=0.99)
    assert lo99 < lo95
    assert hi99 > hi95


# --------------------------------------------------------------------- #
# wilson_halfwidth_at_p                                                 #
# --------------------------------------------------------------------- #


def test_wilson_halfwidth_at_p_matches_doc_value() -> None:
    """``HW(p=0.5, n=250) ≈ 0.0615`` — the headline number in MDE_TABLE.md."""
    hw = wilson_halfwidth_at_p(0.5, 250)
    assert hw == pytest.approx(0.0615, abs=1e-4)


def test_wilson_halfwidth_at_p_consistent_with_wilson_ci() -> None:
    """``wilson_halfwidth_at_p`` is just ``(hi - lo) / 2`` of ``wilson_ci``."""
    for p, n in [(0.10, 100), (0.25, 250), (0.50, 250), (0.95, 500)]:
        successes = round(p * n)
        lo, hi = wilson_ci(successes, n)
        hw_direct = (hi - lo) / 2.0
        hw_helper = wilson_halfwidth_at_p(p, n)
        assert hw_helper == pytest.approx(hw_direct, abs=1e-12)


def test_wilson_halfwidth_at_p_shrinks_with_n() -> None:
    """HW scales like 1/√n: doubling n shrinks HW by ≈ √2."""
    hw_250 = wilson_halfwidth_at_p(0.5, 250)
    hw_1000 = wilson_halfwidth_at_p(0.5, 1000)
    ratio = hw_250 / hw_1000
    assert 1.9 < ratio < 2.1, f"HW(N=250)/HW(N=1000) = {ratio}, expected ≈ 2.0"


def test_wilson_halfwidth_at_p_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="p must be in"):
        wilson_halfwidth_at_p(1.5, 100)
    with pytest.raises(ValueError, match="p must be in"):
        wilson_halfwidth_at_p(-0.1, 100)
    with pytest.raises(ValueError, match="n must be positive"):
        wilson_halfwidth_at_p(0.5, 0)
    with pytest.raises(ValueError, match="alpha must be in"):
        wilson_halfwidth_at_p(0.5, 100, alpha=1.5)


# --------------------------------------------------------------------- #
# mcnemar_paired                                                        #
# --------------------------------------------------------------------- #


def test_mcnemar_paired_no_disagreements_returns_one() -> None:
    """``b = c = 0`` carries no information — pvalue must be 1.0."""
    stat, p = mcnemar_paired(0, 0)
    assert stat == 0.0
    assert p == 1.0


def test_mcnemar_paired_exact_branch_against_hand_computed_binomial() -> None:
    """``b=4, c=1`` → exact two-sided p = 2 · P(X ≤ 1 | n=5, p=0.5).

    Hand calc: P(X=0) + P(X=1) at n=5,p=0.5
            = (1 + 5) / 32 = 6/32 = 0.1875.
    Two-sided p = min(1.0, 2 · 0.1875) = 0.375.
    """
    stat, p = mcnemar_paired(4, 1, exact=True)
    assert stat == 1.0  # min(b, c)
    assert p == pytest.approx(0.375, abs=1e-12)
    # Cross-check against scipy's own binomial CDF.
    expected = 2.0 * float(scipy_stats.binom.cdf(1, 5, 0.5))
    assert p == pytest.approx(min(1.0, expected), abs=1e-12)


def test_mcnemar_paired_exact_symmetric_table_pvalue_one() -> None:
    """``b = c`` is the maximally non-significant configuration; p == 1."""
    _, p = mcnemar_paired(3, 3, exact=True)
    assert p == 1.0


def test_mcnemar_paired_chi2_branch_used_above_threshold() -> None:
    """At ``b + c > 25`` the chi-square branch is used and matches the
    closed-form continuity-corrected formula.
    """
    b, c = 30, 10  # n = 40 > 25 ⇒ chi-square branch
    stat, p = mcnemar_paired(b, c, exact=True)
    expected_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    assert stat == pytest.approx(expected_chi2)
    expected_p = float(scipy_stats.chi2.sf(expected_chi2, df=1))
    assert p == pytest.approx(expected_p)


def test_mcnemar_paired_extreme_imbalance_is_significant() -> None:
    """``b=30, c=0`` is overwhelmingly significant under either branch.

    Chosen above the n=25 exact/chi2 threshold so we exercise the chi2
    branch directly with ``exact=True``. (b=20, c=0 routes through the
    exact branch and gives p ≈ 2e-6; the chi2 approximation at that n is
    ≈ 2e-5, just above the 1e-5 cutoff.)
    """
    _, p_chi2_route = mcnemar_paired(30, 0, exact=True)  # n=30 > 25 ⇒ chi2
    _, p_chi2_forced = mcnemar_paired(30, 0, exact=False)
    assert p_chi2_route == p_chi2_forced
    assert p_chi2_route < 1e-6


def test_mcnemar_paired_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_paired(-1, 5)
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_paired(5, -1)


# --------------------------------------------------------------------- #
# bootstrap_pivotal_ci                                                  #
# --------------------------------------------------------------------- #


def test_bootstrap_pivotal_ci_deterministic_under_fixed_seed() -> None:
    """Same ``seed`` → bitwise-identical CI."""
    values = np.array([True, False, True, True, False, True, True, False, True, False])
    a = bootstrap_pivotal_ci(values, n_resamples=1_000, seed=42)
    b = bootstrap_pivotal_ci(values, n_resamples=1_000, seed=42)
    assert a == b


def test_bootstrap_pivotal_ci_centered_on_point_estimate() -> None:
    """For a symmetric statistic the pivotal CI is centered on θ_hat.

    For a moderate-n Bernoulli the resampled-mean distribution is
    approximately symmetric, so the pivotal CI's midpoint should be
    within ~1 pp of the empirical mean.
    """
    rng = np.random.default_rng(0)
    n = 500
    values = rng.uniform(size=n) < 0.4
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=2_000, seed=1)
    midpoint = (lo + hi) / 2.0
    assert abs(midpoint - float(values.mean())) < 0.01


def test_bootstrap_pivotal_ci_close_to_wilson_at_large_n() -> None:
    """Pivotal bootstrap converges to Wilson at large n on Bernoulli data."""
    rng = np.random.default_rng(2)
    n = 5_000
    values = rng.uniform(size=n) < 0.5
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=2_000, seed=3)
    wlo, whi = wilson_ci(int(values.sum()), n)
    assert abs(lo - wlo) < 0.01
    assert abs(hi - whi) < 0.01


def test_bootstrap_pivotal_ci_floats_supported() -> None:
    """Continuous values (e.g. episode returns) are supported."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.7, scale=0.2, size=200)
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=1_000, seed=0)
    assert lo < 0.7 < hi


def test_bootstrap_pivotal_ci_zero_variance_input_returns_point() -> None:
    """``[True]*N`` has zero variance — the pivotal CI collapses to ``(1, 1)``."""
    values = np.ones(100, dtype=bool)
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=500, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_pivotal_ci_small_n_resamples_still_runs() -> None:
    """``n_resamples=10`` is wasteful but must not crash; quantile is well-defined."""
    values = np.array([True, False, True, False, True])
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=10, seed=0)
    assert lo <= hi


def test_bootstrap_pivotal_ci_n1_input_returns_point() -> None:
    """N=1: every resample returns the same value → CI collapses to a point."""
    values = np.array([True])
    lo, hi = bootstrap_pivotal_ci(values, n_resamples=100, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_pivotal_ci_rejects_invalid_inputs() -> None:
    values = np.array([True, False])
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_pivotal_ci(np.zeros((3, 4)), seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_pivotal_ci(np.array([], dtype=bool), seed=0)
    with pytest.raises(ValueError, match="alpha must be in"):
        bootstrap_pivotal_ci(values, alpha=1.5, seed=0)
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_pivotal_ci(values, n_resamples=0, seed=0)


# --------------------------------------------------------------------- #
# paired_diff_ci                                                        #
# --------------------------------------------------------------------- #


def test_paired_diff_ci_zero_when_arrays_identical() -> None:
    a = np.array([True, False, True, True, False, True, False])
    lo, hi = paired_diff_ci(a, a, n_resamples=500, seed=0)
    assert lo == 0.0
    assert hi == 0.0


def test_paired_diff_ci_full_separation_collapses_to_one() -> None:
    a = np.ones(50, dtype=bool)
    b = np.zeros(50, dtype=bool)
    lo, hi = paired_diff_ci(a, b, n_resamples=500, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_paired_diff_ci_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(size=200) < 0.6
    b = rng.uniform(size=200) < 0.4
    ci1 = paired_diff_ci(a, b, n_resamples=1_000, seed=7)
    ci2 = paired_diff_ci(a, b, n_resamples=1_000, seed=7)
    assert ci1 == ci2


def test_paired_diff_ci_centered_on_observed_delta() -> None:
    rng = np.random.default_rng(0)
    n = 500
    a = rng.uniform(size=n) < 0.6
    b = rng.uniform(size=n) < 0.4
    delta = float(a.mean() - b.mean())
    lo, hi = paired_diff_ci(a, b, n_resamples=2_000, seed=1)
    midpoint = (lo + hi) / 2.0
    assert abs(midpoint - delta) < 0.01


def test_paired_diff_ci_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share shape"):
        paired_diff_ci(np.array([True, False]), np.array([True]), seed=0)


def test_paired_diff_ci_rejects_invalid_inputs() -> None:
    a = np.array([True, False])
    b = np.array([True, False])
    with pytest.raises(ValueError, match="1-D"):
        paired_diff_ci(np.zeros((2, 2)), np.zeros((2, 2)), seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        paired_diff_ci(np.array([], dtype=bool), np.array([], dtype=bool), seed=0)
    with pytest.raises(ValueError, match="alpha must be in"):
        paired_diff_ci(a, b, alpha=0.0, seed=0)
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        paired_diff_ci(a, b, n_resamples=0, seed=0)
