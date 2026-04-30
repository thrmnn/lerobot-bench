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

from lerobot_bench.stats import (
    bootstrap_ci,
    cohens_h,
    paired_delta_bootstrap,
    paired_wilcoxon,
    wilson_ci,
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
