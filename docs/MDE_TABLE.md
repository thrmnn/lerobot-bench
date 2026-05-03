# MDE table — minimum detectable difference at the planned sweep N

**Premortem mitigation #4** (pre-sweep methodological check). This doc
is the source of truth for every "is this comparison powered?" claim
in `paper/main.tex` § Methods, `notebooks/01-write-finding.ipynb`
cell 5, and the leaderboard "inconclusive" gate.

The contracted sweep design is **5 seeds × 50 episodes = N=250 binary
outcomes per cell**. Cells that the auto-downscope rule shrinks land at
N ∈ {50, 100, 250} depending on per-step latency; the down-scoped
variants are tabulated below so a slow VLA cell's CI half-width is
visible at a glance before the writeup builds claims on it.

All numbers in this doc are produced by `lerobot_bench.stats.wilson_ci`
and `lerobot_bench.stats.paired_delta_bootstrap` (no closed-form
approximations except where explicitly labeled "closed-form bound").
The simulation cells use `numpy.random.default_rng(seed=42)` so the
table is reproducible; re-running the simulation script with the same
seed produces the same numbers to the bit.

## TL;DR (the four numbers the writeup quotes)

| Quantity | Value | Source |
|---|---|---|
| Wilson 95% half-width at p=0.5, N=250 | **0.0615** | `wilson_ci(125, 250)` |
| Wilson "inconclusive band" 2·HW at p=0.5, N=250 | **0.1230** | 2 × above |
| Empirical paired MDE at p=0.5, N=250, ρ=0 (80% power) | **0.15** | bootstrap simulation, 1000 outer × 2000 boot |
| Empirical paired MDE at p=0.5, N=250, ρ=0.3 (80% power) | **0.15** | same |

The paper's previous text ("0.062 → |Δ| < 0.124") is rounded to 3 dp
but the precise numbers are **0.0615 / 0.123**. The paper has been
updated to the precise values with a citation to this doc.

## 1. Per-cell Wilson 95% CI half-width at N=250

Computed by `wilson_ci(round(p·N), N)` with the closed-form Wilson
score interval (Wilson, 1927). The "inconclusive band" column is
`2 · half-width`: the smallest paired Δ that two cells with
**marginal** Wilson half-widths up to HW each could produce purely
from independent sampling noise (loose upper bound; the simulation
in §2 gives the operational MDE).

| p | Wilson lower | Wilson upper | half-width | inconclusive band (2·HW) |
|---:|---:|---:|---:|---:|
| 0.02 | 0.0086 | 0.0460 | 0.0187 | 0.0374 |
| 0.05 | 0.0277 | 0.0820 | 0.0272 | 0.0543 |
| 0.10 | 0.0687 | 0.1435 | 0.0374 | 0.0748 |
| 0.25 | 0.1986 | 0.3051 | 0.0533 | 0.1065 |
| 0.50 | 0.4385 | 0.5615 | 0.0615 | 0.1230 |
| 0.75 | 0.6949 | 0.8014 | 0.0533 | 0.1065 |
| 0.90 | 0.8565 | 0.9313 | 0.0374 | 0.0748 |
| 0.95 | 0.9180 | 0.9723 | 0.0272 | 0.0543 |
| 0.98 | 0.9540 | 0.9914 | 0.0187 | 0.0374 |

Wilson half-widths are symmetric around p=0.5 by construction: `HW(p)`
matches `HW(1-p)` to within the rounding induced by the integer
`successes = round(p·N)` discretisation. The `test_mde_consistency.py`
suite asserts this symmetry.

## 2. Two-cell paired comparison MDE at N=250

The relevant statistic for cross-cell ranking is the **paired**
Δsuccess `mean(a) − mean(b)`. The marginal Wilson half-width above is
a loose proxy: under positive correlation between the two cells'
per-episode outcomes (same env reset, same scenario, two policies
both either solving or failing the same hard episode) the SE on the
delta shrinks by `sqrt(1 − ρ)` relative to the independence baseline.

We report two MDE estimates:

### 2a. Closed-form independence bound (loose)

For two-sided α=0.05 and 80% power,
`MDE_indep(p) = (z_{0.975} + z_{0.8}) · sqrt(2 p(1-p) / N)`
with `z_{0.975} = 1.96`, `z_{0.8} = 0.8416`. Under correlation ρ this
multiplies by `sqrt(1 − ρ)`.

| p | SE(Δ) (ρ=0) | MDE (ρ=0) | MDE (ρ=0.3) | MDE (ρ=0.5) |
|---:|---:|---:|---:|---:|
| 0.02 | 0.0125 | 0.0351 | 0.0294 | 0.0248 |
| 0.05 | 0.0195 | 0.0546 | 0.0457 | 0.0386 |
| 0.10 | 0.0268 | 0.0752 | 0.0629 | 0.0532 |
| 0.25 | 0.0387 | 0.1085 | 0.0908 | 0.0767 |
| 0.50 | 0.0447 | 0.1253 | 0.1048 | 0.0886 |
| 0.75 | 0.0387 | 0.1085 | 0.0908 | 0.0767 |
| 0.90 | 0.0268 | 0.0752 | 0.0629 | 0.0532 |
| 0.95 | 0.0195 | 0.0546 | 0.0457 | 0.0386 |
| 0.98 | 0.0125 | 0.0351 | 0.0294 | 0.0248 |

### 2b. Empirical bootstrap MDE (simulation)

For each `(p, ρ)`, simulate 1000 paired cells of N=250 paired Bernoulli
draws with marginals `(p+Δ, p)` and Pearson correlation ρ (Gaussian-copula
threshold construction; verified to recover ρ to 3 dp at n=10⁵). For
each simulated cell, run `paired_delta_bootstrap(a, b, n_resamples=2000,
rng=np.random.default_rng(42))` and record whether the 95% CI on Δ
excludes zero. The smallest Δ in {0.01, 0.02, 0.05, 0.10, 0.15, 0.20}
achieving ≥80% rejection rate is the MDE.

> **Reproducibility.** Master RNG `numpy.random.default_rng(seed=42)`;
> the simulation script lives at `scripts/calibrate_mde.py` (see
> §6). Production CIs in the leaderboard use `n_resamples=10_000` for
> tighter percentile estimation; the simulation uses 2000 to keep the
> 1000-outer × 27-cell power table under 5 minutes — Monte-Carlo
> noise on the MDE column is roughly ±1 step at the chosen Δ grid.

| p | MDE @ ρ=0 | power@MDE | MDE @ ρ=0.3 | power@MDE | MDE @ ρ=0.5 | power@MDE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.10 | 0.995 | 0.05 | 0.921 | 0.05 | 0.991 |
| 0.05 | 0.10 | 0.973 | 0.10 | 0.997 | 0.05 | 0.850 |
| 0.10 | 0.10 | 0.874 | 0.10 | 0.967 | 0.10 | 0.996 |
| 0.25 | 0.15 | 0.953 | 0.10 | 0.824 | 0.10 | 0.939 |
| 0.50 | 0.15 | 0.910 | 0.15 | 0.981 | 0.10 | 0.873 |
| 0.75 | 0.15 | 0.995 | 0.10 | 0.910 | 0.10 | 0.976 |
| 0.90 | 0.10 | 1.000 | 0.10 | 1.000 | 0.05 | 0.856 |
| 0.95 | 0.05 | 0.998 | 0.05 | 1.000 | 0.05 | 0.999 |
| 0.98 | >0.02 | — | >0.02 | — | >0.02 | — |

Notes on the table:
- The MDE at p=0.50 is **0.15** under ρ=0 and **0.15** under ρ=0.3.
  The closed-form bound said 0.125 and 0.105 respectively; the
  simulation is more conservative because the bootstrap CI is wider
  than the normal-approximation interval at this N (the percentile
  bootstrap pays a small tax for not assuming normality, which is the
  whole point of using it for a Bernoulli proportion).
- At the extremes p ∈ {0.02, 0.98} the Δ-grid resolution becomes
  load-bearing — the true MDE is between the listed Δ rows. The
  practical takeaway is that near-saturated cells (p ≥ 0.95) can
  resolve a 5-pp delta but anything tighter requires more episodes.
- ρ ≥ 0.5 corresponds to "policies fail the same hard episodes"; for
  the planned matrix this is the realistic bound — DiffPolicy and ACT
  both struggle on visually-occluded PushT scenes; they share enough
  episode-level signal that the independence bound is meaningfully
  conservative.

## 3. Down-scoped cell variants — Wilson half-width at N ∈ {50, 100, 250, 500}

If Day-0b calibration (`scripts/calibrate.py`) forces a per-policy
auto-downscope below n_ep=50, the per-cell N drops accordingly. N=500
is included as a what-if for a v2 design that doubles episodes per
seed.

### N=50 (5 seeds × 10 episodes — the absolute floor before a policy is dropped)

| p | half-width | inconclusive band (2·HW) |
|---:|---:|---:|
| 0.02 | 0.0507 | 0.1014 |
| 0.05 | 0.0618 | 0.1236 |
| 0.10 | 0.0851 | 0.1701 |
| 0.25 | 0.1156 | 0.2312 |
| 0.50 | 0.1336 | 0.2671 |
| 0.75 | 0.1156 | 0.2312 |
| 0.90 | 0.0851 | 0.1701 |
| 0.95 | 0.0618 | 0.1236 |
| 0.98 | 0.0507 | 0.1014 |

### N=100 (5 seeds × 20 episodes)

| p | half-width | inconclusive band (2·HW) |
|---:|---:|---:|
| 0.02 | 0.0323 | 0.0645 |
| 0.05 | 0.0451 | 0.0902 |
| 0.10 | 0.0596 | 0.1191 |
| 0.25 | 0.0838 | 0.1676 |
| 0.50 | 0.0962 | 0.1923 |
| 0.75 | 0.0838 | 0.1676 |
| 0.90 | 0.0596 | 0.1191 |
| 0.95 | 0.0451 | 0.0902 |
| 0.98 | 0.0323 | 0.0645 |

### N=250 (the contract; reprised from §1 for easy comparison)

| p | half-width | inconclusive band (2·HW) |
|---:|---:|---:|
| 0.02 | 0.0187 | 0.0374 |
| 0.05 | 0.0272 | 0.0543 |
| 0.10 | 0.0374 | 0.0748 |
| 0.25 | 0.0533 | 0.1065 |
| 0.50 | **0.0615** | **0.1230** |
| 0.75 | 0.0533 | 0.1065 |
| 0.90 | 0.0374 | 0.0748 |
| 0.95 | 0.0272 | 0.0543 |
| 0.98 | 0.0187 | 0.0374 |

### N=500 (v2 stretch: 5 seeds × 100 episodes)

| p | half-width | inconclusive band (2·HW) |
|---:|---:|---:|
| 0.02 | 0.0128 | 0.0255 |
| 0.05 | 0.0193 | 0.0387 |
| 0.10 | 0.0264 | 0.0527 |
| 0.25 | 0.0379 | 0.0757 |
| 0.50 | 0.0437 | 0.0873 |
| 0.75 | 0.0379 | 0.0757 |
| 0.90 | 0.0264 | 0.0527 |
| 0.95 | 0.0193 | 0.0387 |
| 0.98 | 0.0128 | 0.0255 |

Going from N=250 to N=500 buys roughly a 1.4× tightening on every
cell — meaningful for the p ≈ 0.5 cells, marginal for the saturated
ones. Going from N=250 to N=100 widens by 1.56× and a 5-pp delta in
the middle of the success-rate range stops being resolvable.

## 4. The "inconclusive at this N" rule

The paper Methods section states: at N=250, Wilson half-width at
p=0.5 is `≈ 0.062` so any `|Δ| < 2·HW ≈ 0.124` is inconclusive.

**Verified.** Computed values (from §1):
- HW(p=0.5, N=250) = **0.0615** (paper rounded to 0.062 — fine)
- 2·HW(p=0.5, N=250) = **0.1230** (paper rounded to 0.124 — off by
  0.001; updated to the precise value in `paper/main.tex`)

The threshold scales with `p̂` (it is **not** a flat 0.124 across all
cells). The leaderboard gate must use the per-cell threshold derived
from the higher of the two cells' p̂:

```python
hw = wilson_half_width(p_hat=max(p_hat_a, p_hat_b), n=N)
inconclusive = abs(delta_hat) < 2 * hw
```

Per-cell thresholds at N=250:

| max(p̂_a, p̂_b) | 2·HW threshold |
|---:|---:|
| 0.02 | 0.0374 |
| 0.05 | 0.0543 |
| 0.10 | 0.0748 |
| 0.25 | 0.1065 |
| 0.50 | **0.1230** ← worst case (used in paper text) |
| 0.75 | 0.1065 |
| 0.90 | 0.0748 |
| 0.95 | 0.0543 |
| 0.98 | 0.0374 |

A comparison between two cells both at p̂=0.95 has a much tighter
2·HW band (0.054) than two cells at p̂=0.5 (0.123). Reporting a
flat 0.124 across all comparisons under-claims power on the
saturated and near-zero cells. The notebook gate (cell 5, see §5)
uses the per-cell threshold; the paper text quotes the worst-case
0.123 for clarity.

## 5. Practical interpretation

If any (policy, env) pair has observed `|Δp̂|` smaller than the MDE
band at the cell's observed `max(p̂_a, p̂_b)`, the leaderboard MUST
display that pair as **"inconclusive at N=250"** rather than as a
ranked ordering. The opposite mistake — ranking two cells whose
sampling noise alone could produce the observed delta — is a paper-
killing error and the precise scenario stats-rigor-reviewer was
chartered to prevent.

The notebook (`notebooks/01-write-finding.ipynb` cell 5) has been
updated to add an `inconclusive_at_N` boolean column to
`paired_df` derived from the per-cell Wilson half-width table above.
Any pair with `inconclusive_at_N == True` should be rendered in
neutral grey in the leaderboard table and excluded from the
"headline finding" sentence in the abstract.

## 6. Reproducing the numbers in this doc

The Wilson tables (§1, §3) are deterministic — re-run

```python
from lerobot_bench.stats import wilson_ci
for p in [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]:
    for N in [50, 100, 250, 500]:
        s = round(p * N)
        lo, hi = wilson_ci(s, N)
        print(p, N, lo, hi, (hi - lo) / 2)
```

The simulation table (§2b) is reproducible under
`numpy.random.default_rng(seed=42)`; the simulation script is
`scripts/calibrate_mde.py` (creates the same JSON consumed by the
markdown above). On a 2026 laptop CPU it takes about 5 minutes for
the full 9 × 3 × 6 grid.

The unit tests in `tests/test_mde_consistency.py` assert that:
- `wilson_ci(125, 250)` half-width matches the **0.0615** quoted
  here (and that the paper's `0.062 / 0.124` numbers are within
  rounding of the precise values).
- `wilson_ci(12, 250)` and `wilson_ci(238, 250)` half-widths are
  symmetric (0.0272 each, within float rounding).
- The simulated paired MDE at `(p=0.5, N=250, ρ=0, seed=42)` lands
  in the expected range [0.10, 0.20] (loose; the simulation has
  Monte-Carlo noise but is fully deterministic under the fixed
  seed).
- The "inconclusive band" column matches `2 × half-width` row by row
  for the table in §1.
