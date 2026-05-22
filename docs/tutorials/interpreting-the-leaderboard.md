# Interpreting the leaderboard

The `lerobot-bench` leaderboard is a grid of `(policy, env)` cells. Each
cell looks like a single number — a success rate — but it is really a
**statistical claim**: "this policy solves this task at roughly this
rate, with this much uncertainty." Reading a cell well means reading the
uncertainty, not just the point estimate.

This tutorial walks through one cell, then a comparison of two cells,
and names the four mistakes the benchmark's design exists to prevent:
over-reading a noisy number, ranking a tie, misreading `n/a`, and
forgetting the floor.

> **Just want the math?** [`docs/MDE_TABLE.md`](../MDE_TABLE.md) is the
> source of truth for every confidence-interval and minimum-detectable-
> effect number quoted here. **Want to compute these yourself?** See
> [`examples/read_results.py`](../../examples/read_results.py) and
> [`examples/compare_two_policies.py`](../../examples/compare_two_policies.py)
> — they call the exact `lerobot_bench.stats` functions this tutorial
> describes.

## 1. What one cell is made of

A **cell** is one `(policy, env)` pair. Its leaderboard success rate is
the pooled outcome of **5 seeds × 50 episodes = 250 binary outcomes**
(`success` ∈ {True, False}). The unit of evidence is the *episode*, not
the seed — all 250 episodes are pooled into one flat list before any
statistic is computed. (Bootstrapping over the 5 seeds instead would
give wildly inflated intervals; see the `src/lerobot_bench/stats.py`
module docstring on "wrong granularity".)

Each cell on the leaderboard shows three things:

```
   success rate          95% CI
       0.64        [0.58, 0.70]   N=250
        │                 │
   point estimate    Wilson interval
```

- **Success rate** — `successes / N`. The maximum-likelihood point
  estimate. On its own it is the *least* informative part of the cell.
- **95% confidence interval** — the Wilson score interval
  (`lerobot_bench.stats.wilson_ci`). The plausible range for the true
  success rate given this much data. The leaderboard's error bars.
- **N** — the episode count behind the cell. Usually 250; some slow
  cells are auto-downscoped to N=100 or N=50 (more on this below).

## 2. A worked example: one cell

Take a real-ish cell — `act` on `aloha_transfer_cube`, N=250, with 160
successes:

```
success rate = 160 / 250 = 0.640
Wilson 95% CI = [0.579, 0.697]     half-width ±0.059
```

`examples/read_results.py` prints exactly this from a parquet:

```bash
python examples/read_results.py --policy act --env aloha_transfer_cube
```

How to read it:

- The policy succeeds **about 64%** of the time on this task.
- But the **true** rate could plausibly be anywhere from **58% to
  70%**. The leaderboard's `0.64` is a point on that interval, not a
  fact.
- The interval **half-width is ±0.059** — about 6 percentage points
  (pp). That is the resolution of this cell. Any claim finer than ~6 pp
  about this single cell is not supported by the data.

### What a wide CI means

CI width is driven by **N** and by the success rate **p̂** (it is widest
near p̂ = 0.5). At the contracted N=250:

| p̂ | Wilson half-width | so the CI spans... |
|---:|---:|---|
| 0.05 | ±0.027 | a near-zero cell is *tightly* pinned |
| 0.25 | ±0.053 | |
| 0.50 | **±0.062** | the *widest* — worst-case resolution |
| 0.75 | ±0.053 | |
| 0.95 | ±0.027 | a near-saturated cell is tightly pinned |

(From [`docs/MDE_TABLE.md`](../MDE_TABLE.md) § 1.)

A **wide CI** — say a downscoped N=50 cell at p̂≈0.5, half-width
±0.134 — means the cell barely constrains the truth at all: the policy
could be a coin flip or could be solidly two-thirds successful, and 50
episodes cannot tell those apart. Treat a wide-CI cell as a *rough
indication*, never as a ranked position. The leaderboard shows the
interval precisely so a wide one is visible at a glance.

### Downscoped cells (N < 250)

Some VLA cells are slow. The **auto-downscope** rule (DESIGN.md §
Methodology) trims a slow cell's episode budget so the whole sweep
fits — those cells land at N=100 or N=50, **documented, never silently
truncated**. A downscoped cell has a *wider* CI; `docs/MDE_TABLE.md` § 3
tabulates the half-widths at every N so you can see the cost. Always
check the `N` next to a cell before comparing it to a full-N cell.

## 3. Comparing two cells — the MDE check

The whole point of a leaderboard is ranking. But "policy A's number is
bigger than policy B's" is **not** a valid ranking. You need to know
whether the gap is real or is just sampling noise.

### The minimum detectable effect (MDE)

The **MDE** is the smallest success-rate gap the benchmark can resolve
at a given N. Concretely, the leaderboard's rule:

```python
hw = wilson_halfwidth_at_p(p_hat=max(p_hat_a, p_hat_b), n=N)
mde_band = 2 * hw
inconclusive = abs(delta) < mde_band
```

If two cells' observed gap `|Δ|` is **smaller than `2 × HW`** (the
half-width at the higher of the two success rates), the ordering is
**inconclusive at this N**: sampling noise alone could have produced
that gap. At N=250, the worst-case band (both cells near p̂=0.5) is
**±0.123** — so a gap under ~12 pp between two mid-range cells is not a
ranking. The band is *tighter* for saturated or near-zero cells (±0.054
at p̂≈0.95). See [`docs/MDE_TABLE.md`](../MDE_TABLE.md) § 4 for the
per-p̂ thresholds.

A leaderboard pair flagged inconclusive is rendered in neutral grey and
**excluded from any "X beats Y" claim** in the paper and the Space.

### A worked comparison

Two LIBERO VLAs on `libero_spatial`, N=250 each, run through
`examples/compare_two_policies.py`:

```
smolvla_libero  success rate: 0.600
xvla_libero     success rate: 0.504
Δsuccess (A − B): +0.096
  paired 95% CI:  [+0.008, +0.184]
  Wilcoxon:       p = 0.033
  Cohen's h:      +0.193  (small)
  MDE band (2·HW at p̂=0.60, N=250): ±0.121

VERDICT: INCONCLUSIVE at N=250.
```

This is the instructive case. A naive read says "A wins — the Wilcoxon
p-value is 0.033, below 0.05." **The benchmark says inconclusive.** Why?

- The observed gap `|Δ| = 0.096` is **inside** the MDE band `±0.121`. A
  9.6-pp gap between two cells near p̂=0.6 is within what 250 episodes
  of noise can manufacture.
- The paired 95% CI on Δ — `[+0.008, +0.184]` — *barely* excludes zero.
  Its lower edge is 0.008: one unlucky resample from "no difference".
- **Cohen's h = 0.19 is a *small* effect.** Even taking the delta at
  face value, this is not a meaningful capability gap. A "significant"
  small effect is still small — always report `h` next to a Δ
  (`lerobot_bench.stats.cohens_h`).

The conservative verdict wins. Ranking these two cells off this data
would be the **paper-killing error** `docs/MDE_TABLE.md` § 5 was written
to prevent. Report them as **tied at N=250**.

### What "resolved" looks like instead

A comparison is reportable when `|Δ|` clears the MDE band **and** the
paired CI excludes zero with room to spare. Then quote the delta *with*
its CI *and* Cohen's h — never the bare delta. The right framing is "A
beats B by 18 pp (95% CI [12, 24], Cohen's h = 0.5, medium effect)",
not "A is better than B".

### Why *paired*

When both cells were run under the **identical seeding contract** —
same `env.reset` seeds, same episode count — episode `i` of cell A and
episode `i` of cell B faced the *same* scenario. Comparing them
**paired** (`paired_diff_ci`, `paired_wilcoxon`,
`paired_delta_bootstrap`) cancels the per-episode difficulty that both
policies share, which tightens the CI on Δ. If the contracts differ or
the episode counts don't match, fall back to two independent
`bootstrap_pivotal_ci` calls and accept the wider interval — the stats
module docstrings spell out this guard.

## 4. Reading the special cases

### `n/a` — not an empty cell, a deliberate one

`n/a` in a `(policy, env)` cell means **no run was attempted**, and the
benchmark distinguishes two reasons:

1. **Incompatible pair.** Each policy declares an `env_compat` list.
   `act` runs only on `aloha_transfer_cube`; `diffusion_policy` only on
   `pusht`; the LIBERO VLAs only on the four `libero_*` suites. A cell
   outside that list is `n/a` *by design* — the policy was never
   trained for that task. The v1 matrix is 22 runnable cells out of the
   6 × 6 grid for exactly this reason (README § v1 scope).
2. **Deferred.** The Pi0 family is `n/a` in v1 because it overflows the
   host RAM budget on the reference hardware — deferred to v1.1, not a
   capability claim.

`n/a` is **never** "the policy scored zero". A scored-zero cell shows
`0.00` with a CI, and that is a real, informative result. Do not
conflate the two.

### The no-op / random floor

Two policies on the leaderboard author no weights:

- **`no_op`** — emits a zero action every step.
- **`random`** — emits a random action every step.

These are the **floor**. They run on *every* env, and their job is to
calibrate "how hard is this task if you do nothing / flail." A real
policy's success rate is only meaningful **relative to the floor**:

- If `act` scores 0.64 on a task where `no_op` scores 0.00, the policy
  is doing real work — all 64 pp are earned.
- If a policy scores 0.30 on a task where `random` *also* scores ~0.28
  (the two cells' CIs overlap), the policy has **not** demonstrably
  beaten chance on that task. The headline number is an illusion of
  competence; run the MDE check against the floor cell before believing
  it.

Always glance at the `no_op` and `random` rows for an env before
reading any policy's number above them. A leaderboard cell that does not
clear its own floor is not a result.

## 5. A checklist for reading any cell

When you look at a leaderboard cell, ask, in order:

1. **What is N?** 250, or downscoped to 100 / 50? A smaller N means a
   wider CI — check it.
2. **How wide is the CI?** The half-width is the cell's resolution. Do
   not make claims finer than it.
3. **Does it clear the floor?** Compare to the `no_op` / `random` cell
   for the same env. If the CIs overlap, the policy has not beaten
   chance.
4. **For a comparison, did `|Δ|` clear the MDE band?** If not, it is
   *inconclusive at this N* — a tie, not a ranking.
5. **Is the effect size meaningful?** A statistically resolved 2-pp
   delta is still a small effect. Report Cohen's h alongside any Δ.

A leaderboard read that survives this checklist is a claim you can
defend. One that skips it is just a number that looks bigger than
another number.

## See also

- [`docs/MDE_TABLE.md`](../MDE_TABLE.md) — Wilson half-widths and MDE
  bands at every N; the math behind every threshold above.
- [`examples/read_results.py`](../../examples/read_results.py) —
  compute a cell's success rate + Wilson CI yourself.
- [`examples/compare_two_policies.py`](../../examples/compare_two_policies.py)
  — run the paired comparison + MDE check yourself.
- [`docs/DESIGN.md`](../DESIGN.md) § Methodology — the seeding contract,
  bootstrap protocol, and auto-downscope rule.
