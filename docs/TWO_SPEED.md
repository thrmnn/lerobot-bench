# Two-speed operating model

**Status:** framing doc · planned, not shipped
**Companions:** [`docs/PIPELINE_ROADMAP.md`](PIPELINE_ROADMAP.md) §0, §6 · [`docs/WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md) · [`docs/DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md)

`lerobot-bench` runs at two speeds, on purpose. This document names the two lanes,
says what belongs in each, and pins down the **one** sanctioned write that crosses
from research into the production benchmark.

The point of the split is simple: the production benchmark's value *is* its
stability — reproducible cells, locked SHAs, a published dataset other people cite.
World-model research, by contrast, needs to churn: new model classes, planning
horizons, latent-dynamics experiments. Letting that churn into the prod bench would
trade away the stability the benchmark exists to provide. So they run on separate
clocks.

---

## The fast lane — production benchmark

Everything that ships under the `lerobot-bench` name and feeds the public
leaderboard.

- **Scope.** The 6×6 policy×env matrix (110 cells executed), the 5 public-leaderboard
  policies (xvla deferred to v1.1), the eval contract
  `(policy, env, seed, n_eps) -> CellResult`, the statistics layer (Wilson +
  bootstrap CIs, MDE bounds, paired comparisons), the failure taxonomy, the HF
  Space + Hub dataset, the arxiv writeup, the upstream eval-module PR.
- **Clock.** Ships and stays stable. Changes are additive and versioned
  (`v1.0.x`, `v1.1`, …); old data stays published.
- **Bar to merge.** Tests green, locked SHAs, every leaderboard number traceable to
  a parquet row. This is the lane the roadmap's §§1–5 describe.
- **Lives in.** This repo (`lerobot-bench`), this toolchain, this CI.

## The slow lane — world-model research track

The exploratory effort to evaluate world-model / JEPA-style planners *as policies*.

- **Scope.** Planner model classes, latent dynamics, planning horizons, and the
  research code that produces a callable exposing `act(obs) -> action`. Defined in
  [`docs/WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md).
- **Clock.** Its own. Iterates freely; not gated behind the prod bench's release
  cadence and does not gate it either.
- **Lives in.** A **separate repo** with its **own toolchain and dependencies** — it
  does not impose its compute profile or dependency set on `lerobot-bench` users.
- **Status.** Planned, not shipped. No world-model `kind` dispatch exists in
  `src/lerobot_bench/eval.py` today; adding one is explicitly out of scope for the
  v1 wave.

---

## The boundary rule

> **The only write from the slow lane into the fast lane is a single gated adapter
> PR, held off the public leaderboard until the planner is explicitly promoted.**

Concretely:

1. **One adapter PR, not a stream of changes.** When a planner is mature enough to
   benchmark, it enters the bench through exactly one PR that wires a world-model
   `kind` into `load_policy` — a future dispatch branch in
   `src/lerobot_bench/eval.py`, alongside the existing baseline / `repo_id`
   branches. The planner is then scored by the *unchanged* eval machinery, exactly
   like any other policy. No new statistics, no new success rules, no bespoke path.

2. **Gated off the leaderboard by default.** The adapter cell lands **behind the
   leaderboard filter** — the same `src/lerobot_bench/leaderboard_filter.py`
   mechanism that defers `xvla_libero`. It is executed in the sweep and kept in the
   raw parquet for reproducibility, but **excluded from the public board** on read.

3. **Promotion is deliberate.** Moving a world-model cell onto the public
   leaderboard is a separate, reviewed decision — never an automatic consequence of
   a cell running green. Until promoted, world-model cells are exploratory, clearly
   labelled, and carry no leaderboard standing.

This is the same posture `lerobot-bench` already takes with deferred policies: run
it, keep the raw rows for reproducibility, but don't let an unvetted number stand on
the public board. See [`docs/DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md) for the
existing precedent (xvla, pi-family).

---

## Why this matters

- **The benchmark stays citable.** Researchers building on the leaderboard get a
  stable target; the WM track can not silently shift a published number.
- **Research stays fast.** The slow lane is free to break things in its own repo
  without a benchmark-release gate.
- **One audited seam.** A single gated adapter PR is easy to review, easy to revert,
  and easy to reason about — instead of a diffuse coupling between a research
  codebase and a public artifact.

---

*Maintainer: keep this in lock-step with [`docs/PIPELINE_ROADMAP.md`](PIPELINE_ROADMAP.md)
§0/§6 and [`docs/WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md). If the boundary rule
changes — e.g. a second sanctioned write is introduced — update this doc first, then
the roadmap.*
