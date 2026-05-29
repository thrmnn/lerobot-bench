# Pipeline Roadmap — beyond v1.0

**Status:** draft · 2026-05-26 · target reviewer: maintainer
**Companion:** the deck at `paper/deck/` (slides 07, 08, 20) and `docs/DEFERRED_POLICIES.md`.

This document plans how `lerobot-bench` evolves past the v1.0 sweep. It is organised so that **methodology robustness comes first** — the deck's "replication gap" claim is only as good as the assumptions behind it, and several of those assumptions are not yet verified to the standard we want for a public benchmark.

---

## 0 · Why this doc exists

The v1.0 sweep produced strong, quotable numbers:

- SmolVLA on `libero_10`: paper **0.71** → measured **0.252** (Δ −45.8 pp, Wilson 95% CI [.202, .309])
- SmolVLA on `libero_object`: paper **0.94** → measured **0.528** (Δ −41.2 pp)
- Diffusion Policy on `pusht`: paper **0.83** → measured **0.816** (Δ −1.4 pp, replicates)
- ACT on `aloha_transfer_cube`: paper rate not given → measured **0.016** (does not beat random)

These claims are **statistically tight** (N=250 puts MDE ≈ 12.3 pp; the largest gaps are 4× that). They are **deterministically reproducible** on our hardware (seed triple → bit-identical parquet rows). The XVLA debugging effort (PRs #71, #74) gave us direct evidence that **Hub processor wiring is a real failure mode**, and SmolVLA passed that audit.

But they are **single-lab, single-hardware, single-task-id** numbers. Before we ride this finding into a paper headline, every assumption that could partially explain the gap needs to be tested in the open. **§1 of this doc is the audit plan; §§2–5 are the coverage expansion.**

---

## 1 · v1.0.1 — methodology audit (before any new envs)

Goal: turn each "warn" entry from deck slide 08 into a "verified" entry on slide 07.

Each item is a small PR + a parquet/notebook artifact. None of them require new sweeps; most are documentation + code audits against existing artifacts.

### 1.1 Task-id coverage audit

> **Open question:** "SmolVLA paper may report per-suite averages over all 10 LIBERO tasks; we run task_id=0 only."

- [ ] Read the SmolVLA-LIBERO model card and any cited paper. Document **precisely** what aggregation the published 0.71 / 0.94 numbers use.
- [ ] If they're per-suite averages: re-state the deck claim as **"on the hardest task in each LIBERO suite, the gap is N pp"** rather than the apples-to-oranges comparison.
- [ ] Write `docs/CLAIM_AUDIT_SMOLVLA.md` enumerating every published number, its aggregation, and our matching slice.
- [ ] **Outcome gate:** if our task_id=0 is materially harder than the suite average, soften the deck headline accordingly *before* publication.

**Risk if skipped:** the deck's headline becomes a strawman.

### 1.2 Inference-settings audit

> **Open question:** "Action-chunk horizon and sampling strategy use lerobot defaults — may not match paper."

- [ ] Log every inference hyperparameter actually used (`chunk_size`, `temperature`, `top_p`, action-aggregation strategy) for each policy in `results.parquet`.
- [ ] Cross-reference against each paper's reported inference settings.
- [ ] For any mismatch with a plausible behavioural impact: re-run a small probe sweep (1 seed × 50 ep) with the paper's settings; report the delta.
- [ ] If the delta closes >5 pp of the gap, **update the deck headline** and re-run the full cell.

**Risk if skipped:** "they used chunk=10, we used chunk=20" could be the entire 46 pp story for SmolVLA.

### 1.3 Episode-termination + success-threshold parity

> **Open question:** "Are we using the same success criterion as the paper?"

- [ ] Document the success criterion lerobot's env factory uses for each env (LIBERO, Aloha, PushT).
- [ ] Compare to each paper's described success criterion.
- [ ] Same for max-episode-length / timeout policy.
- [ ] If different: re-run a probe with the paper's criterion.

**Risk if skipped:** an off-by-tolerance threshold could flip many successes/failures.

### 1.4 Independent SmolVLA replication

> **Open question:** "Only one lab measured this."

- [ ] Reach out to one external user (issue posted on the SmolVLA model card asking for an independent eval slice).
- [ ] Provide them `scripts/run_one.py` + the exact pinned SHA + the inference settings from 1.2.
- [ ] Accept their parquet rows into `results-external/` and add a "third-party replication" column to the dashboard.
- [ ] **Outcome gate:** if external numbers are wildly different (>15 pp), pause v1.1 work and investigate.

**Risk if skipped:** "lab variance" is the most likely silent confounder; without external numbers, we can't bound it.

### 1.5 Cross-hardware probe

> **Open question:** "Single laptop GPU; CUDA non-determinism could shift behaviour."

- [ ] Run one cell (e.g. SmolVLA × libero_goal) on a different GPU (RTX A6000 on the lab cluster).
- [ ] Compare per-episode success against the laptop run.
- [ ] If they diverge: add a "calibrated MDE band" column = paired-bootstrap variance across hardware.

**Risk if skipped:** the published numbers are accurate but not portable; a reviewer with different hardware might disagree.

### 1.6 Negative-control probe

- [ ] Run SmolVLA on its **own training distribution** (if we can get a public eval split) — should reproduce 100%-ish.
- [ ] If it doesn't: there's a wiring bug or distribution-shift artifact we haven't found, and v1.0's headline is suspect.

**Bundle exit criteria for v1.0.1:** all six items either closed-green or explicitly waived in `docs/CLAIM_AUDIT_SMOLVLA.md`. Ship a `v1.0.1` patch that updates the deck headline + paper + README with the audit results.

---

## 2 · v1.1 — coverage breadth (after v1.0.1 ships)

Goal: address the "smolvla coverage skew" critique. Currently 4 of 6 non-baseline cells are SmolVLA — that's *where Hub checkpoints exist*, but a benchmark with that distribution is fragile to the SmolVLA story changing.

### 2.1 All-tasks LIBERO

- Run **all 10 task_ids** per LIBERO suite for SmolVLA (and XVLA once §3 lands).
- That's `4 suites × 10 tasks × 5 seeds × 50 ep = 10,000 LIBERO eps per VLA policy`. Calibrate first to confirm fits in a weekend sweep.
- Aggregate two ways: per-suite mean (matches papers) and per-task-id (matches us). Publish both.

### 2.2 XVLA bug 3

- See `docs/DEFERRED_POLICIES.md`. Suspected chunked-action layout vs `empty_camera_0` handling.
- Engage upstream issue `huggingface/lerobot#3674`.
- Once resolved, run the same 4-suite × 10-task matrix as 2.1.
- Re-enable XVLA on the live leaderboards (revert the filter from PR #82).

### 2.3 Multi-comparison correction

- The matrix is now 22+ cells across 5 policies — frequentist hypothesis-testing on the family needs Holm-Bonferroni correction.
- Add `scripts/family_correction.py` that emits `(per-cell p_raw, per-cell p_adj, family α)`.
- Add a "significance after correction" column to the dashboard.

### 2.4 Replication scatter as a first-class figure

- Promote the supplementary deck slide S1 (paper-vs-measured scatter) to a top-level figure in the paper and a dashboard panel.
- Script: `scripts/replication_scatter.py` reads `results.parquet` + `MODEL_CARDS.md` paper_rates, emits SVG + PNG.

### 2.5 Continuous CI sweep

- Add a nightly GitHub Actions job that runs **one cell per policy** as a regression test.
- Catches regressions in lerobot upstream that would silently change numbers.

**Exit criteria for v1.1:** dataset bumps to `lerobot-bench-v1.1`. Deck slide 04 matrix re-renders with denser cells. Headline gap claim is restated with full-suite numbers.

---

## 3 · v1.2 — new envs + Pi-family

### 3.1 New environments (pick 2–3, not all)

Candidates, prioritised by relevance + Hub checkpoint availability:

| env | priority | why |
| --- | --- | --- |
| `aloha_insertion` | **high** | second ACT-paper env; closes ACT's solo-env coverage |
| `robomimic_lift` / `_can` / `_square` | **high** | widely-used benchmark; gives cross-paper comparison |
| `metaworld_pick_place` | medium | RL-canonical; brings non-imitation policies into scope |
| `robocasa_kitchen` | medium | recent, large-scale; tests scaling |
| `pusht_variants` | low | small variation on pusht; mostly Diffusion Policy stress test |

Each new env requires a calibration pass + a `MODEL_CARDS.md` entry with the published rates we're comparing to.

### 3.2 Pi-family with quantization

- pi-0 and pi-0.5 are large; full-precision exceeds laptop VRAM.
- Two paths to evaluate: 4-bit GPTQ, or `accelerate device_map="auto"` streaming.
- First step: a deferral note (already in `docs/DEFERRED_POLICIES.md`); v1.2 actually executes one of the two paths.
- Add wall-time + peak-VRAM columns to the parquet schema for cost-aware comparisons.

### 3.3 Cross-policy on overlapping envs

- Where Hub checkpoints exist, **deliberately test policies off-home-env** (e.g. SmolVLA on PushT, Diffusion Policy on LIBERO if a checkpoint surfaces).
- This is the only way to claim cross-policy comparisons properly.

**Exit criteria for v1.2:** matrix is no longer sparse along the SmolVLA row; pi-family appears at least in one cell.

---

## 4 · v1.3 — sim-to-real bridge

Goal: take the protocol off MuJoCo onto physical hardware.

- Target hardware: **Koch v1.1** and **SO-100** arms (both well-supported in lerobot).
- Re-run a subset of the matrix (1 policy × 1 task, then expand) on real hardware.
- Re-engineer episode termination + success detection for the physical setup (vision-based success classifier, or instrumented success switch).
- Sparser matrix is fine; the **statistics infrastructure carries over unchanged**.

This is the test that distinguishes "robot software" from "sim software". If our protocol survives the embodiment gap, the contribution claim from deck slide 22 ("the protocol is the contribution") gets meaningfully stronger.

---

## 5 · vNext — community + scaling laws

Once v1.3 is out:

### 5.1 Community submissions

- Open the leaderboard to external checkpoint submissions: PR your `MODEL_CARDS.md` entry + a calibrated sweep manifest.
- CI runs the cell(s) on a shared HF Space-attached GPU; results land in the published dataset.
- This is the GitHub Discussions → Hub flow; needs an admin-review step to gate the merge.

### 5.2 Inference-cost as a benchmark dimension

- Three columns we don't currently expose but should: peak VRAM, wall-clock per episode, FLOPs/step.
- A scatter plot of "success rate vs inference cost" is way more useful than success rate alone.

### 5.3 Scaling laws

- Across pi-0, pi-0.5, SmolVLA, XVLA: plot success vs param count vs training data size.
- Does the replication gap narrow at scale? Stay constant? Widen?
- This is publishable once v1.2 lands.

### 5.4 Online learning track

- Use the deterministic-ceiling baseline (which env _can_ achieve given perfect info) as a critic.
- Small-batch RL fine-tuning that pushes a pretrained policy toward its paper number.
- Publishes a "before/after" delta column on the leaderboard.

---

## 6 · How this gets executed

- **One milestone at a time.** v1.0.1 (audit) must close before v1.1 starts — otherwise we keep building on numbers we haven't validated.
- **Each milestone bumps the dataset version** (`lerobot-bench-v1.0.1`, `…-v1.1`, etc.). Old data stays published; new data is additive.
- **Every milestone re-renders the deck and paper.** The "what we verified" slide 07 grows; the "open questions" slide 08 shrinks. Track on `docs/CLAIM_AUDIT_SMOLVLA.md`.
- **PR template** for new milestones: methodology audit checklist on top, then expansion items.
- **Each policy/env addition lands in a single PR** that includes: model card update, calibration entry, one full cell, and one paragraph in this document.

---

## 7 · What's intentionally *not* on this roadmap

- **Tuning the policies themselves.** This is a benchmark, not a methods paper. If a policy under-performs, we report it; we don't fix it.
- **A new training loop.** Out of scope until §5.4 (online learning track), and even then we treat it as a separately-scored fine-tune column rather than replacing the baseline checkpoint.
- **Comparing to closed-source models.** If a checkpoint isn't downloadable + auditable, it doesn't enter the matrix.
- **Continuous-action regression metrics.** Binary success per episode is the contract; nothing else.

---

*Maintainer: keep this file in lock-step with the deck (slides 07, 08, 20) and `docs/DEFERRED_POLICIES.md`. When v1.0.1 ships, archive the §1 list under `docs/AUDITS/` and start the next round.*
