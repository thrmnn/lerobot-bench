# Pipeline Roadmap — beyond v1.0

**Status:** draft · 2026-05-29 (post-v1.0.2 cascade) · target reviewer: maintainer
**Companion:** the deck at `paper/deck/` (slides 07, 08, 20) and `docs/DEFERRED_POLICIES.md`.

This document plans how `lerobot-bench` evolves past the v1.0 sweep. It is organised so that **methodology robustness comes first** — the deck's "replication gap" claim is only as good as the assumptions behind it, and several of those assumptions are not yet verified to the standard we want for a public benchmark.

---

## Current focus / critical path (as of v1.0.2)

**THE critical path is the publish chain** — pushing the Hub dataset + Space so the public-facing surfaces (README, site, deck, paper dataset citation, upstream-issue repro links) stop pointing at dead 401 links. Everything else (v1.1 coverage, upstream PR, the V3 distribution moment) is gated behind this single chain going live. The paper builds clean (4 body + 1 refs page, audit numbers baked in) and the public surfaces are content-ready; they are hard-blocked only on the Hub artifacts existing.

**Ordered steps (see `## Publish runbook` below for exact commands):**

1. Create the Hub **dataset** repo `thrmnn/lerobot-bench-v1` + upload its card (`docs/HUB_DATASET_README.md` → dataset `README.md`). *Gates everything downstream.*
2. **Publish** artifacts from `results/sweep-full/` (parquet + manifest + 5247 videos) with `scripts/publish_results.py`.
3. Create + **deploy the Space** `thrmnn/lerobot-bench` (renders non-empty only after step 2 lands the parquet).
4. **README URL unblock** — drop the dataset-TODO comment once the parquet is live.
5. **v1.0.0 git tag + GitHub release** (release.yml auto-creates the release on tag push).

**Triage of the work that remains:**

- **Gated on user (author-side, one-time):** arXiv submission + category lock (fills the README/deck/site arXiv links + BibTeX); driving any interactive HF auth. The publish chain itself is runnable now (`huggingface-cli whoami` = `thrmnn`, write token live).
- **Gated on external resources:** §1.4 independent replication (external lab), §1.5 cross-hardware probe (second GPU, e.g. RTX A6000), the full v1.1 10-task LIBERO sweep (multi-day GPU), XVLA bug-3 (upstream `huggingface/lerobot#3674` + open-ended debug), pi-family quantization if no quantized Hub checkpoint exists.
- **Autonomous-doable now:** the publish chain (steps 1–5, modulo the two broken Make targets flagged in the runbook); §1.6 negative-control probe (cheap, local GPU) or its waiver; flipping CLAIM_AUDIT / SUCCESS_CRITERION `Status: Open` headers to closed; the #125 probe-override→config refactor; merging the SO-100 env repo PR #1 + adding CI; the v1.1 calibration probe to size the 10-task sweep; a `device_map="auto"` load spike for pi-family.

**Decision to surface before the tag (step 5):** the CHANGELOG `[Unreleased]` section holds **v1.0.1 + v1.0.2** work, but the version triple (`VERSION` / `__version__` / `pyproject`) reads `1.0.0`. release.yml gates on `tag == VERSION`, so either bump the triple to `1.0.2` and tag `v1.0.2`, or tag `v1.0.0` and treat the audit cascade as a later release. Resolve this before tagging.

---

## Publish runbook

Exact commands for the chain above. **Two Make targets are broken as written — use the explicit forms here, not `make publish` / `make space-deploy`.**

### Step 1 — Create the Hub dataset repo (gates everything)

`publish_results.py` does **not** create the repo; `_check_hub_auth` calls `repo_info(...)` and exits `4` if the dataset is missing. Create it first, then upload the card:

```
huggingface-cli login                                       # write scope; auth is live as thrmnn
huggingface-cli repo create lerobot-bench-v1 --type dataset
# upload docs/HUB_DATASET_README.md as the dataset's README.md (has the configs: results.parquet front-matter)
```

### Step 2 — Publish artifacts

`make publish SWEEP=…` is **broken** (the target ignores `SWEEP`; the three `--*-path` flags are `required=True`, so argparse errors). Invoke directly:

```
# dry-run first: stages + writes _provenance.json, no network
python scripts/publish_results.py \
  --results-path results/sweep-full/results.parquet \
  --manifest-path results/sweep-full/sweep_manifest.json \
  --videos-dir results/sweep-full/videos \
  --dry-run
# then drop --dry-run to upload
```

Only `results.parquet` / `sweep_manifest.json` / `videos/*.mp4` are staged (the `results-act-rerun` / `results-xvla-sanity*` parquets in the same dir are ignored). The raw parquet ships with the `xvla_libero` rows intact **by design** (`leaderboard_filter.py` — reproducibility); the Space filters xvla on read. Exit codes: `0` ok / `2` partial (oversized MP4 skipped) / `3` bad inputs / `4` auth-or-repo-missing / `5` mid-upload fail.

### Step 3 — Create + deploy the Space

`make space-deploy` is **insufficient** — there is no `hf-space` remote and `space/` is not a standalone repo (it's part of the parent monorepo), so a flat `git push hf-space main` would nest `app.py` under `space/` where the Space can't find it. Create the Space, add the remote, and push the **subtree** so `space/`'s contents land at the Space root:

```
huggingface-cli repo create lerobot-bench --type space --space_sdk gradio
git remote add hf-space https://huggingface.co/spaces/thrmnn/lerobot-bench
git subtree push --prefix space hf-space main
```

The Space reads the dataset purely by URL (`…/lerobot-bench-v1/resolve/main/results.parquet`, videos via `resolve/main/…mp4`), so it renders non-empty only after step 2. `requirements.txt` pins the project at a GitHub SHA — confirm that SHA is still an ancestor of `main` before deploy.

### Step 4 — README URL unblock

Once the parquet is live, drop the dataset-TODO comment in `README.md` (the HF Dataset badge + Quick-links dataset URL already point at the final location). The HF Space badge + live-leaderboard links go live after step 3. The BibTeX `<!-- TODO -->` stays blocked on the arXiv ID (out of the publish chain). Land via a normal PR to main.

### Step 5 — v1.0.0 git tag + GitHub release

After resolving the CHANGELOG-vs-tag numbering decision and moving `[Unreleased]` under the chosen version on main:

```
git tag v1.0.0          # or v1.0.2 per the decision above
git push origin main --tags
```

`release.yml` triggers on the tag push, gates on `VERSION == __version__ == pyproject == tag` (the `v` prefix is stripped), then auto-creates the GitHub Release with generated notes + `dist/*` — **no manual `gh release create` needed.** TestPyPI only fires on manual `workflow_dispatch`.

**Dependency order:** step 1 → step 2 → {step 3, step 4} → step 5.

---

## 0 · Two-speed framing

This roadmap describes the **fast lane**: the production benchmark — shipping, testing, coverage breadth, the publish chain. There is also a **slow lane**: the world-model / JEPA planner research track, which runs in its own repo on its own clock and writes into the bench through exactly one gated adapter PR (held off the leaderboard until promoted). The two-speed operating model — what belongs in each lane, and the single sanctioned write across the boundary — is specified in [`docs/TWO_SPEED.md`](TWO_SPEED.md); the research track itself in [`docs/WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md). Everything in §§1–5 below is fast-lane work; §6 is where the slow lane lands when (and only when) a planner is promoted.

---

## 0.1 · Why this doc exists

The v1.0 sweep produced strong, quotable numbers:

- SmolVLA on `libero_10`: paper **0.71** → measured **0.252** (Δ −45.8 pp, Wilson 95% CI [.202, .309])
- SmolVLA on `libero_object`: paper **0.94** → measured **0.528** (Δ −41.2 pp)
- Diffusion Policy on `pusht`: 0.654 Hub-card reference → measured **0.816** (+16.2 pp; success-rule over-count, not a genuine lift)
- ACT on `aloha_transfer_cube`: paper rate not given → measured **0.016** (does not beat random)

These claims are **statistically tight** (N=250 puts MDE ≈ 12.3 pp; the largest gaps are 4× that). They are **deterministically reproducible** on our hardware (seed triple → bit-identical parquet rows). The XVLA debugging effort (PRs #71, #74) gave us direct evidence that **Hub processor wiring is a real failure mode**, and SmolVLA passed that audit.

But they are **single-lab, single-hardware, single-task-id** numbers. Before we ride this finding into a paper headline, every assumption that could partially explain the gap needs to be tested in the open. **§1 of this doc is the audit plan; §§2–5 are the coverage expansion.**

**Post-cascade update (v1.0.2):** the v1.0.1 audit ran (see §1 below — §§1.1/1.2/1.3 closed, §§1.4/1.5/1.6 open on external/cheap-local resources). The ACT `aloha_transfer_cube` cell above is now understood as an **inference-config artifact, not an architecture failure**: with `temporal_ensemble_coeff` set, the probe lifts it **0.016 → 0.764** (Wilson [.708, .812], CIs disjoint from the broken cell by an order of magnitude). The SmolVLA `libero_10` headline is reframed as a **task-coverage scope mismatch** (paper = 10-task suite avg; v1 = `task_id=0`), and a cap=600 probe confirms the step budget is **not** the bottleneck (`0.252 → 0.256`). Both findings are baked into the paper abstract.

The **v1.0.2 consolidation** also shipped, beyond the audit numbers: probe-script dedup into `scripts/probes/_common.py`; the `leaderboard_filter` module that drops `xvla_libero` on read; a deck symbol fix; the paper trimmed to **4 body pages** (+1 refs); the figure pipeline (`src/lerobot_bench/figures.py` `replication_scatter()` + `scripts/render_figures.py`) with the scatter embedded in paper §S1 and the site Results section; and a doc-reconciliation pass fixing the deck ACT contradiction and stale references.

---

## 1 · v1.0.1 — methodology audit (mostly closed; ran during the cascade)

Goal: turn each "warn" entry from deck slide 08 into a "verified" entry on slide 07.

**Status after the v1.0.2 cascade: 3 of 6 closed, 3 open.** §§1.1/1.2/1.3 are closed-green (with the §1.3 caveat below). §§1.4/1.5 are deferred-in-prose and need external resources to *close-green* — but only doc-only **waiver entries in `docs/CLAIM_AUDIT_SMOLVLA.md`** to satisfy the gate's waiver form. §1.6 is a cheap local probe and is neither run nor waived yet. No new full sweeps are required to meet the gate.

> **Doc hygiene:** the headers of `docs/CLAIM_AUDIT_SMOLVLA.md` and `docs/SUCCESS_CRITERION_AUDIT.md` still read `Status: Open` despite the audits being substantively complete — flip to Closed/Resolved when declaring the gate met. Also confirm the PR #91 headline reframe propagated to deck slide 07 + paper §results.

### 1.1 Task-id coverage audit — CLOSED (documented mismatch)

> **Open question:** "SmolVLA paper may report per-suite averages over all 10 LIBERO tasks; we run task_id=0 only."

- **Closed.** `docs/CLAIM_AUDIT_SMOLVLA.md` (PR #84) traces SmolVLA §4.1 (Shukor 2025, arXiv:2506.01844): the published 0.71/0.94 are **suite-averages over 10 tasks × 10 trials**; v1 ran `task_id=0` × 250 ep. The per-task fact (0.252 [.202, .309]) is bit-reproducible; the apples-to-apples suite-avg comparison is **not identified at v1 coverage**. Headline reframed via PR #91. The 10-task remedy is correctly pushed to v1.1 (§2.1).
- Closed *as documented mismatch*, not *as resolved* — the deck/paper headline must stay in the softened form.

### 1.2 Inference-settings audit — CLOSED (audit + both load-bearing probes resolved)

> **Open question:** "Action-chunk horizon and sampling strategy use lerobot defaults — may not match paper."

- **Closed.** Static audit `docs/INFERENCE_AUDIT.md` (PR #86) found 7 mismatches; the load-bearing one is ACT (`n_action_steps` 100 vs 1, `temporal_ensemble_coeff` None vs 0.01). Probe artifact `results/probes/act-aloha-temporal-ensemble/summary.json` lifts the ACT cell **0.016 → 0.764** [.708, .812] — disjoint Wilson CIs, so an inference-config artifact, not architecture failure. README/MODEL_CARDS filled (commit aacf3a2). pi0fast/xvla chunk_size mismatches concern deferred policies, not v1-blocking.

### 1.3 Episode-termination + success-threshold parity — CLOSED (audit + canonical infra + cap probe)

> **Open question:** "Are we using the same success criterion as the paper?"

- **Closed (with a soft underbelly).** Audit `docs/SUCCESS_CRITERION_AUDIT.md` (PR #89): PushT + Aloha rule mismatches (bench `final_reward>=thr` over-counts vs sticky); LIBERO rule bit-equivalent but step-cap 520 vs canonical 600. Infra `docs/CANONICAL_CRITERIA.md` (PR #90): `--canonical` flag + `EnvSpec.with_criterion()` in `src/lerobot_bench/envs.py`. Cap probe `results/probes/smolvla-libero-10-cap600/summary.json` → **0.252 → 0.256** (Δ +0.4pp, within Wilson half-width); cap-hits 74.8% → 74.4% — policy stuck-while-trying, **step budget is not the bottleneck**.
- **Residue:** the cap probe covers only `libero_10`. The PushT/Aloha **rule-axis** over-count and SUCCESS_CRITERION_AUDIT §9 items 1/2/4 (PushT decomposition, ACT strict-transfer rate, Hub-card repro) are **not yet empirically probed** — scheduled for v1.1 (parquet schema bump + rule switch). Closeable as "documented + infra-built + primary cap probe done"; headline claims survive directionally.

### 1.4 Independent SmolVLA replication — OPEN (needs external lab)

> **Open question:** "Only one lab measured this."

- **Open.** Requires an EXTERNAL LAB to run `scripts/run_one.py` at the pinned SHA + audited inference settings; their parquet lands in `results-external/` (does not exist yet). Outcome gate: pause v1.1 if external diverges >15pp.
- **Waivable, not closed:** per the bundle exit criteria this can be waived in `CLAIM_AUDIT_SMOLVLA.md` with a doc-only edit; external resources are needed only to upgrade waived → closed-green.

### 1.5 Cross-hardware probe — OPEN (needs external GPU)

> **Open question:** "Single laptop GPU; CUDA non-determinism could shift behaviour."

- **Open.** Requires a second GPU (roadmap names RTX A6000 on the lab cluster): run e.g. SmolVLA × libero_goal, compare per-episode success, emit a calibrated-MDE band.
- **Waivable, not closed** — same as §1.4: the waiver is doc-only; close-green needs the external GPU.

### 1.6 Negative-control probe — OPEN (cheap local probe, least attention)

- **Open and under-specified.** Run SmolVLA on its **own training distribution** (public eval split) → expect ~100%. No artifact, no probe script, no PROBE_RESULTS mention — this item has had the least attention of the six. It is neither run, probed, nor waived. It is **local-GPU-feasible and cheap** — run it, or write an explicit waiver entry. This is the smallest real risk on the gate.

**Bundle exit criteria for v1.0.1:** all six items either closed-green or explicitly waived in `docs/CLAIM_AUDIT_SMOLVLA.md`. **Net: the gate is achievable with doc-only edits (waivers for §§1.4/1.5 + status-header flips) plus at most one cheap local probe (§1.6) — no external lab/GPU is needed to meet the waiver form.** The `Status: Open` headers on CLAIM_AUDIT / SUCCESS_CRITERION are the clearest signal the gate is not yet formally declared met.

---

## 2 · v1.1 — coverage breadth (after v1.0.1 ships)

Goal: address the "smolvla coverage skew" critique. Currently 4 of 6 non-baseline cells are SmolVLA — that's *where Hub checkpoints exist*, but a benchmark with that distribution is fragile to the SmolVLA story changing.

### 2.1 All-tasks LIBERO

- Run **all 10 task_ids** per LIBERO suite for SmolVLA (and XVLA once §3 lands).
- That's `4 suites × 10 tasks × 5 seeds × 50 ep = 10,000 LIBERO eps per VLA policy`. Calibrate first to confirm fits in a weekend sweep.
- Aggregate two ways: per-suite mean (matches papers) and per-task-id (matches us). Publish both.
- **Readiness:** infra-ready — `configs/envs.yaml` already takes `factory_kwargs.task_ids: [0]`; switching to `[0..9]` is a config edit, no source change. This is the **single largest compute line item** (≈10× the v1 LIBERO eps; multi-day GPU). **Next action:** a 1-suite × 10-task calibration probe to get per-episode wallclock and size the overnight sweep. Directly resolves the smolvla single-task-vs-paper-10-task caveat; the cap=600 probe already ruled out truncation as the confounder, so task coverage is the remaining gap.

### 2.2 XVLA bug 3

- See `docs/DEFERRED_POLICIES.md`. Suspected chunked-action layout vs `empty_camera_0` handling.
- Engage upstream issue `huggingface/lerobot#3674`.
- Once resolved, run the same 4-suite × 10-task matrix as 2.1.
- Re-enable XVLA on the live leaderboards (revert the filter from PR #82).
- **Readiness:** NOT autonomous — highest-uncertainty item. Bugs 1+2 (6D→axis-angle postproc, ImageNet preproc) are patched and confirmed firing (PRs #71/#74), but XVLA is still **0/10 across all 4 LIBERO suites**. The `control_mode=absolute` hypothesis (upstream lerobot#3401) was tested — 0/5, full timeout — and ruled out. Remaining candidates (none isolated): chunked-action layout, `empty_camera_0` handling, tokenizer/prompt contract. Fix path needs upstream coordination (#3674) + lerobot-bench task #62 (remove loader-side patching). Locked SHA preserved.

### 2.3 Multi-comparison correction

- The matrix is now 22+ cells across 5 policies — frequentist hypothesis-testing on the family needs Holm-Bonferroni correction.
- Add `scripts/family_correction.py` that emits `(per-cell p_raw, per-cell p_adj, family α)`.
- Add a "significance after correction" column to the dashboard.

### 2.4 Replication scatter as a first-class figure

- Promote the supplementary deck slide S1 (paper-vs-measured scatter) to a top-level figure in the paper and a dashboard panel.
- **Shipped:** `src/lerobot_bench/figures.py` `replication_scatter()` + `scripts/render_figures.py` read `results.parquet` + `MODEL_CARDS.md` paper_rates, emit SVG/PNG/PDF (paper/deck/web variants). Embedded in paper §S1 + the site Results section.

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

- Two paths to evaluate: 4-bit GPTQ, or `accelerate device_map="auto"` streaming.
- First step: a deferral note (already in `docs/DEFERRED_POLICIES.md`); v1.2 actually executes one of the two paths.
- Add wall-time + peak-VRAM columns to the parquet schema for cost-aware comparisons.
- **Readiness / framing correction:** the deferral driver is **host CPU RAM, not VRAM**. PaliGemma-3B fits the 8GB 4060 at inference (<4 GB steady-state VRAM); the blocker is the **~30 GB host-RAM spike during `from_pretrained` weight-conversion staging** against a 31 GB box. Frame pi-family cost around the load-time RAM spike. **Autonomous-doable as an engineering spike:** try `device_map="auto"` first (no new checkpoint). The GPTQ path is hard-gated if no quantized Hub checkpoint exists yet. Locked SHAs ready for pi0/pi0.5/pi0fast (note pi0fast license is unspecified on Hub — treat as all-rights-reserved, publish risk).

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

**Sim-bridge progress (`lerobot-env-so100-pickplace` repo):** v0.1 (env + factory + smoke) done; phases 1+2 committed. **Open PR #1** (`feat/phase2-task-tuning`, +661/−50: wrist cam, dense shaping reward, scripted IK controller, determinism/contract tests, headless conftest) is **ready to merge — 20 tests pass; merging it + adding CI is autonomous now.** **Phase-3 grasp physics is the real gate:** the scripted controller cannot complete a grasp (SO-100's tiny 8mm×2mm finger pads on a hinged arc slide off the cube within ~20 steps even at μ=3.0/24N). Fixing it needs enlarged finger pads / high-`solimp` stickiness / a push-to-target reformulation — a physics gap, not a config tweak. Dense reward keeps the env leaderboard-usable despite 0% scripted success. Phases 3/4/5 (grasp → ACT-on-sim-demos → sim-to-real) are research-substantive.

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

## 6 · World-model / JEPA planner track (slow lane, exploratory)

This is the **slow lane** — research-grade, not on the v1 critical path. It is **planned, not shipped**, and it deliberately lives outside the production benchmark. The framing rule (see [`docs/TWO_SPEED.md`](TWO_SPEED.md)) is: world-model research moves on its own clock, in its own repo, with its own toolchain; the **only** write into this benchmark is a single gated adapter PR, held off the public leaderboard until a planner is explicitly promoted.

### 6.1 What it is

- Evaluate world-model / JEPA-style planners **as policies** through the existing eval contract: a planner exposes `act(obs) -> action` and is scored cell-by-cell with the same `(policy, env, seed, n_eps) -> CellResult` machinery (Wilson + bootstrap CIs, MDE bounds, failure taxonomy) as every other policy.
- The benchmark stays policy-agnostic: a planner is just another callable behind the eval loop. No new statistics, no new success rules.

### 6.2 Why a separate lane (and a separate repo)

- Research churn (model classes, planning horizons, latent dynamics) would destabilise a benchmark whose value is its stability. The two-speed split keeps the prod bench shippable while the WM work iterates freely.
- The research track owns its dependencies and compute profile; it does not impose them on `lerobot-bench` users. Repo split + toolchain rationale live in [`docs/WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md).

### 6.3 The single sanctioned write: a gated adapter PR

- When a planner is mature enough to benchmark, it enters via **one** adapter PR that wires a WM `kind` into `load_policy` (a future dispatch branch in `src/lerobot_bench/eval.py`, alongside the existing baseline / `repo_id` branches).
- That adapter lands **behind the leaderboard filter** (the same `leaderboard_filter.py` mechanism that defers `xvla_libero`): executed in the sweep, published in the raw parquet for reproducibility, but **excluded from the public board** until explicitly promoted.
- Promotion is a deliberate, reviewed step — not an automatic consequence of the cell running green. Until then the WM cells are exploratory, clearly labelled, and carry no leaderboard standing.

### 6.4 Readiness

- **Not autonomous, not v1-blocking.** This section is a placeholder for the slow lane; no WM `kind` dispatch exists in `eval.py` today, and adding one is explicitly out of scope for the v1 wave. Sequencing: the prod bench must publish (top of this doc) and the v1.0.1 audit gate must close before any WM cell is taken seriously.

---

## 7 · How this gets executed

- **Publish first.** The publish chain (top of this doc) is the live priority — the public surfaces are content-ready and hard-blocked on the Hub artifacts existing.
- **One milestone at a time.** v1.0.1 (audit) must close before v1.1 starts — otherwise we keep building on numbers we haven't validated. Per §1, the gate is achievable with doc-only waivers + one cheap local probe (§1.6).
- **Each milestone bumps the dataset version** (`lerobot-bench-v1.0.1`, `…-v1.1`, etc.). Old data stays published; new data is additive.
- **Every milestone re-renders the deck and paper.** The "what we verified" slide 07 grows; the "open questions" slide 08 shrinks. Track on `docs/CLAIM_AUDIT_SMOLVLA.md`.
- **PR template** for new milestones: methodology audit checklist on top, then expansion items.
- **Each policy/env addition lands in a single PR** that includes: model card update, calibration entry, one full cell, and one paragraph in this document.
- **Refactor on deck (task #125):** promote probe overrides (max_steps caps, control_mode, chunk_size, temporal-ensemble flags) from runtime `dataclasses.replace`/monkeypatch (in `scripts/probes/`) into declared `EnvSpec`/`PolicySpec` overlay fields — mirroring the shipped `canonical:` sub-block + `EnvSpec.with_criterion`. Pure refactor against synthetic-testable code, no GPU, autonomous.

---

## 8 · What's intentionally *not* on this roadmap

- **Tuning the policies themselves.** This is a benchmark, not a methods paper. If a policy under-performs, we report it; we don't fix it.
- **A new training loop.** Out of scope until §5.4 (online learning track), and even then we treat it as a separately-scored fine-tune column rather than replacing the baseline checkpoint.
- **Comparing to closed-source models.** If a checkpoint isn't downloadable + auditable, it doesn't enter the matrix.
- **Continuous-action regression metrics.** Binary success per episode is the contract; nothing else.

---

*Maintainer: keep this file in lock-step with the deck (slides 07, 08, 20) and `docs/DEFERRED_POLICIES.md`. When v1.0.1 ships, archive the §1 list under `docs/AUDITS/` and start the next round.*
