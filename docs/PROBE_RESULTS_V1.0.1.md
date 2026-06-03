# v1.0.1 audit probe results

> ⚠️ **CORRECTION (v1.0.2) — Probe 1's causal conclusion was WRONG and has been SUPERSEDED.**
>
> The original Probe 1 below concluded that "Hub-default inference settings hide ~75 pp of ACT's competence" and that **temporal ensembling** recovered the cell from 0.016 to 0.764. **That causal story is false.** A controlled 2×2 ablation (normalization {buggy, fixed} × inference {Hub-default, paper-settings}, each cell N=250) shows the recovery is **100% a normalization fix on our end, 0% temporal ensembling**. The 0.016 was a **bench-side normalization bug in our own eval harness** — it silently skipped applying dataset normalization stats to `observation.images.top`, feeding ACT un-normalized image observations. We caught and fixed it in #51. On the fixed harness, Hub-default vs. paper inference settings are statistically indistinguishable (0.812 vs. 0.768) — **ensembling is a wash, not the cause.** See the CORRECTION banner inside Probe 1 for the ablation table and the canonical number. The original Probe-1 text is preserved below for the historical record but its verdict no longer holds.
>
> **Status: BOTH probes RESOLVED** (Probe 1's *resolution* now reframed as the norm-fix story above).
>
> **Probe 1 — ACT × aloha (RE-FRAMED):** the v1.0.0 `0.016` was **our own normalization bug**, fixed in #51. Canonical number is **ACT × `aloha_transfer_cube` = 0.824 [0.772, 0.866]** (N=250, Hub-default inference, normalization fixed; from `results.parquet`). Temporal ensembling does not move the cell.
>
> **Probe 2 — SmolVLA × libero_10 at canonical cap=600:** essentially no change. Pooled rate **0.252 → 0.256** (Δ +0.4 pp, within Wilson half-width). Cap-hits stay high (74.4% at cap=600 vs. 74.8% at cap=520) — the policy is **stuck-while-still-trying**, not slow-but-eventually-correct. The v1.0.1 "lower bound at our cap" caveat was technically correct but the lower bound essentially equals the value; extending the cap does not recover successes. The smolvla scope caveat (single-task vs. paper 10-task) is **NOT** resolved by this probe and remains v1.1 work.
>
> Last update: 2026-06-03 (Probe 1 reframed to the normalization-fix conclusion).

## What this doc is

The v1.0.1 methodology audit (PRs #84, #86, #89) identified three places where the v1 sweep measurement is **scope-narrower than the source paper's claim**, and PR #91 restated the v1 headline framing accordingly. This doc holds the **empirical resolution** of two of those three caveats — the actual numbers you get when you re-run the affected cells under the paper-canonical settings.

| Audit | What the v1 sweep ran | What the paper / canonical protocol uses | Probe |
|---|---|---|---|
| PR #84 | smolvla × LIBERO at `task_id=0`, 5 seeds × 50 ep | 10 tasks averaged × 10 trials/task | _no probe_ (scope mismatch, not a setting flip) |
| PR #86 | act × aloha_transfer_cube with Hub-default `temporal_ensemble_coeff=None, n_action_steps=100` | Paper `coeff=0.01, n_action_steps=1` | `scripts/probes/probe_act_temporal_ensemble.py` (task #121) |
| PR #89 | LIBERO step caps `{spatial=280, object=280, goal=300, libero_10=520}` | Canonical LIBERO `max_steps=600` for all four suites | `scripts/probes/probe_smolvla_libero_canonical_cap.py` (task #122) — currently runs libero_10 only |

The PR #84 scope mismatch can only be resolved by a 10-task sweep (deferred to v1.1) — there is no single setting to flip. The other two are runnable now on the v1 GPU and produce a parquet under `results/probes/<probe>/`.

## Probe 1 — ACT × Aloha (task #121) ✅ RESOLVED — ⚠️ CAUSAL CONCLUSION SUPERSEDED

> ⚠️ **CORRECTION (v1.0.2): the temporal-ensembling interpretation in the original Probe-1 text below was WRONG. The recovery was a normalization bug we fixed on our end, not temporal ensembling.**
>
> **What was wrong.** The original Probe 1 framed `0.016 → 0.764` as a *temporal-ensembling* effect ("Hub default was hiding ACT's competence", "+26 pp over paper"). That causal claim is false. The original "0.764 probe" ran on **post-#51 code** — i.e., it already carried the normalization fix — so it conflated the norm fix with the inference-setting change and wrongly attributed the entire jump to ensembling.
>
> **The truth (ablation-backed).** The v1.0.0 `0.016` was a **bench-side normalization bug in our own eval harness**: it silently skipped applying dataset normalization stats to `observation.images.top`, feeding ACT un-normalized image observations. We caught and fixed it in **#51** (`_recover_dataset_stats_from_safetensors` now disambiguates buffer names against config `feature_keys`). A controlled 2×2 ablation isolates the cause — normalization {buggy, fixed} × inference {Hub-default, paper-settings}, each cell **N=250 (5 seeds × 50 ep)**:
>
> | normalization | inference settings | pooled success |
> |---|---|---|
> | **buggy** | Hub-default (`coeff=None`, `n_action_steps=100`) | **0.016** |
> | **buggy** | paper (`coeff=0.01`, `n_action_steps=1`) | **0.016** |
> | **fixed** | Hub-default (`coeff=None`, `n_action_steps=100`) | **0.812** |
> | **fixed** | paper (`coeff=0.01`, `n_action_steps=1`) | **0.768** |
>
> Source: `results/probes/act-norm-ablation/{buggy_hub,buggy_paper,fixed_hub,fixed_paper}/summary.json`; probe script `scripts/probes/probe_act_normalization_ablation.py`.
>
> **Verdict: recovery is 100% the normalization fix, 0% temporal ensembling.** On **broken** normalization, switching to paper inference settings does **nothing** (0.016 → 0.016). On **fixed** normalization, Hub-default vs. paper settings are statistically **indistinguishable** (0.812 vs. 0.768, overlapping Wilson CIs) — **temporal ensembling is a wash**, not the cause.
>
> **Canonical number (UNCHANGED, headline):** **ACT × `aloha_transfer_cube` = 0.824 [0.772, 0.866]** (N=250, Hub-default inference, normalization fixed; from `results.parquet`). The ablation's `fixed + Hub-default` cell (0.812) is the same condition measured in a separate N=250 run — the two are consistent within CI.
>
> **Tone note for future reads:** this was a bug in *our* harness, not a bad Hub default. We caught it, fixed it (#51), and the leaderboard number stands.

---

_The remainder of this section is the **historical** Probe-1 record. Its methodology description is accurate; its **causal verdict** (attributing the jump to temporal ensembling) is **superseded** by the CORRECTION above and retained only for provenance._

**Source paper:** Zhao et al., _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_ (RSS 2023). Table I shows ACT with overlapping action-chunk weighted averaging at `k=0.01` (`temporal_ensemble_coeff` in lerobot terms) and `n_action_steps=1`.

**v1 sweep value:** **0.016** [0.006, 0.040] pooled across 5 seeds × 50 ep on `aloha_transfer_cube`. _(Now known to be the bench-side normalization bug, not an inference-setting effect — see CORRECTION.)_

**Original probe value (superseded interpretation):** **0.764** [0.708, 0.812] pooled — _at the time read as "paper inference settings," but this run was on post-#51 code and the jump was the normalization fix, not ensembling._

| Seed | v1.0.0 reading | original probe run | Δ |
|---|---|---|---|
| 0 | 0.02 | **0.92** | +0.90 |
| 1 | 0.04 | **0.80** | +0.76 |
| 2 | 0.00 | **0.76** | +0.76 |
| 3 | 0.02 | **0.66** | +0.64 |
| 4 | 0.00 | **0.68** | +0.68 |
| **pooled** | **0.016** | **0.764** | **+0.748** |
| **Wilson 95% CI** | [0.006, 0.040] | **[0.708, 0.812]** | — |
| **across-seed stdev** | 0.018 | 0.104 | — |

> ⚠️ **Superseded verdict (do not cite).** The original text concluded "Hub default was hiding ACT's competence" via temporal ensembling and called it "a documented quirk of the ACT inference pipeline." The controlled ablation refutes this: the jump is the normalization fix, and ensembling is a wash. See the CORRECTION banner above.

**Probe methodology (historical, accurate).** `scripts/probes/probe_act_temporal_ensemble.py` monkey-patches `lerobot.configs.policies.PreTrainedConfig.from_pretrained` to set `cfg.temporal_ensemble_coeff = 0.01` and `cfg.n_action_steps = 1` on ACT configs only, before the policy is instantiated. The rest of the pipeline — observation preprocessing, action postprocessing, render path — is identical to the v1 sweep. Seeds (0-4) and N=50/seed match the v1 contract for direct comparability. _(For the current causal evidence, see `scripts/probes/probe_act_normalization_ablation.py` and `results/probes/act-norm-ablation/`.)_

**Probe wall-clock.** ~50 minutes on 1× RTX 4060 (vs. ~12 minutes for the same cell at `n_action_steps=100`; the 4× wall-clock cost is the price of inference-every-step).

## Probe 2 — SmolVLA × LIBERO-10 canonical step cap (task #122) ✅ RESOLVED

**Canonical reference:** Liu et al., _LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning_ (NeurIPS 2023 D&B). The canonical LIBERO termination rule uses `max_steps=600` for every suite (spatial, object, goal, 10).

**v1 sweep value:** **0.252** [0.202, 0.309] pooled across 5 seeds × 50 ep on `libero_10`, with `max_steps=520`. 74.8% of failed episodes hit the cap.

**v1.0.2 probe value:** **0.256** [0.206, 0.314] pooled across 5 seeds × 50 ep at `max_steps=600`. Source: `results/probes/smolvla-libero-10-cap600/summary.json`.

| Seed | v1.0.0 (cap=520) | v1.0.2 probe (cap=600) | Δ | cap-hits @ 600 |
|---|---|---|---|---|
| 0 | 0.22 | 0.24 | +0.02 | 38/50 |
| 1 | 0.26 | 0.24 | −0.02 | 38/50 |
| 2 | 0.26 | 0.32 | +0.06 | 34/50 |
| 3 | 0.26 | 0.24 | −0.02 | 38/50 |
| 4 | 0.26 | 0.24 | −0.02 | 38/50 |
| **pooled** | **0.252** | **0.256** | **+0.004** | **186/250 (74.4%)** |
| **Wilson 95% CI** | [0.202, 0.309] | **[0.206, 0.314]** | — | — |
| **across-seed stdev** | 0.018 | 0.036 | — | — |

**Verdict: this is the "scaffold outcome (1) variant" — probe ≈ v1, but cap-hits stay high at cap=600.** The 520 → 600 step-cap bump produced essentially no recovery (Δ = +0.4 pp, well within the Wilson half-width of either CI). Cap-hits dropped only fractionally (74.8% at 520 → 74.4% at 600) — so the cap *is* binding at both budgets, but extending it doesn't translate into more successes. The policy is **stuck-while-still-trying** (drift-style failure mid-task) rather than slow-but-eventually-correct.

**v1.0.2 framing implication.** The v1.0.1 "lower bound at our caps" caveat in the README + MODEL_CARDS + paper was technically correct but somewhat misleading — it implied "more time would help materially." It does not. The 0.252 reading is approximately the policy's true rate at all reasonable step caps for this single-task setup. The correct reframe:

> SmolVLA × `libero_10` measures **0.252** [0.202, 0.309] at v1's cap=520 and **0.256** [0.206, 0.314] at canonical cap=600 — essentially the same number. The policy is bottlenecked by behavior (drift on long-horizon `libero_10` task 0), not by truncation. The cap=520 reading is not materially under-counting the policy's competence at this task; what bounds it is policy behavior, not the step budget.

The **scope** caveat (single-task vs. paper's 10-task average, PR #84) still stands and is **not** addressed by this probe — it requires the all-10-tasks LIBERO sweep planned for v1.1.

**Probe methodology.** `scripts/probes/probe_smolvla_libero_canonical_cap.py` calls `dataclasses.replace` on the env spec to set `max_steps=600`, then runs through the standard `run_cell_from_specs` pipeline. The cap-hit count is captured per-seed (`n_steps == 600 and not success`). 5 seeds × 50 ep × 1 cell = 250 episodes, matches v1 contract for direct comparability.

**Probe wall-clock.** ~1h45 on 1× RTX 4060 (vs. ~1h15 for the cap=520 cell in v1 — the 30-min overhead is the per-episode tail that hits the new 80-step-larger cap). SmolVLA forward pass per step is ~30 ms on the 4060; with ~70% of episodes running the full 600 steps, the linear overhead vs. cap=520 is bounded by `(600-520) × 0.030 s × 250 ep × 0.7 cap-hit-rate ≈ 700 s ≈ 12 min`; the additional 18 min was extra inference on the additional successful-near-the-cap episodes plus env reset overhead.

**Important caveat for future reads.** This probe holds the LIBERO step cap (the env caveat) constant. It does NOT hold the SCOPE caveat (`task_id=0` only) constant — that would require an all-10-tasks sweep. The two v1.0.1 caveats on smolvla × LIBERO compound; this probe resolves one and leaves the other as v1.1 work.

## What this doc UNBLOCKS in the v1.0.2 release

Once both probe summaries land:

- [ ] **README.md** — for ACT × aloha: lead with the canonical **0.824 [0.772, 0.866]** and the honest "we found and fixed our own normalization bug (#51)" framing; **do NOT** describe it as a temporal-ensembling recovery. For SmolVLA × libero_10: fill in the cap=600 number + 1-line interpretation.
- [ ] **docs/MODEL_CARDS.md** — for `act`: replace any "temporal-ensembling / Hub-default inference artifact" framing with the normalization-fix story (#51) and the 2×2 ablation. For `smolvla_libero`: fill in the cap=600 empirical number.
- [ ] **paper/main.tex** §sec:results-audit — same numeric fill-in.
- [ ] **paper/deck/index.html** slide 06 (smolvla libero_10 careful read) — add the cap=600 number alongside the cap=520 number with the cap-hit count.
- [ ] **dashboard / Space** — no change needed; v1 rows stay published; v1.0.2 is a doc-only refinement of the framing.

## What this doc does NOT resolve

- The **PR #84 SmolVLA 10-task scope mismatch** can only be closed by a 10-task LIBERO sweep. That work lives in v1.1 (`feat/v1.1-canonical-criteria` PR #90 provides the `--canonical` infrastructure; a 10-task sweep run is a separate execution task).
- **Cross-hardware repeatability** (different GPU / driver / CUDA arithmetic) — listed as §1.5 of the audit roadmap (`docs/PIPELINE_ROADMAP.md`), deferred to v1.1.
- **External replication** by a third party — §1.4, deferred.

## Reproducibility

Both probes write standard `RESULT_SCHEMA`-compatible parquet rows. To rerun:

```bash
# ACT probe
python scripts/probes/probe_act_temporal_ensemble.py
# Writes: results/probes/act-aloha-temporal-ensemble/{results.parquet,summary.json,videos/}

# SmolVLA libero_10 cap=600 probe
python scripts/probes/probe_smolvla_libero_canonical_cap.py
# Writes: results/probes/smolvla-libero-10-cap600/{results.parquet,summary.json,videos/}
```

Each probe is deterministic given (policy_sha, env, seed, n_episodes). Re-running with identical inputs reproduces the parquet bit-for-bit (same seeding contract as the main sweep — see `docs/DESIGN.md` § Methodology).

---

_Companion docs:_ [`docs/CLAIM_AUDIT_SMOLVLA.md`](CLAIM_AUDIT_SMOLVLA.md) (PR #84 audit report), [`docs/INFERENCE_AUDIT.md`](INFERENCE_AUDIT.md) (PR #86), [`docs/SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md) (PR #89), [`docs/CANONICAL_CRITERIA.md`](CANONICAL_CRITERIA.md) (PR #90 implementation), [`docs/PIPELINE_ROADMAP.md`](PIPELINE_ROADMAP.md) (full v1.0.1 → v1.1 plan).
