# v1.0.1 audit probe results

> **Status: scaffold — numbers fill in once `scripts/probes/` complete.**
>
> Last update: 2026-05-27 _(probes in flight; this doc lands as a PR with placeholder rows; a follow-up PR replaces `TBD` with the empirical numbers + Wilson CIs once both probes finish)_.

## What this doc is

The v1.0.1 methodology audit (PRs #84, #86, #89) identified three places where the v1 sweep measurement is **scope-narrower than the source paper's claim**, and PR #91 restated the v1 headline framing accordingly. This doc holds the **empirical resolution** of two of those three caveats — the actual numbers you get when you re-run the affected cells under the paper-canonical settings.

| Audit | What the v1 sweep ran | What the paper / canonical protocol uses | Probe |
|---|---|---|---|
| PR #84 | smolvla × LIBERO at `task_id=0`, 5 seeds × 50 ep | 10 tasks averaged × 10 trials/task | _no probe_ (scope mismatch, not a setting flip) |
| PR #86 | act × aloha_transfer_cube with Hub-default `temporal_ensemble_coeff=None, n_action_steps=100` | Paper `coeff=0.01, n_action_steps=1` | `scripts/probes/probe_act_temporal_ensemble.py` (task #121) |
| PR #89 | LIBERO step caps `{spatial=280, object=280, goal=300, libero_10=520}` | Canonical LIBERO `max_steps=600` for all four suites | `scripts/probes/probe_smolvla_libero_canonical_cap.py` (task #122) — currently runs libero_10 only |

The PR #84 scope mismatch can only be resolved by a 10-task sweep (deferred to v1.1) — there is no single setting to flip. The other two are runnable now on the v1 GPU and produce a parquet under `results/probes/<probe>/`.

## Probe 1 — ACT × Aloha temporal-ensembling (task #121)

**Source paper:** Zhao et al., _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_ (RSS 2023). Table I shows ACT with overlapping action-chunk weighted averaging at `k=0.01` (`temporal_ensemble_coeff` in lerobot terms) and `n_action_steps=1`.

**v1 sweep value:** **0.016** [0.005, 0.046] pooled across 5 seeds × 50 ep on `aloha_transfer_cube`.

**v1.0.2 probe value:** TBD _(see `results/probes/act-aloha-temporal-ensemble/summary.json` once `probe_act_temporal_ensemble.py` completes)._

| Seed | v1.0.0 default | v1.0.2 probe (paper settings) | Δ |
|---|---|---|---|
| 0 | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| **pooled** | **0.016** | **TBD** | **TBD** |
| **Wilson 95% CI** | [0.005, 0.046] | TBD | — |

**Probe methodology.** `scripts/probes/probe_act_temporal_ensemble.py` monkey-patches `lerobot.configs.policies.PreTrainedConfig.from_pretrained` to set `cfg.temporal_ensemble_coeff = 0.01` and `cfg.n_action_steps = 1` on ACT configs only, before the policy is instantiated. The rest of the pipeline — observation preprocessing, action postprocessing, render path — is identical to the v1 sweep. Seeds (0-4) and N=50/seed match the v1 contract for direct comparability. See `scripts/probes/probe_act_temporal_ensemble.py` docstring for the override mechanism.

**Reading the result.** Three possible outcomes, with what each implies for the v1.0.2 framing:

1. **Probe ≈ v1 (within ±5pp).** The architecture itself does not reach the paper's number on `aloha_transfer_cube` from the Hub checkpoint at any inference setting we tried. v1.0.2 framing: keep the 0.016, drop the "Hub-default inference settings" caveat, flag a deeper investigation as v1.1 work.
2. **Probe materially higher (e.g. > 0.15).** The Hub default was hiding most of the architecture's competence on this env. v1.0.2 framing: lead with "ACT × aloha at paper inference settings: probe_value", report v1 default as "v1.0.0 default inference: 0.016", and remove the "act fails on aloha" framing from the live leaderboard surface.
3. **Probe between (5pp < Δ < 15pp).** Partial recovery. v1.0.2 framing: report both numbers side-by-side; do not pick one as "the" number for ACT on aloha.

## Probe 2 — SmolVLA × LIBERO-10 canonical step cap (task #122)

**Canonical reference:** Liu et al., _LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning_ (NeurIPS 2023 D&B). The canonical LIBERO termination rule uses `max_steps=600` for every suite (spatial, object, goal, 10).

**v1 sweep value:** **0.252** [0.202, 0.309] pooled across 5 seeds × 50 ep on `libero_10`, with `max_steps=520`. 74.8% of failed episodes hit the cap.

**v1.0.2 probe value:** TBD _(see `results/probes/smolvla-libero-10-cap600/summary.json` once `probe_smolvla_libero_canonical_cap.py` completes)._

| Seed | v1.0.0 (cap=520) | v1.0.2 probe (cap=600) | Δ | cap-hits @ 600 |
|---|---|---|---|---|
| 0 | TBD | TBD | TBD | TBD/50 |
| 1 | TBD | TBD | TBD | TBD/50 |
| 2 | TBD | TBD | TBD | TBD/50 |
| 3 | TBD | TBD | TBD | TBD/50 |
| 4 | TBD | TBD | TBD | TBD/50 |
| **pooled** | **0.252** | **TBD** | **TBD** | **TBD/250** |
| **Wilson 95% CI** | [0.202, 0.309] | TBD | — | — |

**Probe methodology.** `scripts/probes/probe_smolvla_libero_canonical_cap.py` calls `dataclasses.replace` on the env spec to set `max_steps=600`, then runs through the standard `run_cell_from_specs` pipeline. The cap-hit count is captured per-seed (`n_steps == 600 and not success`) so we can quantify how much of the v1 lower-bound gap was cap-truncation vs. policy failure.

**Reading the result.** Same three-bucket logic as Probe 1, plus the cap-hit count tells us the policy's typical episode length when it isn't successful:

1. **Probe ≈ v1 (within ±5pp), low cap-hits at 600.** The 520 → 600 bump didn't reach the additional successes because most failures aren't time-bounded — the policy gets stuck, not slow. v1.0.2 framing: keep "lower bound" but note that the lower-bound gap is small.
2. **Probe materially higher, high cap-hits at 600 too.** Cap is still binding; canonical 600 is itself a lower bound at the v1.0.2 measurement; the policy probably could finish more tasks with longer episodes. v1.0.2 framing: report cap=600 number, flag that even canonical LIBERO under-measures this policy.
3. **Probe materially higher, near-zero cap-hits at 600.** The policy succeeds when given enough time; 520 was eating the tail of slow-but-correct rollouts. v1.0.2 framing: replace 0.252 with the cap=600 number as the canonical reading; the lower-bound claim was justified and now resolved.

## What this doc UNBLOCKS in the v1.0.2 release

Once both probe summaries land:

- [ ] **README.md** — replace the "probe pending" sentences in the Methodology caveats table (rows 2 + 3) with the empirical numbers + a 1-line interpretation per probe.
- [ ] **docs/MODEL_CARDS.md** — replace the "v1.0.2 probe pending" footnotes under the `act` and `smolvla_libero` paper-vs-measured paragraphs with the empirical numbers.
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
