# Capability-Ladder artifact index

Every ladder number cited in the paper and blog maps to a committed,
reproducible artifact in this repo. This index is the lookup table: cited
number -> file -> how to regenerate it.

All rungs share the same eval contract (5 seeds, per-cell
`n_episodes_per_seed`, fixed revision SHA, the cell's canonical success
rule). Wilson 95% CIs come from `embodimetry.stats.wilson_ci`. The
canonical multi-cell sweep parquet (`results/sweep-full/`) lives on the HF
Hub, not in git (it is large and gitignored); the per-rung summary JSONs
below pin the exact slice each number is read from.

| Rung | Cell | N | Success | Wilson 95% CI | Committed artifact |
|---|---|---|---|---|---|
| L0 | Diffusion x pusht (auto-downscoped, n_ep=25) | 125 | 0.816 (102/125) | [0.739, 0.874] | `results/sweep-full/` slice (HF Hub); pinned via paper Table + sweep parquet |
| L0 / L1 baseline | ACT zero-shot x aloha_transfer_cube | 250 | 0.824 (206/250) | [0.772, 0.866] | [`results/ladder/act_aloha_l1.summary.json`](../../results/ladder/act_aloha_l1.summary.json) (`zero_shot`) |
| L1 | ACT fine-tuned x aloha_transfer_cube | 250 | 0.864 (216/250) | [0.816, 0.901] | [`results/ladder/act_aloha_l1.summary.json`](../../results/ladder/act_aloha_l1.summary.json) (`finetuned`) |
| L1 (negative) | SmolVLA zero-shot x libero_10 | 250 | 0.252 (63/250) | [0.202, 0.309] | [`results/ladder/smolvla_libero10_l1.summary.json`](../../results/ladder/smolvla_libero10_l1.summary.json) (`zero_shot`) |
| L1 (negative) | SmolVLA-LoRA fine-tuned x libero_10 | seed-0 partials | ~0.00 | — (collapse; see doc) | [`results/ladder/smolvla_libero10_l1.summary.json`](../../results/ladder/smolvla_libero10_l1.summary.json) + [`L1_smolvla_lora_collapse.md`](L1_smolvla_lora_collapse.md) |
| L2 | classical (scripted) x pusht_state | 250 | 0.012 (3/250) | [0.004, 0.035] | [`results/ladder/classical_pusht_l2.summary.json`](../../results/ladder/classical_pusht_l2.summary.json) + `classical_pusht_l2.parquet` |

## L1 lift (ACT)

`Delta = +0.040` (0.864 - 0.824), Cohen's `h = 0.110`. The fine-tuned and
zero-shot Wilson intervals overlap on `[0.816, 0.866]`; `h` is below the
N=250 MDE. This is reported as a within-noise shift, not a fine-tuning win
-- the honest result of continuing to fine-tune a policy already near its
ceiling on its own training distribution. Source:
`results/ladder/act_aloha_l1.summary.json` (`lift`).

## L2 classical quality (the "last 50% of precision" claim)

The scripted PushT state-feedback controller clears the strict canonical
bar (`coverage > 0.95`) on only **3/250 = 0.012** episodes, yet its
per-episode **max-coverage** distribution shows it is genuinely competent:

- mean max-coverage **0.505**, median **0.501**, p90 0.699, max 0.988

i.e. on an average rollout it pushes the block ~halfway to the target
footprint but cannot finish to the strict bar. The full max-coverage
histogram and percentiles are in
`results/ladder/classical_pusht_l2.summary.json` (`max_coverage`); the
per-episode rows are in `results/ladder/classical_pusht_l2.parquet`.

This is the **PushT classical** rung. It is distinct from the
SO-100 "0% scripted grasp" figure in
[`docs/PIPELINE_ROADMAP.md`](../PIPELINE_ROADMAP.md), which is a different
env (SO-100 grasp) and a different number -- do not conflate them.

## Regenerating each artifact

```bash
# L1 ACT fine-tune (GPU)
python scripts/run_ladder_l1_finetune.py

# L1 SmolVLA-LoRA negative (GPU, ~8 GB)
python scripts/run_ladder_l1_smolvla_lora.py

# L2 classical PushT (CPU-only, deterministic, no learned weights)
python scripts/run_ladder_l2_classical.py
```

L2 is deterministic (fixed per-episode seeding
`env.reset(seed=seed_idx*1000 + e)`), so a fresh clone reproduces
`3/250 = 0.012` and mean max-coverage `0.505` bit-for-bit.
