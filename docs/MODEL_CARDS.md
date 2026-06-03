# Model cards

One entry per policy in the v1 leaderboard. Repo IDs and revision SHAs
were locked at Day 0a; the failure-mode column is populated at Day 7
from the failure-taxonomy labeling pass.

The **v1 roster is five policies in the leaderboard** (`act`,
`diffusion_policy`, `smolvla_libero`, and the `no_op` / `random` floor
baselines), plus `xvla_libero` which was *executed* but is **deferred
from the v1 leaderboard** (see the xvla card below and
[`DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md)). The `pi0` family
(`pi0`, `pi0.5`, `pi0fast`) is also **deferred to v1.1** — see
[`DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md) for both deferrals and
the locked SHAs that carry forward.

> VRAM at inference and calibrated ms/step are back-filled from
> `results/calibration-2026-05-12.json`; the per-cell success rates are
> from `results/sweep-full/results.parquet`. Per-mode failure-taxonomy
> labels (overshoot / slip / drift / …) are a Day-7 hand-labeling pass and
> are **not yet in the parquet** — the cards report the label-free signal
> already on disk (cap-hit-on-failures, the timeout proxy) and flag where
> a per-mode breakdown still awaits labeling. Per `docs/CEO-PLAN.md`, a
> `repo_id` or `revision_sha` left unknown at lock-in is a sweep blocker;
> those are all resolved. Numbers below are pulled from
> `configs/policies.yaml`, the cited primary sources, and the calibration
> / sweep artifacts named above — none are invented.

---

## ACT (Action Chunking Transformer)

- **Architecture**: Action-chunking transformer — a CVAE-conditioned
  encoder–decoder transformer that, at each inference call, predicts a
  *chunk* of future actions (a fixed horizon of ~100 steps) rather than
  a single action, and executes them with temporal ensembling to
  smooth chunk boundaries. Visuomotor: RGB camera inputs plus joint
  proprioception, continuous joint-space action output.
- **Repo ID**: `lerobot/act_aloha_sim_transfer_cube_human`
- **Revision SHA**: `ba73b2766f1371cdc133ca4efb97eb090d744625`
  (locked 2026-05-03; Hub `lastModified` 2025-03-06)
- **License**: apache-2.0
- **Parameter scale**: ≈80M parameters (ACT as introduced by Zhao et
  al. 2023 is an ~80M-parameter model; the LeRobot checkpoint uses the
  default ACT config and is in the same class). Exact count for this
  checkpoint is not separately published on the Hub card.
- **Training data**: Aloha *Transfer Cube* simulation task, **human
  teleoperation** demonstrations (the `_human` suffix in the repo ID).
  The companion `_scripted` checkpoint trains on scripted-policy data;
  this checkpoint is the human-data variant.
- **Envs supported**: `aloha_transfer_cube` only. This checkpoint is
  Aloha-trained; a PushT ACT variant would be a separate Hub entry and
  is not in the v1 roster.
- **Inference precision**: fp32
- **VRAM @ inference (RTX 4060 8GB)**: **≈266 MB peak** (`results/calibration-2026-05-12.json`, `act × aloha_transfer_cube`). The lightest cell in the matrix.
- **Mean ms/step (calibrated)**: **≈18.4 ms/step** (p95 20.6 ms; 20-step probe on `aloha_transfer_cube`, same calibration JSON).
- **Paper-reported success**: `aloha_transfer_cube` = **0.50**
  (Zhao et al. 2023, Table I, row "ACT (Ours)", Cube Transfer (sim) /
  Transfer subtask, human-teleop training column; 3 seeds × 50
  evaluations). The LeRobot Hub model card additionally reports
  **83.0%** on 500 episodes from a LeRobot retraining; the model card
  itself acknowledges the gap from the paper and attributes it to
  `gym-aloha` success heuristics. lerobot-bench re-runs at 5 seeds ×
  50 episodes per cell.
- **Known failure modes**: Per-mode taxonomy labels (overshoot / slip /
  etc.) are a Day-7 hand-labeling pass and not yet in the parquet
  (`failure_mode` is a future schema bump). Caveat on any pre-#51 run:
  the original `act × aloha_transfer_cube` = 0.016 row was produced by a
  normalization bug *in our own harness* (image observations were fed
  un-normalized; see PR #51 and the audit row below), under which every
  failed episode ran the full step cap. That 100%-cap-hit signature is an
  artefact of the bug, **not** the policy's intrinsic failure mode — do
  not read it as ACT's failure signature. The corrected run
  (normalization fixed) succeeds the large majority of episodes.
- **Source paper**: Zhao et al., "Learning Fine-Grained Bimanual
  Manipulation with Low-Cost Hardware", RSS 2023
  ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)).
- **Caveats**: Paper vs. Hub-card success rates differ by ~33pp on the
  same nominal task — a head-to-head against any other number requires
  matching the success heuristic and episode budget, which is exactly
  the comparability gap lerobot-bench exists to close.
- **Paper vs. measured (v1.0.2 correction — supersedes the v1.0.1
  "temporal ensembling" framing)**: the canonical leaderboard number is
  `act × aloha_transfer_cube` = **0.824** [0.772, 0.866] (Wilson 95% CI,
  N=250, Hub-default inference, normalization fixed; `results.parquet`),
  comfortably above Zhao et al. 2023's Table I value of 0.50. The earlier
  v1.0.0 reading of **0.016** was **a normalization bug on our end**, not
  an inference-config story: our eval harness silently skipped applying
  dataset normalization stats to `observation.images.top`, feeding ACT
  un-normalized image observations. We caught and fixed it in PR #51
  (`_recover_dataset_stats_from_safetensors`; see the wiring caveat
  below). A controlled 2×2 ablation
  (`scripts/probes/probe_act_normalization_ablation.py`, each cell
  N=250 = 5 seeds × 50 ep,
  `results/probes/act-norm-ablation/`) isolates the cause:

  | normalization | inference     | success |
  | ------------- | ------------- | ------- |
  | buggy         | Hub-default   | 0.016   |
  | buggy         | paper-settings| 0.016   |
  | fixed         | Hub-default   | 0.812   |
  | fixed         | paper-settings| 0.768   |

  **Verdict: the recovery is 100% the normalization fix, 0% temporal
  ensembling.** On broken normalization, switching to paper inference
  settings does nothing (0.016 → 0.016). On fixed normalization,
  Hub-default vs. paper settings are statistically indistinguishable
  (0.812 vs. 0.768, overlapping Wilson CIs) — temporal ensembling is a
  **wash**, not the cause. (The earlier "0.764 probe" ran on post-#51
  code, so it already had the norm fix; it wrongly attributed the gain to
  the inference-setting change.) The ablation's fixed+Hub cell (0.812) is
  the same condition as the canonical 0.824, measured in a separate N=250
  run and consistent within CI. The v1.0.0 leaderboard row stays as the
  pre-#51 reading for audit-trail integrity.
  Full audit: [`docs/INFERENCE_AUDIT.md`](INFERENCE_AUDIT.md),
  [`docs/PROBE_RESULTS_V1.0.1.md`](PROBE_RESULTS_V1.0.1.md);
  Aloha success-rule audit (the bench accepts reward ∈ {1..4} while the
  paper's Transfer subtask requires `reward == 4`): PR #89 / [`docs/SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md).
- **Wiring caveat (PR #51)**: The legacy-checkpoint stats recovery in
  `_recover_dataset_stats_from_safetensors` reversed only the first
  underscore when mapping `buffer_*` keys back to feature keys, so
  `observation.images_top` was not being matched to
  `observation.images.top` and image normalization stats were silently
  dropped for this checkpoint. See `src/lerobot_bench/eval.py`
  → `_buffer_name_to_feature_key`. Downstream consumers loading via the
  pre-PR-#51 code path will get under-normalized images and ~2pp lower
  success on `aloha_transfer_cube`.

## Diffusion Policy

- **Architecture**: Diffusion visuomotor policy — represents the
  action distribution as a conditional denoising diffusion process; a
  noise-prediction network (the LeRobot checkpoint uses the CNN-based
  U-Net variant) is conditioned on a short history of RGB observations
  and proprioception, and an action sequence is produced by iterative
  denoising at inference time.
- **Repo ID**: `lerobot/diffusion_pusht`
- **Revision SHA**: `84a7c23178445c6bbf7e1a884ff497017910f653`
  (locked 2026-05-03; Hub `lastModified` 2025-03-06)
- **License**: apache-2.0
- **Parameter scale**: ≈260M parameters (the CNN-based Diffusion Policy
  on image observations as reported by Chi et al. 2023 is ~260M; the
  LeRobot checkpoint uses the default Diffusion Policy config and is in
  the same class). Exact count for this checkpoint is not separately
  published on the Hub card.
- **Training data**: PushT task demonstrations (teleoperated pushing of
  a T-shaped block to a target pose), the standard PushT demonstration
  set used by the Diffusion Policy paper and the LeRobot `pusht`
  dataset.
- **Envs supported**: `pusht` only. This checkpoint is PushT-trained;
  an Aloha Diffusion Policy variant would be a separate Hub entry and
  is not in the v1 roster.
- **Inference precision**: fp32
- **VRAM @ inference (RTX 4060 8GB)**: **≈1098 MB peak** (`results/calibration-2026-05-12.json`, `diffusion_policy × pusht`).
- **Mean ms/step (calibrated)**: **≈228 ms/step**, but **p95 ≈1505 ms/step** (20-step probe on `pusht`, same calibration JSON). The heavy iterative-denoising tail (mean ≪ p95) is what tripped the `mean_step_ms > 100` auto-downscope rule — this cell ran at 25 episodes/seed (N=125), not 50.
- **Paper-reported success**: `pusht` = **0.654** — from the LeRobot
  Hub model card for `lerobot/diffusion_pusht` (evaluation section):
  65.4% over 500 episodes on `gym-pusht`, where success := max overlap
  ≥ 95%. Note this is **not** the number in the original paper: Chi et
  al. 2023 (Table 2) report PushT as *target-area coverage* (0.91 max
  / 0.84 avg-of-last-10 for the image-CNN policy), a continuous metric
  that is not directly comparable to the binary success threshold
  `gym-pusht` and lerobot-bench use. The Hub-card number is the
  apples-to-apples reference. lerobot-bench re-runs at 5 seeds × 50
  episodes per cell.
- **Known failure modes**: Per-mode taxonomy labels are a Day-7 pass and
  not yet in the parquet. Label-free signal: at the bench's `final_reward
  >= 0.95` rule the cell succeeds on 81.6% of episodes (N=125); its
  failures are dominated by **timeout** (mode 3) — the T-block ends near
  but below the coverage threshold at the 300-step cap. The audit below
  records a 34.4% cap-hit rate among the bench's near-converged episodes,
  which is the window the lax-vs-sticky success rule opens. Drift/overshoot
  vs. genuine timeout cannot be separated until the rollouts are
  hand-labeled in v1.0.2.
- **Source paper**: Chi et al., "Diffusion Policy: Visuomotor Policy
  Learning via Action Diffusion", RSS 2023
  ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)).
- **Caveats**: The paper's coverage metric and the benchmark's binary
  success metric measure different things; do not compare the 0.91
  coverage figure against any binary success rate.
- **Paper vs. measured (v1.0.1 audit, PR #89)**: lerobot-bench v1
  measures `diffusion_policy × pusht` = **0.816** [0.739, 0.874]
  (N=125 after auto-downscope), **+16.2 pp above** the 0.654
  Hub-card reference. The audit flags this gap as a **success-rule
  over-count**, not a genuine performance lift: our success rule
  is `final_reward >= 0.95` (≡ coverage ≥ 0.9025), while the
  PushT Hub card uses `any(coverage > 0.95)` over the rollout
  (sticky-true). 34.4% of our `diffusion_policy × pusht` episodes hit
  the step cap, opening a window for the lax threshold to fire on
  near-converged but non-terminated trajectories — i.e. the bench may
  be **over-counting** relative to the canonical rule. PR #90 ships
  the canonical `sticky_is_success` rule as a selectable option; the
  re-run probe under it is queued for v1.0.2. Full audit:
  [`docs/SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md).

## SmolVLA (LIBERO finetune)

- **Architecture**: Vision-language-action model — a compact VLA that
  pairs a vision-language backbone (the SmolVLM-2 lineage) with a
  flow-matching action-expert head that decodes language-conditioned
  visual observations into continuous action chunks. Designed to be
  small and fast enough for commodity hardware.
- **Repo ID**: `lerobot/smolvla_libero`
- **Revision SHA**: `31d453f7edd78c839a8bbc39744a292686daf0de`
  (locked 2026-05-03 via `huggingface_hub.HfApi().model_info(repo_id).sha`)
- **License**: apache-2.0
- **Parameter scale**: ≈0.45B parameters. The checkpoint inherits from
  `lerobot/smolvla_base` (0.45B) — its `config.json` `pretrained_path`
  field points there — so the apples-to-apples paper row is the
  SmolVLA-0.45B variant (not the 0.24B or 2.25B variants in the same
  paper table). Smallest VLA in the v1 matrix.
- **Training data**: LIBERO. Finetuned from the `smolvla_base`
  pretrained checkpoint on LIBERO task demonstrations; the public
  README does not break the suite mix down per-suite. All four LIBERO
  suites run under lerobot's shared observation contract.
- **Envs supported**: `libero_spatial`, `libero_object`, `libero_goal`,
  `libero_10`.
- **Inference precision**: bf16
- **VRAM @ inference (RTX 4060 8GB)**: **≈921 MB peak** across all four
  LIBERO suites (`results/calibration-2026-05-12.json`; `libero_10` 920 MB,
  the other three 922 MB). Smallest GPU footprint of the three VLAs.
- **Mean ms/step (calibrated)**: **≈20–21 ms/step** on `libero_spatial`,
  `libero_object`, `libero_goal`; **≈42.5 ms/step** on `libero_10`
  (longer chunks / horizon), per `results/calibration-2026-05-12.json`.
  All four stayed under the `mean_step_ms > 100` downscope threshold, so
  every SmolVLA cell ran the full 50 episodes/seed (N=250). (Supersedes
  the earlier ~160 ms/step smoke-run estimate.)
- **Paper-reported success** (Shukor et al. 2025, Table 2, row
  "SmolVLA (0.45B), VLA Pt. = No"; protocol: 10 trials per task,
  binary scoring — 1 only if the task is fully completed):
  - `libero_spatial` = **0.90**
  - `libero_object`  = **0.96**
  - `libero_goal`    = **0.92**
  - `libero_10`      = **0.71**  (paper's "Long" suite; Avg 87.3)

  lerobot-bench re-runs at 5 seeds × 50 episodes per cell, ~5× more
  rollouts than the paper, so CI widths should be tighter.
- **Known failure modes**: Per-mode taxonomy labels are a Day-7 pass and
  not yet in the parquet. Label-free signal from the audit cap-hit-on-
  failures rates (`docs/SUCCESS_CRITERION_AUDIT.md`): `libero_spatial`
  22.4%, `libero_object` 47.2%, `libero_goal` 7.2%, `libero_10` **74.8%**.
  The low-cap-hit suites (`goal` especially) fail *early* — overshoot /
  slip / premature-release territory — while `libero_10` failures are
  cap-bound. The v1.0.2 probe characterised the `libero_10` cap-bound
  failures as **drift** (mode 6): re-running at canonical cap=600 recovered
  no successes (74.4% cap-hit even at 600), so the policy is stuck-while-
  drifting mid-task, not slow-but-eventually-correct. Per-mode counts on
  the other suites await the Day-7 labeling pass.
- **Source paper**: Shukor et al., "SmolVLA: A Vision-Language-Action
  Model for Affordable and Efficient Robotics", 2025
  ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)).
- **Caveats**: Per-suite training mix is not published, so a low score
  on one LIBERO suite cannot be cleanly attributed to data scarcity vs.
  task difficulty. The `libero_10` (long-horizon) score is the lowest
  cell in the paper — long-horizon compositional tasks are the known
  weak point of this policy class.
- **Paper vs. measured (v1.0.1 audit, PRs #84 + #89)**: lerobot-bench
  v1 measures, at `task_id=0` × 5 seeds × 50 episodes per suite
  (N=250 binary outcomes per cell):
  - `libero_spatial` = **0.776** [0.720, 0.823] (paper 0.90)
  - `libero_object`  = **0.528** [0.466, 0.589] (paper 0.96)
  - `libero_goal`    = **0.928** [0.889, 0.954] (paper 0.92)
  - `libero_10`      = **0.252** [0.202, 0.309] (paper 0.71)

  **The paper-vs-measured deltas are not apples-to-apples** and should
  not be read as replication failures. Two audit-confirmed scope
  mismatches:
  - **Task coverage (PR #84)**: the SmolVLA paper Table 2 reports
    **per-suite averages over 10 tasks × 10 trials per task = 100-ep
    suite averages**. We ran **one task (`task_id=0`) × 5 seeds × 50
    episodes = 250 single-task episodes** per suite. The two scalars
    measure different quantities: ours is a single-task envelope claim
    ("at least one of the 10 tasks scores well below the published
    suite average"), the paper's is a suite-averaged claim. v1.1 will
    expand to all 10 tasks per suite and close the apples-to-apples
    question. Full audit: [`docs/CLAIM_AUDIT_SMOLVLA.md`](CLAIM_AUDIT_SMOLVLA.md).
  - **Step caps (PR #89; v1.0.2 probe PR #108 RESOLVED for `libero_10`)**:
    canonical LIBERO uses `max_steps=600` for every suite; v1 inherited
    lerobot's tighter caps `{spatial=280, object=280, goal=300, libero_10=520}`.
    Cap-hit rates on **failed** episodes are 22.4%, 47.2%, 7.2%, and
    **74.8%** respectively. **The v1.0.2 probe re-ran `libero_10` at
    canonical cap=600 and measured 0.256 [0.206, 0.314] vs. v1's 0.252
    at cap=520 — Δ +0.4 pp, essentially zero. Cap-hits stayed at 74.4%
    even at 600.** Verdict: the cap is binding at both budgets but
    extending it doesn't recover successes. The 0.252 reading is
    **policy-bottlenecked, not cap-bottlenecked** — the policy is
    stuck-while-still-trying (drift-style failure mid-task), not
    slow-but-eventually-correct. The earlier "lower bound at our cap"
    framing was technically correct but the lower bound essentially
    equals the value. The other three LIBERO suites (`spatial`,
    `object`, `goal`) have far lower cap-hit rates and their numbers
    remain provisional lower bounds pending re-runs at cap=600 in v1.1.
    Full audit: [`docs/SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md);
    probe: [`docs/PROBE_RESULTS_V1.0.1.md`](PROBE_RESULTS_V1.0.1.md).

  Both caveats compound on the `libero_10` headline cell.
  The within-protocol measurements are bit-reproducible and tight
  (Wilson half-width 0.054 at N=250 on `libero_10`); what's contested
  is the scope of what they imply about the paper's number, not the
  number we measured.

## XVLA (LIBERO)

- **Deferred to v1.1.** Executed in the v1 sweep but excluded from the
  v1 leaderboard — two upstream Hub-artifact wiring bugs were patched
  in our loader (PR #71, PR #74) and a third unresolved issue still
  produces 0/10 rollouts across all 4 LIBERO suites. See
  [`docs/DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md) for the full
  account and v1.1 plan.
- **Architecture**: Vision-language-action model — X-VLA is a
  soft-prompted transformer designed for scalable cross-embodiment
  control; per-embodiment learnable "soft prompt" tokens condition a
  shared transformer trunk, and a flow-matching action head emits
  continuous actions. The LeRobot checkpoint is the LIBERO-specialized
  variant.
- **Repo ID**: `lerobot/xvla-libero`
- **Revision SHA**: `12e8783e996944f5c97e490d37d4c145484ed70a`
  (locked 2026-05-03 via `huggingface_hub.HfApi().model_info(repo_id).sha`)
- **License**: apache-2.0
- **Parameter scale**: ≈0.9B parameters. The Hub README links X-VLA as
  the original reference; the LeRobot checkpoint is the
  LIBERO-specialized 0.9B model (the "X-VLA (Ours), 0.9B" row of the
  paper's Table 2).
- **Training data**: LIBERO. The LIBERO-specialized full-finetune of
  X-VLA; per-suite mix is not separately documented on the Hub card.
  All four LIBERO suites run under lerobot's shared observation
  contract.
- **Envs supported**: `libero_spatial`, `libero_object`, `libero_goal`,
  `libero_10`.
- **Inference precision**: bf16
- **VRAM @ inference (RTX 4060 8GB)**: **≈3522 MB peak** across the four
  LIBERO suites (`results/calibration-2026-05-12.json`). The heaviest GPU
  footprint in the matrix — ~3.8× SmolVLA's.
- **Mean ms/step (calibrated)**: **≈53–77 ms/step** on `libero_spatial` /
  `libero_object` / `libero_goal`; **≈164 ms/step** on `libero_10`
  (`results/calibration-2026-05-12.json`). The `libero_10` mean crossed
  the `mean_step_ms > 100` threshold, so that cell was auto-downscoped to
  25 episodes/seed (N=125) at dispatch. NB: these are *load-and-step*
  latencies; xvla still produces 0/10 successful rollouts (see below), so
  the numbers measure inference cost, not useful work.
- **Paper-reported success** (Bu et al. 2025, Table 2, "X-VLA (Ours),
  0.9B" row, full-finetune LIBERO columns; the paper does not state an
  explicit episode count for the LIBERO eval — the underlying LIBERO
  benchmark uses 10 trials per task, Liu et al. 2023):
  - `libero_spatial` = **0.982**
  - `libero_object`  = **0.986**
  - `libero_goal`    = **0.978**
  - `libero_10`      = **0.976**  (paper's "Long" suite; Avg 98.1)

  lerobot-bench re-runs at 5 seeds × 50 episodes per cell.
- **Known failure modes**: **Not taxonomy-classifiable.** xvla scores
  **0/10 across all four LIBERO suites** in our rollouts (success rate
  0.000 on every cell), and that 0% is attributed to an unresolved
  Hub-artifact / inference-pipeline wiring bug (see the deferral driver
  below), not to a policy-level failure mode. Assigning overshoot / slip /
  drift labels to rollouts produced by a mis-wired loader would be
  meaningless, so the failure-mode field is left explicitly **unmeasurable
  until the wiring is fixed** (v1.1) rather than back-filled. The cell is
  excluded from the published leaderboard for the same reason. See
  [`docs/DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md).
- **Source paper**: Bu et al., "X-VLA: Soft-Prompted Transformer as
  Scalable Cross-Embodiment Vision-Language-Action Model", 2025
  ([arXiv:2510.10274](https://arxiv.org/abs/2510.10274)).
- **Caveats**: Paper-reported LIBERO scores are near-saturated
  (≥97.6% on all four suites). Near-ceiling cells leave little MDE
  headroom at N=250 — a re-run delta against the paper number, if
  small, may be inconclusive rather than a genuine regression.
- **Wiring caveat (PR #71)**: The upstream `lerobot/xvla-libero` Hub
  checkpoint ships `policy_postprocessor.json` exported via the generic
  `make_xvla_pre_post_processors` rather than
  `make_xvla_libero_pre_post_processors`, so the rotation-6D →
  axis-angle step required for this checkpoint is **missing**. Our
  eval pipeline patches the postprocessor at load time by inserting
  `XVLARotation6DToAxisAngleProcessorStep` before the trailing
  `DeviceProcessorStep` — see
  `src/lerobot_bench/eval.py:_patch_postprocessor_for_policy`.
  Downstream consumers loading this checkpoint via vanilla
  `lerobot.policies.factory.make_pre_post_processors` will silently
  get zero-success rollouts. A complementary upstream PR is planned
  (lerobot-bench task #62).
- **Wiring caveat (PR #74)**: The upstream `lerobot/xvla-libero` Hub
  `policy_preprocessor.json` declares `norm_map: {VISUAL: IDENTITY}`,
  so images **skip** ImageNet normalization on the input side even
  though the model was trained against ImageNet-normalized images. Our
  eval pipeline patches the preprocessor at load time by inserting
  `XVLAImageNetNormalizeProcessorStep` before the trailing
  `DeviceProcessorStep`. Downstream consumers loading this checkpoint
  via vanilla `lerobot.policies.factory.make_pre_post_processors` will
  silently get un-normalized image inputs. Same upstream PR (task #62)
  is planned to fix this end-to-end on the Hub side.
- **Unresolved (deferral driver)**: Even with both patches above
  firing at load time (log lines `patched xvla preprocessor` /
  `patched xvla postprocessor` confirmed), xvla on LIBERO still scores
  0/10 across all 4 suites in our sanity rollout. A third bug,
  most likely in the model-arch / inference pipeline (e.g. the
  chunked-action layout or empty-camera placeholder), remains
  unresolved and is out of scope for the v1 window. See
  [`docs/DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md) for the full
  account and v1.1 plan.

## no-op (baseline)

- **Architecture**: Weights-free baseline. Emits a zero action every
  step — no model, no inference.
- **Repo ID**: n/a (no checkpoint)
- **Revision SHA**: n/a
- **License**: MIT (the baseline implementation in this repo; there is
  no upstream model to license)
- **Parameter scale**: 0
- **Training data**: none
- **Envs supported**: `pusht`, `aloha_transfer_cube`, `libero_spatial`,
  `libero_object`, `libero_goal`, `libero_10` (all six v1 envs)
- **Inference precision**: n/a
- **VRAM @ inference**: 0
- **Mean ms/step**: ~0.1 ms (Python loop overhead only)
- **Paper-reported success**: n/a — baseline, no published reference.
- **Purpose / interpretation**: Leaderboard floor. A policy that fails
  to beat no-op on a given env has not learned that env. Note that on
  some envs a zero action is not the same as "do nothing useful" — it
  holds the arm in place, which can incidentally satisfy a small
  fraction of a coverage-style reward; the random baseline and no-op
  together bracket the trivial-policy floor.
- **Known failure modes**: n/a — by construction it fails every task
  that requires motion.

## random (baseline)

- **Architecture**: Weights-free baseline. Uniformly samples an action
  from the env action space each step — no model, no inference.
- **Repo ID**: n/a (no checkpoint)
- **Revision SHA**: n/a
- **License**: MIT (the baseline implementation in this repo; there is
  no upstream model to license)
- **Parameter scale**: 0
- **Training data**: none
- **Envs supported**: `pusht`, `aloha_transfer_cube`, `libero_spatial`,
  `libero_object`, `libero_goal`, `libero_10` (all six v1 envs)
- **Inference precision**: n/a
- **VRAM @ inference**: 0
- **Mean ms/step**: negligible (a single action-space sample per step)
- **Paper-reported success**: n/a — baseline, no published reference.
- **Purpose / interpretation**: Cheap stochastic floor. PushT in
  particular has a coverage-style reward whose expected value under
  uniform action is empirically non-zero, so the random baseline
  guards against reading a non-zero success rate as evidence of
  learning. Action sampling inherits the per-cell torch generator and
  is reproducible under the seeding contract.
- **Known failure modes**: n/a — its outcomes are sampling noise, not
  taxonomy-classifiable failures.

---

## Deferred policies (v1.1)

The `pi0` family is **not in the v1 roster**. The cold-load host-RAM
footprint of the PaliGemma-3B backbone exceeds the headroom on the v1
reference machine (see `paper/main.tex` Limitations: ~30GB host RAM
during `from_pretrained` weight-conversion staging, despite <4GB peak
VRAM at steady state). Locked SHAs are kept here so v1.1 onboarding
does not need a fresh lock-in pass.

| Policy   | Repo ID                               | Revision SHA (locked 2026-05-03)           | License | Notes |
|----------|---------------------------------------|--------------------------------------------|---------|-------|
| Pi0      | `lerobot/pi0_libero_finetuned_v044`   | `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` | gemma   | Pi0 LIBERO finetune (~8.9k DL). Review Gemma terms before redistribution. |
| Pi0.5    | `lerobot/pi05_libero_finetuned_v044`  | `dbf8a3f794a9c4297b44f40b752712f50073d945` | gemma   | Pi0.5 LIBERO finetune (~17.5k DL); most-downloaded Pi-class LIBERO finetune at lock-in. |
| Pi0Fast  | `lerobot/pi0fast-libero`              | `840f4b503f4c09110421c33c810a85b6684fd658` | unspecified | License unspecified on Hub card — treat as "all rights reserved" until clarified upstream. Logged as a publish risk. |

- **Architecture (all three)**: VLA action heads on a PaliGemma-class
  vision-language backbone. Pi0 uses a flow-matching action expert;
  Pi0Fast uses an autoregressive action tokenization for faster
  inference; Pi0.5 is the successor checkpoint in the same family.
- **Parameter scale**: ≈3B parameters (PaliGemma-3B backbone class).
- **Source**: Physical Intelligence, mirrored under the `lerobot` Hub
  org. Source paper: Black et al., "$\pi_0$: A Vision-Language-Action
  Flow Model for General Robot Control", 2024
  ([arXiv:2410.24164](https://arxiv.org/abs/2410.24164)).
- **v1.1 plan**: onboard via a quantized checkpoint (4-bit/8-bit GPTQ)
  or `accelerate` `device_map="auto"` streaming load to bring cold-load
  RAM under 12GB. Paper-reported success rates and Day-7 failure modes
  will be filled when these policies enter the sweep.
