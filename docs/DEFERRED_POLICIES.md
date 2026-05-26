# Deferred policies (v1.1)

This document collects every policy that was *intended* for the v1
leaderboard but is being **deferred to v1.1**. Each section names the
policy (or family), the root cause, what has already been done about it,
and what a v1.1 onboarding pass would still need to land.

Locked Hub revision SHAs are preserved here so v1.1 does not require a
fresh lock-in round.

---

## Pi-family

The `pi0` family (`pi0`, `pi0.5`, `pi0fast`) is **not in the v1
roster**. The cold-load host-RAM footprint of the PaliGemma-3B backbone
exceeds the headroom on the v1 reference machine: ~30 GB host CPU RAM
during `from_pretrained` weight-conversion staging under HF
Transformers' default path, despite a steady-state peak of <4 GB VRAM.
On the 32 GB WSL2 calibration host this overflows the budget available
alongside other tenant workloads (see `paper/main.tex` Limitations).

Locked SHAs (kept so v1.1 onboarding does not need a fresh lock-in
pass):

| Policy   | Repo ID                               | Revision SHA (locked 2026-05-03)           | License     | Notes |
|----------|---------------------------------------|--------------------------------------------|-------------|-------|
| Pi0      | `lerobot/pi0_libero_finetuned_v044`   | `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` | gemma       | Pi0 LIBERO finetune (~8.9k DL). Review Gemma terms before redistribution. |
| Pi0.5    | `lerobot/pi05_libero_finetuned_v044`  | `dbf8a3f794a9c4297b44f40b752712f50073d945` | gemma       | Pi0.5 LIBERO finetune (~17.5k DL); most-downloaded Pi-class LIBERO finetune at lock-in. |
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
  RAM under 12 GB. Paper-reported success rates and Day-7 failure
  modes will be filled when these policies enter the sweep.

---

## xvla

The `xvla_libero` policy (Hub: `lerobot/xvla-libero`, revision
`12e8783e996944f5c97e490d37d4c145484ed70a`) is **deferred to v1.1**.
Two upstream Hub-artifact bugs were found and patched in our loader,
but a third deeper issue still causes 0% rollouts. Rather than block
v1 on continued investigation, xvla joins the pi-family in the
deferred-policies pile so v1 ships with a clean, honest story.

### Bug 1 — postprocessor missing the rotation-6D → axis-angle step (patched, PR #71)

The Hub checkpoint ships `policy_postprocessor.json` exported via the
generic `make_xvla_pre_post_processors` rather than the
LIBERO-specialized `make_xvla_libero_pre_post_processors`. The
rotation-6D → axis-angle conversion step required for the LIBERO action
contract is therefore **absent** from the on-disk postprocessor.

**Our loader patches this** at load time by inserting
`XVLARotation6DToAxisAngleProcessorStep` before the trailing
`DeviceProcessorStep`. The log line `patched xvla postprocessor` is
emitted whenever this fires.

### Bug 2 — preprocessor skipping ImageNet normalization (patched, PR #74)

The Hub `policy_preprocessor.json` declares
`norm_map: {VISUAL: IDENTITY}`, so images **skip** ImageNet
normalization on the input side. The model was trained against
ImageNet-normalized images.

**Our loader patches this** by inserting
`XVLAImageNetNormalizeProcessorStep` before the trailing
`DeviceProcessorStep`. The log line `patched xvla preprocessor` is
emitted whenever this fires.

### Bug 3 — unresolved (deferral driver)

With both patches above confirmed firing at load time, xvla on LIBERO
**still scores 0/10 across all 4 envs** in our sanity rollout. See
`results/sweep-full/results-xvla-sanity2.parquet` (present once the
investigation closes; the run was killed mid-stream after
`libero_spatial` confirmed 0/10).

The remaining issue is most likely in the model-arch / inference
pipeline — candidate hypotheses include the chunked-action layout, the
empty-camera placeholder handling, or a tokenizer / language-prompt
contract mismatch — but isolating it is **out of scope for the v1
window**.

### What to expect when reproducing

Anyone reproducing the xvla rows today should expect **0% xvla**
across all LIBERO suites until the upstream PR (lerobot-bench task #62,
also in flight) resolves the Hub-artifact bugs end-to-end and the third
unresolved issue is run down.

### v1.1 plan

- Land the upstream PR (task #62) so the Hub artifacts no longer
  require loader-side patching.
- Root-cause and fix the third issue (suspected: inference-pipeline /
  action-layout mismatch).
- Re-run the xvla × LIBERO cells under the same N=5×50 contract and
  publish the row alongside the v1.1 dataset.
