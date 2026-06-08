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

**Probe — `control_mode=absolute` (ruled out):** Closed upstream issue
[huggingface/lerobot#3401](https://github.com/huggingface/lerobot/issues/3401)
documents the same 0%-success symptom and was resolved by the reporter
with `--env.control_mode=absolute` (X-VLA is trained on absolute end-effector
poses, not deltas). We tested this hypothesis: 5 episodes of
`xvla_libero × libero_spatial × seed 0` with `control_mode: absolute`
patched into `configs/envs.yaml::libero_spatial.factory_kwargs`. Result:
**0/5**, `n_steps=280` (full timeout) on every episode. The patch is
reverted; absolute-pose mode is not Bug 3.

### What to expect when reproducing

Anyone reproducing the xvla rows today should expect **0% xvla**
across all LIBERO suites until the upstream PR (embodimetry task #62,
also in flight) resolves the Hub-artifact bugs end-to-end and the third
unresolved issue is run down.

### v1.1 plan

- Land the upstream PR (task #62) so the Hub artifacts no longer
  require loader-side patching.
- Root-cause and fix the third issue (suspected: inference-pipeline /
  action-layout mismatch).
- Re-run the xvla × LIBERO cells under the same N=5×50 contract and
  publish the row alongside the v1.1 dataset.

---

## vqbet_pusht (config-schema mismatch)

Investigated 2026-06-08 during the matrix-expansion pass as the *second*
vision policy on the shared `pusht` env — the most valuable possible
addition (a real cross-architecture head-to-head vs `diffusion_policy`
under the identical 3×96×96 image + (2,) agent-pos contract and
coverage > 0.95 success rule). Hub eval_info reports `pc_success = 57.0`.

**Blocker (genuine, reproduced on CPU):** the Hub `config.json` for
`lerobot/vqbet_pusht` (revision `15c2d0af889c401c7e5db7a07499d3b498afc276`)
carries a field — `mlp_hidden_dim: 1024` — that the pinned
`lerobot==0.5.1` `VQBeTConfig` dataclass does **not** declare. draccus
rejects the unknown field at `PreTrainedConfig.from_pretrained`:

```
draccus.utils.DecodingError: The fields `mlp_hidden_dim` are not valid for VQBeTConfig
```

So the spec **cannot run today** under the pinned dependency, and per the
explicit-not-silent contract it is left out of `configs/policies.yaml`
rather than added as a non-runnable row. (`mlp_hidden_dim=1024` happens to
match the 0.5.1 default, so the value is benign — only the schema is
stale.)

**v1.1 onboarding (low effort, isolated):** add a config-key allowlist
filter in `eval._load_pretrained_policy` that drops Hub config keys not
present in the target `*Config` dataclass before draccus parse (the same
class of Hub-JSON wiring fix as the xvla loader patches). Then re-lock the
SHA, smoke-load on CPU, and run `vqbet_pusht × pusht` under the N=5×50
contract for the diffusion-vs-VQ-BeT head-to-head. Locked SHA preserved
here so onboarding needs no fresh lock-in:

| Policy      | Repo ID              | Revision SHA (locked 2026-06-08)           | License     |
|-------------|----------------------|--------------------------------------------|-------------|
| vqbet_pusht | `lerobot/vqbet_pusht`| `15c2d0af889c401c7e5db7a07499d3b498afc276` | apache-2.0  |
