# Model cards

One entry per policy in the v1 leaderboard. Filled at Day 0a (repo IDs +
revision SHAs locked) and again at Day 7 (failure-mode column populated
from the failure taxonomy labeling pass).

> Anything not yet locked is marked **TBD**. Do not silently fill —
> if a value is unknown at lockin, that's a sweep blocker per
> `docs/CEO-PLAN.md`.

---

## Diffusion Policy

- **Repo ID**: `lerobot/diffusion_pusht`
- **Revision SHA**: `84a7c23178445c6bbf7e1a884ff497017910f653` (locked 2026-05-03; lastModified 2025-03-06)
- **License**: apache-2.0
- **Envs supported**: PushT (this checkpoint is PushT-trained only; an Aloha variant would be a separate Hub entry not yet listed)
- **Inference precision**: fp32
- **VRAM @ inference (RTX 4060 8GB)**: TBD (Day 0b calibration)
- **Mean ms/step (calibrated)**: TBD
- **Known failure modes**: TBD (Day 7 taxonomy)
- **Source paper**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023

## ACT (Action Chunking Transformer)

- **Repo ID**: `lerobot/act_aloha_sim_transfer_cube_human`
- **Revision SHA**: `ba73b2766f1371cdc133ca4efb97eb090d744625` (locked 2026-05-03; lastModified 2025-03-06)
- **License**: apache-2.0
- **Envs supported**: Aloha transfer-cube (this checkpoint is Aloha-trained only; a PushT variant would be a separate Hub entry not yet listed)
- **Inference precision**: fp32
- **VRAM @ inference**: TBD
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source paper**: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023

## SmolVLA (libero finetune)

- **Repo ID**: `lerobot/smolvla_libero`
- **Revision SHA**: `31d453f7edd78c839a8bbc39744a292686daf0de` (locked 2026-05-03 via `huggingface_hub.HfApi().model_info(repo_id).sha`)
- **License**: apache-2.0
- **Envs supported**: libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: bf16
- **VRAM @ inference (RTX 4060 8GB)**: TBD (Day 0b calibration)
- **Mean ms/step (calibrated)**: ~160 ms/step measured on a smoke run (12.75s wallclock / 79 steps for libero_spatial seed 0). Full per-cell calibration deferred.
- **Known failure modes**: TBD (Day 7 taxonomy)
- **Source**: HuggingFace TB

## Pi0 (libero finetune v0.4.4)

- **Repo ID**: `lerobot/pi0_libero_finetuned_v044`
- **Revision SHA**: `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` (locked 2026-05-03)
- **License**: gemma — review terms before redistribution; the underlying LM is Gemma-licensed.
- **Envs supported**: libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: bf16
- **VRAM @ inference**: ~6 GB target (3B params @ bf16, plus activations + KV cache). **Drop policy if OOM at bf16** — no quantization in v1.
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source**: Physical Intelligence (lerobot mirror)

## Pi0.5 (libero finetune v0.4.4)

- **Repo ID**: `lerobot/pi05_libero_finetuned_v044`
- **Revision SHA**: `dbf8a3f794a9c4297b44f40b752712f50073d945` (locked 2026-05-03)
- **License**: gemma — review terms before redistribution.
- **Envs supported**: libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: bf16
- **VRAM @ inference**: ~6 GB target (Pi0-class). Drop if OOM at bf16.
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source**: Physical Intelligence (lerobot mirror)
- **Notes**: The most-downloaded Pi-class libero finetune on the Hub at lockin (~17.5k DL).

## Pi0Fast (libero)

- **Repo ID**: `lerobot/pi0fast-libero`
- **Revision SHA**: `840f4b503f4c09110421c33c810a85b6684fd658` (locked 2026-05-03)
- **License**: unspecified on Hub card — treat as "all rights reserved" until clarified upstream. Logged as risk for the v1 publish.
- **Envs supported**: libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: bf16
- **VRAM @ inference**: TBD (Pi0Fast claims faster inference than Pi0 baseline)
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source**: Physical Intelligence (lerobot mirror)

## XVLA (libero)

- **Repo ID**: `lerobot/xvla-libero`
- **Revision SHA**: `12e8783e996944f5c97e490d37d4c145484ed70a` (locked 2026-05-03)
- **License**: apache-2.0
- **Envs supported**: libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: bf16
- **VRAM @ inference**: TBD
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source**: lerobot mirror

## no-op (baseline)

- **Repo ID**: n/a — emits zero action every step
- **Revision SHA**: n/a
- **License**: n/a
- **Envs supported**: pusht, aloha_transfer_cube, libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: n/a
- **VRAM @ inference**: 0
- **Mean ms/step**: ~0.1 ms (Python overhead only)
- **Purpose**: floor on the leaderboard. A policy that fails to beat no-op on a given env has not learned that env.

## random (baseline, optional)

- **Repo ID**: n/a — uniform-sampled action per step
- **Revision SHA**: n/a
- **License**: n/a
- **Envs supported**: pusht, aloha_transfer_cube, libero_spatial, libero_object, libero_goal, libero_10
- **Inference precision**: n/a
- **Purpose**: cheap stochastic baseline. Included only if calibration leaves headroom.
