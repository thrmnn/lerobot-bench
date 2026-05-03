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

## SmolVLA

- **Repo ID**: `HuggingFaceTB/SmolVLA-...` (exact ID locked Day 0a)
- **Revision SHA**: TBD
- **License**: TBD
- **Envs supported**: Aloha, Libero
- **Inference precision**: fp16
- **VRAM @ inference**: TBD
- **Mean ms/step**: TBD (likely 0.5-2 s/step on RTX 4060 — see DESIGN.md § Open Questions Q2)
- **Known failure modes**: TBD
- **Source**: HuggingFace TB

## Pi0

- **Repo ID**: `lerobot/pi0_...` (exact ID locked Day 0a)
- **Revision SHA**: TBD
- **License**: TBD
- **Envs supported**: Aloha, Libero
- **Inference precision**: fp16
- **VRAM @ inference**: ~6 GB target (3B params @ fp16, plus activations + KV cache). **Drop policy if OOM at fp16** — no quantization in v1.
- **Mean ms/step**: TBD
- **Known failure modes**: TBD
- **Source**: Physical Intelligence

## no-op (baseline)

- **Repo ID**: n/a — emits zero action every step
- **Revision SHA**: n/a
- **License**: n/a
- **Envs supported**: PushT, Aloha, Libero
- **Inference precision**: n/a
- **VRAM @ inference**: 0
- **Mean ms/step**: ~0.1 ms (Python overhead only)
- **Purpose**: floor on the leaderboard. A policy that fails to beat no-op on a given env has not learned that env.

## random (baseline, optional)

- **Repo ID**: n/a — uniform-sampled action per step
- **Revision SHA**: n/a
- **License**: n/a
- **Envs supported**: PushT, Aloha, Libero
- **Inference precision**: n/a
- **Purpose**: cheap stochastic baseline. Included only if calibration leaves headroom.
