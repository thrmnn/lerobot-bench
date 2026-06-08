# Matrix-expansion eval plan (2026-06-08)

Integration-only pass: new policies/envs are **wired into the registry and
load-smoked**, but the GPU eval runs are **gated behind the running Wall
job (#198)** and queue for after it frees. This doc is the prioritized
launch list for the orchestrator.

All new cells are on the **expansion axis** — deliberately **gated off
`V1_POLICIES`** (the frozen v1 leaderboard). They widen the cross-policy
comparison surface; they do not change any published v1 row.

## Seeding / scoring contract (unchanged)

Every cell below runs the standard contract: 5 seeds × ≤50 episodes,
per-cell seeding `seed_idx*1000`, per-episode `env.reset(seed=seed_idx*1000+e)`,
cell-boundary checkpoint flush. Scoring uses the canonical criterion per
env (PushT sticky-is-success, Aloha `reward==4`, LIBERO 600-step cap) when
run with `--canonical`, else v1_legacy.

## What was integrated

| Policy        | Repo ID                                | Locked SHA   | Env(s)                        | Compares against (shared env)                          |
|---------------|----------------------------------------|--------------|-------------------------------|--------------------------------------------------------|
| `act_insertion` | `lerobot/act_aloha_sim_insertion_human` | `33259aa8…` | `aloha_insertion` (NEW env)   | baselines on a second aloha task; generalizes the ACT finding off transfer_cube |
| `pi05_libero` | `lerobot/pi05-libero`                  | `10522ae3…`  | 4 LIBERO suites               | `pi05_libero_finetuned_v044`, `pi0`, `pi0fast`, `smolvla`, `xvla` — second pi05 variant |

New env `aloha_insertion` (`gym_aloha/AlohaInsertion-v0`, max_steps 400,
canonical `reward==4`) extends the `aloha` family to two tasks. Baselines
(`no_op`, `random`) are obs-agnostic and run on it automatically once their
`env_compat` is extended — see "Baseline coverage" below.

CPU load-smoke verified (off-GPU, #198 untouched): `act_insertion`
instantiates a `PI05Policy`-free ACT checkpoint through the pinned lerobot
0.5.1 loader and resets cleanly; `pi05_libero` config resolves to
`PI05Policy` identically to the known-good `pi05_libero_finetuned_v044`.

## Prioritized eval cells (launch after #198)

GPU-hour estimates are grounded in: Hub `eval_info.eval_ep_s`
(act-insertion ≈ 5.15 s/episode at the Hub eval cap) and the DESIGN.md
VLA-latency note (Pi-class ≈ 0.5–2 s/step on the RTX 4060). Each cell =
5 seeds × 50 episodes = 250 episodes unless auto-downscoped.

### Tier 1 — cheap, clean, high-confidence

| # | Cell                                | Episodes | Est. wall-time            | Value |
|---|-------------------------------------|----------|---------------------------|-------|
| 1 | `act_insertion × aloha_insertion`   | 250      | ≈ 0.4–0.7 GPU-h (ACT ≈ 5 s/ep; ACT is fast, ~50 ms/step) | Second aloha task; checks the ACT normalization-fix finding generalizes off transfer_cube. CPU-load-verified. |
| 2 | `no_op × aloha_insertion`           | 250      | ≈ 0.1 GPU-h (CPU-bound)   | Floor on the new env (needed for a meaningful success-rate reference). |
| 3 | `random × aloha_insertion`          | 250      | ≈ 0.1 GPU-h (CPU-bound)   | Random-action reference on the new env. |

**Tier 1 total: ≈ 0.6–0.9 GPU-hours.** Run these first — they need no VRAM
headroom beyond ACT and finish fastest.

### Tier 2 — the cross-checkpoint VLA comparison (the headline value)

`pi05_libero` is a Pi-class VLA: same cold-load host-RAM footprint caveat
as the rest of the Pi family (see `docs/DEFERRED_POLICIES.md` § Pi-family —
~30 GB host RAM during weight-conversion staging, <4 GB steady-state VRAM).
Run on the **capped wrapper** (`scripts/run_capped.sh`, cgroup MemoryMax)
to protect the WSL2 host. Expect the auto-downscope rule to possibly drop
`libero_10` to 25 ep/seed if a cell exceeds the 3-h budget.

| # | Cell                            | Episodes | Est. wall-time              | Value |
|---|---------------------------------|----------|-----------------------------|-------|
| 4 | `pi05_libero × libero_spatial`  | 250      | ≈ 0.7–2.7 GPU-h (280 steps × ~0.5–2 s) | Cross-checkpoint vs `pi05_libero_finetuned_v044` — does a different pi05 finetune run move the suite numbers? |
| 5 | `pi05_libero × libero_object`   | 250      | ≈ 0.7–2.7 GPU-h             | Same, object suite. |
| 6 | `pi05_libero × libero_goal`     | 250      | ≈ 0.8–2.9 GPU-h (300 steps) | Same, goal suite. |
| 7 | `pi05_libero × libero_10`       | 125–250  | ≈ 1.3–5.0 GPU-h (520 steps; may auto-downscope to 125) | Long-horizon — the suite where checkpoints diverge most. |

**Tier 2 total: ≈ 3.5–13 GPU-hours** (wide band driven by the 0.5–2 s/step
VLA-latency uncertainty; tighten after a 20-step calibration spike, per
DESIGN.md Day-0b). Run on `run_capped.sh`; let auto-downscope handle
`libero_10`.

### Calibration first (recommended)

Before Tier 2, run the Day-0b 20-step calibration spike for `pi05_libero`
on one LIBERO suite to pin `mean_ms_per_step` + `vram_peak_mb`, then the
auto-downscope rule in `scripts/run_sweep.py` locks the exact episode
counts. This converts the wide Tier-2 band into a firm number.

## Total estimate

- **Tier 1:** ≈ 0.6–0.9 GPU-hours (run unconditionally; fast, low-risk).
- **Tier 2:** ≈ 3.5–13 GPU-hours (gated on a calibration spike + capped
  runner; auto-downscope may cut the upper end).
- **Combined:** ≈ 4–14 GPU-hours, all gated behind #198.

## Investigated and rejected (honest short list)

| Candidate                        | Why rejected |
|----------------------------------|--------------|
| `lerobot/vqbet_pusht`            | **Highest-value miss.** Second vision policy on shared `pusht` (head-to-head vs diffusion). Config carries `mlp_hidden_dim`, which the pinned 0.5.1 `VQBeTConfig` rejects → draccus `DecodingError`. Genuinely un-runnable today. Deferred with a config-key-allowlist fix; SHA preserved in `docs/DEFERRED_POLICIES.md`. |
| `lerobot/VLA-JEPA-LIBERO`        | `type=vla_jepa` is not in lerobot 0.5.1 `get_policy_class` (supported: tdmpc/diffusion/act/multi_task_dit/vqbet/pi0/pi05/pi0_fast/smolvla/wall_x/sac). Can't instantiate without a newer lerobot — pinned dep forbids the bump. |
| `lerobot/lingbot_va_libero_long` | `type=lingbot_va` not in 0.5.1 `get_policy_class`. Same blocker. |
| `lerobot/diffusion_pusht_keypoints` | `config.json` has empty `input_features` and a pre-0.5 schema that won't even parse (`ParsingError`, missing `type` key); keypoints obs is not in our `pusht` env contract anyway. |
| `lerobot/smolvla_libero_plus`    | Trained on the augmented "libero_plus" suite; viable to load (same smolvla loader) but a *suite-mismatched* checkpoint — its comparison vs the existing `smolvla_libero` on the standard 4 suites is muddier than the clean pi05-vs-pi05 cross-checkpoint comparison. Held as a tier-2 backup, not wired, to keep the integrated list short and high-confidence. |
| `lerobot/pi05_libero_finetuned_quantiles_v044`, `lerobot/pi0fast-libero-v044`, `lerobot/pi0_libero_base`, `lerobot/pi05_libero_base` | Loadable, but they are minor variants of checkpoints already in the registry (the Pi family) — they add little new cross-policy signal beyond what `pi05_libero` already provides, and the Pi family is itself VRAM-gated. Not wired; revisit if the orchestrator wants a finetune-vs-base ablation. |
| `lerobot/act_aloha_sim_insertion_human` ↔ no head-to-head | Accepted anyway (Tier 1) but flagged honestly: `aloha_insertion` has **no second learned policy** — only ACT + baselines. It widens the family / tests generalization, but it is not a cross-policy head-to-head like the LIBERO suites are. |

## Baseline coverage note

`aloha_insertion` needs `no_op` / `random` in its support set for a
meaningful reference. Their `env_compat` currently lists
`aloha_transfer_cube` but not `aloha_insertion`; extend both baselines'
`env_compat` to include `aloha_insertion` **before** launching Tier 1
cells 2–3. (Deferred out of this integration PR only if it would touch
V1_POLICIES-adjacent fixtures — it does not, so it is folded into this
PR; see the policies.yaml diff.)
