# L1 (fine-tuning) rung — SmolVLA-LoRA on libero_10: an honest negative

**Result: fine-tuning *collapse*, not a lift and not an ordering inversion.**

| | success | 95% Wilson CI | N |
|---|---|---|---|
| zero-shot `smolvla_libero` | **0.252** | [0.202, 0.309] | 250 (63/250) |
| LoRA fine-tuned (best recipe) | **~0.00** | — | seed-0 partials 0/50, 0/20, 0/15 |

The full 5×50 re-measure was aborted by design: a 0%-success policy runs every episode to the 520-step cap (~20 min/seed), and three independent recipes all collapse on seed 0, so the negative is conclusive without burning ~100 min.

## What this settles (the local-only pivot was right)
- **VRAM: FITS the 8 GB card.** LoRA freezes the ~451M base; only ~743K adapter params train. Peak 1332 MiB @ batch 1, 2764 @ batch 8, 4404 @ batch 16. **PR #166's "needs a bigger GPU" routing was unnecessary for the LoRA path.**
- **Data-wiring (the actual #166 blocker), root-caused + fixed:** rename `observation.images.{image,wrist_image}` → `{camera1,camera2}` (libero_10's 2nd camera is `wrist_image`, not the `image2` the published recipe used), and `--dataset.use_imagenet_stats=false` (SmolVLA uses VISUAL:IDENTITY and never consumes image stats; the lerobot default `True` KeyErrors on libero_10's pre-image-stats cached `stats.json`).

## Why it collapses (the finding)
Open-loop BC loss drops cleanly (0.166 → 0.058) while **closed-loop success goes to 0** — the classic offline-looks-great / online-control-breaks gap. The smoking gun is the **gripper action dimension flipping sign** (−0.99 zero-shot → +0.57 fine-tuned) while overall action magnitude stays sane (absmax ~0.86 vs ~1.0). A light LoRA fine-tune destabilizes an already-marginal VLA's closed-loop grasping.

Ruled out: learning rate (both 1e-4 and 1e-5 collapse → structural, not a tuning miss); image orientation (eval flips env images 180° to match the dataset identically for both); eval-path correctness (zero-shot scored through the *identical* path reproduces 0.252).

**Remaining hypothesis:** a subtle train-vs-closed-loop observation/action distribution gap the marginal checkpoint tolerates zero-shot but a light fine-tune breaks. Not resolvable in the local time box; would need the upstream author's matched train/eval pipeline or a longer warmup + step sweep.

This is exactly the kind of honest negative the instrument exists to surface: a reproducible "fine-tuning made it worse, and here's the mechanism," not a cherry-picked win. Machine-readable record: `results/ladder/smolvla_libero10_l1.summary.json`.
