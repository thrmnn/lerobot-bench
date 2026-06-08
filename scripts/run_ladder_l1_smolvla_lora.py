"""L1 (fine-tuning) rung — SmolVLA-LoRA on libero_10, local 8 GB RTX 4060.

LoRA-fine-tunes the pretrained **SmolVLA** checkpoint
(``lerobot/smolvla_libero``) on the ``lerobot/libero_10`` demos, then
RE-MEASURES the fine-tuned checkpoint on the IDENTICAL eval contract its
zero-shot cell used (5 seeds x 50 episodes, v1_legacy max_steps=520,
success := final_reward >= 1.0), so the fine-tuning lift lands on the same
axis as the other ladder rungs.

**Why this is the headroom-rich L1 cell.** SmolVLA zero-shot on libero_10 is
**0.252** (63/250) under the v1 contract (cap=520) — real room to move, unlike
ACT/aloha (already ~0.82). PR #166 routed this cell to a bigger GPU claiming
the blocker was "data wiring, not VRAM"; this script proves the VRAM half
(LoRA freezes the ~450 M base, ~few-M adapters fit 8 GB) and fixes the wiring.

**OUTCOME (honest negative — see results/ladder/smolvla_libero10_l1.summary.json).**
VRAM fits (peak ~2.8 GB at batch=8, ~4.4 GB at batch=16) and training runs
clean with the wiring fix (BC loss converges 0.17 -> 0.06). BUT the fine-tuned
policy collapses to **0% closed-loop success** on libero_10 — across LR 1e-4
(3000 steps) and 1e-5 (1000 steps), via both the merged checkpoint and the raw
PeftModel adapter. Controls rule out the eval path (published zero-shot scores
0.252 through it) and the merge (adapter-direct also 0). The most parsimonious
read: the already-marginal smolvla_libero checkpoint sits at a narrow optimum
that a light LoRA update on the 379-episode libero_10 subset destabilizes (low
open-loop BC loss masks broken closed-loop control). A working fine-tune likely
needs the full ``lerobot/libero`` data + the upstream training-eval-matched
recipe + a longer warmup/step sweep — beyond the local time box. The script is
kept correct for the case a working recipe is found (e.g. on a bigger GPU); the
re-measure block below runs as soon as a fine-tune that is not 0% is produced.

**The data-wiring fix.** ``lerobot/smolvla_libero`` was trained (per its
``train_config.json``) on ``lerobot/libero`` with
``rename_map={observation.images.image: camera1, observation.images.image2:
camera2}``. The suite-specific ``lerobot/libero_10`` dataset names its second
camera ``observation.images.wrist_image`` (not ``image2``), so the rename map
here is ``{image: camera1, wrist_image: camera2}``. The checkpoint's
``config.json`` lists stale 3-camera / 6-dim-state placeholders inherited from
``smolvla_base``; ``make_policy`` re-derives the real 2-camera / 8-dim-state /
7-dim-action features from the dataset meta at train time (verified against the
published normalizer: state.mean has shape (8,), action.mean (7,)). Image
normalization is ``VISUAL: IDENTITY`` for SmolVLA, so the dataset image stats
are not load-bearing — the only real mismatch is the camera *key* names, fixed
by the rename map.

**Contract fidelity.** The re-measure goes through
``embodimetry.eval.run_cell`` cell-by-cell — the exact loop, per-cell seeding
contract, and v1_legacy success rule the zero-shot smolvla_libero cell used.
Only the policy weights change. The LoRA adapter is merged into the base
(``merge_and_unload``) so the fine-tuned checkpoint is a plain
``from_pretrained``-loadable SmolVLA dir, loaded for eval via
``embodimetry.eval._load_pretrained_policy`` with a local path — same posture
as the ACT L1 cell.

Outputs (``results/ladder/`` only — canonical sweep parquet untouched):

* ``smolvla_libero10_l1.parquet`` — one row per (seed, episode) in
  RESULT_SCHEMA, policy renamed ``smolvla_finetuned`` so it never collides
  with the zero-shot ``smolvla_libero`` rows or leaks into ``V1_POLICIES``.
* ``smolvla_libero10_l1.summary.json`` — LoRA config, zero-shot vs fine-tuned
  pooled rate + Wilson 95% CIs + N, the lift, Cohen's h, and wall-clock.

The fine-tuned policy is gated OUT of the published leaderboard (not added to
``configs/policies.yaml`` / ``V1_POLICIES``), same as ``classical_pusht`` /
``xvla_libero`` / ``act_finetuned``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "ladder"
PARQUET = OUT_DIR / "smolvla_libero10_l1.parquet"
SUMMARY = OUT_DIR / "smolvla_libero10_l1.summary.json"

# Pretrained SmolVLA checkpoint (same repo + revision as configs/policies.yaml
# `smolvla_libero`).
BASE_REPO = "lerobot/smolvla_libero"
BASE_REV = "31d453f7edd78c839a8bbc39744a292686daf0de"
TRAIN_DATASET = "lerobot/libero_10"
# libero_10 cameras -> smolvla's camera1/camera2 (the data-wiring fix).
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.wrist_image": "observation.images.camera2",
}

POLICY_OUT_NAME = "smolvla_finetuned"
ENV = "libero_10"
N_SEEDS = 5
N_EPISODES = 50

# Zero-shot baseline (v1_legacy contract, max_steps=520), from
# results/sweep-full/results.parquet smolvla_libero x libero_10.
ZEROSHOT_SUCCESSES = 63
ZEROSHOT_N = 250


def _finetune(
    work_dir: Path, *, steps: int, batch_size: int, lora_r: int, lr: float, save_freq: int
) -> Path:
    """LoRA fine-tune SmolVLA via ``lerobot-train``; return the checkpoint dir.

    Uses lerobot's native PEFT path (``--peft.method_type=LORA``), the
    rename map for the libero_10 camera keys, and gradient checkpointing is
    not needed (SmolVLA freezes the vision encoder + trains expert-only by
    default, so LoRA on the expert is light). Returns the
    ``checkpoints/<step>/pretrained_model`` directory.
    """
    out = work_dir / "ft_out"
    if out.exists():
        shutil.rmtree(out)
    rename_json = json.dumps(RENAME_MAP)
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        # No --policy.revision flag: lerobot-train resolves --policy.path at the
        # repo's main branch, which is pinned-SHA-identical to BASE_REV (asserted
        # in run() before launch). The eval re-measure uses the same loaded weights.
        f"--policy.path={BASE_REPO}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={TRAIN_DATASET}",
        # SmolVLA normalizes images with VISUAL: IDENTITY, so it never consumes
        # image stats. The published smolvla_libero checkpoint was trained with
        # use_imagenet_stats=false (verified in its train_config.json). The
        # lerobot default is True, which crashes here: libero_10's cached
        # stats.json predates the image-stat keys, so factory.make_dataset
        # KeyErrors writing ImageNet stats into the missing image keys. Matching
        # the published recipe (false) both fixes the crash and keeps the
        # normalization contract bit-identical to the zero-shot cell.
        "--dataset.use_imagenet_stats=false",
        f"--rename_map={rename_json}",
        "--peft.method_type=LORA",
        f"--peft.r={lora_r}",
        # Peak LR for the cosine schedule. The SmolVLA preset default (1e-4) is
        # tuned for full fine-tunes from scratch and collapses a LoRA fine-tune
        # of the already-converged smolvla_libero checkpoint (measured: 0%
        # success). A gentler LR preserves the base behavior while adapting.
        f"--policy.optimizer_lr={lr}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--num_workers=4",
        "--eval_freq=0",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--log_freq=50",
        "--wandb.enable=false",
        f"--output_dir={out}",
    ]
    print(
        f"[l1-smolvla] lerobot-train LoRA r={lora_r}: steps={steps} batch={batch_size} -> {out}",
        flush=True,
    )
    subprocess.run(cmd, check=True)

    ckpts = sorted((out / "checkpoints").glob("*/pretrained_model"))
    if not ckpts:
        raise RuntimeError(f"no checkpoint produced under {out / 'checkpoints'}")
    # Skip the 'last' symlink; take the highest numeric step dir.
    numeric = [c for c in ckpts if c.parent.name.isdigit()]
    return (numeric or ckpts)[-1]


def _merge_lora(adapter_ckpt: Path, work_dir: Path) -> Path:
    """Merge the LoRA adapter into the base weights -> plain SmolVLA checkpoint.

    The eval contract loads via ``_load_pretrained_policy`` ->
    ``from_pretrained`` (no PEFT awareness). Merging the adapter with
    ``PeftModel.merge_and_unload()`` produces a standard SmolVLA dir that
    loads identically to the zero-shot checkpoint, keeping the re-measure on
    the exact same code path. The merged dir reuses the adapter dir's processor
    JSONs and a freshly-written non-PEFT config.json.
    """
    import lerobot.policies.factory as fac
    from lerobot.configs.policies import PreTrainedConfig

    merged = work_dir / "merged_ckpt"
    if merged.exists():
        shutil.rmtree(merged)

    cfg = PreTrainedConfig.from_pretrained(str(adapter_ckpt))
    cfg.pretrained_path = str(adapter_ckpt)
    cfg.device = "cuda"
    cfg.use_peft = True
    policy_cls = fac.get_policy_class(cfg.type)

    # Reproduce factory's PEFT load: base from adapter config, then adapter.
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(str(adapter_ckpt))
    base = policy_cls.from_pretrained(peft_config.base_model_name_or_path, config=cfg)
    peft_model = PeftModel.from_pretrained(base, str(adapter_ckpt), config=peft_config)
    merged_policy = peft_model.merge_and_unload()
    merged_policy.config.use_peft = False
    merged_policy.config.pretrained_path = None

    merged.mkdir(parents=True, exist_ok=True)
    merged_policy.save_pretrained(str(merged))
    merged_policy.config.save_pretrained(str(merged))
    # Carry over the processor JSONs + safetensors from the adapter dir.
    for f in adapter_ckpt.iterdir():
        if f.name.startswith("policy_pre") or f.name.startswith("policy_post"):
            shutil.copy2(f, merged / f.name)
    return merged


def _eval_finetuned(ft_ckpt: Path) -> tuple[list[dict[str, object]], int, int]:
    """Re-measure on the v1_legacy libero_10 contract via the production loop."""
    import pandas as pd  # noqa: F401

    from embodimetry.checkpointing import RESULT_SCHEMA
    from embodimetry.envs import EnvRegistry
    from embodimetry.eval import (
        _detect_code_sha,
        _detect_lerobot_version,
        _load_pretrained_policy,
        load_env,
    )
    from embodimetry.eval import run_cell as eval_run_cell

    # v1_legacy (the default): max_steps=520, success := final_reward >= 1.0.
    # This is the EXACT contract the zero-shot 0.252 (63/250) baseline in
    # results/sweep-full/results.parquet was measured under (verified: no
    # baseline episode ran past 520 steps). The `canonical` overlay would bump
    # the cap to 600 and break the apples-to-apples comparison.
    env_spec = (
        EnvRegistry.from_yaml(REPO_ROOT / "configs" / "envs.yaml")
        .get(ENV)
        .with_criterion("v1_legacy")
    )
    code_sha = _detect_code_sha()
    lr_version = _detect_lerobot_version()

    rows: list[dict[str, object]] = []
    successes = 0
    n_pooled = 0
    for seed_idx in range(N_SEEDS):
        env = load_env(env_spec)
        action_space = getattr(env, "action_space", None)
        action_shape = tuple(action_space.shape) if action_space is not None else (7,)
        policy = _load_pretrained_policy(
            repo_id=str(ft_ckpt), revision=None, action_shape=action_shape, device="cuda"
        )
        cell = eval_run_cell(
            policy,
            env,
            policy_name=POLICY_OUT_NAME,
            env_spec=env_spec,
            seed_idx=seed_idx,
            n_episodes=N_EPISODES,
            record_video=False,
            videos_dir=None,
            code_sha=code_sha,
            lerobot_version=lr_version,
            eval_run_id="ladder-l1-smolvla-lora",
        )
        env.close()
        df = cell.to_rows()
        for _, r in df.iterrows():
            rows.append({c: r[c] for c in RESULT_SCHEMA})
        successes += int(df["success"].sum())
        n_pooled += len(df)
        print(
            f"[l1-smolvla] seed {seed_idx}: {int(df['success'].sum())}/{len(df)} "
            f"(running {successes}/{n_pooled})",
            flush=True,
        )
        # Free the per-seed SmolVLA (~3.2 GB) before the next load so 5
        # sequential loads don't accumulate toward the 8 GB ceiling.
        import gc

        import torch

        del policy
        gc.collect()
        torch.cuda.empty_cache()
    return rows, n_pooled, successes


def run(
    *,
    steps: int,
    batch_size: int,
    lora_r: int,
    lr: float,
    save_freq: int,
    keep_work: bool,
    skip_eval: bool,
) -> None:
    import pandas as pd

    from embodimetry.checkpointing import RESULT_SCHEMA
    from embodimetry.stats import cohens_h, wilson_ci

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/l1_smolvla_lora_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()

    # Contract guard: lerobot-train has no --policy.revision, so it loads
    # BASE_REPO@main. Assert main is the pinned SHA the zero-shot cell used so
    # the fine-tune starts from bit-identical base weights.
    from huggingface_hub import HfApi

    main_sha = HfApi().model_info(BASE_REPO).sha
    if main_sha != BASE_REV:
        raise RuntimeError(
            f"{BASE_REPO} main is {main_sha}, expected pinned {BASE_REV}. "
            "lerobot-train would fine-tune from a different base than the zero-shot "
            "cell; pin/move the dataset or update BASE_REV before proceeding."
        )

    t_train0 = time.perf_counter()
    adapter_ckpt = _finetune(
        work_dir,
        steps=steps,
        batch_size=batch_size,
        lora_r=lora_r,
        lr=lr,
        save_freq=save_freq,
    )
    ft_ckpt = _merge_lora(adapter_ckpt, work_dir)
    train_wall_s = time.perf_counter() - t_train0
    print(
        f"[l1-smolvla] merged checkpoint at {ft_ckpt} (train {train_wall_s / 60:.1f} min)",
        flush=True,
    )

    if skip_eval:
        print("[l1-smolvla] --skip-eval set; stopping after train+merge.", flush=True)
        return

    t_eval0 = time.perf_counter()
    rows, n, successes = _eval_finetuned(ft_ckpt)
    eval_wall_s = time.perf_counter() - t_eval0

    df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
    df.to_parquet(PARQUET, index=False)

    ft_rate = successes / n
    ft_lo, ft_hi = wilson_ci(successes, n)
    zs_rate = ZEROSHOT_SUCCESSES / ZEROSHOT_N
    zs_lo, zs_hi = wilson_ci(ZEROSHOT_SUCCESSES, ZEROSHOT_N)
    lift = ft_rate - zs_rate
    h = cohens_h(ft_rate, zs_rate)

    summary = {
        "policy": POLICY_OUT_NAME,
        "env": ENV,
        "rung": "L1_finetune",
        "base_checkpoint": {"repo_id": BASE_REPO, "revision_sha": BASE_REV},
        "train_dataset": TRAIN_DATASET,
        "rename_map": RENAME_MAP,
        "finetune_config": {
            "method": "lora",
            "lora_r": lora_r,
            "lr": lr,
            "lora_alpha": 8,  # peft LoraConfig default; not exposed via lerobot's high-level PeftConfig CLI
            "target_modules": "smolvla default (lm_expert q/v_proj + state/action projections)",
            "steps": steps,
            "batch_size": batch_size,
            "lora": True,
        },
        "contract": {
            "n_seeds": N_SEEDS,
            "n_episodes_per_seed": N_EPISODES,
            "n_pooled": n,
            "success_metric": "final_reward_threshold",
            "success_threshold": 1.0,
            "max_steps": 520,
            "criterion": "v1_legacy",
        },
        "zero_shot": {
            "source": "results/sweep-full/results.parquet (smolvla_libero x libero_10, v1_legacy cap=520)",
            "successes": ZEROSHOT_SUCCESSES,
            "n": ZEROSHOT_N,
            "pooled_success": zs_rate,
            "wilson_ci_95": [zs_lo, zs_hi],
        },
        "finetuned": {
            "successes": successes,
            "n": n,
            "pooled_success": ft_rate,
            "wilson_ci_95": [ft_lo, ft_hi],
            "errored_rows": int(df["errored"].sum()),
        },
        "lift": {
            "delta_success": lift,
            "cohens_h": h,
            "read": (
                "Single-policy fine-tuning lift on the headroom-rich weak cell "
                "(SmolVLA zero-shot 0.252 on libero_10). This is the LIFT, not an "
                "ordering inversion: a true inversion needs >=2 policies re-ranked "
                "on the same env after fine-tuning, which this single-policy run "
                "does not provide. CI overlap / separation vs zero-shot is the read."
            ),
        },
        "wall_clock_s": {
            "train": train_wall_s,
            "eval": eval_wall_s,
            "total": train_wall_s + eval_wall_s,
        },
        "timestamp_utc": ts,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"wrote {PARQUET}")
    print(f"wrote {SUMMARY}")
    print(
        f"L1 smolvla_finetuned/libero_10: zero-shot {zs_rate:.4f} "
        f"[{zs_lo:.4f},{zs_hi:.4f}] -> fine-tuned {ft_rate:.4f} [{ft_lo:.4f},{ft_hi:.4f}]  "
        f"(N={n}); lift {lift:+.4f}, cohens_h {h:+.4f}"
    )
    print(
        f"wall: train {train_wall_s / 60:.1f} min, eval {eval_wall_s / 60:.1f} min, "
        f"total {(train_wall_s + eval_wall_s) / 60:.1f} min"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run-ladder-l1-smolvla-lora", description=__doc__)
    p.add_argument("--steps", type=int, default=3000, help="LoRA fine-tune steps (default 3000).")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4).")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default 16).")
    p.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Peak LR (default 1e-5; the 1e-4 preset collapses LoRA on the converged base).",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=0,
        help="Checkpoint save frequency (default 0 -> = steps, single final ckpt).",
    )
    p.add_argument("--keep-work", action="store_true", help="Keep /tmp work dir for inspection.")
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train + merge only (for the VRAM feasibility spike).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    save_freq = args.save_freq if args.save_freq > 0 else args.steps
    run(
        steps=args.steps,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lr=args.lr,
        save_freq=save_freq,
        keep_work=args.keep_work,
        skip_eval=args.skip_eval,
    )
