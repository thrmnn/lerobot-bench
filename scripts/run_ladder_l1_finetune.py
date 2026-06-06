"""L1 (fine-tuning) rung of the Capability-Ladder Audit.

Fine-tune a *pretrained* policy on a small demo budget, then RE-MEASURE the
fine-tuned checkpoint on the IDENTICAL eval contract its zero-shot cell used,
to quantify the fine-tuning lift on the same axis as the other rungs (L0
zero-shot, L2 classical control).

**Why ACT on aloha_transfer_cube.** The feasibility spike (see the L1 PR
body) ruled the 8 GB RTX 4060 budget: full-finetune of ACT (~50 M params,
ResNet18 backbone) peaks at ~2.8 GB VRAM at batch 8 and trains at ~0.16
s/step — trivially feasible. SmolVLA (~450 M) full-FT does not fit 8 GB and
its LoRA path on `lerobot/libero_10` is blocked by a dataset-stats / camera-key
mismatch (the dataset ships no image normalization stats and a 2-camera vs
3-camera layout), so SmolVLA L1 is routed to a bigger GPU. ACT is the one
turnkey, contract-matching, VRAM-feasible local L1 cell.

**Contract fidelity.** The re-measure goes through ``embodimetry.eval.run_cell``
— the exact loop, per-cell seeding contract, and canonical
``sticky_reward_eq`` (reward == 4.0) success rule the zero-shot ACT cell used
(5 seeds x 50 episodes, max_steps=400). The only thing that changes between
the zero-shot and fine-tuned cells is the policy weights.

**Honesty note on the baseline.** ACT zero-shot on aloha_transfer_cube under
the canonical contract is **0.824** (206/250) — the corrected re-run
(``results/sweep-full/results-act-rerun.parquet``), NOT the 0.016 the buggy
canonical ``results.parquet`` ships (a normalization-key drop). We compare the
fine-tuned rate against 0.824. Continued fine-tuning of an already-converged
policy on its *own* training data is expected to yield a small / near-zero
lift; that is itself a valid, legible L1 result and is reported as such.

Outputs (``results/ladder/`` only — canonical sweep parquet is never touched):

* ``act_aloha_l1.parquet`` — one row per (seed, episode) in the canonical
  RESULT_SCHEMA, policy renamed ``act_finetuned`` so it never collides with
  the zero-shot ``act`` rows or leaks into ``V1_POLICIES``.
* ``act_aloha_l1.summary.json`` — fine-tune config (steps, demos, batch),
  zero-shot vs fine-tuned pooled rate + Wilson 95% CIs + N, the lift, Cohen's
  h, and wall-clock.

The fine-tuned checkpoint is a plain ``from_pretrained``-loadable ACT
checkpoint, loaded for eval via :func:`embodimetry.eval._load_pretrained_policy`
with a local path (revision=None). It is gated OUT of the published leaderboard
(not added to ``configs/policies.yaml`` / ``V1_POLICIES``), same posture as
``classical_pusht`` and ``xvla_libero``.
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

# Headless MuJoCo: gym-aloha renders the top camera inside get_observation on
# every reset/step, so a GL backend is required even with record_video=False.
# egl is the canonical bench setting (glfw segfaults on WSL2); osmesa is the
# CPU fallback. setdefault so an operator override wins.
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "ladder"
PARQUET = OUT_DIR / "act_aloha_l1.parquet"
SUMMARY = OUT_DIR / "act_aloha_l1.summary.json"

# Pretrained ACT checkpoint (same repo + revision as configs/policies.yaml `act`).
BASE_REPO = "lerobot/act_aloha_sim_transfer_cube_human"
BASE_REV = "ba73b2766f1371cdc133ca4efb97eb090d744625"
TRAIN_DATASET = "lerobot/aloha_sim_transfer_cube_human"

# Re-measure contract — IDENTICAL to the zero-shot `act` / `aloha_transfer_cube`
# canonical cell.
POLICY_OUT_NAME = "act_finetuned"
ENV = "aloha_transfer_cube"
N_SEEDS = 5
N_EPISODES = 50

# Genuine corrected zero-shot baseline (canonical contract), from
# results/sweep-full/results-act-rerun.parquet. NOT the buggy 0.016.
ZEROSHOT_SUCCESSES = 206
ZEROSHOT_N = 250


def _materialize_base_checkpoint(work_dir: Path) -> Path:
    """Download the legacy ACT checkpoint locally and write the processor JSONs.

    The Hub checkpoint predates lerobot's processor-pipeline split, so it has
    no ``policy_preprocessor.json`` — ``lerobot-train --policy.path=<hub repo>``
    404s on it. We recover the normalization stats from the legacy safetensors
    buffers (the same path :mod:`embodimetry.eval` uses for zero-shot eval) and
    save proper processor JSONs into a local copy, so the trainer can resume
    from it cleanly.
    """
    import lerobot.policies.factory as fac
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig

    from embodimetry.eval import _recover_dataset_stats_from_safetensors

    src = snapshot_download(BASE_REPO, revision=BASE_REV)
    local = work_dir / "base_ckpt"
    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(src, local)

    cfg = PreTrainedConfig.from_pretrained(str(local))
    feature_keys = (*cfg.input_features.keys(), *cfg.output_features.keys())
    stats = _recover_dataset_stats_from_safetensors(BASE_REPO, BASE_REV, feature_keys=feature_keys)
    pre, post = fac.make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(str(local))
    post.save_pretrained(str(local))
    return local


def _finetune(base_ckpt: Path, work_dir: Path, *, steps: int, batch_size: int) -> Path:
    """Run ``lerobot-train`` from the base checkpoint; return the fine-tuned dir.

    Saves a single checkpoint at the final step. Returns the
    ``checkpoints/<step>/pretrained_model`` directory, which is a plain
    ``from_pretrained``-loadable ACT checkpoint (config + weights + processors).
    """
    out = work_dir / "ft_out"
    if out.exists():
        shutil.rmtree(out)
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={base_ckpt}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={TRAIN_DATASET}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--num_workers=4",
        "--eval_freq=0",
        f"--save_freq={steps}",
        "--save_checkpoint=true",
        "--log_freq=100",
        "--wandb.enable=false",
        f"--output_dir={out}",
    ]
    print(f"[l1] lerobot-train: steps={steps} batch={batch_size} -> {out}", flush=True)
    subprocess.run(cmd, check=True)

    step_id = f"{steps:06d}"
    ft = out / "checkpoints" / step_id / "pretrained_model"
    if not ft.is_dir():
        # Fall back to whatever single checkpoint was saved.
        ckpts = sorted((out / "checkpoints").glob("*/pretrained_model"))
        if not ckpts:
            raise RuntimeError(f"no checkpoint produced under {out / 'checkpoints'}")
        ft = ckpts[-1]
    return ft


def _eval_finetuned(ft_ckpt: Path) -> tuple[list[dict[str, object]], int, int]:
    """Re-measure the fine-tuned checkpoint on the canonical aloha contract.

    Goes through the production eval loop (``run_cell``) cell-by-cell so the
    seeding contract and canonical ``sticky_reward_eq`` rule are bit-identical
    to the zero-shot cell. Returns (rows, n_pooled, successes).
    """
    import pandas as pd  # noqa: F401  (RESULT_SCHEMA columns)

    from embodimetry.checkpointing import RESULT_SCHEMA
    from embodimetry.envs import EnvRegistry
    from embodimetry.eval import (
        _detect_code_sha,
        _detect_lerobot_version,
        _load_pretrained_policy,
        load_env,
    )
    from embodimetry.eval import run_cell as eval_run_cell

    env_spec = (
        EnvRegistry.from_yaml(REPO_ROOT / "configs" / "envs.yaml")
        .get(ENV)
        .with_criterion("canonical")
    )
    code_sha = _detect_code_sha()
    lr_version = _detect_lerobot_version()

    rows: list[dict[str, object]] = []
    successes = 0
    n_pooled = 0
    for seed_idx in range(N_SEEDS):
        env = load_env(env_spec)
        action_space = getattr(env, "action_space", None)
        action_shape = tuple(action_space.shape) if action_space is not None else (14,)
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
            eval_run_id="ladder-l1-finetune",
        )
        env.close()
        df = cell.to_rows()
        for _, r in df.iterrows():
            rows.append({c: r[c] for c in RESULT_SCHEMA})
        successes += int(df["success"].sum())
        n_pooled += len(df)
        print(
            f"[l1] seed {seed_idx}: {int(df['success'].sum())}/{len(df)} "
            f"(running {successes}/{n_pooled})",
            flush=True,
        )
    return rows, n_pooled, successes


def run(*, steps: int, batch_size: int, keep_work: bool) -> None:
    import pandas as pd

    from embodimetry.checkpointing import RESULT_SCHEMA
    from embodimetry.stats import cohens_h, wilson_ci

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/l1_finetune_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()

    t_train0 = time.perf_counter()
    base_ckpt = _materialize_base_checkpoint(work_dir)
    ft_ckpt = _finetune(base_ckpt, work_dir, steps=steps, batch_size=batch_size)
    train_wall_s = time.perf_counter() - t_train0

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
        "finetune_config": {
            "method": "full_finetune",
            "steps": steps,
            "batch_size": batch_size,
            "n_demo_episodes": 50,
            "lora": False,
        },
        "contract": {
            "n_seeds": N_SEEDS,
            "n_episodes_per_seed": N_EPISODES,
            "n_pooled": n,
            "success_metric": "sticky_reward_eq",
            "strict_reward_value": 4.0,
            "max_steps": 400,
            "criterion": "canonical",
        },
        "zero_shot": {
            "source": "results/sweep-full/results-act-rerun.parquet (corrected canonical re-run)",
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
            "note": (
                "fine-tuned minus corrected zero-shot. ACT is already converged "
                "on its own training data, so a near-zero lift is the expected, "
                "honest result for this cell."
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
        f"L1 act_finetuned/aloha_transfer_cube: zero-shot {zs_rate:.4f} "
        f"[{zs_lo:.4f},{zs_hi:.4f}] -> fine-tuned {ft_rate:.4f} [{ft_lo:.4f},{ft_hi:.4f}]  "
        f"(N={n}); lift {lift:+.4f}, cohens_h {h:+.4f}"
    )
    print(
        f"wall: train {train_wall_s / 60:.1f} min, eval {eval_wall_s / 60:.1f} min, "
        f"total {(train_wall_s + eval_wall_s) / 60:.1f} min"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run-ladder-l1-finetune", description=__doc__)
    p.add_argument("--steps", type=int, default=6000, help="Fine-tune steps (default 6000).")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8).")
    p.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the /tmp work dir (base + fine-tuned checkpoints) for inspection.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(steps=args.steps, batch_size=args.batch_size, keep_work=args.keep_work)
