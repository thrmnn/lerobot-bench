"""Re-measure the L2 classical-control rung (``classical_pusht`` on PushT).

Runs the scripted state-feedback controller through the IDENTICAL eval
contract the learned policies use -- same env (``pusht_state``), same
per-cell seeding contract (``env.reset(seed=seed_idx*1000 + e)``), same
canonical ``sticky_is_success`` (coverage > 0.95) success rule -- and
writes:

* ``results/ladder/classical_pusht_l2.parquet`` -- one row per episode in
  the canonical RESULT_SCHEMA (so it slots into the same tooling as every
  other cell).
* ``results/ladder/classical_pusht_l2.summary.json`` -- pooled success,
  Wilson 95% CI, and the per-episode **max-coverage** distribution, which
  makes the *quality* of the classical effort legible: how close a
  competent scripted controller gets to the strict 0.95 bar.

Max-coverage is the max over the rollout of ``info['coverage']`` -- it is
not part of the parquet schema, so it lives only in the summary JSON.

CPU-only, deterministic, no learned weights. Not on the published v1
leaderboard (``classical_pusht`` is gated out of ``V1_POLICIES``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from embodimetry.checkpointing import RESULT_SCHEMA
from embodimetry.policies_classical import _ClassicalPushTPolicy
from embodimetry.stats import wilson_ci

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "ladder"
PARQUET = OUT_DIR / "classical_pusht_l2.parquet"
SUMMARY = OUT_DIR / "classical_pusht_l2.summary.json"

POLICY = "classical_pusht"
ENV = "pusht_state"
GYM_ID = "gym_pusht/PushT-v0"
N_SEEDS = 5
N_EPISODES = 50
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.95  # coverage > 0.95 (sticky_is_success / env's is_success)


def run() -> None:
    # Deferred so importing this module (and ``--help``) never requires the
    # sim deps; fail loud with an actionable message if they are missing.
    try:
        import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
        import gymnasium as gym
    except ImportError as exc:
        sys.exit(
            f"missing sim runtime: {exc}\n"
            "L2 classical control runs on the PushT sim env. Install the sim "
            'extras: pip install -e ".[sim]" (pulls gym-pusht + gymnasium).'
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = gym.make(GYM_ID, obs_type="state")
    rows: list[dict[str, object]] = []
    max_covs: list[float] = []
    ts = datetime.now(UTC).isoformat()

    for seed_idx in range(N_SEEDS):
        for e in range(N_EPISODES):
            policy = _ClassicalPushTPolicy((2,))
            obs, _info = env.reset(seed=seed_idx * 1000 + e)
            policy.reset()
            sticky_success = False
            max_cov = 0.0
            n_steps = 0
            cumulative_return = 0.0
            for _ in range(MAX_STEPS):
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                n_steps += 1
                cumulative_return += float(reward)
                cov = float(info.get("coverage", 0.0))
                max_cov = max(max_cov, cov)
                if bool(info.get("is_success", False)):
                    sticky_success = True
                if terminated or truncated:
                    break
            max_covs.append(max_cov)
            rows.append(
                {
                    "policy": POLICY,
                    "env": ENV,
                    "seed": seed_idx,
                    "episode_index": e,
                    "success": sticky_success,
                    "return_": cumulative_return,
                    "n_steps": n_steps,
                    "wallclock_s": 0.0,
                    "video_sha256": "",
                    "code_sha": "",
                    "lerobot_version": "",
                    "timestamp_utc": ts,
                    "errored": False,
                    "eval_run_id": "ladder-l2-classical",
                }
            )
    env.close()

    df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
    df.to_parquet(PARQUET, index=False)

    n = len(df)
    successes = int(df["success"].sum())
    lo, hi = wilson_ci(successes, n)
    mc = np.array(max_covs, dtype=float)
    # how many episodes land in each max-coverage band (legibility of quality)
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
    hist = np.histogram(mc, bins=edges)[0].tolist()
    summary = {
        "policy": POLICY,
        "env": ENV,
        "rung": "L2_classical_control",
        "n_seeds": N_SEEDS,
        "n_episodes_per_seed": N_EPISODES,
        "n_pooled": n,
        "success_metric": "sticky_is_success",
        "success_threshold": SUCCESS_THRESHOLD,
        "pooled_success": successes / n,
        "successes": successes,
        "wilson_ci_95": [lo, hi],
        "errored_rows": int(df["errored"].sum()),
        "max_coverage": {
            "mean": float(mc.mean()),
            "median": float(np.median(mc)),
            "p90": float(np.percentile(mc, 90)),
            "p99": float(np.percentile(mc, 99)),
            "max": float(mc.max()),
            "min": float(mc.min()),
            "n_at_or_above_0.90": int((mc >= 0.90).sum()),
            "n_at_or_above_0.95": int((mc > 0.95).sum()),
            "histogram_bins": edges,
            "histogram_counts": hist,
        },
        "timestamp_utc": ts,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {PARQUET}")
    print(f"wrote {SUMMARY}")
    print(f"pooled success {successes}/{n} = {successes / n:.4f}  Wilson95% [{lo:.4f}, {hi:.4f}]")
    print(
        f"max-coverage: mean={mc.mean():.3f} median={np.median(mc):.3f} "
        f"p90={np.percentile(mc, 90):.3f} max={mc.max():.3f}"
    )
    print(f"max-coverage histogram {edges}: {hist}")


def main() -> None:
    argparse.ArgumentParser(
        prog="run-ladder-l2-classical",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    run()


if __name__ == "__main__":
    main()
