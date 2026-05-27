#!/usr/bin/env python3
"""v1.0.1 audit probe (task #122): re-run smolvla x libero_10 with the
canonical LIBERO max_steps=600 instead of our v1 default of 520.

PR #89 identified that 74.8% of failed libero_10 episodes hit our 520-step
cap. The canonical Liu et al. (2023) LIBERO benchmark uses 600 steps for
every suite. This probe puts a number on the lower-bound claim in the v1.0.1
audit: if pooled success on libero_10 jumps from 0.252 at cap=520 to
something materially higher at cap=600, the "lower bound" framing in the
restated README/MODEL_CARDS is empirically validated.

Output: results/probes/smolvla-libero-10-cap600/results.parquet (RESULT_SCHEMA),
plus summary.json for the deck/audit doc.

The override is a 1-line dataclasses.replace on the env spec — no source
mutation. The eval pipeline's gymnasium.make() call uses spec.max_steps as
max_episode_steps, so bumping it propagates cleanly.

One-off probe — promote env_overrides into the canonical run_one CLI in a
follow-up PR (PR #90 already adds --canonical for the v1.1 sweep path).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_bench.checkpointing import append_cell_rows  # noqa: E402
from lerobot_bench.envs import EnvRegistry  # noqa: E402
from lerobot_bench.policies import PolicyRegistry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("probe-smolvla-libero-cap600")

POLICY_NAME = "smolvla_libero"
ENV_NAME = "libero_10"
SEEDS = (0, 1, 2, 3, 4)
N_EPISODES_PER_SEED = 50
CANONICAL_MAX_STEPS = 600
OUT_DIR = REPO_ROOT / "results" / "probes" / "smolvla-libero-10-cap600"
OUT_PARQUET = OUT_DIR / "results.parquet"
VIDEOS_DIR = OUT_DIR / "videos"
SUMMARY_JSON = OUT_DIR / "summary.json"
V1_BASELINE_RATE = 0.252


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PARQUET.exists():
        OUT_PARQUET.unlink()
        logger.info("removed stale parquet %s", OUT_PARQUET)

    policy_reg = PolicyRegistry.from_yaml(REPO_ROOT / "configs" / "policies.yaml")
    env_reg = EnvRegistry.from_yaml(REPO_ROOT / "configs" / "envs.yaml")
    policy_spec = policy_reg.get(POLICY_NAME)
    base_env_spec = env_reg.get(ENV_NAME)

    env_spec = dataclasses.replace(base_env_spec, max_steps=CANONICAL_MAX_STEPS)
    logger.info(
        "PROBE: %s max_steps %d -> %d (canonical LIBERO cap)",
        ENV_NAME,
        base_env_spec.max_steps,
        env_spec.max_steps,
    )

    from lerobot_bench.eval import run_cell_from_specs

    cell_rates: dict[int, float] = {}
    cap_hit_counts: dict[int, int] = {}
    for seed in SEEDS:
        logger.info("probe seed=%d starting", seed)
        cell = run_cell_from_specs(
            policy_spec,
            env_spec,
            seed_idx=seed,
            n_episodes=N_EPISODES_PER_SEED,
            device="cuda",
            record_video=True,
            videos_dir=VIDEOS_DIR,
        )
        cell_rates[seed] = cell.success_rate
        cap_hits = sum(
            1 for e in cell.episodes if not e.success and e.n_steps >= CANONICAL_MAX_STEPS
        )
        cap_hit_counts[seed] = cap_hits
        append_cell_rows(OUT_PARQUET, cell.to_rows())
        logger.info(
            "probe seed=%d success_rate=%.4f cap_hits=%d/%d",
            seed,
            cell.success_rate,
            cap_hits,
            N_EPISODES_PER_SEED,
        )

    overall = sum(cell_rates.values()) / len(cell_rates)
    total_cap_hits = sum(cap_hit_counts.values())
    summary = {
        "policy": POLICY_NAME,
        "env": ENV_NAME,
        "probe": f"max_steps={CANONICAL_MAX_STEPS}",
        "v1_default_max_steps": base_env_spec.max_steps,
        "v1_default_rate": V1_BASELINE_RATE,
        "seeds": SEEDS,
        "n_episodes_per_seed": N_EPISODES_PER_SEED,
        "per_seed_success_rate": cell_rates,
        "per_seed_cap_hits": cap_hit_counts,
        "total_cap_hits": total_cap_hits,
        "pooled_success_rate": overall,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    logger.info(
        "PROBE COMPLETE pooled=%.4f (v1 baseline=%.4f, total_cap_hits=%d)",
        overall,
        V1_BASELINE_RATE,
        total_cap_hits,
    )
    logger.info("summary -> %s", SUMMARY_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
