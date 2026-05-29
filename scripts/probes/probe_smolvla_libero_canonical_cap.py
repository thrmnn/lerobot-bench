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
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    N_EPISODES_PER_SEED,
    SEEDS,
    CellResult,
    run_seeds,
    setup_probe,
    write_summary,
)

logger = logging.getLogger("probe-smolvla-libero-cap600")

PROBE_NAME = "smolvla-libero-10-cap600"
POLICY_NAME = "smolvla_libero"
ENV_NAME = "libero_10"
CANONICAL_MAX_STEPS = 600
V1_BASELINE_RATE = 0.252


def main() -> int:
    ctx = setup_probe(PROBE_NAME, policy_name=POLICY_NAME, env_name=ENV_NAME)
    base_max_steps = ctx.env_spec.max_steps

    env_spec = dataclasses.replace(ctx.env_spec, max_steps=CANONICAL_MAX_STEPS)
    logger.info(
        "PROBE: %s max_steps %d -> %d (canonical LIBERO cap)",
        ENV_NAME,
        base_max_steps,
        env_spec.max_steps,
    )

    cap_hit_counts: dict[int, int] = {}

    def count_cap_hits(seed: int, cell: CellResult) -> None:
        cap_hits = sum(
            1 for e in cell.episodes if not e.success and e.n_steps >= CANONICAL_MAX_STEPS
        )
        cap_hit_counts[seed] = cap_hits
        logger.info(
            "probe seed=%d cap_hits=%d/%d",
            seed,
            cap_hits,
            N_EPISODES_PER_SEED,
        )

    cell_rates = run_seeds(ctx, env_spec=env_spec, on_cell=count_cap_hits)
    overall = sum(cell_rates.values()) / len(cell_rates)
    total_cap_hits = sum(cap_hit_counts.values())
    write_summary(
        ctx,
        {
            "policy": POLICY_NAME,
            "env": ENV_NAME,
            "probe": f"max_steps={CANONICAL_MAX_STEPS}",
            "v1_default_max_steps": base_max_steps,
            "v1_default_rate": V1_BASELINE_RATE,
            "seeds": SEEDS,
            "n_episodes_per_seed": N_EPISODES_PER_SEED,
            "per_seed_success_rate": cell_rates,
            "per_seed_cap_hits": cap_hit_counts,
            "total_cap_hits": total_cap_hits,
            "pooled_success_rate": overall,
        },
    )
    logger.info(
        "PROBE COMPLETE pooled=%.4f (v1 baseline=%.4f, total_cap_hits=%d)",
        overall,
        V1_BASELINE_RATE,
        total_cap_hits,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
