#!/usr/bin/env python3
"""v1.0.1 audit probe (task #121): re-run act × aloha_transfer_cube with the
paper's temporal-ensembling inference settings (temporal_ensemble_coeff=0.01,
n_action_steps=1) instead of the Hub default (coeff=None, n_action_steps=100).

PR #86 identified this as the most plausible explanation for our v1 measurement
of 0.016 on this cell. This probe puts a number on it. If the result jumps
materially (e.g. > 0.15), the v1.0.2 framing for ACT should mirror what the
README/MODEL_CARDS now say for SmolVLA: the cell measurement was real but the
inference settings were apples-to-oranges with the paper.

Output: results/probes/act-aloha-temporal-ensemble/results.parquet (RESULT_SCHEMA),
plus a one-line summary JSON for the deck/audit doc.

This is a one-off probe script, not a permanent feature. If the result is
interesting, promote the override mechanism to PolicySpec.policy_overrides
and add an entry to configs/policies.yaml in a follow-up PR.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    N_EPISODES_PER_SEED,
    SEEDS,
    run_seeds,
    setup_probe,
    write_summary,
)

logger = logging.getLogger("probe-act-temporal-ensemble")

PROBE_NAME = "act-aloha-temporal-ensemble"
POLICY_NAME = "act"
ENV_NAME = "aloha_transfer_cube"
V1_BASELINE_RATE = 0.016


def _patch_act_inference_settings() -> None:
    """Monkey-patch lerobot's PreTrainedConfig.from_pretrained to set
    temporal_ensemble_coeff=0.01 and n_action_steps=1 on ACT configs only.

    Patched in-place so the rest of the bench pipeline (load_policy ->
    _load_pretrained_policy -> policy_cls.from_pretrained(config=cfg))
    picks up the override transparently. Non-ACT configs pass through.
    """
    from lerobot.configs.policies import PreTrainedConfig

    original = PreTrainedConfig.from_pretrained

    def patched(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = original(*args, **kwargs)
        if cfg.type == "act":
            old_coeff = getattr(cfg, "temporal_ensemble_coeff", None)
            old_nstep = getattr(cfg, "n_action_steps", None)
            cfg.temporal_ensemble_coeff = 0.01
            cfg.n_action_steps = 1
            logger.info(
                "PROBE: patched ACT cfg temporal_ensemble_coeff %r -> 0.01, n_action_steps %r -> 1",
                old_coeff,
                old_nstep,
            )
        return cfg

    PreTrainedConfig.from_pretrained = classmethod(patched)


def main() -> int:
    ctx = setup_probe(PROBE_NAME, policy_name=POLICY_NAME, env_name=ENV_NAME)

    _patch_act_inference_settings()

    cell_rates = run_seeds(ctx)
    overall = sum(cell_rates.values()) / len(cell_rates)
    write_summary(
        ctx,
        {
            "policy": POLICY_NAME,
            "env": ENV_NAME,
            "probe": "temporal_ensemble_coeff=0.01,n_action_steps=1",
            "v1_default_rate": V1_BASELINE_RATE,
            "seeds": SEEDS,
            "n_episodes_per_seed": N_EPISODES_PER_SEED,
            "per_seed_success_rate": cell_rates,
            "pooled_success_rate": overall,
        },
    )
    logger.info("PROBE COMPLETE pooled=%.4f (v1 baseline=%.4f)", overall, V1_BASELINE_RATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
