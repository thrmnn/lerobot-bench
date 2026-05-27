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
logger = logging.getLogger("probe-act-temporal-ensemble")

POLICY_NAME = "act"
ENV_NAME = "aloha_transfer_cube"
SEEDS = (0, 1, 2, 3, 4)
N_EPISODES_PER_SEED = 50
OUT_DIR = REPO_ROOT / "results" / "probes" / "act-aloha-temporal-ensemble"
OUT_PARQUET = OUT_DIR / "results.parquet"
VIDEOS_DIR = OUT_DIR / "videos"
SUMMARY_JSON = OUT_DIR / "summary.json"


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PARQUET.exists():
        OUT_PARQUET.unlink()
        logger.info("removed stale parquet %s", OUT_PARQUET)

    policy_reg = PolicyRegistry.from_yaml(REPO_ROOT / "configs" / "policies.yaml")
    env_reg = EnvRegistry.from_yaml(REPO_ROOT / "configs" / "envs.yaml")
    policy_spec = policy_reg.get(POLICY_NAME)
    env_spec = env_reg.get(ENV_NAME)

    _patch_act_inference_settings()

    from lerobot_bench.eval import run_cell_from_specs

    cell_rates: dict[int, float] = {}
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
        append_cell_rows(OUT_PARQUET, cell.to_rows())
        logger.info("probe seed=%d success_rate=%.4f", seed, cell.success_rate)

    overall = sum(cell_rates.values()) / len(cell_rates)
    summary = {
        "policy": POLICY_NAME,
        "env": ENV_NAME,
        "probe": "temporal_ensemble_coeff=0.01,n_action_steps=1",
        "v1_default_rate": 0.016,
        "seeds": SEEDS,
        "n_episodes_per_seed": N_EPISODES_PER_SEED,
        "per_seed_success_rate": cell_rates,
        "pooled_success_rate": overall,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    logger.info("PROBE COMPLETE pooled=%.4f (v1 baseline=%.4f)", overall, 0.016)
    logger.info("summary -> %s", SUMMARY_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
