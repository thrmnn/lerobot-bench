#!/usr/bin/env python3
"""Controlled 2x2 ablation: decompose the ACT x aloha_transfer_cube lift into
(normalization fix) vs (temporal ensembling), at a SINGLE code_sha with fresh seeds.

Background: the v1.0 sweep scored ACT x aloha 0.016; a post-fix Hub-default rerun
scored 0.824 and a post-fix paper-settings probe scored 0.764. Those three come
from different code_shas / configs, so the causal decomposition is confounded.
This probe runs all four cells of the 2x2 at the CURRENT code_sha:

    factor N (normalization recovery): buggy | fixed
    factor E (action execution):       hub   | paper

  - N=buggy recreates the #51 bug by patching eval._buffer_name_to_feature_key
    to the old first-underscore-only mapping (observation_images_top ->
    observation.images_top, an unknown key the normalizer silently skips ->
    top camera fed un-normalized). N=fixed is the current disambiguated mapping.
  - E=hub uses the Hub defaults (temporal_ensemble_coeff=None, n_action_steps=100,
    100-step open-loop chunks). E=paper patches the ACT config to the paper's
    settings (temporal_ensemble_coeff=0.01, n_action_steps=1, per-step ensemble).

Run one cell per subprocess (clean monkeypatch + VRAM isolation):

    python scripts/probes/probe_act_normalization_ablation.py --norm buggy --settings hub
    python scripts/probes/probe_act_normalization_ablation.py --norm fixed --settings hub
    python scripts/probes/probe_act_normalization_ablation.py --norm buggy --settings paper
    python scripts/probes/probe_act_normalization_ablation.py --norm fixed --settings paper

--validate-only runs the CPU toggle check (no GPU, no rollouts) and exits.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless MuJoCo; default glfw segfaults on WSL2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import lerobot_bench.eval as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_seeds, setup_probe, write_summary  # noqa: E402

logger = logging.getLogger("probe.act_norm_ablation")

POLICY_NAME = "act"
ENV_NAME = "aloha_transfer_cube"

# The two-dot camera key the #51 bug mishandled.
_TOP_CAM_KEY = "observation.images.top"


def _patch_normalization_buggy() -> None:
    """Recreate the EFFECT of the #51 bug at the current code_sha.

    Pre-#51, the buffer ``buffer_observation_images_top`` recovered under the
    wrong name ``observation.images_top``; lerobot's NormalizerProcessorStep
    keys on ``observation.images.top``, didn't find it, and silently skipped
    normalizing the top camera -> raw [0,1] pixels -> garbage features -> ~0%.
    #51 also added a guard (eval.py:739) that hard-errors on the unmapped key,
    so simply mis-mapping no longer reproduces the silent skip. We instead drop
    the top-camera stat from the recovered dict: behaviorally identical (the
    normalizer skips an absent key exactly as it skipped a mis-named one), it
    passes the guard (no extra key), and it varies ONLY normalization."""

    original = ev._recover_dataset_stats_from_safetensors

    def buggy(repo_id, revision, feature_keys=()):  # type: ignore[no-untyped-def]
        stats = original(repo_id, revision, feature_keys=feature_keys)
        stats.pop(_TOP_CAM_KEY, None)
        return stats

    ev._recover_dataset_stats_from_safetensors = buggy  # type: ignore[attr-defined]
    logger.info(
        "PROBE: patched eval._recover_dataset_stats_from_safetensors -> drop %s", _TOP_CAM_KEY
    )


def _patch_settings_paper() -> None:
    """Patch ACT config to the paper's inference settings (per-step temporal
    ensemble), matching probe_act_temporal_ensemble.py."""
    from lerobot.configs.policies import PreTrainedConfig

    original = PreTrainedConfig.from_pretrained

    def patched(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = original(*args, **kwargs)
        if cfg.type == "act":
            cfg.temporal_ensemble_coeff = 0.01
            cfg.n_action_steps = 1
            logger.info("PROBE: patched ACT cfg -> temporal_ensemble_coeff=0.01, n_action_steps=1")
        return cfg

    PreTrainedConfig.from_pretrained = classmethod(patched)


def _validate() -> int:
    """Light check that the buggy patch installs and removes the top-cam stat.
    Behavioral validation is --smoke (buggy+hub -> ~0, fixed+hub -> high)."""
    before = ev._recover_dataset_stats_from_safetensors
    _patch_normalization_buggy()
    assert ev._recover_dataset_stats_from_safetensors is not before, "buggy patch did not install"
    print(f"VALIDATE OK: buggy patch installed (drops {_TOP_CAM_KEY}); confirm effect via --smoke")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", choices=["buggy", "fixed"], required=False)
    ap.add_argument("--settings", choices=["hub", "paper"], required=False)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed x 5 ep sanity (no full run)")
    args = ap.parse_args()

    if args.validate_only:
        return _validate()
    if not args.norm or not args.settings:
        ap.error("--norm and --settings are required unless --validate-only")

    if args.norm == "buggy":
        _patch_normalization_buggy()
    if args.settings == "paper":
        _patch_settings_paper()

    suffix = "-smoke" if args.smoke else ""
    cell_name = f"act-norm-ablation{suffix}/{args.norm}_{args.settings}"
    ctx = setup_probe(cell_name, policy_name=POLICY_NAME, env_name=ENV_NAME)
    cell_rates = run_seeds(ctx, seeds=(0,), n_episodes_per_seed=5) if args.smoke else run_seeds(ctx)
    pooled = sum(cell_rates.values()) / len(cell_rates)
    write_summary(
        ctx,
        {
            "policy": POLICY_NAME,
            "env": ENV_NAME,
            "ablation_cell": f"norm={args.norm},settings={args.settings}",
            "norm": args.norm,
            "settings": args.settings,
            "per_seed_success_rate": cell_rates,
            "pooled_success_rate": pooled,
        },
    )
    logger.info(
        "ABLATION CELL COMPLETE norm=%s settings=%s pooled=%.4f", args.norm, args.settings, pooled
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
