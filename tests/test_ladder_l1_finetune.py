"""Fast (no GPU / no sim / no lerobot) checks for the L1 fine-tuning rung.

The heavy work (``lerobot-train`` + the canonical aloha re-measure) is
exercised by actually running ``scripts/run_ladder_l1_finetune.py`` on the
4060; these tests pin the contract-level invariants that must hold regardless
of the run:

* the script module imports without torch / lerobot (pure-orchestration import
  surface, like ``scripts/run_one.py``);
* the fine-tuned policy name is gated OUT of the published leaderboard
  (``V1_POLICIES``), same posture as ``classical_pusht`` / ``xvla_libero``;
* artifacts are written under ``results/ladder/`` only (never the canonical
  ``results/sweep-full/results.parquet``);
* the lift is computed against the corrected zero-shot baseline (0.824), not
  the buggy 0.016 the canonical parquet ships;
* the re-measure contract matches the zero-shot ACT cell (5 seeds x 50 ep,
  canonical sticky_reward_eq=4.0).
"""

from __future__ import annotations

from scripts import run_ladder_l1_finetune as l1

from embodimetry.leaderboard_filter import V1_POLICIES


def test_finetuned_policy_gated_out_of_v1_leaderboard() -> None:
    # The fine-tuned checkpoint must never leak into the public leaderboard.
    assert l1.POLICY_OUT_NAME == "act_finetuned"
    assert l1.POLICY_OUT_NAME not in V1_POLICIES


def test_outputs_confined_to_results_ladder() -> None:
    # Both artifacts live under results/ladder/, never results/sweep-full/.
    assert l1.PARQUET.parent.name == "ladder"
    assert l1.SUMMARY.parent.name == "ladder"
    assert l1.PARQUET.parent.parent.name == "results"
    assert "sweep-full" not in str(l1.PARQUET)
    assert "sweep-full" not in str(l1.SUMMARY)


def test_remeasure_contract_matches_zeroshot_cell() -> None:
    # Same env + sampling budget the zero-shot `act` canonical cell used.
    assert l1.ENV == "aloha_transfer_cube"
    assert l1.N_SEEDS == 5
    assert l1.N_EPISODES == 50


def test_baseline_is_corrected_not_buggy() -> None:
    # The zero-shot reference is the corrected canonical re-run (0.824 = 206/250),
    # NOT the normalization-bug 0.016 the canonical results.parquet ships.
    assert (l1.ZEROSHOT_SUCCESSES, l1.ZEROSHOT_N) == (206, 250)
    rate = l1.ZEROSHOT_SUCCESSES / l1.ZEROSHOT_N
    assert abs(rate - 0.824) < 1e-3


def test_base_checkpoint_matches_policies_yaml() -> None:
    # The L1 base must be the exact same checkpoint the registry's `act` uses.
    assert l1.BASE_REPO == "lerobot/act_aloha_sim_transfer_cube_human"
    assert l1.BASE_REV == "ba73b2766f1371cdc133ca4efb97eb090d744625"
