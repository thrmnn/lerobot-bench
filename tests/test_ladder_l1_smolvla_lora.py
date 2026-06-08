"""Fast (no GPU / no sim / no lerobot) checks for the L1 SmolVLA-LoRA rung.

The heavy work (the LoRA ``lerobot-train`` + the canonical libero_10 re-measure)
runs on the 4060 via ``scripts/run_ladder_l1_smolvla_lora.py``; these tests pin
the contract-level invariants that must hold regardless of the run:

* the script imports without torch / lerobot (pure-orchestration import surface);
* the fine-tuned policy name is gated OUT of the published leaderboard
  (``V1_POLICIES``), same posture as ``classical_pusht`` / ``act_finetuned``;
* artifacts are written under ``results/ladder/`` only (never the canonical
  ``results/sweep-full/results.parquet``);
* the zero-shot baseline matches the canonical smolvla_libero x libero_10 cell
  (63/250 = 0.252);
* the re-measure contract matches the zero-shot cell (5 seeds x 50 ep, canonical
  max_steps=600);
* the base checkpoint matches the registry's ``smolvla_libero`` entry;
* the data-wiring rename map maps the libero_10 camera keys to the SmolVLA
  camera1/camera2 keys (the fix that unblocks training).
"""

from __future__ import annotations

from scripts import run_ladder_l1_smolvla_lora as l1

from embodimetry.leaderboard_filter import V1_POLICIES


def test_finetuned_policy_gated_out_of_v1_leaderboard() -> None:
    assert l1.POLICY_OUT_NAME == "smolvla_finetuned"
    assert l1.POLICY_OUT_NAME not in V1_POLICIES


def test_outputs_confined_to_results_ladder() -> None:
    assert l1.PARQUET.parent.name == "ladder"
    assert l1.SUMMARY.parent.name == "ladder"
    assert l1.PARQUET.parent.parent.name == "results"
    assert "sweep-full" not in str(l1.PARQUET)
    assert "sweep-full" not in str(l1.SUMMARY)


def test_remeasure_contract_matches_zeroshot_cell() -> None:
    assert l1.ENV == "libero_10"
    assert l1.N_SEEDS == 5
    assert l1.N_EPISODES == 50


def test_zeroshot_baseline_matches_canonical_cell() -> None:
    # 63/250 = 0.252, the canonical smolvla_libero x libero_10 zero-shot rate.
    assert (l1.ZEROSHOT_SUCCESSES, l1.ZEROSHOT_N) == (63, 250)
    rate = l1.ZEROSHOT_SUCCESSES / l1.ZEROSHOT_N
    assert abs(rate - 0.252) < 1e-3


def test_base_checkpoint_matches_policies_yaml() -> None:
    # The L1 base must be the exact same checkpoint the registry's
    # `smolvla_libero` uses (SHA also asserted == repo main at run time).
    assert l1.BASE_REPO == "lerobot/smolvla_libero"
    assert l1.BASE_REV == "31d453f7edd78c839a8bbc39744a292686daf0de"
    assert l1.TRAIN_DATASET == "lerobot/libero_10"


def test_rename_map_fixes_camera_key_mismatch() -> None:
    # libero_10 ships `image` + `wrist_image`; SmolVLA expects `camera1` +
    # `camera2`. The rename map is the data-wiring fix that unblocks training.
    assert l1.RENAME_MAP == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.wrist_image": "observation.images.camera2",
    }
