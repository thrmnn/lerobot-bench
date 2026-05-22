"""Unit tests for ``scripts/review_results.py``.

Pure pandas + config-YAML analysis — no torch, no env, no GPU. These
run in default CI. Each of the five anomaly checks gets a synthetic
in-memory cell built against the *real* ``configs/policies.yaml`` and
``configs/envs.yaml`` registries (so paper-reported numbers and
``max_steps`` are the genuine values the tool will see in production).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts import review_results as rr

from lerobot_bench.envs import EnvRegistry
from lerobot_bench.policies import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"


# --------------------------------------------------------------------- #
# Fixtures / builders                                                   #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def policies() -> PolicyRegistry:
    return PolicyRegistry.from_yaml(POLICIES_YAML)


@pytest.fixture(scope="module")
def envs() -> EnvRegistry:
    return EnvRegistry.from_yaml(ENVS_YAML)


def _cell_rows(
    policy: str,
    env: str,
    seed: int,
    successes: list[bool],
    n_steps: list[int],
) -> list[dict[str, Any]]:
    """Build one synthetic cell's worth of episode rows."""
    assert len(successes) == len(n_steps)
    return [
        {
            "policy": policy,
            "env": env,
            "seed": seed,
            "episode_index": i,
            "success": s,
            "return_": 1.0 if s else 0.0,
            "n_steps": steps,
            "wallclock_s": 1.0,
            "video_sha256": "",
            "code_sha": "deadbeef",
            "lerobot_version": "0.5.1",
            "timestamp_utc": "2026-05-22T00:00:00+00:00",
        }
        for i, (s, steps) in enumerate(zip(successes, n_steps, strict=True))
    ]


def _frame(*cells: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in cells:
        rows.extend(c)
    return pd.DataFrame(rows)


def _flags_for(reviews: list[rr.CellReview], policy: str, env: str, seed: int) -> list[str]:
    for r in reviews:
        if r.policy == policy and r.env == env and r.seed == seed:
            return r.flags
    raise AssertionError(f"no review for {policy}×{env}×seed{seed}")


# --------------------------------------------------------------------- #
# Clean cell — no flag                                                  #
# --------------------------------------------------------------------- #


def test_clean_cell_not_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # diffusion_policy on pusht near its paper-reported 0.654.
    succ = [True] * 17 + [False] * 8  # 68%
    steps = [100 + i for i in range(25)]
    df = _frame(_cell_rows("diffusion_policy", "pusht", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    assert len(reviews) == 1
    assert reviews[0].flags == []
    assert reviews[0].flagged is False


# --------------------------------------------------------------------- #
# Check 1 — far from paper                                              #
# --------------------------------------------------------------------- #


def test_far_from_paper_below(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # act paper-reported 0.50 on aloha_transfer_cube; here ~4% with varied steps
    # so the never-succeeds / degenerate checks do not also fire.
    succ = [True] * 2 + [False] * 48
    steps = [200 + i for i in range(50)]
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert any(f.startswith("FAR-FROM-PAPER (well-below)") for f in flags)


def test_far_from_paper_above(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    succ = [True] * 49 + [False]  # 98% vs paper 50%
    steps = [200 + i for i in range(50)]
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert any(f.startswith("FAR-FROM-PAPER (well-above)") for f in flags)


def test_near_paper_within_threshold_not_flagged(
    policies: PolicyRegistry, envs: EnvRegistry
) -> None:
    # act at ~54% — within 0.25 of paper 0.50, no flag.
    succ = [True] * 27 + [False] * 23
    steps = [200 + i for i in range(50)]
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert not any(f.startswith("FAR-FROM-PAPER") for f in flags)


# --------------------------------------------------------------------- #
# Check 2 — baseline above floor                                        #
# --------------------------------------------------------------------- #


def test_baseline_above_floor_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # no_op scoring 40% on aloha — way above the 0.15 floor.
    succ = [True] * 20 + [False] * 30
    steps = [200 + i for i in range(50)]
    df = _frame(_cell_rows("no_op", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "no_op", "aloha_transfer_cube", 0)
    assert any(f.startswith("BASELINE-ABOVE-FLOOR") for f in flags)


def test_baseline_at_floor_not_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # no_op at 0% on aloha — the expected baseline floor, no flag.
    succ = [False] * 50
    steps = [200 + i for i in range(50)]
    df = _frame(_cell_rows("no_op", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "no_op", "aloha_transfer_cube", 0)
    assert not any(f.startswith("BASELINE-ABOVE-FLOOR") for f in flags)


def test_pusht_random_not_false_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # PushT random has a known small non-zero floor; 20% must not flag
    # because the env-specific PushT bar is higher than the generic 0.15.
    succ = [True] * 5 + [False] * 20  # 20%
    steps = [50 + i for i in range(25)]
    df = _frame(_cell_rows("random", "pusht", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "random", "pusht", 0)
    assert not any(f.startswith("BASELINE-ABOVE-FLOOR") for f in flags)


def test_pusht_random_high_still_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # Above even the bumped PushT bar (0.35) — still a flag.
    succ = [True] * 15 + [False] * 10  # 60%
    steps = [50 + i for i in range(25)]
    df = _frame(_cell_rows("random", "pusht", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "random", "pusht", 0)
    assert any(f.startswith("BASELINE-ABOVE-FLOOR") for f in flags)


# --------------------------------------------------------------------- #
# Check 3 — never-succeeds / all-max-steps                              #
# --------------------------------------------------------------------- #


def test_never_succeeds_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # aloha_transfer_cube max_steps == 400; act inert.
    succ = [False] * 50
    steps = [400] * 50
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert any(f.startswith("NEVER-SUCCEEDS") for f in flags)


def test_never_succeeds_not_flagged_when_steps_vary(
    policies: PolicyRegistry, envs: EnvRegistry
) -> None:
    # 0 successes but episodes ended early — not the all-max-steps signature.
    succ = [False] * 50
    steps = [350 + i for i in range(50)]
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert not any(f.startswith("NEVER-SUCCEEDS") for f in flags)


# --------------------------------------------------------------------- #
# Check 4 — seed disagreement                                           #
# --------------------------------------------------------------------- #


def test_seed_disagreement_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # diffusion_policy seed 0 at ~16%, seed 1 at ~80% — spread 0.64 > 0.30.
    lo = _cell_rows(
        "diffusion_policy", "pusht", 0, [True] * 4 + [False] * 21, list(range(100, 125))
    )
    hi = _cell_rows(
        "diffusion_policy", "pusht", 1, [True] * 20 + [False] * 5, list(range(100, 125))
    )
    df = _frame(lo, hi)
    reviews = rr.review_cells(df, policies, envs)
    for seed in (0, 1):
        flags = _flags_for(reviews, "diffusion_policy", "pusht", seed)
        assert any(f.startswith("SEED-DISAGREEMENT") for f in flags)


def test_seeds_in_agreement_not_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    a = _cell_rows("diffusion_policy", "pusht", 0, [True] * 17 + [False] * 8, list(range(100, 125)))
    b = _cell_rows("diffusion_policy", "pusht", 1, [True] * 18 + [False] * 7, list(range(100, 125)))
    df = _frame(a, b)
    reviews = rr.review_cells(df, policies, envs)
    for seed in (0, 1):
        flags = _flags_for(reviews, "diffusion_policy", "pusht", seed)
        assert not any(f.startswith("SEED-DISAGREEMENT") for f in flags)


# --------------------------------------------------------------------- #
# Check 5 — degenerate (identical episodes)                             #
# --------------------------------------------------------------------- #


def test_degenerate_identical_flagged(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # Every episode identical: same success, same n_steps (but steps not
    # max_steps, so this is the degenerate signature, not never-succeeds).
    succ = [True] * 50
    steps = [123] * 50
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert any(f.startswith("DEGENERATE") for f in flags)


def test_degenerate_not_flagged_when_steps_vary(
    policies: PolicyRegistry, envs: EnvRegistry
) -> None:
    succ = [True] * 50
    steps = [100 + i for i in range(50)]
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "act", "aloha_transfer_cube", 0)
    assert not any(f.startswith("DEGENERATE") for f in flags)


def test_degenerate_baseline_exempt(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    # no_op produces identical episodes by construction — that is the
    # expected floor, not a determinism bug.
    succ = [False] * 50
    steps = [400] * 50
    df = _frame(_cell_rows("no_op", "aloha_transfer_cube", 0, succ, steps))
    reviews = rr.review_cells(df, policies, envs)
    flags = _flags_for(reviews, "no_op", "aloha_transfer_cube", 0)
    assert not any(f.startswith("DEGENERATE") for f in flags)


# --------------------------------------------------------------------- #
# Exit codes                                                            #
# --------------------------------------------------------------------- #


def _write_parquet(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "results.parquet"
    df.to_parquet(p)
    return p


def test_exit_code_zero_when_clean(tmp_path: Path) -> None:
    succ = [True] * 17 + [False] * 8
    steps = [100 + i for i in range(25)]
    df = _frame(_cell_rows("diffusion_policy", "pusht", 0, succ, steps))
    results = _write_parquet(tmp_path, df)
    code = rr.run(
        [
            "--results",
            str(results),
            "--manifest",
            str(tmp_path / "absent.json"),
            "--policies",
            str(POLICIES_YAML),
            "--envs",
            str(ENVS_YAML),
        ]
    )
    assert code == rr.EXIT_OK


def test_exit_code_one_when_flagged(tmp_path: Path) -> None:
    succ = [False] * 50
    steps = [400] * 50
    df = _frame(_cell_rows("act", "aloha_transfer_cube", 0, succ, steps))
    results = _write_parquet(tmp_path, df)
    code = rr.run(
        [
            "--results",
            str(results),
            "--manifest",
            str(tmp_path / "absent.json"),
            "--policies",
            str(POLICIES_YAML),
            "--envs",
            str(ENVS_YAML),
        ]
    )
    assert code == rr.EXIT_ANOMALIES


def test_exit_code_two_when_results_missing(tmp_path: Path) -> None:
    code = rr.run(
        [
            "--results",
            str(tmp_path / "nope.parquet"),
            "--policies",
            str(POLICIES_YAML),
            "--envs",
            str(ENVS_YAML),
        ]
    )
    assert code == rr.EXIT_CANNOT_RUN


def test_exit_code_two_when_results_empty(tmp_path: Path) -> None:
    empty = pd.DataFrame(
        columns=["policy", "env", "seed", "episode_index", "success", "n_steps", "wallclock_s"]
    )
    results = _write_parquet(tmp_path, empty)
    code = rr.run(
        [
            "--results",
            str(results),
            "--policies",
            str(POLICIES_YAML),
            "--envs",
            str(ENVS_YAML),
        ]
    )
    assert code == rr.EXIT_CANNOT_RUN


# --------------------------------------------------------------------- #
# Report rendering                                                      #
# --------------------------------------------------------------------- #


def test_report_header_counts(policies: PolicyRegistry, envs: EnvRegistry) -> None:
    clean = _cell_rows(
        "diffusion_policy", "pusht", 0, [True] * 17 + [False] * 8, list(range(100, 125))
    )
    bad = _cell_rows("act", "aloha_transfer_cube", 0, [False] * 50, [400] * 50)
    reviews = rr.review_cells(_frame(clean, bad), policies, envs)
    report = rr.render_report(reviews)
    assert "2 cells reviewed, 1 flagged" in report
    assert "ANOMALIES" in report
