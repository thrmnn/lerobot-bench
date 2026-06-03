"""Tests for ``scripts/merge_corrected_act_rows.py``.

Builds a synthetic canonical + act-rerun parquet pair matching
``RESULT_SCHEMA`` (the stale act×aloha cell pools to ~0.016, the rerun to
~0.824) and asserts the splice:
  * preserves total row count and cell count,
  * flips the act×aloha pooled rate from ~0.016 to ~0.824,
  * leaves every other (non act×aloha) cell byte-for-byte untouched,
  * is idempotent (running it on an already-merged canonical is a no-op),
  * and that the publish preflight gate rejects the stale parquet but
    accepts the merged one.

No torch, no network: pure pandas fixtures on tmp_path.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scripts import merge_corrected_act_rows as mod

from lerobot_bench.checkpointing import RESULT_SCHEMA, load_results

EXPECTED_RERUN_ROWS = mod.EXPECTED_RERUN_ROWS  # 250 (5 seeds × 50 episodes)


# --------------------------------------------------------------------- #
# Fixture builders                                                      #
# --------------------------------------------------------------------- #


def _row(
    *,
    policy: str,
    env: str,
    seed: int,
    episode_index: int,
    success: bool,
    code_sha: str = "deadbeef",
) -> dict:
    """One RESULT_SCHEMA-shaped row (optional columns included)."""
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": episode_index,
        "success": success,
        "return_": 1.0 if success else 0.0,
        "n_steps": 10,
        "wallclock_s": 0.05,
        "video_sha256": f"{policy}_{env}_{seed}_{episode_index}",
        "code_sha": code_sha,
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-01T00:00:00+00:00",
        "errored": False,
        "eval_run_id": "",
    }


def _act_aloha_block(*, n_success: int, n_total_per_seed: int = 50, code_sha: str) -> list[dict]:
    """5 seeds × ``n_total_per_seed`` act×aloha rows; ``n_success`` total True."""
    rows: list[dict] = []
    flipped = 0
    for seed in range(5):
        for ep in range(n_total_per_seed):
            success = flipped < n_success
            if success:
                flipped += 1
            rows.append(
                _row(
                    policy="act",
                    env="aloha_transfer_cube",
                    seed=seed,
                    episode_index=ep,
                    success=success,
                    code_sha=code_sha,
                )
            )
    return rows


def _other_cells() -> list[dict]:
    """Two unrelated cells that must survive the splice untouched."""
    rows: list[dict] = []
    for ep in range(5):
        rows.append(_row(policy="diffusion", env="pusht", seed=0, episode_index=ep, success=True))
        rows.append(
            _row(
                policy="random", env="aloha_transfer_cube", seed=0, episode_index=ep, success=False
            )
        )
    return rows


def _canonical_df() -> pd.DataFrame:
    # Stale act×aloha: 4/250 success -> pooled 0.016.
    rows = _act_aloha_block(n_success=4, code_sha="staleeee") + _other_cells()
    return pd.DataFrame(rows)[list(RESULT_SCHEMA)]


def _rerun_df() -> pd.DataFrame:
    # Corrected act×aloha: 206/250 success -> pooled 0.824.
    rows = _act_aloha_block(n_success=206, code_sha="7361d96")
    return pd.DataFrame(rows)[list(RESULT_SCHEMA)]


# --------------------------------------------------------------------- #
# Core merge                                                            #
# --------------------------------------------------------------------- #


def test_pooled_rates_set_up_as_documented() -> None:
    canon = _canonical_df()
    rerun = _rerun_df()
    assert mod._pooled_rate(canon[mod._act_aloha_mask(canon)]) == pytest.approx(0.016, abs=1e-6)
    assert mod._pooled_rate(rerun[mod._act_aloha_mask(rerun)]) == pytest.approx(0.824, abs=1e-6)


def test_merge_preserves_rows_and_flips_rate() -> None:
    canon = _canonical_df()
    rerun = _rerun_df()
    n_before = len(canon)
    cells_before = mod._cell_count(canon)

    merged = mod.merge_corrected_act_rows(canon, rerun)

    assert len(merged) == n_before
    assert mod._cell_count(merged) == cells_before
    after = mod._pooled_rate(merged[mod._act_aloha_mask(merged)])
    assert after == pytest.approx(0.824, abs=1e-6)


def test_other_cells_untouched() -> None:
    canon = _canonical_df()
    rerun = _rerun_df()
    merged = mod.merge_corrected_act_rows(canon, rerun)

    def _non_act_aloha(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[~mod._act_aloha_mask(df)].copy()
        return sub.sort_values(["policy", "env", "seed", "episode_index"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(_non_act_aloha(canon), _non_act_aloha(merged))


def test_merge_is_idempotent() -> None:
    canon = _canonical_df()
    rerun = _rerun_df()
    once = mod.merge_corrected_act_rows(canon, rerun)
    twice = mod.merge_corrected_act_rows(once, rerun)
    pd.testing.assert_frame_equal(once, twice)


def test_rejects_wrong_rerun_row_count() -> None:
    canon = _canonical_df()
    short = _rerun_df().iloc[:-1]  # 249 rows
    with pytest.raises(ValueError, match="act×aloha rows"):
        mod.merge_corrected_act_rows(canon, short)


def test_rejects_rerun_with_wrong_rate() -> None:
    canon = _canonical_df()
    bad = pd.DataFrame(_act_aloha_block(n_success=4, code_sha="7361d96"))[list(RESULT_SCHEMA)]
    with pytest.raises(ValueError, match="outside"):
        mod.merge_corrected_act_rows(canon, bad)


# --------------------------------------------------------------------- #
# File round-trip + dry-run                                            #
# --------------------------------------------------------------------- #


def _write(path, df: pd.DataFrame) -> None:
    df.to_parquet(path, index=False, engine="pyarrow")


def test_process_file_writes_and_dry_run(tmp_path) -> None:
    canon_path = tmp_path / "results.parquet"
    rerun_path = tmp_path / "results-act-rerun.parquet"
    _write(canon_path, _canonical_df())
    _write(rerun_path, _rerun_df())

    before, after, n_rows = mod.process_file(canon_path, rerun_path, dry_run=True)
    assert before == pytest.approx(0.016, abs=1e-6)
    assert after == pytest.approx(0.824, abs=1e-6)
    # Dry-run must not mutate the file.
    on_disk = load_results(canon_path)
    assert mod._pooled_rate(on_disk[mod._act_aloha_mask(on_disk)]) == pytest.approx(0.016, abs=1e-6)

    mod.process_file(canon_path, rerun_path, dry_run=False)
    written = load_results(canon_path)
    assert mod._pooled_rate(written[mod._act_aloha_mask(written)]) == pytest.approx(0.824, abs=1e-6)
    assert len(written) == n_rows


# --------------------------------------------------------------------- #
# Publish preflight gate                                                #
# --------------------------------------------------------------------- #


def test_preflight_rejects_stale_accepts_merged(tmp_path) -> None:
    from scripts import publish_results as pr

    manifest = tmp_path / "sweep_manifest.json"
    manifest.write_text("{}")

    stale_path = tmp_path / "results.parquet"
    _write(stale_path, _canonical_df())
    res = pr._preflight(
        results_path=stale_path,
        manifest_path=manifest,
        videos_dir=tmp_path / "videos",
        skip_videos=True,
    )
    assert res.error is not None
    assert "Stale pre-#51" in res.error

    merged = mod.merge_corrected_act_rows(_canonical_df(), _rerun_df())
    _write(stale_path, merged)
    res2 = pr._preflight(
        results_path=stale_path,
        manifest_path=manifest,
        videos_dir=tmp_path / "videos",
        skip_videos=True,
    )
    assert res2.error is None
