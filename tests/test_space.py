"""Tests for ``space/_helpers.py``.

The Space lives in its own deploy target (``huggingface.co/spaces/...``)
and runs on the free CPU tier without our pytest job; we cannot import
``space/app.py`` here because Gradio is intentionally NOT in the
project's ``[dev]`` extras (the pytest fast job must stay snappy and
laptop-friendly). What we *can* test is the data-loading and rendering
helpers in ``space/_helpers.py``, which is gradio-free by design.

These tests exercise:

* schema validation against the canonical RESULT_SCHEMA
* leaderboard aggregation maths (Wilson CI half-width matches stats.py)
* deterministic sort order
* empty parquet → empty leaderboard, no exception
* Hub URL formatting
* methodology markdown contents
* an AST guard that ``_helpers.py`` does not import ``gradio`` at
  module load — protects the contract that pytest fast can run without
  the heavy Spaces dep.

The companion ``space-smoke.yml`` workflow boots ``app.py`` for real on
every PR that touches ``space/**`` so we still catch Gradio-side
regressions; this file only covers the parts a pure-Python unit test
can reasonably reach.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.stats import wilson_ci

# space/ is a sibling of tests/, not on the default sys.path. Add it so
# we can import the helpers as ``_helpers`` directly. The test file is
# the only reason the path mutation exists; the Space itself runs from
# inside the space/ working directory on HF and finds _helpers naturally.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPACE_DIR = _REPO_ROOT / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from _helpers import (  # noqa: E402  (sys.path mutation must precede import)
    DEFAULT_PARQUET_URL,
    HUB_DATASET_REPO,
    HUB_RAW_PREFIX,
    LEADERBOARD_COLUMNS,
    clear_results_cache,
    compute_leaderboard_table,
    episode_metadata,
    filter_episodes,
    format_video_url,
    list_unique,
    load_results_df,
    render_methodology_md,
)

# --------------------------------------------------------------------- #
# Synthetic-parquet fixture                                             #
# --------------------------------------------------------------------- #


def _row(
    policy: str,
    env: str,
    seed: int,
    ep: int,
    *,
    success: bool = True,
    return_: float = 1.0,
    n_steps: int = 10,
    wallclock_s: float = 0.5,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": ep,
        "success": success,
        "return_": return_,
        "n_steps": n_steps,
        "wallclock_s": wallclock_s,
        "video_sha256": "deadbeef",
        "code_sha": "cafef00d",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-03T00:00:00Z",
    }


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(RESULT_SCHEMA))


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """3 policies × 2 envs × 3 seeds × 50 episodes synthetic parquet.

    Success rates are deliberately stratified per policy so the
    leaderboard sort order is checkable: ``good`` succeeds always,
    ``mid`` half the time, ``bad`` never. Same per env to keep the
    asserts simple; cross-env sort tie-break exercises the secondary
    sort key.
    """
    rows: list[dict[str, Any]] = []
    for policy, success_rate in [("good", 1.0), ("mid", 0.5), ("bad", 0.0)]:
        for env in ["pusht", "aloha"]:
            for seed in [0, 1, 2]:
                for ep in range(50):
                    # Deterministic interleaved success pattern so the
                    # rate matches exactly without a random source.
                    succeed = ep < int(50 * success_rate)
                    rows.append(_row(policy, env, seed, ep, success=succeed))
    path = tmp_path / "results.parquet"
    _df(rows).to_parquet(path, index=False)
    # Reset the lru_cache between tests so parquet reads see fresh
    # files. The cache is keyed on the path string, so different
    # tmp_path values would already get separate entries; clearing
    # here is belt-and-suspenders.
    clear_results_cache()
    return path


# --------------------------------------------------------------------- #
# load_results_df                                                       #
# --------------------------------------------------------------------- #


def test_load_results_df_roundtrip(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    assert tuple(df.columns) == RESULT_SCHEMA
    # 3 policies * 2 envs * 3 seeds * 50 eps = 900 rows.
    assert len(df) == 900


def test_load_results_df_missing_path_returns_empty(tmp_path: Path) -> None:
    """Missing local parquet → empty DataFrame, no exception.

    Mirrors the empty-state contract: the Space must boot even when
    the dataset hasn't been published yet.
    """
    clear_results_cache()
    df = load_results_df(tmp_path / "does-not-exist.parquet")
    assert df.empty
    assert tuple(df.columns) == RESULT_SCHEMA


def test_load_results_df_wrong_columns_raises(tmp_path: Path) -> None:
    """Schema drift between sweep and Space is loud, not silent."""
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"unrelated": [1, 2, 3]}).to_parquet(path, index=False)
    clear_results_cache()
    with pytest.raises(ValueError, match=r"wrong columns"):
        load_results_df(path)


# --------------------------------------------------------------------- #
# compute_leaderboard_table                                             #
# --------------------------------------------------------------------- #


def test_leaderboard_row_count_and_sort(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    table = compute_leaderboard_table(df)

    # 3 policies * 2 envs = 6 cells.
    assert len(table) == 6
    # Sort: success_rate descending.
    rates = list(table["success_rate"])
    assert rates == sorted(rates, reverse=True)
    # Top two rows are "good"; bottom two are "bad".
    assert set(table.head(2)["policy"]) == {"good"}
    assert set(table.tail(2)["policy"]) == {"bad"}


def test_leaderboard_wilson_half_width_matches_stats(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    table = compute_leaderboard_table(df)

    # Pick the "mid" / "pusht" cell: 3 seeds × 25 successes = 75 / 150.
    row = table[(table["policy"] == "mid") & (table["env"] == "pusht")].iloc[0]
    assert int(row["n_episodes"]) == 150
    assert int(row["n_successes"]) == 75
    lo_ref, hi_ref = wilson_ci(75, 150, ci=0.95)
    assert row["ci_low"] == pytest.approx(lo_ref)
    assert row["ci_high"] == pytest.approx(hi_ref)
    assert row["ci_half_width"] == pytest.approx((hi_ref - lo_ref) / 2.0)


def test_leaderboard_empty_input_returns_empty_table() -> None:
    """Premortem: parquet is empty, table must still render its columns."""
    empty = pd.DataFrame(columns=list(RESULT_SCHEMA))
    table = compute_leaderboard_table(empty)
    assert list(table.columns) == list(LEADERBOARD_COLUMNS)
    assert len(table) == 0


def test_leaderboard_columns_are_canonical(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    table = compute_leaderboard_table(df)
    assert tuple(table.columns) == LEADERBOARD_COLUMNS


# --------------------------------------------------------------------- #
# Browse Rollouts helpers                                               #
# --------------------------------------------------------------------- #


def test_list_unique_returns_sorted_strings(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    assert list_unique(df, "policy") == ["bad", "good", "mid"]
    assert list_unique(df, "env") == ["aloha", "pusht"]
    # Seeds are ints in the parquet; helper normalises to strings for
    # the dropdown.
    assert list_unique(df, "seed") == ["0", "1", "2"]


def test_filter_episodes_returns_sorted_cell(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    cell = filter_episodes(df, policy="mid", env="aloha", seed="1")
    assert len(cell) == 50
    assert list(cell["episode_index"]) == list(range(50))


def test_filter_episodes_no_match_is_empty(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    cell = filter_episodes(df, policy="ghost", env="aloha", seed="1")
    assert cell.empty


# --------------------------------------------------------------------- #
# format_video_url                                                      #
# --------------------------------------------------------------------- #


def test_format_video_url_canonical() -> None:
    url = format_video_url("diffusion_policy", "pusht", 0, 7)
    expected = f"{HUB_RAW_PREFIX}/videos/diffusion_policy/pusht/seed0/episode7.mp4"
    assert url == expected
    # Sanity: anchored at the published dataset.
    assert HUB_DATASET_REPO in url
    assert url.startswith("https://huggingface.co/datasets/")


def test_format_video_url_rejects_negative() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        format_video_url("p", "e", -1, 0)
    with pytest.raises(ValueError, match="episode must be non-negative"):
        format_video_url("p", "e", 0, -1)


def test_default_parquet_url_resolves_under_dataset_repo() -> None:
    """The cached parquet URL points inside the published dataset."""
    assert DEFAULT_PARQUET_URL.startswith(HUB_RAW_PREFIX)
    assert DEFAULT_PARQUET_URL.endswith("/results.parquet")


# --------------------------------------------------------------------- #
# episode_metadata                                                      #
# --------------------------------------------------------------------- #


def test_episode_metadata_projects_human_friendly_keys() -> None:
    row = pd.Series(_row("p", "e", 2, 7, success=True, return_=0.42, n_steps=99))
    meta = episode_metadata(row)
    assert meta["policy"] == "p"
    assert meta["env"] == "e"
    assert meta["seed"] == 2
    assert meta["episode_index"] == 7
    assert meta["success"] is True
    # Renamed for display: parquet column ``return_`` → JSON key ``return``.
    assert meta["return"] == pytest.approx(0.42)
    assert meta["n_steps"] == 99


# --------------------------------------------------------------------- #
# Methodology markdown                                                  #
# --------------------------------------------------------------------- #


def test_methodology_md_contains_required_terms() -> None:
    md = render_methodology_md()
    assert isinstance(md, str)
    assert len(md) > 200
    # Load-bearing terms — the methodology tab must explain seeding,
    # CI math, and the lerobot version pin. If anyone deletes one of
    # these the test fails loudly.
    for term in ["seed", "Wilson", "bootstrap", "lerobot==0.5.1"]:
        assert term in md, f"methodology missing {term!r}"


def test_methodology_links_back_to_repo() -> None:
    md = render_methodology_md()
    assert "github.com/thrmnn/lerobot-bench" in md


# --------------------------------------------------------------------- #
# AST guard: _helpers.py must NOT import gradio at module load          #
# --------------------------------------------------------------------- #


def test_helpers_does_not_import_gradio() -> None:
    """Static guard: gradio is heavy and not in the repo's [dev] extras.

    If anyone wires a ``import gradio`` into _helpers.py, the project's
    pytest job will start failing at collection time on machines that
    haven't installed Gradio. Catch it here at the AST level so the
    error message names the offending module.
    """
    src = (_SPACE_DIR / "_helpers.py").read_text()
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "gradio":
                    offenders.append(alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "gradio"
        ):
            offenders.append(node.module)
    assert not offenders, f"_helpers.py must not import gradio at module load; found {offenders}"
