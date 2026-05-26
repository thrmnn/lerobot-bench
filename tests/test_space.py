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
    FAILURE_MODES,
    HUB_DATASET_REPO,
    HUB_RAW_PREFIX,
    LEADERBOARD_COLUMNS,
    PAIRED_COLUMNS,
    V1_POLICIES,
    clear_results_cache,
    compute_failure_counts,
    compute_leaderboard_table,
    compute_paired_table,
    episode_metadata,
    filter_episodes,
    filter_to_v1_policies,
    format_video_url,
    list_unique,
    load_results_df,
    narrate_top_finding,
    parse_failure_taxonomy_md,
    render_failure_panel_markdown,
    render_methodology_md,
    render_v1_status,
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
    leaderboard sort order is checkable: ``act`` succeeds always,
    ``diffusion_policy`` half the time, ``random`` never. Policy names
    are drawn from :data:`V1_POLICIES` because :func:`load_results_df`
    drops non-v1 rows on read (xvla deferral, PR #76); fake names like
    ``good``/``bad`` would get filtered out before any assert ran.
    """
    rows: list[dict[str, Any]] = []
    for policy, success_rate in [("act", 1.0), ("diffusion_policy", 0.5), ("random", 0.0)]:
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
    # Top two rows are the always-succeed policy; bottom two are
    # never-succeed.
    assert set(table.head(2)["policy"]) == {"act"}
    assert set(table.tail(2)["policy"]) == {"random"}


def test_leaderboard_wilson_half_width_matches_stats(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    table = compute_leaderboard_table(df)

    # Pick the half-success / pusht cell: 3 seeds × 25 successes = 75 / 150.
    row = table[(table["policy"] == "diffusion_policy") & (table["env"] == "pusht")].iloc[0]
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
    assert list_unique(df, "policy") == ["act", "diffusion_policy", "random"]
    assert list_unique(df, "env") == ["aloha", "pusht"]
    # Seeds are ints in the parquet; helper normalises to strings for
    # the dropdown.
    assert list_unique(df, "seed") == ["0", "1", "2"]


def test_filter_episodes_returns_sorted_cell(synthetic_parquet: Path) -> None:
    df = load_results_df(synthetic_parquet)
    cell = filter_episodes(df, policy="diffusion_policy", env="aloha", seed="1")
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


# --------------------------------------------------------------------- #
# Paired comparison                                                     #
# --------------------------------------------------------------------- #


def _make_paired_parquet(
    tmp_path: Path,
    *,
    policy_a: str = "act",
    policy_b: str = "random",
    env: str = "pusht",
    n_seeds: int = 5,
    n_episodes_per_seed: int = 10,
    a_success_rate: float = 0.8,
    b_success_rate: float = 0.2,
) -> Path:
    """Build a tiny synthetic parquet for paired-comparison tests.

    Defaults to ``act`` / ``random`` because :func:`load_results_df`
    drops rows whose policy is not in :data:`V1_POLICIES`; the old
    placeholder names ``A`` / ``B`` would be silently filtered out.
    """
    rows: list[dict[str, Any]] = []
    cutoff_a = int(n_episodes_per_seed * a_success_rate)
    cutoff_b = int(n_episodes_per_seed * b_success_rate)
    for seed in range(n_seeds):
        for ep in range(n_episodes_per_seed):
            rows.append(_row(policy_a, env, seed, ep, success=(ep < cutoff_a)))
            rows.append(_row(policy_b, env, seed, ep, success=(ep < cutoff_b)))
    path = tmp_path / "paired.parquet"
    _df(rows).to_parquet(path, index=False)
    clear_results_cache()
    return path


def test_paired_comparison_table_renders_synthetic_parquet(tmp_path: Path) -> None:
    """2 policies × 1 env × 50 rows → table has the canonical columns +
    delta sign matches the synthetic gap.
    """
    path = _make_paired_parquet(
        tmp_path,
        policy_a="act",
        policy_b="random",
        n_seeds=5,
        n_episodes_per_seed=50,
        a_success_rate=0.8,
        b_success_rate=0.2,
    )
    df = load_results_df(path)
    table = compute_paired_table(df, policy_a="act", policy_b="random")

    # Right column set and exactly one row (single env).
    assert tuple(table.columns) == PAIRED_COLUMNS
    assert len(table) == 1

    row = table.iloc[0]
    assert row["env"] == "pusht"
    assert int(row["n_A"]) == 250
    assert int(row["n_B"]) == 250
    # act is the stronger policy → delta should be ≈ +0.6 and clearly
    # exceed the per-cell MDE bound at N=250.
    assert float(row["delta"]) == pytest.approx(0.6, abs=1e-9)
    assert float(row["delta"]) > 0
    assert bool(row["clears_MDE"]) is True

    # Bootstrap CI bounds bracket the true delta.
    assert float(row["ci_low_delta"]) < float(row["delta"]) < float(row["ci_high_delta"])


def test_paired_comparison_handles_n_a_env(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Policy A has rows for env X but B doesn't → env is excluded with
    a log line, not a crash.
    """
    rows: list[dict[str, Any]] = []
    # Both policies in pusht; only act in aloha.
    for seed in range(2):
        for ep in range(20):
            rows.append(_row("act", "pusht", seed, ep, success=True))
            rows.append(_row("random", "pusht", seed, ep, success=False))
            rows.append(_row("act", "aloha", seed, ep, success=True))
    path = tmp_path / "asym.parquet"
    _df(rows).to_parquet(path, index=False)
    clear_results_cache()

    df = load_results_df(path)

    # No explicit ``envs``: shared-env logic drops aloha silently.
    table = compute_paired_table(df, policy_a="act", policy_b="random")
    assert list(table["env"]) == ["pusht"]
    assert len(table) == 1

    # Explicit ``envs`` requesting both: aloha is dropped with a log line.
    with caplog.at_level("INFO", logger="_helpers"):
        table2 = compute_paired_table(
            df, policy_a="act", policy_b="random", envs=["pusht", "aloha"]
        )
    assert list(table2["env"]) == ["pusht"]
    # Log mentions the dropped env.
    log_messages = [r.getMessage() for r in caplog.records]
    assert any("aloha" in m for m in log_messages), log_messages


def test_narrate_top_finding_describes_winning_env(tmp_path: Path) -> None:
    """The narration sentence names the env, both policies, and the delta."""
    path = _make_paired_parquet(
        tmp_path,
        a_success_rate=0.9,
        b_success_rate=0.1,
        n_episodes_per_seed=50,
    )
    df = load_results_df(path)
    table = compute_paired_table(df, policy_a="act", policy_b="random")
    sentence = narrate_top_finding(table, policy_a="act", policy_b="random")
    assert "pusht" in sentence
    assert "act" in sentence and "random" in sentence
    # Either "outperforms" or "underperforms" — act > random here.
    assert "outperforms" in sentence
    # Sample size is reported.
    assert "N=" in sentence


def test_narrate_top_finding_honest_when_below_mde(tmp_path: Path) -> None:
    """If no env clears MDE, the narration says so honestly."""
    # Equal-success policies → delta ≈ 0, well below MDE.
    path = _make_paired_parquet(
        tmp_path,
        a_success_rate=0.5,
        b_success_rate=0.5,
        n_episodes_per_seed=50,
    )
    df = load_results_df(path)
    table = compute_paired_table(df, policy_a="act", policy_b="random")
    sentence = narrate_top_finding(table, policy_a="act", policy_b="random")
    assert "No env clears the MDE bound" in sentence


def test_paired_comparison_empty_inputs_returns_canonical_empty() -> None:
    """Empty df → empty frame with the canonical column set."""
    empty = pd.DataFrame(columns=list(RESULT_SCHEMA))
    table = compute_paired_table(empty, policy_a="act", policy_b="random")
    assert tuple(table.columns) == PAIRED_COLUMNS
    assert len(table) == 0


# --------------------------------------------------------------------- #
# Failure taxonomy renderer                                             #
# --------------------------------------------------------------------- #


def test_failure_taxonomy_renders_categories_from_markdown() -> None:
    """Parse ``docs/FAILURE_TAXONOMY.md`` → list of category strings."""
    categories = parse_failure_taxonomy_md()
    # The doc declares the six canonical modes; the parser should
    # return all six in document order.
    assert len(categories) == len(FAILURE_MODES)
    parsed_labels = [c["label"] for c in categories]
    assert parsed_labels == list(FAILURE_MODES)
    # Human-readable name corresponds to the heading text — each one
    # should be non-empty and contain at least one character.
    for cat in categories:
        assert cat["name"]
        assert isinstance(cat["summary"], str)


def test_failure_taxonomy_renders_categories_from_explicit_path(
    tmp_path: Path,
) -> None:
    """Caller can pass a path; parser handles a small synthetic doc."""
    src = tmp_path / "TAXONOMY.md"
    src.write_text(
        "# Header\n\n"
        "### 1. First mode\n\n"
        "**Definition.** First-mode lead paragraph.\n\n"
        "Second paragraph not in summary.\n\n"
        "### 2. Second mode\n\n"
        "**Definition.** Second-mode lead paragraph.\n",
        encoding="utf-8",
    )
    cats = parse_failure_taxonomy_md(src)
    assert len(cats) == 2
    assert cats[0]["name"] == "First mode"
    assert cats[0]["summary"].startswith("First-mode lead paragraph")
    assert cats[1]["summary"].startswith("Second-mode lead paragraph")


def test_failure_taxonomy_empty_state_when_no_labels(synthetic_parquet: Path) -> None:
    """Empty parquet (or one without ``failure_label``) → empty-state
    string from the markdown renderer.
    """
    df = load_results_df(synthetic_parquet)
    counts = compute_failure_counts(df)
    assert counts.empty
    assert list(counts.columns) == ["policy", "env", "mode", "count"]

    categories = parse_failure_taxonomy_md()
    md = render_failure_panel_markdown(categories, counts)
    assert "No failure labels yet" in md
    # The categories list still renders so the panel is useful pre-labels.
    for cat in categories:
        assert cat["label"] in md or cat["name"] in md


def test_failure_taxonomy_counts_aggregate_when_labels_present(tmp_path: Path) -> None:
    """When a ``failure_label`` column is present and rows are failures,
    counts aggregate per (policy, env, mode).
    """
    rows: list[dict[str, Any]] = []
    for ep in range(10):
        # All failures with rotating labels — two cells worth of rows.
        rows.append(_row("A", "pusht", 0, ep, success=False))
        rows.append(_row("A", "pusht", 0, 10 + ep, success=False))
    df = _df(rows)
    # Inject failure_label column. First 10 rows labeled "drift", next
    # 10 labeled "timeout".
    df["failure_label"] = ["drift"] * 10 + ["timeout"] * 10
    counts = compute_failure_counts(df)
    assert len(counts) == 2
    by_mode = dict(zip(counts["mode"], counts["count"], strict=True))
    assert by_mode["drift"] == 10
    assert by_mode["timeout"] == 10
    # render_failure_panel_markdown does NOT show the empty-state string.
    md = render_failure_panel_markdown(parse_failure_taxonomy_md(), counts)
    assert "No failure labels yet" not in md
    assert "20" in md  # total label count surfaces in the summary line


def test_failure_taxonomy_drops_unknown_labels(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Labels outside FAILURE_MODES are dropped with a log line."""
    rows = [_row("A", "pusht", 0, ep, success=False) for ep in range(4)]
    df = _df(rows)
    df["failure_label"] = ["drift", "drift", "made_up_mode", "another_invented"]
    with caplog.at_level("INFO", logger="_helpers"):
        counts = compute_failure_counts(df)
    assert int(counts["count"].sum()) == 2
    assert "made_up_mode" in " ".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------- #
# Leaderboard sort order                                                #
# --------------------------------------------------------------------- #


def test_leaderboard_sorted_by_mean_success_rate(tmp_path: Path) -> None:
    """Synthetic parquet with three v1 policies → sorted output matches
    expected order.

    Per-policy mean success rates: smolvla_libero=0.7, act=0.5,
    diffusion_policy=0.3. The sort key is per-policy mean descending,
    so the table reads smolvla_libero-rows, act-rows, diffusion_policy-
    rows from top to bottom regardless of per-cell rate within a policy.
    Names drawn from :data:`V1_POLICIES` because the loader filters out
    non-v1 rows.
    """
    rows: list[dict[str, Any]] = []
    targets = {"act": 0.5, "diffusion_policy": 0.3, "smolvla_libero": 0.7}
    for policy, rate in targets.items():
        for env in ["pusht", "aloha"]:
            for seed in [0, 1]:
                for ep in range(10):
                    rows.append(_row(policy, env, seed, ep, success=(ep < int(10 * rate))))
    path = tmp_path / "ordered.parquet"
    _df(rows).to_parquet(path, index=False)
    clear_results_cache()

    df = load_results_df(path)
    table = compute_leaderboard_table(df)

    # Six cells total (3 policies × 2 envs). Policy order top→bottom
    # follows the per-policy mean descending: smolvla, act, diffusion.
    assert len(table) == 6
    policy_order = list(table["policy"])
    assert policy_order[:2] == ["smolvla_libero", "smolvla_libero"]
    assert policy_order[2:4] == ["act", "act"]
    assert policy_order[4:6] == ["diffusion_policy", "diffusion_policy"]
    # Within each policy, envs are alphabetical.
    assert list(table.iloc[:2]["env"]) == ["aloha", "pusht"]
    assert list(table.iloc[2:4]["env"]) == ["aloha", "pusht"]
    assert list(table.iloc[4:6]["env"]) == ["aloha", "pusht"]


# --------------------------------------------------------------------- #
# v1 status badge                                                       #
# --------------------------------------------------------------------- #


def test_render_v1_status_default_mentions_v1_and_pi0() -> None:
    md = render_v1_status()
    assert "v1" in md
    # Pi0 deferral is the load-bearing footnote.
    assert "Pi0" in md
    assert "v1.1" in md


def test_render_v1_status_uses_manifest_counts_when_passed() -> None:
    manifest = {
        "cells": [
            {"status": "completed"},
            {"status": "completed"},
            {"status": "pending"},
            {"status": "failed"},
        ]
    }
    md = render_v1_status(manifest)
    assert "2/4 cells completed" in md
    assert "completed=2" in md and "pending=1" in md and "failed=1" in md


# --------------------------------------------------------------------- #
# v1 policy filter (xvla_libero deferral, PR #76)                       #
# --------------------------------------------------------------------- #


def test_v1_policies_excludes_xvla() -> None:
    """xvla_libero is deferred to v1.1; the leaderboard tuple must not list it."""
    assert "xvla_libero" not in V1_POLICIES
    # Sanity: the v1 roster is exactly the five publicly-shipped policies.
    assert set(V1_POLICIES) == {
        "act",
        "diffusion_policy",
        "smolvla_libero",
        "no_op",
        "random",
    }


def test_filter_to_v1_policies_drops_xvla_rows() -> None:
    """The standalone helper drops non-v1 rows without touching the rest."""
    df = pd.DataFrame(
        {
            "policy": ["act", "xvla_libero", "diffusion_policy", "pi0", "random"],
            "env": ["pusht"] * 5,
            "success": [True, False, True, False, True],
        }
    )
    out = filter_to_v1_policies(df)
    assert set(out["policy"]) == {"act", "diffusion_policy", "random"}
    assert "xvla_libero" not in set(out["policy"])
    assert "pi0" not in set(out["policy"])


def test_filter_to_v1_policies_empty_passthrough() -> None:
    """Empty / missing-column input returns the frame unchanged."""
    empty = pd.DataFrame(columns=["policy", "env", "success"])
    assert filter_to_v1_policies(empty).empty
    # No policy column → no filter applied (degrades to identity).
    no_col = pd.DataFrame({"other": [1, 2, 3]})
    assert len(filter_to_v1_policies(no_col)) == 3


def test_load_results_df_filters_xvla_but_parquet_preserves_it(tmp_path: Path) -> None:
    """The published parquet still carries xvla rows for reproducibility;
    :func:`load_results_df` drops them so the Space's leaderboard, paired
    comparisons, and rollout dropdowns never see them.
    """
    rows: list[dict[str, Any]] = []
    for policy in ("act", "xvla_libero", "diffusion_policy"):
        for ep in range(5):
            rows.append(_row(policy, "pusht", 0, ep, success=(ep < 3)))
    path = tmp_path / "with-xvla.parquet"
    _df(rows).to_parquet(path, index=False)
    clear_results_cache()

    # Raw parquet on disk still has every policy — reproducibility contract.
    raw = pd.read_parquet(path)
    assert "xvla_libero" in set(raw["policy"])

    # But the loader gates v1 surfaces to V1_POLICIES.
    df = load_results_df(path)
    assert "xvla_libero" not in set(df["policy"])
    assert set(df["policy"]) == {"act", "diffusion_policy"}

    # Downstream leaderboard never surfaces an xvla row.
    table = compute_leaderboard_table(df)
    assert "xvla_libero" not in set(table["policy"])
