"""Tests for ``lerobot_bench.figures`` and ``scripts/render_figures.py``.

Headless / fast: ``matplotlib.use("Agg")`` is set at import; no torch /
lerobot / gym deps. Synthetic data is generated per test so the suite
runs without ``results/sweep-full/results.parquet`` on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Skip the whole module if matplotlib isn't installed: the figure-pipeline
# requires it but the bench's minimal CI install path ("fast" pytest job)
# does not pull matplotlib. Mirrors pytest.importorskip("lerobot") etc.
# elsewhere in this test tree.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pandas as pd  # noqa: E402  -- after pytest.importorskip("matplotlib") guard

from lerobot_bench import figures as fig_mod  # noqa: E402
from lerobot_bench.figures import (  # noqa: E402
    MDE_BAND,
    STYLES,
    act_norm_ablation_2x2,
    act_probe_bar,
    apply_style,
    forest_plot,
    replication_scatter,
)
from lerobot_bench.policies import PolicyRegistry, PolicySpec  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _synthetic_df() -> pd.DataFrame:
    """Two policies x two envs x 5 seeds x 50 ep (incl xvla, which must be filtered)."""
    rows: list[dict[str, object]] = []
    spec = [
        ("act", "aloha_transfer_cube", 0.20),
        ("smolvla_libero", "libero_spatial", 0.80),
        ("random", "pusht", 0.05),
        ("xvla_libero", "libero_spatial", 0.99),  # must be filtered out
    ]
    for policy, env, p in spec:
        for seed in range(5):
            for ep in range(50):
                rows.append(
                    {
                        "policy": policy,
                        "env": env,
                        "seed": seed,
                        "episode_index": ep,
                        "success": bool(((seed * 50 + ep) % 100) < int(p * 100)),
                    }
                )
    return pd.DataFrame(rows)


def _toy_registry() -> PolicyRegistry:
    specs = {
        "act": PolicySpec(
            name="act",
            is_baseline=False,
            env_compat=("aloha_transfer_cube",),
            repo_id="x",
            revision_sha="y",
            paper_reported_success={"aloha_transfer_cube": 0.50},
        ),
        "smolvla_libero": PolicySpec(
            name="smolvla_libero",
            is_baseline=False,
            env_compat=("libero_spatial",),
            repo_id="x",
            revision_sha="y",
            paper_reported_success={"libero_spatial": 0.90},
        ),
    }
    return PolicyRegistry(specs)


def test_apply_style_returns_copy_not_reference() -> None:
    s = apply_style("paper")
    s["palette"]["ok"] = "#ff00ff"
    s["font_size"] = 999
    again = apply_style("paper")
    assert again["palette"]["ok"] != "#ff00ff"
    assert again["font_size"] != 999


def test_styles_have_required_keys() -> None:
    required = {"figsize", "font_family", "palette", "bg", "dpi", "formats"}
    required_palette = {"ok", "warm", "fail", "muted"}
    for name, s in STYLES.items():
        missing = required - set(s)
        assert not missing, f"{name} missing keys {missing}"
        palette_missing = required_palette - set(s["palette"])
        assert not palette_missing, f"{name}.palette missing {palette_missing}"


def test_forest_plot_produces_file_per_format(tmp_path: Path) -> None:
    df = _synthetic_df()
    for style in STYLES:
        paths = forest_plot(df, style=style, out_dir=tmp_path)
        assert len(paths) == len(STYLES[style]["formats"])
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
            assert p.parent == tmp_path / style


def test_act_probe_bar_loads_from_summary_json_if_present(tmp_path: Path) -> None:
    summary = {
        "per_seed_success_rate": {"0": 0.10, "1": 0.10, "2": 0.10, "3": 0.10, "4": 0.10},
        "pooled_success_rate": 0.10,
        "n_episodes_per_seed": 50,
        "v1_default_rate": 0.05,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    data = fig_mod._load_probe_data(summary_path)
    assert data["probe"]["rate"] == pytest.approx(0.10)
    assert data["probe"]["per_seed"] == [0.10] * 5
    assert data["v1_default"]["rate"] == pytest.approx(0.05)
    paths = act_probe_bar(style="web", out_dir=tmp_path, summary_path=summary_path)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_act_probe_bar_falls_back_to_hardcoded(tmp_path: Path) -> None:
    paths = act_probe_bar(
        style="paper", out_dir=tmp_path, summary_path=tmp_path / "does-not-exist.json"
    )
    assert all(p.exists() for p in paths)
    data = fig_mod._load_probe_data(tmp_path / "does-not-exist.json")
    assert data["probe"]["rate"] == pytest.approx(0.764)
    assert data["v1_default"]["rate"] == pytest.approx(0.016)


def test_act_norm_ablation_2x2_produces_file_per_format(tmp_path: Path) -> None:
    for style in STYLES:
        paths = act_norm_ablation_2x2(style=style, out_dir=tmp_path)
        assert len(paths) == len(STYLES[style]["formats"])
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
            assert p.parent == tmp_path / style
            assert p.stem == "act_norm_ablation"


def test_act_norm_ablation_2x2_uses_canonical_cells() -> None:
    cells = fig_mod._ACT_NORM_ABLATION
    assert cells[(0, 0)]["rate"] == pytest.approx(0.016)  # buggy + hub
    assert cells[(0, 1)]["rate"] == pytest.approx(0.016)  # buggy + paper
    assert cells[(1, 0)]["rate"] == pytest.approx(0.812)  # fixed + hub
    assert cells[(1, 1)]["rate"] == pytest.approx(0.768)  # fixed + paper
    assert cells[(1, 0)]["ci"] == (0.759, 0.856)
    assert cells[(1, 1)]["ci"] == (0.712, 0.816)


def test_replication_scatter_filters_xvla(tmp_path: Path) -> None:
    df = _synthetic_df()
    assert (df["policy"] == "xvla_libero").any()
    rows = fig_mod._collect_replication_rows(fig_mod._filter_leaderboard(df), _toy_registry())
    assert all(r["policy"] != "xvla_libero" for r in rows)
    paths = replication_scatter(df, style="web", out_dir=tmp_path, registry=_toy_registry())
    assert all(p.exists() for p in paths)


def test_replication_scatter_greyscale_for_inside_MDE() -> None:
    rows = [
        {
            "policy": "act",
            "env": "e",
            "paper": 0.50,
            "measured": 0.55,
            "lo": 0.5,
            "hi": 0.6,
            "n": 250,
        },
        {
            "policy": "act",
            "env": "e",
            "paper": 0.50,
            "measured": 0.95,
            "lo": 0.92,
            "hi": 0.97,
            "n": 250,
        },
    ]
    assert abs(rows[0]["measured"] - rows[0]["paper"]) < MDE_BAND
    assert abs(rows[1]["measured"] - rows[1]["paper"]) >= MDE_BAND


def test_cli_renders_all_9_with_defaults(tmp_path: Path) -> None:
    df = _synthetic_df()
    results_path = tmp_path / "results.parquet"
    df.to_parquet(results_path)
    out_dir = tmp_path / "figures"
    env = {"PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "render_figures.py"),
            "--results",
            str(results_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    fig_names = (
        "forest_plot",
        "act_probe_bar",
        "act_norm_ablation",
        "replication_scatter",
    )
    expected: list[Path] = []
    for fig_name in fig_names:
        for style, style_dict in STYLES.items():
            for ext in style_dict["formats"]:
                expected.append(out_dir / style / f"{fig_name}.{ext}")
    for p in expected:
        assert p.exists(), f"missing: {p}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert p.stat().st_size > 0
    # 4 figs x (paper(2) + deck(1) + web(1)) = 16 files
    assert len(expected) == len(fig_names) * 4
