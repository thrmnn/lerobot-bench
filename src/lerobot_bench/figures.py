"""Parameterized figure-generation pipeline for lerobot-bench.

Three canonical figures (forest plot, ACT temporal-ensembling probe bar,
paper-vs-measured replication scatter) render at three target styles
(`paper`, `deck`, `web`). Each figure function takes a ``style`` kwarg
and writes the rendered file(s) to ``out_dir / style / name.<ext>`` per
the style's ``formats`` tuple. Style dicts are the only public surface
for visual configuration — figure functions must NOT take colors as
kwargs.

Data sources (cited inline in each function's docstring):

- Forest plot: ``results/sweep-full/results.parquet`` (PR #74 sweep).
- ACT probe bar: ``results/probes/act-aloha-temporal-ensemble/summary.json``
  when present, else hardcoded values from
  ``docs/PROBE_RESULTS_V1.0.1.md`` Table "Probe 1".
- Replication scatter: parquet + ``configs/policies.yaml``
  ``paper_reported_success`` blocks (Zhao 2023, Shukor 2025, etc. —
  see policies.yaml comments for per-cell citations).

Every render call writes to ``paper/figures/{style}/{name}.{ext}``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lerobot_bench.policies import PolicyRegistry
from lerobot_bench.stats import wilson_ci, wilson_halfwidth_at_p

Style = Literal["paper", "deck", "web"]

# Sort order for the leaderboard cells. xvla_libero is intentionally
# absent — deferred from the v1 leaderboard per PR #82 / xvla v1.1
# deferral memo, and exclude_xvla() drops it from inputs.
_POLICY_ORDER: tuple[str, ...] = (
    "act",
    "diffusion_policy",
    "smolvla_libero",
    "no_op",
    "random",
)
_ENV_ORDER: tuple[str, ...] = (
    "pusht",
    "aloha_transfer_cube",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Minimum-detectable-effect band half-width at p=0.5, N=250 (DESIGN.md
# § Methodology: 2 * wilson_halfwidth_at_p(0.5, 250) ≈ 0.123). Used by
# replication_scatter() to greyscale "indistinguishable from paper"
# cells. Computed at module load so tests can read it.
MDE_BAND: float = 2.0 * wilson_halfwidth_at_p(0.5, 250)


PAPER_STYLE: dict[str, Any] = {
    "figsize": (3.5, 2.5),
    "font_family": "serif",
    "font_size": 8,
    "line_width": 0.8,
    "palette": {"ok": "#2c7fb8", "warm": "#d95f02", "fail": "#c91414", "muted": "#888888"},
    "bg": "white",
    "fg": "#1a1a1a",
    "dpi": 300,
    "formats": ("svg", "pdf"),
}
DECK_STYLE: dict[str, Any] = {
    "figsize": (8.0, 4.5),
    "font_family": "Instrument Sans, sans-serif",
    "font_size": 18,
    "line_width": 2.0,
    "palette": {"ok": "#34d399", "warm": "#fbbf24", "fail": "#f87171", "muted": "#a78bfa"},
    "bg": "#0a0d12",
    "fg": "#f5f7fa",
    "dpi": 120,
    "formats": ("png",),
}
WEB_STYLE: dict[str, Any] = {
    "figsize": (6.0, 4.0),
    "font_family": "Instrument Sans, system-ui, sans-serif",
    "font_size": 12,
    "line_width": 1.2,
    "palette": {"ok": "#34d399", "warm": "#fbbf24", "fail": "#f87171", "muted": "#a78bfa"},
    "bg": "transparent",
    "fg": "#1a1a1a",
    "dpi": 96,
    "formats": ("svg",),
}

STYLES: dict[Style, dict[str, Any]] = {
    "paper": PAPER_STYLE,
    "deck": DECK_STYLE,
    "web": WEB_STYLE,
}

# Per-policy color (used by forest plot + replication scatter). Pulled
# from a small qualitative ramp so colours hold up at print-grayscale —
# the paper style overrides this with a darker, B&W-safe ramp.
_POLICY_COLORS_PAPER: dict[str, str] = {
    "act": "#1b9e77",
    "diffusion_policy": "#7570b3",
    "smolvla_libero": "#d95f02",
    "no_op": "#666666",
    "random": "#999999",
}
_POLICY_COLORS_DARK: dict[str, str] = {
    "act": "#34d399",
    "diffusion_policy": "#7aa3ff",
    "smolvla_libero": "#fbbf24",
    "no_op": "#7d8593",
    "random": "#a78bfa",
}


def apply_style(style: Style) -> dict[str, Any]:
    """Return a deep copy of the named style dict and apply rcParams.

    Returning a copy (not a reference) so callers can mutate the returned
    dict (e.g. to override a single color in a one-off render) without
    poisoning the module-level STYLES singleton — tests pin this.

    The matplotlib rcParams are mutated in place; the caller is expected
    to wrap repeated renders inside a single ``apply_style`` to amortise
    the cost. Reverting rcParams is the caller's responsibility (e.g.
    ``with plt.rc_context(...)``) — these figures all save + close
    inside a single function call so cross-figure rcParam leakage is not
    observed in practice.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLES)}")
    s = copy.deepcopy(STYLES[style])
    plt.rcParams["font.family"] = s["font_family"]
    plt.rcParams["font.size"] = s["font_size"]
    plt.rcParams["axes.linewidth"] = s["line_width"]
    plt.rcParams["axes.edgecolor"] = s["fg"]
    plt.rcParams["axes.labelcolor"] = s["fg"]
    plt.rcParams["xtick.color"] = s["fg"]
    plt.rcParams["ytick.color"] = s["fg"]
    plt.rcParams["text.color"] = s["fg"]
    plt.rcParams["axes.titlesize"] = s["font_size"] + 1
    plt.rcParams["savefig.dpi"] = s["dpi"]
    return s


def _policy_color_map(style: Style) -> dict[str, str]:
    return _POLICY_COLORS_PAPER if style == "paper" else _POLICY_COLORS_DARK


def _apply_bg(fig: Figure, style_dict: dict[str, Any]) -> None:
    bg = style_dict["bg"]
    if bg == "transparent":
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg)


def _save_all(fig: Figure, name: str, style: Style, out_dir: Path) -> list[Path]:
    """Save ``fig`` in every format of ``style`` under ``out_dir / style /``.

    Returns the list of written paths in the order they appear in
    ``STYLES[style]['formats']``. Each path is logged + sized so the
    CLI can render the operator-facing summary table.
    """
    style_dict = STYLES[style]
    target_dir = out_dir / style
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    transparent = style_dict["bg"] == "transparent"
    for ext in style_dict["formats"]:
        path = target_dir / f"{name}.{ext}"
        fig.savefig(
            path,
            dpi=style_dict["dpi"],
            bbox_inches="tight",
            transparent=transparent,
            facecolor=(style_dict["bg"] if not transparent else "none"),
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _filter_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Drop xvla rows (deferred from leaderboard per PR #82)."""
    return df.loc[~df["policy"].str.startswith("xvla"), :].copy()


# --------------------------------------------------------------------- #
# Figure 1 — forest plot                                                #
# --------------------------------------------------------------------- #


def forest_plot(df: pd.DataFrame, *, style: Style, out_dir: Path) -> list[Path]:
    """Per-cell success rate + Wilson 95% CI forest plot.

    Source: ``results/sweep-full/results.parquet`` (PR #74). Each cell
    is one ``(policy, env)`` pair; pooled rate across the cell's
    ``5 × n_episodes_per_seed`` episodes, with Wilson 95% CI per
    ``lerobot_bench.stats.wilson_ci`` (Wilson 1927; Agresti & Coull 1998).

    xvla_libero rows are excluded (deferred from the v1 leaderboard per
    PR #82). The vertical dotted line is the random-baseline pooled rate
    across all envs, for "is this policy beating random?" reference.
    """
    df = _filter_leaderboard(df)
    rows: list[dict[str, Any]] = []
    for (policy, env), grp in df.groupby(["policy", "env"], sort=False):
        n = len(grp)
        k = int(grp["success"].sum())
        rate = k / n if n else 0.0
        lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
        rows.append({"policy": policy, "env": env, "rate": rate, "lo": lo, "hi": hi, "n": n})

    cells = pd.DataFrame(rows)
    p_rank = {p: i for i, p in enumerate(_POLICY_ORDER)}
    e_rank = {e: i for i, e in enumerate(_ENV_ORDER)}
    cells["_p"] = cells["policy"].map(lambda p: p_rank.get(str(p), 99))
    cells["_e"] = cells["env"].map(lambda e: e_rank.get(str(e), 99))
    cells = cells.sort_values(["_p", "_e"], ascending=[True, True]).reset_index(drop=True)

    s = apply_style(style)
    # The forest plot's row count drives height: a fixed figsize cramps
    # all 17 labels into 2.5 inches. Scale height to keep ~0.22 in/row
    # at all styles, capped at 3x the style's default height so the deck
    # still fits a slide.
    base_w, base_h = s["figsize"]
    per_row = 0.22 if style == "paper" else 0.35
    height = max(base_h, min(3.0 * base_h, per_row * len(cells) + 1.0))
    fig, ax = plt.subplots(figsize=(base_w, height))

    color_map = _policy_color_map(style)
    y = np.arange(len(cells))
    labels = [f"{r['policy']} x {r['env']}" for _, r in cells.iterrows()]
    colors = [color_map.get(str(r["policy"]), s["palette"]["muted"]) for _, r in cells.iterrows()]
    rates = cells["rate"].to_numpy(dtype=float)
    # Clip to handle float noise from Wilson at k=0 / k=n (lo/hi can
    # drift ~1e-19 past the rate, which trips matplotlib's xerr>=0 check).
    err_lo = np.clip(rates - cells["lo"].to_numpy(dtype=float), 0.0, None)
    err_hi = np.clip(cells["hi"].to_numpy(dtype=float) - rates, 0.0, None)

    for i, color in enumerate(colors):
        ax.errorbar(
            rates[i],
            y[i],
            xerr=[[err_lo[i]], [err_hi[i]]],
            fmt="none",
            ecolor=color,
            elinewidth=s["line_width"],
            capsize=2.5,
            alpha=0.9,
        )
    ax.scatter(rates, y, c=colors, s=22, zorder=3, edgecolors=s["fg"], linewidths=0.4)

    random_rate = cells.loc[cells["policy"] == "random", "rate"].mean()
    if pd.notna(random_rate):
        ax.axvline(
            float(random_rate),
            color=s["palette"]["muted"],
            linestyle=":",
            linewidth=s["line_width"],
            alpha=0.7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(6, s["font_size"] - 2))
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("success rate")
    ax.set_title("Per-cell success rates - 95% Wilson CI - N=250/cell")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _apply_bg(fig, s)
    fig.tight_layout()
    return _save_all(fig, "forest_plot", style, out_dir)


# --------------------------------------------------------------------- #
# Figure 2 — ACT temporal-ensembling probe (headline finding)           #
# --------------------------------------------------------------------- #

# Fallback hardcode from docs/PROBE_RESULTS_V1.0.1.md table "Probe 1"
# (RESOLVED). Used when summary.json is unavailable. Wilson CIs are
# fixed to the doc-published values; per-seed rates match the table.
_PROBE_FALLBACK: dict[str, Any] = {
    "v1_default": {
        "label": "v1.0.0 Hub default\nn_action_steps=100",
        "rate": 0.016,
        "ci": (0.006, 0.040),
        "per_seed": [0.02, 0.04, 0.00, 0.02, 0.00],
    },
    "probe": {
        "label": "v1.0.2 paper\ncoeff=0.01, n_action_steps=1",
        "rate": 0.764,
        "ci": (0.708, 0.812),
        "per_seed": [0.92, 0.80, 0.76, 0.66, 0.68],
    },
}


def _load_probe_data(summary_path: Path | None) -> dict[str, Any]:
    """Load probe summary or return the hardcoded fallback.

    The shipped summary.json has only ``per_seed_success_rate`` and
    ``pooled_success_rate`` for the probe arm; the v1-default arm and
    Wilson CIs come from the doc (PROBE_RESULTS_V1.0.1.md). We merge the
    probe arm in when summary.json is present, keeping the v1-default
    arm + both CIs from the doc.
    """
    data = copy.deepcopy(_PROBE_FALLBACK)
    if summary_path is None or not summary_path.exists():
        return data
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return data
    per_seed = summary.get("per_seed_success_rate")
    if isinstance(per_seed, dict):
        seeds_sorted = sorted(per_seed.keys(), key=lambda k: int(k))
        data["probe"]["per_seed"] = [float(per_seed[k]) for k in seeds_sorted]
    pooled = summary.get("pooled_success_rate")
    if isinstance(pooled, (int, float)):
        n_total = 5 * int(summary.get("n_episodes_per_seed", 50))
        k = round(float(pooled) * n_total)
        data["probe"]["rate"] = float(pooled)
        data["probe"]["ci"] = wilson_ci(k, n_total)
    v1_default = summary.get("v1_default_rate")
    if isinstance(v1_default, (int, float)):
        n_total = 5 * int(summary.get("n_episodes_per_seed", 50))
        k = round(float(v1_default) * n_total)
        data["v1_default"]["rate"] = float(v1_default)
        data["v1_default"]["ci"] = wilson_ci(k, n_total)
    return data


def act_probe_bar(
    *,
    style: Style,
    out_dir: Path,
    summary_path: Path | None = None,
) -> list[Path]:
    """ACT × aloha_transfer_cube headline finding: settings, not architecture.

    Side-by-side bars compare the v1.0.0 Hub-default inference settings
    (``n_action_steps=100``, no temporal ensembling) against the paper-
    canonical settings (``temporal_ensemble_coeff=0.01``,
    ``n_action_steps=1``) on the same checkpoint + same env. Wilson 95%
    CIs are disjoint by an order of magnitude; this is the v1.0.2
    headline figure.

    Source data: ``results/probes/act-aloha-temporal-ensemble/summary.json``
    when present, else hardcoded fallback from
    ``docs/PROBE_RESULTS_V1.0.1.md`` Table "Probe 1". Paper reference
    Zhao et al. 2023 ("Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware", RSS) reports 0.50 on the human-teleop column for
    this checkpoint — overlaid as a dotted horizontal line.
    """
    if summary_path is None:
        summary_path = Path("results/probes/act-aloha-temporal-ensemble/summary.json")
    data = _load_probe_data(summary_path)
    s = apply_style(style)
    fig, ax = plt.subplots(figsize=s["figsize"])

    arms = ["v1_default", "probe"]
    x = np.array([0.0, 1.0])
    rates = np.array([data[a]["rate"] for a in arms], dtype=float)
    cis = [data[a]["ci"] for a in arms]
    err_lo = rates - np.array([c[0] for c in cis])
    err_hi = np.array([c[1] for c in cis]) - rates
    bar_colors = [s["palette"]["fail"], s["palette"]["ok"]]
    ax.bar(x, rates, width=0.6, color=bar_colors, edgecolor=s["fg"], linewidth=s["line_width"])
    ax.errorbar(
        x, rates, yerr=[err_lo, err_hi], fmt="none", ecolor=s["fg"], capsize=4, elinewidth=1.2
    )

    rng = np.random.default_rng(0)
    for xi, arm in zip(x, arms, strict=True):
        per_seed = np.asarray(data[arm]["per_seed"], dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=per_seed.size)
        ax.scatter(
            xi + jitter,
            per_seed,
            s=18,
            color=s["fg"],
            edgecolors=s["bg"] if s["bg"] != "transparent" else "white",
            linewidths=0.5,
            zorder=4,
            alpha=0.85,
        )

    ax.axhline(0.50, color=s["palette"]["muted"], linestyle=":", linewidth=s["line_width"])
    ax.text(
        -0.30,
        0.52,
        "paper (Zhao 2023): 0.50",
        va="bottom",
        ha="left",
        fontsize=max(6, s["font_size"] - 5),
        color=s["palette"]["muted"],
    )
    ax.set_xlim(-0.45, 1.45)

    delta = rates[1] - rates[0]
    ax.annotate(
        f"+{delta * 100:.1f} pp",
        xy=(0.5, max(rates) + 0.05),
        ha="center",
        fontsize=s["font_size"],
        fontweight="bold",
        color=s["palette"]["ok"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([data[a]["label"] for a in arms], fontsize=max(7, s["font_size"] - 2))
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ACT x aloha_transfer_cube - inference settings are the load-bearing variable")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _apply_bg(fig, s)
    fig.tight_layout()
    return _save_all(fig, "act_probe_bar", style, out_dir)


# --------------------------------------------------------------------- #
# Figure 3 — paper-vs-measured replication scatter                      #
# --------------------------------------------------------------------- #


def _collect_replication_rows(df: pd.DataFrame, registry: PolicyRegistry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in registry:
        if spec.paper_reported_success is None:
            continue
        for env, paper_rate in spec.paper_reported_success.items():
            if paper_rate is None:
                continue
            grp = df[(df["policy"] == spec.name) & (df["env"] == env)]
            if grp.empty:
                continue
            n = len(grp)
            k = int(grp["success"].sum())
            measured = k / n
            lo, hi = wilson_ci(k, n)
            rows.append(
                {
                    "policy": spec.name,
                    "env": env,
                    "paper": float(paper_rate),
                    "measured": measured,
                    "lo": lo,
                    "hi": hi,
                    "n": n,
                    "inside_mde": abs(measured - float(paper_rate)) < MDE_BAND,
                }
            )
    return rows


def replication_scatter(
    df: pd.DataFrame,
    *,
    style: Style,
    out_dir: Path,
    registry: PolicyRegistry | None = None,
) -> list[Path]:
    """Paper-reported vs measured success rate per cell, with Wilson CIs.

    For every ``(policy, env)`` cell that has a published
    ``paper_reported_success`` rate in ``configs/policies.yaml``, plot
    one point at ``(paper, measured)`` with vertical Wilson 95% error
    bars. Cells where ``|measured - paper|`` is below the MDE band
    (``2 * wilson_halfwidth_at_p(0.5, 250) ~= 0.123``) are greyed out —
    those are within the noise floor of the bench and "agree with paper"
    is the right reading. Cells outside the band are colored by policy.

    xvla rows are filtered upstream (deferred from leaderboard, PR #82).
    See ``configs/policies.yaml`` comments for per-cell citations (Zhao
    2023, Chi 2023 / Hub card, Shukor 2025, etc.).
    """
    df = _filter_leaderboard(df)
    if registry is None:
        registry = PolicyRegistry.from_yaml(Path("configs/policies.yaml"))
    rows = _collect_replication_rows(df, registry)

    s = apply_style(style)
    fig, ax = plt.subplots(figsize=s["figsize"])
    ax.plot([0, 1], [0, 1], linestyle="--", color=s["palette"]["muted"], linewidth=s["line_width"])

    color_map = _policy_color_map(style)
    for row in rows:
        paper = float(row["paper"])
        measured = float(row["measured"])
        err = [
            [max(0.0, measured - float(row["lo"]))],
            [max(0.0, float(row["hi"]) - measured)],
        ]
        color = (
            s["palette"]["muted"]
            if row["inside_mde"]
            else color_map.get(str(row["policy"]), s["palette"]["muted"])
        )
        ax.errorbar(
            paper,
            measured,
            yerr=err,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
            elinewidth=s["line_width"],
        )
        label_fontsize = max(5, s["font_size"] - 4)
        ax.annotate(
            f"{row['policy']}/{row['env']}",
            xy=(paper, measured),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=label_fontsize,
            color=s["fg"],
            alpha=0.75,
        )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("paper-reported success")
    ax.set_ylabel("measured success (N=250 per cell)")
    ax.set_title("Paper-reported vs measured - N=250 each - grey = inside MDE band")
    ax.grid(True, linestyle=":", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _apply_bg(fig, s)
    fig.tight_layout()
    return _save_all(fig, "replication_scatter", style, out_dir)


# Public re-exports for the CLI.
FIGURES: dict[str, Any] = {
    "forest_plot": forest_plot,
    "act_probe_bar": act_probe_bar,
    "replication_scatter": replication_scatter,
}


def _as_style(name: str) -> Style:
    if name == "paper":
        return "paper"
    if name == "deck":
        return "deck"
    if name == "web":
        return "web"
    raise ValueError(f"unknown style {name!r}")
