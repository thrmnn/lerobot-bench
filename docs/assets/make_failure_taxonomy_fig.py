"""Render the failure-taxonomy distribution from the hand-labeled sample.

Reads ``failure-taxonomy-labels.csv`` (the real labels assigned from the
shipped rollout artifacts) and writes ``fig-failure-taxonomy.{svg,png}``.

This is an honest, small labeled set: the only per-rollout evidence that
ships in the repo is the curated showcase MP4s/PNGs under
``docs/assets/rollouts/`` -- there is no per-episode parquet on disk, so
labels are assigned by visual inspection of those artifacts, not by an
episode-record join. The figure reflects exactly that sample size.

Run: ``python docs/assets/make_failure_taxonomy_fig.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "failure-taxonomy-labels.csv"

# Observed-mode -> display label. These are the modes that actually
# appear in the labeled artifacts, NOT the abstract six-mode scheme.
MODE_LABELS = {
    "never_contacts": "Never contacts object\n(random baseline)",
    "grasp_miss": "Grasp miss\n(reaches, no secure grasp)",
    "no_decisive_progress": "No decisive progress\n(long-horizon stall)",
    "wrong_final_pose": "Wrong final pose\n(pushed, mis-oriented)",
}
MODE_ORDER = ["never_contacts", "grasp_miss", "no_decisive_progress", "wrong_final_pose"]
COLORS = {
    "never_contacts": "#9e9e9e",
    "grasp_miss": "#d9534f",
    "no_decisive_progress": "#f0ad4e",
    "wrong_final_pose": "#5bc0de",
}


def main() -> None:
    df = pd.read_csv(CSV)
    # Drop the unmeasurable (xvla wiring-bug) rows from the distribution.
    df = df[df["observed_mode"] != "unmeasurable"].copy()

    counts = df["observed_mode"].value_counts().reindex(MODE_ORDER, fill_value=0)
    total = int(counts.sum())

    labels = [MODE_LABELS[m] for m in MODE_ORDER]
    colors = [COLORS[m] for m in MODE_ORDER]
    values = [int(counts[m]) for m in MODE_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, edgecolor="#333", height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Labeled rollout artifacts (N = {total})", fontsize=10)
    ax.set_xlim(0, max(values) + 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    for i, v in enumerate(values):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=9, color="#222")
    ax.set_title(
        f"Observed failure modes in shipped rollouts (hand-labeled, N={total})",
        fontsize=10.5,
        pad=8,
    )
    ax.grid(axis="x", color="#ddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.01,
        -0.04,
        "Counts are over curated showcase artifacts, not a random sample of "
        "all failures; xvla excluded (wiring bug, not a policy mode).",
        fontsize=7,
        color="#666",
        ha="left",
    )
    fig.tight_layout()
    for ext in ("svg", "png"):
        out = HERE / f"fig-failure-taxonomy.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
