#!/usr/bin/env python3
"""Render the canonical lerobot-bench figures at the three target styles.

Usage::

    render-figures                              # every figure x every style
    render-figures --style paper                # all figures at paper style
    render-figures --figure act_probe_bar       # 1 figure at all 3 styles
    render-figures --style deck --figure forest_plot --out-dir /tmp/check

Default inputs:
- ``--results results/sweep-full/results.parquet``
- ``--out-dir paper/figures``

Each render prints ``<path>\\t<size_bytes>``; the trailing line reports
the wall-clock cost so the operator can spot slow renders.

This script intentionally has zero torch / lerobot / gym imports so it
can run inside any minimal CI image with just matplotlib + pandas.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

# Headless backend before any pyplot import — keeps this script usable
# from CI runners and SSH sessions without a display.
matplotlib.use("Agg")

import pandas as pd

from lerobot_bench.figures import FIGURES, PARQUET_FREE_FIGURES, STYLES, Style, _as_style

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS = _REPO_ROOT / "results" / "sweep-full" / "results.parquet"
_DEFAULT_OUT_DIR = _REPO_ROOT / "paper" / "figures"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="render-figures",
        description="Render the three canonical lerobot-bench figures at paper/deck/web styles.",
    )
    p.add_argument(
        "--style",
        choices=[*sorted(STYLES), "all"],
        default="all",
        help="Render style. Default: all (paper + deck + web).",
    )
    p.add_argument(
        "--figure",
        choices=[*sorted(FIGURES), "all"],
        default="all",
        help="Which figure to render. Default: all 3.",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=_DEFAULT_RESULTS,
        help=f"Path to results parquet. Default: {_DEFAULT_RESULTS}",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {_DEFAULT_OUT_DIR}",
    )
    return p.parse_args(argv)


def _load_results(path: Path, figure: str) -> pd.DataFrame:
    """Load parquet, or return an empty frame for figures that don't need it."""
    if figure in PARQUET_FREE_FIGURES:
        return pd.DataFrame()
    if not path.exists():
        raise SystemExit(
            f"results parquet not found: {path}\n"
            "  (act_probe_bar / act_norm_ablation_2x2 can render "
            "without it; the other figures cannot.)"
        )
    return pd.read_parquet(path)


def _render_one(figure: str, style: Style, df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fn = FIGURES[figure]
    kwargs: dict[str, Any] = {"style": style, "out_dir": out_dir}
    if figure in PARQUET_FREE_FIGURES:
        return list(fn(**kwargs))
    return list(fn(df, **kwargs))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    styles: list[Style] = (
        [_as_style(s) for s in sorted(STYLES)] if args.style == "all" else [_as_style(args.style)]
    )
    figures: list[str] = sorted(FIGURES) if args.figure == "all" else [args.figure]

    # Load parquet once and reuse — read_parquet is the dominant cost
    # for the leaderboard figures.
    needs_df = any(f not in PARQUET_FREE_FIGURES for f in figures)
    df = _load_results(args.results, figure="forest_plot" if needs_df else "act_probe_bar")

    t0 = time.perf_counter()
    written: list[Path] = []
    for figure in figures:
        for style in styles:
            paths = _render_one(figure, style, df, args.out_dir)
            for path in paths:
                size = path.stat().st_size
                print(f"{path}\t{size}")
                written.append(path)
    dt = time.perf_counter() - t0
    print(f"# {len(written)} file(s) in {dt:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
