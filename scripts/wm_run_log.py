#!/usr/bin/env python3
"""Offline-first progress logger for world-model / JEPA training runs.

The exploratory world-model track trains on a single laptop with no
network and no wandb. This module is the minimal, dependency-free writer
the training loop calls to leave a breadcrumb trail the operator
dashboard's **Training** tab can tail:

    results/wm-runs/<run_id>/progress.jsonl

Each line is one JSON record. The schema is a **PROPOSAL** (see
``docs/MONITORING.md`` -- pending the user's confirmation; the WM repo
will import this writer later), kept deliberately minimal::

    {"ts": "2026-06-02T18:00:00+00:00", "run_id": "jepa-pusht-001",
     "step": 1200, "metric": "loss", "value": 0.0431}

Design constraints:

* **Offline-first.** No wandb, no network, no third-party deps -- only
  the stdlib. A training run with no internet still gets full progress
  visibility on the laptop.
* **Append-only.** One ``open(..., "a")`` + ``write`` + ``flush`` per
  record so a ``kill -9`` mid-run leaves a valid prefix (the dashboard's
  reader skips a half-written trailing line).
* **Importable + CLI.** :func:`log_progress` is the API the WM training
  loop calls; the ``__main__`` block is a thin CLI for shell-driven runs
  and smoke tests.

This script writes *only* under ``results/wm-runs/`` -- it never touches
the sweep parquet, the manifest, or ``src/lerobot_bench/``.

Usage::

    # From a training loop (Python):
    from scripts.wm_run_log import log_progress
    log_progress("jepa-pusht-001", step=1200, metric="loss", value=0.0431)

    # From the shell (one record):
    python scripts/wm_run_log.py --run-id jepa-pusht-001 \
        --step 1200 --metric loss --value 0.0431

    # Smoke test: write a few synthetic records and print the path.
    python scripts/wm_run_log.py --run-id smoke --demo
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

# Anchored on the repo root (this file lives in ``scripts/``) so the
# default mirrors the dashboard's ``results/wm-runs/`` discovery. The
# ``WM_RUNS_SUBDIR`` / ``WM_PROGRESS_FILENAME`` names are duplicated from
# ``dashboard/_helpers.py`` on purpose: this writer must not import the
# dashboard (which drags in pandas + the lerobot_bench package). Keep the
# two in sync -- if you rename one, rename both.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results"
WM_RUNS_SUBDIR = "wm-runs"
WM_PROGRESS_FILENAME = "progress.jsonl"

# The proposed record schema. Documented in docs/MONITORING.md as
# PROPOSED pending user confirmation; the dashboard reader tolerates
# extra keys and missing keys, so the WM repo can extend this later
# without breaking the Training tab.
RECORD_KEYS = ("ts", "run_id", "step", "metric", "value")


def progress_path(run_id: str, results_dir: Path | None = None) -> Path:
    """Return the JSONL path for ``run_id`` (does not create it).

    ``results/wm-runs/<run_id>/progress.jsonl``. ``results_dir`` defaults
    to the repo ``results/`` dir so the path matches the dashboard's
    discovery; pass an override in tests to write under ``tmp_path``.
    """
    root = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    return root / WM_RUNS_SUBDIR / run_id / WM_PROGRESS_FILENAME


def make_record(
    run_id: str,
    *,
    step: int,
    metric: str,
    value: float,
    ts: str | None = None,
) -> dict[str, Any]:
    """Build one progress record dict in the proposed schema.

    ``ts`` defaults to the current UTC time in ISO-8601 with a ``+00:00``
    offset (the same shape the sweep manifest uses), so a caller that
    wants a deterministic timestamp (tests) can pass one in.
    """
    timestamp = ts if ts is not None else dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return {
        "ts": timestamp,
        "run_id": run_id,
        "step": int(step),
        "metric": str(metric),
        "value": float(value),
    }


def log_progress(
    run_id: str,
    *,
    step: int,
    metric: str,
    value: float,
    results_dir: Path | None = None,
    ts: str | None = None,
) -> Path:
    """Append one progress record to the run's JSONL and return its path.

    Creates the ``results/wm-runs/<run_id>/`` directory on first call.
    Each record is written + flushed individually so a crash leaves a
    valid prefix. This is the function the WM training loop imports and
    calls once per logged step/metric.

    Returns the path written to so a caller can log it once at startup.
    """
    path = progress_path(run_id, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = make_record(run_id, step=step, metric=metric, value=value, ts=ts)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
    return path


def _demo(run_id: str, results_dir: Path | None = None) -> Path:
    """Write a handful of synthetic records for a smoke test."""
    path = progress_path(run_id, results_dir)
    for step in range(0, 500, 100):
        log_progress(
            run_id,
            step=step,
            metric="loss",
            value=1.0 / (step + 1),
            results_dir=results_dir,
        )
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a world-model training-progress record (offline, no wandb).",
    )
    parser.add_argument("--run-id", required=True, help="run identifier (dir name under wm-runs/)")
    parser.add_argument("--step", type=int, help="training step the metric was measured at")
    parser.add_argument("--metric", help="metric name, e.g. loss / val_loss / lr")
    parser.add_argument("--value", type=float, help="metric value")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"results root (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="write a few synthetic records (smoke test) instead of one real record",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Exit codes:
        0  record(s) written
        2  bad arguments (missing --step/--metric/--value without --demo)
    """
    args = _build_arg_parser().parse_args(argv)

    if args.demo:
        path = _demo(args.run_id, args.results_dir)
        print(f"wrote demo records to {path}")
        return 0

    if args.step is None or args.metric is None or args.value is None:
        print(
            "error: --step, --metric and --value are required (or use --demo).",
            file=sys.stderr,
        )
        return 2

    path = log_progress(
        args.run_id,
        step=args.step,
        metric=args.metric,
        value=args.value,
        results_dir=args.results_dir,
    )
    print(f"appended record to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
