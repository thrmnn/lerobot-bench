"""Merge multiple calibration JSON files into one.

Each input is a calibration report (the same schema written by
``calibrate.py``). Cells are deduped by ``(policy, env)`` — when the
same key appears in multiple files, the LATER file (later in the
argv order) wins. The merged report uses the timestamp of the
LATEST input file by mtime, the git_sha of the FIRST input, and the
union of all unique cells.

Usage:
    scripts/merge_calibration.py a.json b.json c.json --out merged.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged_cells: dict[tuple[str, str], dict] = {}
    timestamp_utc = ""
    git_sha = ""
    lerobot_version = None
    for path in args.inputs:
        if not path.exists():
            print(f"warning: {path} missing, skipping", file=sys.stderr)
            continue
        with path.open() as f:
            r = json.load(f)
        if not git_sha:
            git_sha = r.get("git_sha", "unknown")
        if r.get("lerobot_version"):
            lerobot_version = r["lerobot_version"]
        if r.get("timestamp_utc", "") > timestamp_utc:
            timestamp_utc = r["timestamp_utc"]
        for cell in r["cells"]:
            key = (cell["policy"], cell["env"])
            # Last write wins (later in argv overrides earlier).
            merged_cells[key] = cell

    cells = sorted(merged_cells.values(), key=lambda c: (c["policy"], c["env"]))
    out = {
        "timestamp_utc": timestamp_utc,
        "git_sha": git_sha,
        "lerobot_version": lerobot_version,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    by_status: dict[str, int] = {}
    for c in cells:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
    print(f"merged {len(args.inputs)} files -> {args.out}")
    print(f"  cells: {len(cells)} ({by_status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
