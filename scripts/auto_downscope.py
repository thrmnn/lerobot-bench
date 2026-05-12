"""Map a calibration JSON to ``configs/sweep_full.yaml`` overrides.

Reads ``results/calibration-YYYYMMDD.json`` (or any path), inspects each
``ok``-status cell's ``recommended`` field, and emits the ``policies``,
``envs``, and ``overrides`` blocks matching ``run_sweep.SweepConfig``.

Default: print the merged YAML to stdout.
With ``--apply <sweep.yaml>``: rewrite the file in place, preserving all
non-overridable fields (``seeds``, ``episodes_per_seed``, paths, etc.).

Cells whose ``recommended.episodes < base_episodes`` get an
``overrides[policy][env].n_episodes`` entry. Cells whose
``recommended.seeds < base_seeds`` get a ``seeds_subset`` entry
(``[0, 1, ..., k-1]``) — note: support for ``seeds_subset`` in
``run_sweep.py`` is a follow-up; for Phase-4 v1 we honor only
``n_episodes`` and warn about cells that requested fewer seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BASE_SEEDS = 5
DEFAULT_BASE_EPISODES = 50


def build_overrides(
    cells: list[dict[str, Any]],
    *,
    base_episodes: int = DEFAULT_BASE_EPISODES,
    base_seeds: int = DEFAULT_BASE_SEEDS,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[tuple[str, str, int]]]:
    """Return (overrides, seeds_warnings).

    overrides: nested dict policy -> env -> {n_episodes, seeds_subset?}
    seeds_warnings: list of (policy, env, recommended_seeds) where the
        recommendation requested fewer seeds than the default. Reported
        to stderr so the operator can manually trim seeds in
        sweep_full.yaml until ``run_sweep.py`` learns to honor them.
    """
    overrides: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seeds_warnings: list[tuple[str, str, int]] = []

    for cell in cells:
        if cell["status"] != "ok":
            continue
        rec = cell.get("recommended") or {}
        episodes = int(rec.get("episodes", base_episodes))
        seeds = int(rec.get("seeds", base_seeds))
        entry: dict[str, Any] = {}
        if episodes < base_episodes:
            entry["n_episodes"] = episodes
        if seeds < base_seeds:
            entry["seeds_subset"] = list(range(seeds))
            seeds_warnings.append((cell["policy"], cell["env"], seeds))
        if entry:
            overrides[cell["policy"]][cell["env"]] = entry

    return dict(overrides), seeds_warnings


def merge_into_sweep(
    existing: dict[str, Any],
    overrides: dict[str, dict[str, dict[str, Any]]],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the merged sweep config: union of policies/envs from cells +
    rewritten overrides block. Other top-level fields are preserved.
    """
    out = dict(existing)
    ok_cells = [c for c in cells if c["status"] == "ok"]
    out["policies"] = sorted({c["policy"] for c in ok_cells})
    out["envs"] = sorted({c["env"] for c in ok_cells})
    out["overrides"] = overrides
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_json", type=Path)
    ap.add_argument(
        "--sweep",
        type=Path,
        default=Path("configs/sweep_full.yaml"),
        help="Path to existing sweep YAML to merge into (default: configs/sweep_full.yaml).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the sweep YAML in place. Default is dry-run (print to stdout).",
    )
    ap.add_argument("--base-episodes", type=int, default=DEFAULT_BASE_EPISODES)
    ap.add_argument("--base-seeds", type=int, default=DEFAULT_BASE_SEEDS)
    args = ap.parse_args()

    if not args.calibration_json.exists():
        print(f"error: {args.calibration_json} not found", file=sys.stderr)
        return 2
    if not args.sweep.exists():
        print(f"error: {args.sweep} not found", file=sys.stderr)
        return 2

    with args.calibration_json.open() as f:
        calib = json.load(f)
    with args.sweep.open() as f:
        existing = yaml.safe_load(f)

    overrides, seeds_warnings = build_overrides(
        calib["cells"],
        base_episodes=args.base_episodes,
        base_seeds=args.base_seeds,
    )
    merged = merge_into_sweep(existing, overrides, calib["cells"])

    yaml_out = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)

    if args.apply:
        args.sweep.write_text(yaml_out)
        print(f"wrote {args.sweep}", file=sys.stderr)
    else:
        sys.stdout.write(yaml_out)

    if seeds_warnings:
        print(
            f"\nwarning: {len(seeds_warnings)} cells recommended fewer seeds than default; "
            "run_sweep.py does not yet honor seeds_subset. Trim configs/sweep_full.yaml "
            "manually or wait for follow-up:",
            file=sys.stderr,
        )
        for p, e, s in seeds_warnings:
            print(f"    {p:30s} x {e:25s}  recommended seeds={s}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
