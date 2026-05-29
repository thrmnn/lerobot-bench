#!/usr/bin/env python3
"""Shared boilerplate for the v1.0.1 audit probe scripts.

Both probes (`probe_act_temporal_ensemble.py`, `probe_smolvla_libero_canonical_cap.py`)
share the same skeleton: repo-root/sys.path setup, registry loads, output-dir
creation, parquet-clear-on-restart, the per-seed `run_cell_from_specs` loop, and
a `summary.json` write. The distinct logic is the override each applies (ACT
monkeypatches the config; smolvla replaces the env spec) and a couple of extra
summary fields. This module factors out the common parts; the probes own only
their override and their summary schema.

The `summary.json` schema is consumed by `docs/PROBE_RESULTS_V1.0.1.md` — the
probes build the dict and pass it to `write_summary` verbatim so the on-disk
format is unchanged by this refactor.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_bench.checkpointing import append_cell_rows  # noqa: E402
from lerobot_bench.envs import EnvRegistry, EnvSpec  # noqa: E402
from lerobot_bench.eval import CellResult, run_cell_from_specs  # noqa: E402
from lerobot_bench.policies import PolicyRegistry, PolicySpec  # noqa: E402

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_EPISODES_PER_SEED = 50


@dataclass
class ProbeContext:
    """Resolved paths + registry specs for a single probe run."""

    logger: logging.Logger
    policy_spec: PolicySpec
    env_spec: EnvSpec
    out_dir: Path
    out_parquet: Path
    videos_dir: Path
    summary_json: Path


def setup_probe(probe_name: str, *, policy_name: str, env_name: str) -> ProbeContext:
    """Configure logging, create the output tree (clearing any stale parquet so
    a re-run starts from episode 0), and resolve the policy + env specs from the
    canonical YAML registries.

    The returned `env_spec` is the unmodified registry spec; probes that need an
    override (e.g. `dataclasses.replace(ctx.env_spec, max_steps=600)`) apply it
    after this call and pass the result to `run_seeds`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(probe_name)

    out_dir = REPO_ROOT / "results" / "probes" / probe_name
    out_parquet = out_dir / "results.parquet"
    videos_dir = out_dir / "videos"
    summary_json = out_dir / "summary.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    if out_parquet.exists():
        out_parquet.unlink()
        logger.info("removed stale parquet %s", out_parquet)

    policy_reg = PolicyRegistry.from_yaml(REPO_ROOT / "configs" / "policies.yaml")
    env_reg = EnvRegistry.from_yaml(REPO_ROOT / "configs" / "envs.yaml")

    return ProbeContext(
        logger=logger,
        policy_spec=policy_reg.get(policy_name),
        env_spec=env_reg.get(env_name),
        out_dir=out_dir,
        out_parquet=out_parquet,
        videos_dir=videos_dir,
        summary_json=summary_json,
    )


def run_seeds(
    ctx: ProbeContext,
    *,
    env_spec: EnvSpec | None = None,
    on_cell: Callable[[int, CellResult], None] | None = None,
    seeds: Sequence[int] = SEEDS,
    n_episodes_per_seed: int = N_EPISODES_PER_SEED,
) -> dict[int, float]:
    """Run `run_cell_from_specs` for each seed, appending rows incrementally and
    returning `{seed: success_rate}`.

    `env_spec` overrides the context's spec (smolvla cap-600 probe passes a
    replaced spec). `on_cell` is an optional per-cell callback the probe uses to
    capture extra stats (smolvla counts cap-hits) without re-running the cell.
    """
    spec = env_spec if env_spec is not None else ctx.env_spec
    cell_rates: dict[int, float] = {}
    for seed in seeds:
        ctx.logger.info("probe seed=%d starting", seed)
        cell = run_cell_from_specs(
            ctx.policy_spec,
            spec,
            seed_idx=seed,
            n_episodes=n_episodes_per_seed,
            device="cuda",
            record_video=True,
            videos_dir=ctx.videos_dir,
        )
        cell_rates[seed] = cell.success_rate
        append_cell_rows(ctx.out_parquet, cell.to_rows())
        ctx.logger.info("probe seed=%d success_rate=%.4f", seed, cell.success_rate)
        if on_cell is not None:
            on_cell(seed, cell)
    return cell_rates


def write_summary(ctx: ProbeContext, summary: Mapping[str, Any]) -> None:
    """Persist the probe's `summary.json` verbatim.

    The probe owns the full schema (it varies per probe — smolvla carries
    cap-hit fields ACT does not); this just serialises what it's handed so the
    format `docs/PROBE_RESULTS_V1.0.1.md` reads stays under the probe's control.
    """
    ctx.summary_json.write_text(json.dumps(dict(summary), indent=2))
    ctx.logger.info("summary -> %s", ctx.summary_json)
