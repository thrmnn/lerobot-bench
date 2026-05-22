#!/usr/bin/env python3
"""Verify one published leaderboard cell by re-running it and comparing bit-for-bit.

This is *the* trust feature of a reproducibility benchmark: anyone with the
same ``lerobot`` version, the same pinned checkpoint SHA and the same seed can
take a single ``(policy, env, seed)`` cell off the leaderboard and confirm the
per-episode outcomes are identical -- not statistically close, identical.

The seeding contract (see :doc:`docs/DESIGN.md` § Methodology) guarantees this:
a cell is deterministic given those three inputs, so the per-episode ``success``
sequence and ``n_steps`` sequence MUST match the reference exactly. A mismatch
is a real signal -- lerobot version drift, checkpoint SHA drift, or a
nondeterminism bug -- not noise to be averaged away.

What this script does, in order:

1. Runs the requested cell via the *same code path* as ``run_one.py`` -- a
   subprocess invocation of ``scripts/run_one.py`` writing to a throwaway
   parquet. Going through the real script (rather than importing ``run_one``
   in-process) keeps the reproduce path honest: it exercises exactly what an
   operator running the sweep would exercise, exit codes included.
2. Loads the matching rows from the reference parquet
   (``results/sweep-full/results.parquet`` by default).
3. Compares the two per-episode ``success`` boolean sequences and the two
   ``n_steps`` sequences, episode by episode.
4. Prints a verdict and exits 0 (reproduced) or non-zero (mismatch / error).

Like ``run_one.py``, the heavy imports (torch / lerobot, transitively via the
subprocess) never happen at module scope, and ``pandas`` is imported lazily
inside the comparison functions so ``--help`` and the arg-parsing path stay
light.

Usage::

    python scripts/reproduce_cell.py --policy act --env pusht --seed 0
    python scripts/reproduce_cell.py --policy diffusion_policy --env pusht \\
        --seed 2 --reference results/sweep-full/results.parquet
    python scripts/reproduce_cell.py --policy act --env pusht --seed 0 --dry-run

Exit codes:
    0  REPRODUCED -- every episode's success + n_steps matched the reference
    1  MISMATCH -- at least one episode diverged from the reference
    2  reference parquet missing, or the cell is absent from it
    3  the re-run itself failed (run_one.py exited non-zero)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_REFERENCE = REPO_ROOT / "results" / "sweep-full" / "results.parquet"
DEFAULT_N_EPISODES = 50
DEFAULT_DEVICE = "cuda"
RUN_ONE = REPO_ROOT / "scripts" / "run_one.py"

# Columns of the per-episode results parquet this script touches. The reference
# sweep parquet uses the short names ``policy`` / ``env`` / ``seed``.
KEY_COLS = ("policy", "env", "seed")
EP_COL = "episode_index"
COMPARE_COLS = ("success", "n_steps")


# --------------------------------------------------------------------- #
# Verdict dataclass                                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Divergence:
    """One episode where the re-run disagreed with the reference."""

    episode_index: int
    column: str
    reference_value: object
    reproduced_value: object


@dataclass(frozen=True)
class CompareResult:
    """Outcome of comparing a re-run cell against the reference cell.

    ``reproduced`` is True iff every episode's ``success`` and ``n_steps``
    matched. ``first_divergence`` is the earliest disagreement (by episode
    index, then by column order) and is ``None`` exactly when reproduced.
    """

    reproduced: bool
    n_episodes: int
    first_divergence: Divergence | None
    n_divergent_episodes: int


# --------------------------------------------------------------------- #
# Comparison core (pure, torch-free, unit-tested)                       #
# --------------------------------------------------------------------- #


def _sorted_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Episode-ordered view of a single cell's rows.

    Sorting by ``episode_index`` makes the comparison robust to parquet row
    order -- ``run_one.py`` appends in episode order, but the reference may
    have been merged from multiple shards.
    """
    return df.sort_values(EP_COL).reset_index(drop=True)


def compare_cells(reference: pd.DataFrame, reproduced: pd.DataFrame) -> CompareResult:
    """Compare two single-cell frames episode-by-episode, exactly.

    Both frames must already be filtered to one ``(policy, env, seed)`` cell.
    The episode-index *sets* must match -- a differing episode count is itself
    a divergence (reported as the first missing/extra episode). Within a shared
    episode, ``success`` and ``n_steps`` must be bit-identical.

    Raises:
        ValueError: if either frame is empty (caller should have caught the
            missing-cell case before calling this).
    """
    if reference.empty or reproduced.empty:
        raise ValueError("compare_cells requires non-empty reference and reproduced frames")

    ref = _sorted_cell(reference)
    rep = _sorted_cell(reproduced)

    ref_eps = list(ref[EP_COL])
    rep_eps = list(rep[EP_COL])

    # Episode-set mismatch: report the first index that is in one but not the
    # other. This catches a re-run that produced a different episode count.
    if ref_eps != rep_eps:
        all_eps = sorted(set(ref_eps) | set(rep_eps))
        for ep in all_eps:
            in_ref = ep in ref_eps
            in_rep = ep in rep_eps
            if in_ref != in_rep:
                return CompareResult(
                    reproduced=False,
                    n_episodes=len(all_eps),
                    first_divergence=Divergence(
                        episode_index=ep,
                        column=EP_COL,
                        reference_value="present" if in_ref else "absent",
                        reproduced_value="present" if in_rep else "absent",
                    ),
                    n_divergent_episodes=sum(
                        1 for e in all_eps if (e in ref_eps) != (e in rep_eps)
                    ),
                )

    first: Divergence | None = None
    divergent_eps: set[int] = set()
    for i in range(len(ref)):
        ep = int(cast("int", _scalar(ref.at[i, EP_COL])))
        for col in COMPARE_COLS:
            ref_val = ref.at[i, col]
            rep_val = rep.at[i, col]
            if not _values_equal(ref_val, rep_val):
                divergent_eps.add(ep)
                if first is None:
                    first = Divergence(
                        episode_index=ep,
                        column=col,
                        reference_value=_scalar(ref_val),
                        reproduced_value=_scalar(rep_val),
                    )

    return CompareResult(
        reproduced=first is None,
        n_episodes=len(ref),
        first_divergence=first,
        n_divergent_episodes=len(divergent_eps),
    )


def _values_equal(a: object, b: object) -> bool:
    """Exact equality for the scalar cell values we compare.

    ``success`` is a bool and ``n_steps`` an int; numpy scalar wrappers from
    pandas compare correctly under ``==`` but we coerce to native types first
    so a numpy ``bool_`` and a python ``bool`` never disagree spuriously.
    """
    return _scalar(a) == _scalar(b)


def _scalar(v: object) -> object:
    """Coerce a numpy/pandas scalar to a native python value for clean printing."""
    item = getattr(v, "item", None)
    if callable(item):
        return item()
    return v


def select_cell(df: pd.DataFrame, *, policy: str, env: str, seed: int) -> pd.DataFrame:
    """Rows of ``df`` for exactly one ``(policy, env, seed)`` cell.

    Raises:
        KeyError: if a required key column is absent from the frame.
    """
    missing = [c for c in (*KEY_COLS, EP_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"results frame is missing required column(s): {', '.join(missing)}; "
            f"have: {', '.join(map(str, df.columns))}"
        )
    mask = (df["policy"] == policy) & (df["env"] == env) & (df["seed"] == seed)
    return df.loc[mask].copy()


# --------------------------------------------------------------------- #
# CELL string parsing (shared with the Makefile `reproduce` target)     #
# --------------------------------------------------------------------- #


def parse_cell(cell: str) -> tuple[str, str, int]:
    """Parse a ``policy/env/seed`` string into its three parts.

    This is the helper the ``make reproduce CELL=...`` target leans on. It is
    intentionally strict: exactly two slashes, a non-empty policy and env, and
    an integer seed >= 0. A vague ``CELL`` is an operator typo worth catching
    loudly before a 15-minute re-run starts.

    Raises:
        ValueError: on any malformed input, with a message naming the expected
            ``policy/env/seed`` shape.
    """
    parts = cell.split("/")
    if len(parts) != 3:
        raise ValueError(f"CELL must be 'policy/env/seed' (exactly two slashes); got {cell!r}")
    policy, env, seed_str = (p.strip() for p in parts)
    if not policy or not env:
        raise ValueError(f"CELL has an empty policy or env: {cell!r}")
    try:
        seed = int(seed_str)
    except ValueError:
        raise ValueError(f"CELL seed must be an integer; got {seed_str!r}") from None
    if seed < 0:
        raise ValueError(f"CELL seed must be >= 0; got {seed}")
    return policy, env, seed


# --------------------------------------------------------------------- #
# Re-run via run_one.py subprocess                                      #
# --------------------------------------------------------------------- #


def rerun_cell(
    *,
    policy: str,
    env: str,
    seed: int,
    n_episodes: int,
    device: str,
    out_parquet: Path,
) -> int:
    """Re-run one cell by shelling out to ``scripts/run_one.py``.

    Video rendering is skipped (``--no-record-video``) -- the reproducibility
    check is over ``success`` / ``n_steps``, and skipping the H.264 encode
    keeps the verify run lighter than the original sweep cell. Returns the
    ``run_one.py`` exit code; the caller maps a non-zero code to exit 3.

    The subprocess inherits stdout/stderr so the operator sees ``run_one``'s
    own progress and resume hints live.
    """
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--policy",
        policy,
        "--env",
        env,
        "--seed",
        str(seed),
        "--n-episodes",
        str(n_episodes),
        "--out-parquet",
        str(out_parquet),
        "--no-record-video",
        "--device",
        device,
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return completed.returncode


# --------------------------------------------------------------------- #
# Verdict rendering                                                     #
# --------------------------------------------------------------------- #

_MISMATCH_HINTS = (
    "likely causes:",
    "  - lerobot version drift: the reference was produced with a different",
    "    lerobot release. Confirm `pip show lerobot` reports 0.5.1.",
    "  - checkpoint SHA drift: the policy's pinned revision_sha in",
    "    configs/policies.yaml differs from the one the reference used.",
    "  - nondeterminism: a code path that escaped the seeding contract",
    "    (see docs/DESIGN.md § Methodology -> Seeding contract).",
)


def format_verdict(cell_key: str, result: CompareResult) -> str:
    """Human-readable verdict block for stdout.

    On a match: a single ``REPRODUCED`` line. On a mismatch: the first
    divergent episode with both values, the divergent-episode count, and the
    likely-cause hint list -- everything an operator needs to triage without
    re-reading the docs.
    """
    n = result.n_episodes
    if result.reproduced:
        return f"REPRODUCED ✓  {cell_key}  ({n}/{n} episodes identical)"

    div = result.first_divergence
    assert div is not None  # not-reproduced always carries a divergence
    lines = [
        f"MISMATCH ✗  {cell_key}  ({result.n_divergent_episodes}/{n} episodes diverged)",
        f"  first divergence at episode {div.episode_index}, column '{div.column}':",
        f"    reference   = {div.reference_value!r}",
        f"    reproduced  = {div.reproduced_value!r}",
        "",
        *_MISMATCH_HINTS,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Orchestration                                                         #
# --------------------------------------------------------------------- #


def reproduce(
    *,
    policy: str,
    env: str,
    seed: int,
    reference: Path,
    n_episodes: int,
    device: str,
    dry_run: bool,
) -> int:
    """Run + compare one cell. Returns the process exit code.

    Reference-side checks happen first (fail fast on exit 2 before any
    expensive re-run), then -- unless ``dry_run`` -- the cell is re-run and
    compared.
    """
    cell_key = f"{policy}/{env}/seed{seed}"

    # Lazy: pandas is the only heavy-ish import and the help path skips it.
    import pandas as pd

    if not reference.exists():
        print(
            f"ERROR: reference parquet not found: {reference}\n"
            "  Fetch the published results first, e.g.:\n"
            "    huggingface-cli download thrmnn/lerobot-bench-results-v1 "
            "--repo-type dataset --local-dir results/sweep-full",
            file=sys.stderr,
        )
        return 2

    ref_df = pd.read_parquet(reference)
    try:
        ref_cell = select_cell(ref_df, policy=policy, env=env, seed=seed)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if ref_cell.empty:
        print(
            f"ERROR: cell {cell_key} is not present in the reference parquet "
            f"{reference}.\n"
            "  Check `--policy/--env/--seed` against the leaderboard; the "
            "reference only contains cells that were swept.",
            file=sys.stderr,
        )
        return 2

    ref_n = len(ref_cell)

    if dry_run:
        print(
            f"[reproduce] dry-run: cell {cell_key} found in reference "
            f"({ref_n} episodes). Would re-run via run_one.py and compare "
            f"success + n_steps bit-for-bit. No torch import, no cell run."
        )
        return 0

    # Re-run into a throwaway parquet so the verify never mutates the
    # reference or any sweep output on disk.
    with tempfile.TemporaryDirectory(prefix="reproduce-cell-") as tmpdir:
        out_parquet = Path(tmpdir) / "rerun.parquet"
        print(f"[reproduce] re-running {cell_key} ({ref_n} episodes) ...")
        rc = rerun_cell(
            policy=policy,
            env=env,
            seed=seed,
            n_episodes=ref_n,
            device=device,
            out_parquet=out_parquet,
        )
        # run_one exit 2 = cell ran but some episodes errored -- rows still
        # landed, so the comparison is still meaningful. Any other non-zero
        # means no usable rows.
        if rc not in (0, 2):
            print(
                f"ERROR: the re-run failed (run_one.py exit {rc}); "
                "cannot compare. See run_one output above for the cause.",
                file=sys.stderr,
            )
            return 3
        if not out_parquet.exists():
            print(
                f"ERROR: the re-run produced no parquet at {out_parquet} (run_one.py exit {rc}).",
                file=sys.stderr,
            )
            return 3

        rep_df = pd.read_parquet(out_parquet)
        rep_cell = select_cell(rep_df, policy=policy, env=env, seed=seed)
        result = compare_cells(ref_cell, rep_cell)

    print(format_verdict(cell_key, result))
    return 0 if result.reproduced else 1


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reproduce-cell",
        description=(
            "Re-run one (policy, env, seed) leaderboard cell and verify its "
            "per-episode success + n_steps match the published reference "
            "bit-for-bit. The seed contract makes this an exact check, not a "
            "statistical one."
        ),
    )
    parser.add_argument("--policy", type=str, required=True, help="Policy name from the registry.")
    parser.add_argument("--env", type=str, required=True, help="Env name from the registry.")
    parser.add_argument("--seed", type=int, required=True, help="Seed index (>= 0).")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Reference results parquet to compare against (default: {DEFAULT_REFERENCE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device for the re-run (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check the cell exists in the reference and report its episode "
            "count; do NOT re-run it or import torch."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return reproduce(
        policy=args.policy,
        env=args.env,
        seed=args.seed,
        reference=args.reference,
        n_episodes=DEFAULT_N_EPISODES,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
