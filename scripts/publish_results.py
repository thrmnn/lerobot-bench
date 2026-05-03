#!/usr/bin/env python3
"""Push a sweep's parquet + MP4 corpus + manifest to a public HF Hub dataset.

The last hop in the bench stack: :mod:`scripts.run_sweep` produces
``results.parquet`` + ``sweep_manifest.json`` + a per-cell MP4 corpus
under ``videos/``; this script bundles those artefacts and uploads them
to ``thrmnn/lerobot-bench-results-v1`` so the public Space can read
from the Hub instead of the dev box.

**Idempotence.** ``HfApi.upload_folder`` is content-addressed: re-running
this script with the same inputs produces a single Hub commit (or zero,
if every file already matches by SHA). We do NOT implement a custom
diffing layer -- the Hub already does it correctly. The local
``_provenance.json`` we write is purely an audit trail; its
``published_utc`` field is the only thing that changes between runs
with otherwise-identical inputs.

**Mockability.** The whole ``huggingface_hub`` import is hidden behind
a module-level :data:`_get_hf_api` callable so tests can inject a fake
``HfApi`` without touching the network. Same AST contract as
:mod:`scripts.run_one` and :mod:`scripts.run_sweep`: no top-level
``huggingface_hub`` import.

**The "last line of defense" rule.** ``render.py`` already caps each
MP4 at 2 MiB via the encoder ladder. This script enforces a separate
``--max-video-mib`` cap as belt-and-braces -- if a clip somehow slips
past the encoder cap (e.g. someone hand-edited the videos dir), it is
logged + skipped, not uploaded. The publish step is the last gate
before bytes hit the Hub.

Usage::

    python scripts/publish_results.py \\
        --results-path results/sweep-YYYYMMDD/results.parquet \\
        --manifest-path results/sweep-YYYYMMDD/sweep_manifest.json \\
        --videos-dir results/sweep-YYYYMMDD/videos
    python scripts/publish_results.py --results-path ... --manifest-path ... \\
        --videos-dir ... --hub-repo someone/other-bench-results
    python scripts/publish_results.py --results-path ... --manifest-path ... \\
        --videos-dir ... --dry-run
    python scripts/publish_results.py --results-path ... --manifest-path ... \\
        --videos-dir ... --skip-videos

Exit codes:
    0  success -- upload happened (or dry-run logged the plan)
    2  partial -- some videos were skipped (size cap, missing); manifest
       lists which. Parquet + sweep_manifest still pushed.
    3  inputs missing or schema-malformed
    4  HF Hub auth missing or dataset doesn't exist
    5  hard failure mid-upload (network / rate limit / etc.)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot_bench.checkpointing import RESULT_SCHEMA, load_results

logger = logging.getLogger("publish-results")

# --------------------------------------------------------------------- #
# Defaults                                                              #
# --------------------------------------------------------------------- #

DEFAULT_HUB_REPO = "thrmnn/lerobot-bench-results-v1"
DEFAULT_REVISION = "main"
DEFAULT_MAX_VIDEO_MIB = 2.0  # mirrors the render ladder cap in DESIGN.md

# Glob patterns we ship to the Hub. Anything outside this set is left
# behind on disk -- the publish step is opt-in per file class. Order
# matters only for human-readability; HfApi treats these as a set.
ALLOW_PATTERNS_FULL: tuple[str, ...] = (
    "results.parquet",
    "sweep_manifest.json",
    "_provenance.json",
    "videos/*.mp4",
)
# When the operator passes --skip-videos.
ALLOW_PATTERNS_NO_VIDEOS: tuple[str, ...] = (
    "results.parquet",
    "sweep_manifest.json",
    "_provenance.json",
)


# --------------------------------------------------------------------- #
# HfApi injection point                                                 #
# --------------------------------------------------------------------- #


def _default_get_hf_api() -> Any:
    """Lazy-import ``huggingface_hub.HfApi`` and return an instance.

    Lazy on purpose: importing this module must not pull
    ``huggingface_hub`` into ``sys.modules``. Tests replace
    :data:`_get_hf_api` with a factory returning a ``MagicMock`` so the
    upload path is exercised without ever touching the network.
    """
    from huggingface_hub import HfApi  # local import: AST guard

    return HfApi()


# Module-level injection point. Tests do
# ``monkeypatch.setattr(publish_results, "_get_hf_api", lambda: fake)``
# to drive ``run_publish`` without spawning a real client.
_get_hf_api: Callable[[], Any] = _default_get_hf_api


# --------------------------------------------------------------------- #
# Outcome dataclass                                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PublishOutcome:
    """What :func:`run_publish` observed before exiting.

    Mirrors :class:`scripts.run_one.RunOneOutcome` in shape so tests
    can assert against a structured object instead of parsing
    stdout/stderr. ``staging_dir`` is the local scratch directory we
    handed to ``HfApi.upload_folder`` -- non-None even on dry-run, so
    the operator can inspect what *would* have been pushed.
    """

    exit_code: int
    n_cells: int
    n_episodes: int
    n_videos_uploaded: int
    n_videos_skipped: int
    total_video_bytes: int
    hub_repo: str
    revision: str
    staging_dir: Path | None
    log_message: str
    skipped_videos: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _git_sha() -> str:
    """Best-effort git SHA. Mirrors the helper in :mod:`scripts.run_one`."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _lerobot_version() -> str:
    """Lazy lookup of ``lerobot.__version__``. Returns ``"unknown"`` on miss.

    Same pattern as :mod:`scripts.run_sweep` -- string sentinel so the
    provenance JSON field stays typed as ``str``.
    """
    try:
        import lerobot
    except ImportError:
        return "unknown"
    version = getattr(lerobot, "__version__", None)
    return str(version) if version is not None else "unknown"


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _mib_to_bytes(mib: float) -> int:
    return int(mib * 1024 * 1024)


# --------------------------------------------------------------------- #
# Pre-flight                                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PreflightResult:
    """Structured pre-flight summary. ``error`` non-None means we exit 3."""

    n_cells: int
    n_episodes: int
    referenced_video_shas: tuple[str, ...]
    error: str | None = None


def _preflight(
    *,
    results_path: Path,
    manifest_path: Path,
    videos_dir: Path,
    skip_videos: bool,
) -> _PreflightResult:
    """Validate parquet schema, manifest readable, MP4s referenced exist.

    Returns a :class:`_PreflightResult` whose ``error`` field is the
    one-line message we'll print on exit-3. ``None`` means safe to
    proceed.

    Strategy:
      1. parquet exists and matches :data:`RESULT_SCHEMA` exactly
         (schema-drift is the kind of failure that wastes a sweep night
         if it surfaces later).
      2. manifest JSON parses.
      3. unless ``skip_videos``, every non-empty ``video_sha256`` row
         must have its corresponding MP4 on disk under ``videos_dir``.
         The MP4 filename is derived from the parquet row's
         ``(policy, env, seed, episode_index)`` -- the canonical naming
         documented in :func:`scripts.run_one.render_episodes_to_videos`.
    """
    if not results_path.exists():
        return _PreflightResult(
            n_cells=0,
            n_episodes=0,
            referenced_video_shas=(),
            error=f"results parquet not found: {results_path}",
        )

    try:
        df = load_results(results_path)
    except ValueError as exc:
        return _PreflightResult(
            n_cells=0,
            n_episodes=0,
            referenced_video_shas=(),
            error=f"results parquet schema mismatch: {exc}",
        )

    # Sanity: schema columns line up. ``load_results`` already enforces
    # this; the extra assertion guards against a future refactor where
    # the loader is loosened.
    if set(df.columns) != set(RESULT_SCHEMA):
        return _PreflightResult(
            n_cells=0,
            n_episodes=0,
            referenced_video_shas=(),
            error=(
                f"parquet schema drift: expected {sorted(RESULT_SCHEMA)}, got {sorted(df.columns)}"
            ),
        )

    if not manifest_path.exists():
        return _PreflightResult(
            n_cells=0,
            n_episodes=0,
            referenced_video_shas=(),
            error=f"manifest not found: {manifest_path}",
        )
    try:
        json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return _PreflightResult(
            n_cells=0,
            n_episodes=0,
            referenced_video_shas=(),
            error=f"manifest unreadable: {manifest_path}: {exc}",
        )

    n_cells = df[["policy", "env", "seed"]].drop_duplicates().shape[0] if len(df) else 0
    n_episodes = len(df)

    # Collect non-empty video shas to surface in provenance + verify
    # against on-disk MP4s.
    referenced_shas = tuple(s for s in df["video_sha256"].tolist() if s)

    if not skip_videos and len(df):
        # Every non-empty video_sha256 row must have its MP4 on disk.
        missing: list[str] = []
        for _, row in df.iterrows():
            sha = row["video_sha256"]
            if not sha:
                continue
            expected = videos_dir / _video_filename(
                policy=str(row["policy"]),
                env=str(row["env"]),
                seed=int(row["seed"]),
                episode_index=int(row["episode_index"]),
            )
            if not expected.exists():
                missing.append(str(expected))
            if len(missing) >= 5:  # cap message length
                break
        if missing:
            return _PreflightResult(
                n_cells=n_cells,
                n_episodes=n_episodes,
                referenced_video_shas=referenced_shas,
                error=(f"video_sha256 references missing MP4s (first {len(missing)}): {missing}"),
            )

    return _PreflightResult(
        n_cells=n_cells,
        n_episodes=n_episodes,
        referenced_video_shas=referenced_shas,
        error=None,
    )


def _video_filename(*, policy: str, env: str, seed: int, episode_index: int) -> str:
    """Mirror the naming scheme in ``scripts/run_one.render_episodes_to_videos``.

    Kept in lock-step with that helper -- if the publish step ever
    drifts from the renderer, half the leaderboard's videos will 404.
    Documented as a hard contract in DESIGN.md § Architecture sketch.
    """
    return f"{policy}__{env}__seed{seed}__ep{episode_index:03d}.mp4"


# --------------------------------------------------------------------- #
# Auth check                                                            #
# --------------------------------------------------------------------- #


def _check_hub_auth(api: Any, hub_repo: str) -> str | None:
    """Return ``None`` if auth + repo access are OK, else a short error string.

    We poke ``whoami()`` first (cheap network round-trip; fails fast on
    missing token) then ``repo_info`` to make sure the dataset exists.
    Both raise generic exceptions in real ``HfApi``; we catch the
    superset and surface a one-line message for exit 4.
    """
    try:
        api.whoami()
    except Exception as exc:
        return f"hub auth failed: {exc}"

    try:
        api.repo_info(repo_id=hub_repo, repo_type="dataset")
    except Exception as exc:
        return f"hub dataset '{hub_repo}' inaccessible: {exc}"

    return None


# --------------------------------------------------------------------- #
# Staging                                                               #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _StagedUpload:
    """Result of staging the local files into a single upload-ready dir."""

    staging_dir: Path
    n_videos_uploaded: int
    n_videos_skipped: int
    total_video_bytes: int
    skipped_videos: tuple[str, ...]


def _stage_upload(
    *,
    results_path: Path,
    manifest_path: Path,
    videos_dir: Path,
    target_dir: Path,
    skip_videos: bool,
    max_video_bytes: int,
    referenced_video_shas: tuple[str, ...],
) -> _StagedUpload:
    """Copy parquet + manifest + filtered MP4s into ``target_dir``.

    Why stage? ``HfApi.upload_folder`` glob-matches relative to the
    folder root, so we need the parquet + manifest + videos to live
    under one parent. Copying (not moving) keeps the original sweep
    directory untouched -- if the upload is interrupted, the operator
    can re-run with the same source paths.

    Per-MP4 size cap: any file over ``max_video_bytes`` is excluded
    from the staging dir (and therefore from the upload patterns).
    The render ladder already caps clips at 2 MiB; this is the last
    line of defense against an edited / mis-rendered file slipping in.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. parquet -- always required.
    shutil.copy2(results_path, target_dir / "results.parquet")

    # 2. manifest -- always required.
    shutil.copy2(manifest_path, target_dir / "sweep_manifest.json")

    # 3. videos -- optional + filtered.
    n_uploaded = 0
    n_skipped = 0
    total_bytes = 0
    skipped: list[str] = []

    if skip_videos or not videos_dir.exists():
        return _StagedUpload(
            staging_dir=target_dir,
            n_videos_uploaded=0,
            n_videos_skipped=0,
            total_video_bytes=0,
            skipped_videos=(),
        )

    target_videos_dir = target_dir / "videos"
    target_videos_dir.mkdir(parents=True, exist_ok=True)

    # We deliberately walk the on-disk dir rather than the parquet's
    # video_sha256 list -- this catches orphan MP4s (rendered but never
    # joined to a row) so the operator sees them in the skipped log.
    for mp4 in sorted(videos_dir.glob("*.mp4")):
        size = mp4.stat().st_size
        if size > max_video_bytes:
            n_skipped += 1
            skipped.append(mp4.name)
            logger.warning(
                "skipping oversized MP4 %s (%.2f MiB > cap %.2f MiB)",
                mp4.name,
                size / (1024 * 1024),
                max_video_bytes / (1024 * 1024),
            )
            continue
        shutil.copy2(mp4, target_videos_dir / mp4.name)
        n_uploaded += 1
        total_bytes += size

    # Surface orphan MP4s (not referenced in the parquet) at info level.
    # We still upload them -- the operator may want them for debugging --
    # but the log line makes the divergence visible. The reference set
    # is by SHA, so we can only check whether the *file* matches a known
    # sha; we don't recompute it here.
    if referenced_video_shas:
        logger.info(
            "%d MP4 file(s) staged; %d unique video_sha256 referenced in parquet",
            n_uploaded,
            len(set(referenced_video_shas)),
        )

    return _StagedUpload(
        staging_dir=target_dir,
        n_videos_uploaded=n_uploaded,
        n_videos_skipped=n_skipped,
        total_video_bytes=total_bytes,
        skipped_videos=tuple(skipped),
    )


def _write_provenance(
    *,
    staging_dir: Path,
    n_cells: int,
    n_episodes: int,
    total_video_bytes: int,
    hub_repo: str,
    revision: str,
    skipped_videos: tuple[str, ...],
) -> Path:
    """Write ``_provenance.json`` next to the parquet in the staging dir.

    Mirrors the manifest provenance contract (DESIGN.md § Methodology)
    but scoped to *this publish*, not the sweep itself: ``code_sha``,
    ``lerobot_version``, ``n_cells``, ``n_episodes``,
    ``total_video_bytes``, ``published_utc``, ``hub_repo``,
    ``revision``, plus the list of skipped videos so the operator can
    audit "what didn't make it" without re-reading sweep logs.

    ``published_utc`` is the only field that changes between idempotent
    re-runs. The Hub's content-addressed dedup will still produce zero
    commits if the parquet/manifest/videos haven't changed -- but a new
    ``_provenance.json`` *will* land because its bytes are different.
    Documented as a known idempotence wrinkle.
    """
    provenance = {
        "code_sha": _git_sha(),
        "lerobot_version": _lerobot_version(),
        "n_cells": n_cells,
        "n_episodes": n_episodes,
        "total_video_bytes": total_video_bytes,
        "published_utc": _now_utc(),
        "hub_repo": hub_repo,
        "revision": revision,
        "skipped_videos": list(skipped_videos),
    }
    out = staging_dir / "_provenance.json"
    out.write_text(json.dumps(provenance, indent=2, sort_keys=False) + "\n")
    return out


# --------------------------------------------------------------------- #
# Orchestration                                                         #
# --------------------------------------------------------------------- #


def run_publish(
    *,
    results_path: Path,
    manifest_path: Path,
    videos_dir: Path,
    hub_repo: str,
    revision: str,
    staging_root: Path,
    dry_run: bool,
    skip_videos: bool,
    max_video_mib: float,
    commit_message: str | None,
) -> PublishOutcome:
    """End-to-end publish. Pure orchestration; lazy-imports ``huggingface_hub``.

    Order of operations:

    1. Pre-flight (parquet schema, manifest readable, MP4 references) ->
       exit 3 on any miss.
    2. (Unless dry-run) auth check via ``HfApi.whoami`` + ``repo_info``
       -> exit 4 on miss.
    3. Stage parquet + manifest + filtered MP4s into ``staging_root``.
    4. Write ``_provenance.json`` into the staging dir.
    5. (Unless dry-run) call ``HfApi.upload_folder`` with the
       precomputed ``allow_patterns`` and the operator-provided commit
       message.

    Exit code 2 fires when the staging step skipped any MP4s for being
    oversized -- the upload still happens, the manifest records the
    delta. Exit code 5 wraps any exception out of ``upload_folder``;
    caller can inspect ``log_message`` for the underlying error.
    """
    max_video_bytes = _mib_to_bytes(max_video_mib)

    # 1. Pre-flight.
    pre = _preflight(
        results_path=results_path,
        manifest_path=manifest_path,
        videos_dir=videos_dir,
        skip_videos=skip_videos,
    )
    if pre.error is not None:
        return PublishOutcome(
            exit_code=3,
            n_cells=pre.n_cells,
            n_episodes=pre.n_episodes,
            n_videos_uploaded=0,
            n_videos_skipped=0,
            total_video_bytes=0,
            hub_repo=hub_repo,
            revision=revision,
            staging_dir=None,
            log_message=f"[publish-results] aborted: {pre.error}",
        )

    # 2. Auth (skip in dry-run -- planner stays offline).
    if not dry_run:
        api = _get_hf_api()
        err = _check_hub_auth(api, hub_repo)
        if err is not None:
            return PublishOutcome(
                exit_code=4,
                n_cells=pre.n_cells,
                n_episodes=pre.n_episodes,
                n_videos_uploaded=0,
                n_videos_skipped=0,
                total_video_bytes=0,
                hub_repo=hub_repo,
                revision=revision,
                staging_dir=None,
                log_message=(
                    f"[publish-results] aborted: {err}. "
                    "Run `huggingface-cli login` (write scope) and ensure the "
                    f"dataset {hub_repo} exists."
                ),
            )
    else:
        api = None

    # 3. Stage.
    staged = _stage_upload(
        results_path=results_path,
        manifest_path=manifest_path,
        videos_dir=videos_dir,
        target_dir=staging_root,
        skip_videos=skip_videos,
        max_video_bytes=max_video_bytes,
        referenced_video_shas=pre.referenced_video_shas,
    )

    # 4. Provenance JSON.
    _write_provenance(
        staging_dir=staged.staging_dir,
        n_cells=pre.n_cells,
        n_episodes=pre.n_episodes,
        total_video_bytes=staged.total_video_bytes,
        hub_repo=hub_repo,
        revision=revision,
        skipped_videos=staged.skipped_videos,
    )

    allow_patterns = list(ALLOW_PATTERNS_NO_VIDEOS) if skip_videos else list(ALLOW_PATTERNS_FULL)

    # Determine the partial-upload exit code BEFORE upload so dry-run
    # surfaces it consistently.
    partial = staged.n_videos_skipped > 0

    if dry_run:
        log = (
            f"[publish-results] dry-run: would upload {pre.n_episodes} episode "
            f"row(s) across {pre.n_cells} cell(s), {staged.n_videos_uploaded} "
            f"video(s) ({staged.total_video_bytes} bytes), "
            f"{staged.n_videos_skipped} skipped, to {hub_repo}@{revision}; "
            f"staging at {staged.staging_dir}"
        )
        return PublishOutcome(
            exit_code=2 if partial else 0,
            n_cells=pre.n_cells,
            n_episodes=pre.n_episodes,
            n_videos_uploaded=staged.n_videos_uploaded,
            n_videos_skipped=staged.n_videos_skipped,
            total_video_bytes=staged.total_video_bytes,
            hub_repo=hub_repo,
            revision=revision,
            staging_dir=staged.staging_dir,
            log_message=log,
            skipped_videos=staged.skipped_videos,
        )

    # 5. Upload.
    msg = commit_message or f"Publish sweep results ({_now_utc()})"
    try:
        api.upload_folder(
            repo_id=hub_repo,
            repo_type="dataset",
            folder_path=str(staged.staging_dir),
            allow_patterns=allow_patterns,
            commit_message=msg,
            revision=revision,
        )
    except Exception as exc:
        return PublishOutcome(
            exit_code=5,
            n_cells=pre.n_cells,
            n_episodes=pre.n_episodes,
            n_videos_uploaded=0,
            n_videos_skipped=staged.n_videos_skipped,
            total_video_bytes=staged.total_video_bytes,
            hub_repo=hub_repo,
            revision=revision,
            staging_dir=staged.staging_dir,
            log_message=f"[publish-results] hub upload failed mid-flight: {exc}",
            skipped_videos=staged.skipped_videos,
        )

    log = (
        f"[publish-results] uploaded {pre.n_episodes} episode row(s) across "
        f"{pre.n_cells} cell(s), {staged.n_videos_uploaded} video(s) "
        f"({staged.total_video_bytes} bytes) to {hub_repo}@{revision}"
    )
    if partial:
        log += f"; {staged.n_videos_skipped} video(s) skipped (size cap)"

    return PublishOutcome(
        exit_code=2 if partial else 0,
        n_cells=pre.n_cells,
        n_episodes=pre.n_episodes,
        n_videos_uploaded=staged.n_videos_uploaded,
        n_videos_skipped=staged.n_videos_skipped,
        total_video_bytes=staged.total_video_bytes,
        hub_repo=hub_repo,
        revision=revision,
        staging_dir=staged.staging_dir,
        log_message=log,
        skipped_videos=staged.skipped_videos,
    )


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="publish-results",
        description=(
            "Push a sweep's results.parquet + sweep_manifest.json + videos/*.mp4 "
            "to a public HF Hub dataset. Idempotent: re-running with identical "
            "inputs yields zero or one Hub commits via content-addressed dedup."
        ),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to the sweep's results.parquet.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the sweep's sweep_manifest.json.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing per-episode MP4s (canonical naming from run_one).",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=DEFAULT_HUB_REPO,
        help=f"HF Hub dataset repo (default: {DEFAULT_HUB_REPO}).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help=f"Hub branch (default: {DEFAULT_REVISION}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan + stage + write provenance; do NOT contact the Hub.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Push parquet + manifest only; leave MP4s on local disk.",
    )
    parser.add_argument(
        "--max-video-mib",
        type=float,
        default=DEFAULT_MAX_VIDEO_MIB,
        help=(
            f"Per-MP4 size cap in MiB (default: {DEFAULT_MAX_VIDEO_MIB}). "
            "Files over this size are skipped, not uploaded."
        ),
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Override the default Hub commit message.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Where to assemble the upload bundle. Default: a sibling "
            "'_publish_staging' next to the parquet."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.max_video_mib <= 0:
        print(
            f"[publish-results] aborted: --max-video-mib must be positive (got {args.max_video_mib})",
            file=sys.stderr,
        )
        return 3

    staging_root = (
        args.staging_dir
        if args.staging_dir is not None
        else args.results_path.parent / "_publish_staging"
    )

    outcome = run_publish(
        results_path=args.results_path,
        manifest_path=args.manifest_path,
        videos_dir=args.videos_dir,
        hub_repo=args.hub_repo,
        revision=args.revision,
        staging_root=staging_root,
        dry_run=args.dry_run,
        skip_videos=args.skip_videos,
        max_video_mib=args.max_video_mib,
        commit_message=args.commit_message,
    )

    stream = sys.stdout if outcome.exit_code in {0, 2} else sys.stderr
    print(outcome.log_message, file=stream)

    if outcome.exit_code not in {0, 2}:
        # One-line resume hint, mirroring run_one / run_sweep.
        print(
            f"[publish-results] resume: python scripts/publish_results.py "
            f"--results-path {args.results_path} "
            f"--manifest-path {args.manifest_path} "
            f"--videos-dir {args.videos_dir} "
            f"--hub-repo {args.hub_repo}",
            file=sys.stderr,
        )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
