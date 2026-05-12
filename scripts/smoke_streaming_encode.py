#!/usr/bin/env python3
"""Synthetic smoke: prove the streaming MP4 encode bounds peak RSS.

Recreates the OOM scenario from the live ``make sweep`` cgroup-kill
without needing lerobot / gym-aloha / torch installed. We run
:func:`lerobot_bench.eval.run_cell` against a fake env that returns
realistically-sized frames (480x640x3 uint8 = ~921 KB each) over many
steps. With the streaming encode contract:

  * memory peak should be ~one episode's frames at a time
    (200 frames * 921 KB = ~184 MB) plus the encoder's scratch
  * NOT (n_episodes * frames_per_episode * 921 KB), which is what the
    pre-fix code did and what crashed the live sweep at 18 GB.

Run under :file:`scripts/run_capped.sh` with a cap well below the
pre-fix peak. The script prints the post-cell RSS so the operator can
confirm the contract held. Exit 0 = success; the cgroup itself OOM-kills
on regression.

Usage::

    scripts/run_capped.sh 4G -- python scripts/smoke_streaming_encode.py

The default args (10 episodes x 200 frames @ 480x640x3) mimic the
shape of an aloha cell at roughly 1/5 scale, so a regression would
need ~3.7 GB to buffer the cell and would be clearly out-of-bounds at
the 4 GB cgroup cap.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class _BigFrameEnv:
    """Fake gym-like env: emits ``frame_h * frame_w`` uint8 RGB frames per step."""

    def __init__(self, *, frame_h: int, frame_w: int, max_steps: int) -> None:
        self._h = frame_h
        self._w = frame_w
        self._max = max_steps
        self._step = 0
        self.action_space = type("S", (), {"shape": (7,)})()

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        self._step = 0
        return {"obs": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del action
        self._step += 1
        truncated = self._step >= self._max
        return {"obs": np.zeros(4, dtype=np.float32)}, 0.0, False, truncated, {}

    def render(self) -> np.ndarray:
        # Distinct content each call so the encoder cannot trivially
        # de-dup; fills the rendered frame uniformly with the step
        # counter (sufficient noise floor for libx264).
        return np.full((self._h, self._w, 3), self._step % 256, dtype=np.uint8)

    def close(self) -> None:
        return None


class _ZeroPolicy:
    def __init__(self, action_shape: tuple[int, ...] = (7,)) -> None:
        self._a = np.zeros(action_shape, dtype=np.float32)

    def __call__(self, _obs: Any) -> np.ndarray:
        return self._a.copy()

    def reset(self) -> None:
        return None


def _rss_mb() -> float:
    """Peak resident set size in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(prog="smoke-streaming-encode")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--frame-h", type=int, default=480)
    parser.add_argument("--frame-w", type=int, default=640)
    args = parser.parse_args()

    frame_bytes = args.frame_h * args.frame_w * 3
    per_episode_bytes = frame_bytes * (args.max_steps + 1)  # +1 reset frame
    cell_buffered_gb = per_episode_bytes * args.n_episodes / 1024**3

    print(
        f"[smoke] config: {args.n_episodes} eps x {args.max_steps} steps "
        f"@ {args.frame_h}x{args.frame_w}x3"
    )
    print(
        f"[smoke] per-frame ~{frame_bytes / 1024:.0f} KB, per-episode ~"
        f"{per_episode_bytes / 1024**2:.0f} MB"
    )
    print(f"[smoke] pre-fix worst case (whole cell buffered): ~{cell_buffered_gb:.2f} GB")

    from lerobot_bench.envs import EnvSpec
    from lerobot_bench.eval import run_cell

    env_spec = EnvSpec(
        name="smoke_env",
        family="smoke",
        gym_id="Smoke-v0",
        max_steps=args.max_steps,
        success_threshold=0.5,
        lerobot_module="smoke",
    )

    env = _BigFrameEnv(frame_h=args.frame_h, frame_w=args.frame_w, max_steps=args.max_steps)
    policy = _ZeroPolicy()

    with tempfile.TemporaryDirectory(prefix="streaming-smoke-") as td:
        videos_dir = Path(td) / "videos"
        rss_before = _rss_mb()
        print(f"[smoke] RSS before run_cell: {rss_before:.0f} MB")

        result = run_cell(
            policy,
            env,
            policy_name="smoke",
            env_spec=env_spec,
            seed_idx=0,
            n_episodes=args.n_episodes,
            record_video=True,
            videos_dir=videos_dir,
        )

        rss_after = _rss_mb()
        n_mp4s = len(list(videos_dir.glob("*.mp4")))
        # Every successful episode must have frames=() and a path on disk.
        leaks = sum(1 for ep in result.episodes if len(ep.frames) > 0)
        with_video = sum(1 for ep in result.episodes if ep.video_path is not None)

    print(f"[smoke] RSS after run_cell:  {rss_after:.0f} MB")
    print(f"[smoke] peak delta:           {rss_after - rss_before:+.0f} MB")
    print(f"[smoke] mp4s on disk:         {n_mp4s}")
    print(f"[smoke] episodes with frames left in memory: {leaks}  (expect 0)")
    print(f"[smoke] episodes with video_path set:        {with_video}")

    if leaks > 0:
        print("[smoke] FAIL: streaming-encode contract broken (frames not dropped)")
        return 1
    if n_mp4s != args.n_episodes:
        print(f"[smoke] FAIL: expected {args.n_episodes} MP4s on disk, got {n_mp4s}")
        return 1
    if with_video != args.n_episodes:
        print(
            f"[smoke] FAIL: expected {args.n_episodes} EpisodeResults with "
            f"video_path, got {with_video}"
        )
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
