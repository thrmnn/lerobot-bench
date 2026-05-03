"""Tests for ``lerobot_bench.render``.

These run on synthetic deterministic frame stacks and stay fast (no sim
env, no GPU). Outputs go to ``tmp_path`` only — the git-push hook
blocks writes to ``results/`` and stray ``*.mp4`` files in the worktree.

Long-input fixtures (600 / 1500 frames) are calibrated to land on
specific rungs of :data:`render.RENDER_LADDER`. The chosen synthetic
signal — a sum of sinusoids plus low-amplitude noise — is the smallest
one that pushes the encoder off rung 0 without being so chaotic that
*every* rung overshoots. See the `_make_*_motion` factories.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from numpy.typing import NDArray

from lerobot_bench import render
from lerobot_bench.render import (
    MAX_BYTES,
    RENDER_LADDER,
    EncoderSettings,
    RenderResult,
    RenderSizeError,
    render_episode,
    render_thumbnail_strip,
)

# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _make_frames(t: int = 30, h: int = 256, w: int = 256, seed: int = 0) -> NDArray[np.uint8]:
    """Deterministic uint8 RGB stack. Seeded for byte-identical re-runs."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def _make_smooth_motion(
    t: int,
    h: int = 256,
    w: int = 256,
    seed: int = 0,
    noise_amp: int = 15,
) -> NDArray[np.uint8]:
    """Calibrated long-clip synthetic: sinusoid sweeps + small per-pixel noise.

    Tunable via ``noise_amp`` to put the encoded file on a specific
    ladder rung. Determinism is preserved because every random draw
    comes from a seeded ``np.random.default_rng``.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((t, h, w, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    for i in range(t):
        phase = i * 0.3
        r = 128 + 100 * np.sin(0.05 * xx + phase) * np.cos(0.04 * yy + 1.3 * phase)
        g = 128 + 100 * np.sin(0.07 * yy - 0.7 * phase) * np.cos(0.06 * xx + phase)
        b = 128 + 100 * np.cos(0.05 * (xx + yy) + 1.7 * phase)
        frame = np.stack([r, g, b], axis=-1)
        frame += rng.integers(-noise_amp, noise_amp + 1, size=frame.shape).astype(np.float32)
        out[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------- #
# render_episode                                                        #
# --------------------------------------------------------------------- #


def test_render_episode_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "ep.mp4"
    result = render_episode(_make_frames(), out)
    assert isinstance(result, RenderResult)
    assert out.exists()
    assert out.stat().st_size > 0
    assert result.bytes_written == out.stat().st_size
    assert result.frame_count == 30
    assert isinstance(result.encoder_settings, EncoderSettings)
    assert result.encoder_settings.codec == "libx264"
    assert result.encoder_settings.pixel_format == "yuv420p"
    assert result.encoder_settings.fps == 10
    assert result.encoder_settings.size == 256
    assert result.encoder_settings.crf == 23
    # 30-frame noise comfortably fits at the default settings, so the
    # adaptive ladder must land on rung 0 — no fps drop / no quality dip.
    assert result.encoder_settings.rung_index == 0
    assert RENDER_LADDER[0] == (
        result.encoder_settings.fps,
        result.encoder_settings.crf,
    )


def test_render_episode_under_size_cap(tmp_path: Path) -> None:
    result = render_episode(_make_frames(), tmp_path / "ep.mp4")
    assert result.bytes_written <= MAX_BYTES, (
        f"30-frame ramp encoded to {result.bytes_written} B; expected ≤ {MAX_BYTES} B at crf=23"
    )


def test_render_episode_ffprobe_readable(tmp_path: Path) -> None:
    out = tmp_path / "ep.mp4"
    render_episode(_make_frames(t=30), out)
    # imageio's ffmpeg backend round-trips: shape must match what we wrote.
    arr = iio.imread(out)
    assert arr.dtype == np.uint8
    assert arr.shape[0] == 30
    assert arr.shape[1:] == (256, 256, 3)


def test_render_episode_deterministic(tmp_path: Path) -> None:
    """Same frames + same encoder settings -> identical content_sha256."""
    frames = _make_frames()
    a = render_episode(frames, tmp_path / "a.mp4")
    b = render_episode(frames, tmp_path / "b.mp4")
    assert a.content_sha256 == b.content_sha256, (
        "libx264 + fixed crf is expected to be byte-identical here. "
        "If this flakes, fall back to decode-and-compare-arrays equality."
    )
    # Bonus invariant: round-tripped pixel arrays are exactly equal.
    assert np.array_equal(iio.imread(a.path), iio.imread(b.path))


def test_render_episode_resizes_non_square_input(tmp_path: Path) -> None:
    """A (T, 96, 128, 3) input must be resampled to (T, 256, 256, 3)."""
    frames = _make_frames(t=12, h=96, w=128)
    result = render_episode(frames, tmp_path / "ep.mp4", crf=28)
    arr = iio.imread(result.path)
    assert arr.shape[1:] == (256, 256, 3)
    assert result.frame_count == 12


def test_render_episode_rejects_empty_frames(tmp_path: Path) -> None:
    empty = np.empty((0, 256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one timestep"):
        render_episode(empty, tmp_path / "ep.mp4")


def test_render_episode_rejects_wrong_dtype(tmp_path: Path) -> None:
    bad = np.zeros((4, 256, 256, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        render_episode(bad, tmp_path / "ep.mp4")  # type: ignore[arg-type]


def test_render_episode_rejects_wrong_shape(tmp_path: Path) -> None:
    no_channels = np.zeros((4, 256, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match="4-D"):
        render_episode(no_channels, tmp_path / "ep.mp4")


def test_render_episode_rejects_wrong_channel_count(tmp_path: Path) -> None:
    rgba = np.zeros((4, 256, 256, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3 channels"):
        render_episode(rgba, tmp_path / "ep.mp4")


def test_render_episode_rejects_odd_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even"):
        render_episode(_make_frames(t=4), tmp_path / "ep.mp4", size=255)


def test_render_episode_rejects_nonpositive_fps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps"):
        render_episode(_make_frames(t=4), tmp_path / "ep.mp4", fps=0)


def test_render_episode_oversized_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny cap -> RenderSizeError + cleaned-up file."""
    monkeypatch.setattr(render, "MAX_BYTES", 50)
    out = tmp_path / "ep.mp4"
    with pytest.raises(RenderSizeError) as excinfo:
        render_episode(_make_frames(), out)
    assert excinfo.value.size > 50
    assert excinfo.value.limit == 50
    assert not out.exists(), "oversized clip must be deleted before raising"


# --------------------------------------------------------------------- #
# render_thumbnail_strip                                                #
# --------------------------------------------------------------------- #


def test_render_thumbnail_strip_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "thumbs.png"
    result = render_thumbnail_strip(_make_frames(t=30), out, n_thumbs=6, thumb_size=96)
    assert out.exists()
    assert result.bytes_written == out.stat().st_size
    assert result.frame_count == 6
    assert result.encoder_settings.codec == "png"
    assert result.encoder_settings.fps == 0
    assert result.encoder_settings.crf == 0

    arr = iio.imread(out)
    assert arr.dtype == np.uint8
    assert arr.shape == (96, 6 * 96, 3)


def test_render_thumbnail_strip_n_thumbs_capped_at_T(tmp_path: Path) -> None:
    """If T < n_thumbs, the strip uses every frame (n_used == T)."""
    out = tmp_path / "thumbs.png"
    result = render_thumbnail_strip(_make_frames(t=3), out, n_thumbs=6, thumb_size=64)
    assert result.frame_count == 3
    arr = iio.imread(out)
    assert arr.shape == (64, 3 * 64, 3)


def test_render_thumbnail_strip_single_frame(tmp_path: Path) -> None:
    out = tmp_path / "thumbs.png"
    result = render_thumbnail_strip(_make_frames(t=1), out, n_thumbs=6, thumb_size=32)
    assert result.frame_count == 1
    arr = iio.imread(out)
    assert arr.shape == (32, 32, 3)


def test_render_thumbnail_strip_rejects_empty(tmp_path: Path) -> None:
    empty = np.empty((0, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one timestep"):
        render_thumbnail_strip(empty, tmp_path / "thumbs.png")


# --------------------------------------------------------------------- #
# Adaptive render ladder                                                #
# --------------------------------------------------------------------- #


def test_render_ladder_long_episode_steps_off_rung_0(tmp_path: Path) -> None:
    """600-frame moderate-entropy clip overshoots rung 0; ladder lands on 1 or 2.

    Calibrated against the actual encoder so we verify the algorithm
    end-to-end. If a future libx264 upgrade shifts bitrates this test
    might need its ``noise_amp`` retuned — but the contract being
    asserted (advance off rung 0, fit under MAX_BYTES) is stable.
    """
    frames = _make_smooth_motion(t=600, noise_amp=15)
    out = tmp_path / "ep_600.mp4"
    result = render_episode(frames, out)

    assert result.bytes_written <= MAX_BYTES, (
        f"600-frame clip after ladder walk: {result.bytes_written} B > {MAX_BYTES} B cap"
    )
    assert result.encoder_settings.rung_index in (1, 2), (
        f"expected ladder to advance to rung 1 or 2 for 600-frame moderate input, "
        f"got rung {result.encoder_settings.rung_index} "
        f"(file={result.bytes_written} B)"
    )
    expected_fps, expected_crf = RENDER_LADDER[result.encoder_settings.rung_index]
    assert result.encoder_settings.fps == expected_fps
    assert result.encoder_settings.crf == expected_crf
    assert result.frame_count == 600


def test_render_ladder_very_long_episode_steps_to_higher_rung(tmp_path: Path) -> None:
    """1500-frame clip needs a deeper rung; verify it still fits under cap."""
    frames = _make_smooth_motion(t=1500, noise_amp=20)
    out = tmp_path / "ep_1500.mp4"
    result = render_episode(frames, out)

    assert result.bytes_written <= MAX_BYTES, (
        f"1500-frame clip after ladder walk: {result.bytes_written} B > {MAX_BYTES} B cap"
    )
    # 1500 frames at this entropy needs at least rung 2 to fit; we
    # assert >= 2 rather than exact == 3 so libx264 minor-version
    # bitrate drift doesn't spuriously break us. The "higher rung"
    # contract from the spec is satisfied either way.
    assert result.encoder_settings.rung_index >= 2, (
        f"expected ladder to advance to rung 2 or 3 for 1500-frame input, "
        f"got rung {result.encoder_settings.rung_index} "
        f"(file={result.bytes_written} B)"
    )
    assert result.frame_count == 1500


def test_render_ladder_pathological_input_raises_with_full_log(tmp_path: Path) -> None:
    """1500 frames of pure white noise: every rung overshoots -> RenderSizeError."""
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(1500, 256, 256, 3), dtype=np.uint8)
    out = tmp_path / "ep_noise.mp4"

    with pytest.raises(RenderSizeError) as excinfo:
        render_episode(frames, out)

    err = excinfo.value
    assert err.limit == MAX_BYTES
    # Every rung must be reflected in the attempt log.
    assert len(err.attempts) == len(RENDER_LADDER), (
        f"expected {len(RENDER_LADDER)} ladder attempts in error, got {len(err.attempts)}"
    )
    for (lad_fps, lad_crf), (att_fps, att_crf, att_size) in zip(
        RENDER_LADDER, err.attempts, strict=True
    ):
        assert (lad_fps, lad_crf) == (att_fps, att_crf), (
            f"attempt {att_fps}/{att_crf} did not match ladder rung {lad_fps}/{lad_crf}"
        )
        assert att_size > MAX_BYTES, (
            f"rung ({att_fps}, {att_crf}) produced {att_size} B which fits — "
            "test should have succeeded, not raised"
        )
    msg = str(err)
    for fps, crf in RENDER_LADDER:
        assert f"fps={fps}" in msg and f"crf={crf}" in msg, (
            f"error message missing rung (fps={fps}, crf={crf}): {msg}"
        )
    # File must be cleaned up after the final overshoot too.
    assert not out.exists(), "oversized clip on final rung must be deleted before raising"


def test_render_ladder_long_episode_deterministic_in_subprocess(tmp_path: Path) -> None:
    """Same 600-frame input in two fresh subprocesses -> identical bytes + same rung.

    Subprocess isolation rules out any in-process cache or RNG-state
    leak as the source of byte-identity. Marked under the same fast
    bucket as the rest; ~5-7 s end-to-end on the dev box.
    """
    runner = (
        "import json, sys, numpy as np, pathlib;\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r});\n"
        "from lerobot_bench.render import render_episode;\n"
        "from tests.test_render import _make_smooth_motion;\n"
        "frames = _make_smooth_motion(t=600, noise_amp=15);\n"
        "out = pathlib.Path(sys.argv[1]);\n"
        "res = render_episode(frames, out);\n"
        "print(json.dumps({\n"
        "    'sha': res.content_sha256,\n"
        "    'rung': res.encoder_settings.rung_index,\n"
        "    'fps': res.encoder_settings.fps,\n"
        "    'crf': res.encoder_settings.crf,\n"
        "    'size': res.bytes_written,\n"
        "}))\n"
    )

    def run(out_name: str) -> dict[str, object]:
        proc = subprocess.run(
            [sys.executable, "-c", runner, str(tmp_path / out_name)],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        result: dict[str, object] = json.loads(proc.stdout.strip())
        return result

    a = run("a.mp4")
    b = run("b.mp4")
    assert a["sha"] == b["sha"], (
        f"determinism broken across fresh subprocesses: {a['sha']} != {b['sha']}"
    )
    assert a["rung"] == b["rung"], (
        f"ladder picked different rungs across fresh runs: {a['rung']} != {b['rung']}"
    )
    assert a["fps"] == b["fps"]
    assert a["crf"] == b["crf"]
    assert a["size"] == b["size"]


def test_render_ladder_explicit_overrides_bypass_ladder(tmp_path: Path) -> None:
    """Caller-supplied fps/crf must short-circuit the ladder and report rung_index=-1.

    Preserves the legacy single-shot encode for tests / callers that
    want to pin the encoder settings deterministically (e.g. the
    existing ``test_render_episode_resizes_non_square_input`` calls
    ``crf=28``). The ladder is opt-in by leaving fps/crf at default.
    """
    frames = _make_frames(t=12)
    result = render_episode(frames, tmp_path / "ep.mp4", crf=28)
    assert result.encoder_settings.rung_index == -1
    assert result.encoder_settings.crf == 28
    assert result.encoder_settings.fps == 10  # default fps still honoured
