"""Tests for ``lerobot_bench.render``.

These run on a synthetic 30-frame deterministic noise stack and stay
fast (no sim env, no GPU). Outputs go to ``tmp_path`` only — the
git-push hook blocks writes to ``results/`` and stray ``*.mp4`` files in
the worktree.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from numpy.typing import NDArray

from lerobot_bench import render
from lerobot_bench.render import (
    MAX_BYTES,
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
