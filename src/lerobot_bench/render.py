"""Episode-frame -> small MP4 renderer for the leaderboard Space.

Constraints (see ``docs/DESIGN.md`` § Render pipeline):

* **Per-clip cap**: ``MAX_BYTES`` = 2 MiB. We publish every clip to the
  free HF Hub dataset and the Space streams them on a free CPU tier;
  bigger files break the UX. If a re-encode exceeds the cap we delete
  the file and raise :class:`RenderSizeError` rather than silently
  truncating — the caller is expected to lower fps or raise crf.

* **Encoder**: H.264 baseline (libx264) + ``yuv420p`` for browser
  compatibility, 256x256 px @ 10 fps. ``macro_block_size=1`` is passed
  to imageio so non-multiples of 16 don't get padded; 256 is already
  even so yuv420p chroma subsampling is fine.

* **Determinism**: same input frames + same encoder settings produce a
  byte-identical MP4 in our spike (libx264 + a fixed crf + no PSNR/SSIM
  side outputs). The :data:`RenderResult.content_sha256` field captures
  the on-disk bytes so callers can detect drift if a future ffmpeg
  upgrade breaks byte-identity; the visual content (decoded array) is
  the deeper invariant.

We use :mod:`imageio.v3` end-to-end (no ``subprocess`` / ``ffmpeg``
shell-out) to keep the dep surface honest. Resizing is done by a small
pure-numpy bilinear sampler so we can avoid pulling PIL or scipy just
for this.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

DEFAULT_FPS: int = 10
DEFAULT_SIZE: int = 256
DEFAULT_CRF: int = 23
MAX_BYTES: int = 2 * 1024 * 1024  # 2 MiB

_CODEC: str = "libx264"
_PIXEL_FORMAT: str = "yuv420p"
_PNG_CODEC: str = "png"


@dataclass(frozen=True)
class EncoderSettings:
    """Encoder knobs we burn into the output filename's metadata.

    For PNG thumbnails ``fps`` and ``crf`` are zero and ``codec`` is
    ``"png"``; the dataclass is shared so callers can switch on
    ``codec`` rather than carrying a separate type.
    """

    fps: int
    size: int
    codec: str
    pixel_format: str
    crf: int


@dataclass(frozen=True)
class RenderResult:
    """Metadata returned by every renderer call.

    ``content_sha256`` is the SHA-256 of the on-disk file bytes (hex).
    Callers fold this into the parquet row's ``video_sha256`` column so
    the Space can detect cache-busting drifts.
    """

    path: Path
    bytes_written: int
    frame_count: int
    encoder_settings: EncoderSettings
    content_sha256: str


class RenderSizeError(RuntimeError):
    """Raised when an encoded clip exceeds :data:`MAX_BYTES`.

    Carries the offending size so the caller can log + decide whether
    to drop fps or raise crf and retry.
    """

    def __init__(self, path: Path, size: int, limit: int) -> None:
        super().__init__(
            f"encoded clip {path} is {size} bytes (> {limit} byte cap); "
            f"lower fps or raise crf and re-encode"
        )
        self.path = path
        self.size = size
        self.limit = limit


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


def _validate_frames(frames: NDArray[np.uint8]) -> None:
    if not isinstance(frames, np.ndarray):
        raise TypeError(f"frames must be a numpy.ndarray, got {type(frames).__name__}")
    if frames.dtype != np.uint8:
        raise TypeError(f"frames must be dtype uint8, got {frames.dtype}")
    if frames.ndim != 4:
        raise ValueError(f"frames must be 4-D (T, H, W, 3), got shape {frames.shape}")
    if frames.shape[-1] != 3:
        raise ValueError(f"frames must have 3 channels (RGB), got shape {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("frames must have at least one timestep")


# --------------------------------------------------------------------- #
# Pure-numpy bilinear resize                                            #
# --------------------------------------------------------------------- #


def _resize_bilinear(frame: NDArray[np.uint8], target_h: int, target_w: int) -> NDArray[np.uint8]:
    """Bilinear resample one ``(H, W, 3) uint8`` frame to ``(target_h, target_w, 3)``.

    Pure-numpy so we don't pull PIL or scipy. Pixel-center alignment
    matches OpenCV's default ``INTER_LINEAR`` and torchvision's
    ``antialias=False`` mode within rounding; that's good enough for a
    256-px preview.
    """
    src_h, src_w, _ = frame.shape
    if target_h == src_h and target_w == src_w:
        return frame

    y = (np.arange(target_h, dtype=np.float32) + 0.5) * src_h / target_h - 0.5
    x = (np.arange(target_w, dtype=np.float32) + 0.5) * src_w / target_w - 0.5
    y = np.clip(y, 0.0, src_h - 1.0)
    x = np.clip(x, 0.0, src_w - 1.0)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = (y - y0.astype(np.float32))[:, None, None]
    wx = (x - x0.astype(np.float32))[None, :, None]

    img = frame.astype(np.float32)
    a = img[np.ix_(y0, x0)]
    b = img[np.ix_(y0, x1)]
    c = img[np.ix_(y1, x0)]
    d = img[np.ix_(y1, x1)]

    out = a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx
    clipped: NDArray[np.uint8] = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return clipped


def _resize_stack(frames: NDArray[np.uint8], target_h: int, target_w: int) -> NDArray[np.uint8]:
    """Apply :func:`_resize_bilinear` per frame; no-op when shape already matches."""
    _, h, w, _ = frames.shape
    if h == target_h and w == target_w:
        return frames
    out = np.empty((frames.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = _resize_bilinear(frames[i], target_h, target_w)
    return out


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_episode(
    frames: NDArray[np.uint8],
    out_path: Path,
    *,
    fps: int = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    crf: int = DEFAULT_CRF,
) -> RenderResult:
    """Encode ``frames`` to a small MP4 at ``out_path`` and return metadata.

    Args:
        frames: ``(T, H, W, 3)`` uint8 RGB array. Resized to
            ``size x size`` via numpy bilinear if not already.
        out_path: Destination MP4 path. Parent directory must exist;
            this function does not create it.
        fps: Output frame rate (default 10). The eval loop typically
            renders at 10 fps regardless of sim step rate.
        size: Output square edge in pixels (default 256). Must be even
            for ``yuv420p``; 256 always satisfies that.
        crf: libx264 constant rate factor (default 23). Higher = smaller
            file, lower visual quality. Try 28 if the default exceeds
            :data:`MAX_BYTES`.

    Raises:
        ValueError / TypeError: on malformed ``frames``.
        :class:`RenderSizeError`: if the encoded file is larger than
            :data:`MAX_BYTES`. The file is removed before raising.
    """
    _validate_frames(frames)
    if size % 2 != 0:
        raise ValueError(f"size must be even (yuv420p chroma), got {size}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    resized = _resize_stack(frames, size, size)
    settings = EncoderSettings(
        fps=fps,
        size=size,
        codec=_CODEC,
        pixel_format=_PIXEL_FORMAT,
        crf=crf,
    )

    iio.imwrite(
        out_path,
        resized,
        fps=fps,
        codec=_CODEC,
        pixelformat=_PIXEL_FORMAT,
        macro_block_size=1,
        output_params=["-crf", str(crf), "-preset", "medium"],
    )

    bytes_written = out_path.stat().st_size
    # Re-read the cap each call so monkeypatched-in-tests values are honoured.
    cap = _current_max_bytes()
    if bytes_written > cap:
        out_path.unlink(missing_ok=True)
        raise RenderSizeError(out_path, bytes_written, cap)

    return RenderResult(
        path=out_path,
        bytes_written=bytes_written,
        frame_count=int(resized.shape[0]),
        encoder_settings=settings,
        content_sha256=_sha256_file(out_path),
    )


def render_thumbnail_strip(
    frames: NDArray[np.uint8],
    out_path: Path,
    *,
    n_thumbs: int = 6,
    thumb_size: int = 96,
) -> RenderResult:
    """Write a horizontal PNG strip of ``n_thumbs`` evenly-spaced frames.

    If ``frames`` has fewer than ``n_thumbs`` timesteps the strip uses
    every frame (capped at ``T``). Each thumb is resized to
    ``thumb_size x thumb_size``; the output PNG dimensions are
    ``(thumb_size, n_used * thumb_size, 3)``.

    The :class:`EncoderSettings` returned uses ``codec="png"`` and
    ``fps=crf=0`` — the dataclass is shared with the MP4 path so the
    leaderboard parquet schema is uniform.
    """
    _validate_frames(frames)
    if n_thumbs <= 0:
        raise ValueError(f"n_thumbs must be positive, got {n_thumbs}")
    if thumb_size <= 0:
        raise ValueError(f"thumb_size must be positive, got {thumb_size}")

    t = int(frames.shape[0])
    n_used = min(n_thumbs, t)
    # Evenly-spaced indices including endpoints when n_used > 1.
    if n_used == 1:
        idxs = np.array([0], dtype=np.int64)
    else:
        idxs = np.linspace(0, t - 1, n_used).round().astype(np.int64)

    thumbs = [_resize_bilinear(frames[i], thumb_size, thumb_size) for i in idxs]
    strip = np.concatenate(thumbs, axis=1)  # (thumb_size, n_used*thumb_size, 3)

    iio.imwrite(out_path, strip, extension=".png")

    bytes_written = out_path.stat().st_size
    cap = _current_max_bytes()
    if bytes_written > cap:
        out_path.unlink(missing_ok=True)
        raise RenderSizeError(out_path, bytes_written, cap)

    settings = EncoderSettings(
        fps=0,
        size=thumb_size,
        codec=_PNG_CODEC,
        pixel_format="rgb24",
        crf=0,
    )
    return RenderResult(
        path=out_path,
        bytes_written=bytes_written,
        frame_count=n_used,
        encoder_settings=settings,
        content_sha256=_sha256_file(out_path),
    )


def _current_max_bytes() -> int:
    """Module-level lookup so tests can monkeypatch :data:`MAX_BYTES`."""
    import lerobot_bench.render as _self  # local import to dodge cycle on init

    value = _self.MAX_BYTES
    return int(value)


__all__ = [
    "DEFAULT_CRF",
    "DEFAULT_FPS",
    "DEFAULT_SIZE",
    "MAX_BYTES",
    "EncoderSettings",
    "RenderResult",
    "RenderSizeError",
    "render_episode",
    "render_thumbnail_strip",
]
