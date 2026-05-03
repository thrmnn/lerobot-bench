"""Episode-frame -> small MP4 renderer for the leaderboard Space.

Constraints (see ``docs/DESIGN.md`` § Render pipeline):

* **Per-clip cap**: ``MAX_BYTES`` = 2 MiB. We publish every clip to the
  free HF Hub dataset and the Space streams them on a free CPU tier;
  bigger files break the UX.

* **Adaptive encoder ladder**: the default encoder settings (256 px,
  10 fps, libx264 crf=23) comfortably fit 30-300 frame clips. Real
  Aloha-style episodes can run 1000+ steps and blow the 2 MiB cap at
  those settings. :data:`RENDER_LADDER` defines a small list of
  ``(fps, crf)`` rungs the encoder walks until the encoded file fits
  under :data:`MAX_BYTES`. A successful rung's index is recorded in
  :class:`EncoderSettings` so downstream consumers (the Space, the
  publish script, the failure-taxonomy labeler) know whether to expect
  frame jumps. We *do not* drop input frames — lower fps just means the
  clip plays back faster than wall-clock; that is a documented tradeoff
  over the alternative of dropping samples (see DESIGN.md). The 256 px
  edge and ``yuv420p`` pixel format are fixed; only fps and crf are
  tunable. If every rung overshoots :class:`RenderSizeError` is raised
  with the full attempt log — the file is removed and we never silently
  truncate.

* **Encoder**: H.264 baseline (libx264) + ``yuv420p`` for browser
  compatibility. ``macro_block_size=1`` is passed to imageio so
  non-multiples of 16 don't get padded; 256 is already even so yuv420p
  chroma subsampling is fine.

* **Determinism**: same input frames + same successful rung produce a
  byte-identical MP4 in our spike (libx264 + a fixed crf). The
  :data:`RenderResult.content_sha256` field captures the on-disk bytes
  so callers can detect drift if a future ffmpeg upgrade breaks
  byte-identity; the visual content (decoded array) is the deeper
  invariant.

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

# Adaptive (fps, crf) rungs the encoder walks until the file fits under
# MAX_BYTES. Ordered cheapest-to-quality-cost. Notes:
#   * Rung 0 is the historical default and handles 30-300 frame clips.
#   * Rung 1 halves the playback rate (encoder-level fps; we DO NOT drop
#     input frames). Surprisingly libx264 often produces a *larger* file
#     at the same crf for fewer fps because per-frame quality is held
#     constant and the GOP structure has fewer P/B references — this is
#     why rung 1 alone may not be enough and rungs 2/3 also bump crf.
#   * Rungs 2 and 3 step crf to 28 and 33; visible artifacts at 33 but
#     better than failing to publish the clip.
RENDER_LADDER: tuple[tuple[int, int], ...] = (
    (10, 23),
    (5, 23),
    (5, 28),
    (5, 33),
)

_CODEC: str = "libx264"
_PIXEL_FORMAT: str = "yuv420p"
_PNG_CODEC: str = "png"


@dataclass(frozen=True)
class EncoderSettings:
    """Encoder knobs we burn into the output filename's metadata.

    For PNG thumbnails ``fps``, ``crf`` and ``rung_index`` are zero and
    ``codec`` is ``"png"``; the dataclass is shared so callers can switch
    on ``codec`` rather than carrying a separate type.

    ``rung_index`` is the index into :data:`RENDER_LADDER` that produced
    a fit. ``-1`` means "ladder bypassed" — the caller passed an
    explicit ``fps`` or ``crf`` override and we did a single-shot encode
    instead of walking the ladder. Downstream consumers should treat
    ``rung_index >= 1`` as "playback is faster than wall-clock; expect
    frame jumps relative to the source rate".
    """

    fps: int
    size: int
    codec: str
    pixel_format: str
    crf: int
    rung_index: int = -1


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

    For ladder-mode encodes the message lists every rung that was tried
    and the size it produced, so the failure-taxonomy labeler has
    actionable detail. ``size`` is the size of the last (highest-rung)
    attempt; ``attempts`` carries the full tuple.
    """

    def __init__(
        self,
        path: Path,
        size: int,
        limit: int,
        attempts: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        if attempts:
            ladder_log = ", ".join(
                f"rung {idx} (fps={fps}, crf={crf}) -> {sz} B"
                for idx, (fps, crf, sz) in enumerate(attempts)
            )
            msg = (
                f"encoded clip {path} exceeds {limit} byte cap on every ladder rung: "
                f"{ladder_log}. Last attempt: {size} B."
            )
        else:
            msg = (
                f"encoded clip {path} is {size} bytes (> {limit} byte cap); "
                f"lower fps or raise crf and re-encode"
            )
        super().__init__(msg)
        self.path = path
        self.size = size
        self.limit = limit
        self.attempts = attempts


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


def _encode_once(
    resized: NDArray[np.uint8],
    out_path: Path,
    fps: int,
    crf: int,
) -> int:
    """Encode ``resized`` to ``out_path`` and return the on-disk size in bytes."""
    iio.imwrite(
        out_path,
        resized,
        fps=fps,
        codec=_CODEC,
        pixelformat=_PIXEL_FORMAT,
        macro_block_size=1,
        output_params=["-crf", str(crf), "-preset", "medium"],
    )
    return out_path.stat().st_size


def render_episode(
    frames: NDArray[np.uint8],
    out_path: Path,
    *,
    fps: int | None = None,
    size: int = DEFAULT_SIZE,
    crf: int | None = None,
) -> RenderResult:
    """Encode ``frames`` to a small MP4 at ``out_path`` and return metadata.

    By default this walks :data:`RENDER_LADDER`, encoding at each rung
    until the output fits under :data:`MAX_BYTES`. The successful rung
    index is recorded in :attr:`EncoderSettings.rung_index`.

    If the caller passes an explicit ``fps`` or ``crf`` the ladder is
    bypassed and a single encode is performed at those settings (with
    the other knob defaulting to :data:`DEFAULT_FPS` / :data:`DEFAULT_CRF`).
    The resulting :attr:`EncoderSettings.rung_index` is ``-1``.

    Args:
        frames: ``(T, H, W, 3)`` uint8 RGB array. Resized to
            ``size x size`` via numpy bilinear if not already.
        out_path: Destination MP4 path. Parent directory must exist;
            this function does not create it.
        fps: Output frame rate. ``None`` (default) selects ladder mode;
            an explicit value bypasses the ladder.
        size: Output square edge in pixels (default 256). Must be even
            for ``yuv420p``; 256 always satisfies that.
        crf: libx264 constant rate factor. ``None`` (default) selects
            ladder mode; an explicit value bypasses the ladder.

    Raises:
        ValueError / TypeError: on malformed ``frames``.
        :class:`RenderSizeError`: if every ladder rung overshoots (or,
            in single-shot mode, the one encode overshoots). In both
            cases the file is removed before raising.
    """
    _validate_frames(frames)
    if size % 2 != 0:
        raise ValueError(f"size must be even (yuv420p chroma), got {size}")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    resized = _resize_stack(frames, size, size)
    cap = _current_max_bytes()

    use_ladder = fps is None and crf is None
    if use_ladder:
        attempts: list[tuple[int, int, int]] = []
        for rung_idx, (rung_fps, rung_crf) in enumerate(RENDER_LADDER):
            bytes_written = _encode_once(resized, out_path, rung_fps, rung_crf)
            attempts.append((rung_fps, rung_crf, bytes_written))
            if bytes_written <= cap:
                settings = EncoderSettings(
                    fps=rung_fps,
                    size=size,
                    codec=_CODEC,
                    pixel_format=_PIXEL_FORMAT,
                    crf=rung_crf,
                    rung_index=rung_idx,
                )
                return RenderResult(
                    path=out_path,
                    bytes_written=bytes_written,
                    frame_count=int(resized.shape[0]),
                    encoder_settings=settings,
                    content_sha256=_sha256_file(out_path),
                )
            # File overshoots; remove it before trying the next rung so
            # we never leave a partially-acceptable artifact behind.
            out_path.unlink(missing_ok=True)
        # Ladder exhausted; the last attempt's size + the full log go in
        # the error so the caller knows exactly what was tried.
        last_size = attempts[-1][2] if attempts else 0
        raise RenderSizeError(out_path, last_size, cap, tuple(attempts))

    # Single-shot legacy path: explicit fps/crf bypasses the ladder.
    eff_fps = DEFAULT_FPS if fps is None else fps
    eff_crf = DEFAULT_CRF if crf is None else crf
    bytes_written = _encode_once(resized, out_path, eff_fps, eff_crf)
    if bytes_written > cap:
        out_path.unlink(missing_ok=True)
        raise RenderSizeError(out_path, bytes_written, cap)

    settings = EncoderSettings(
        fps=eff_fps,
        size=size,
        codec=_CODEC,
        pixel_format=_PIXEL_FORMAT,
        crf=eff_crf,
        rung_index=-1,
    )
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
        rung_index=-1,
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
    "RENDER_LADDER",
    "EncoderSettings",
    "RenderResult",
    "RenderSizeError",
    "render_episode",
    "render_thumbnail_strip",
]
