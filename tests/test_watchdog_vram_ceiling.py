"""Unit tests for the watchdog's sustained-VRAM-ceiling guard (no GPU).

The ceiling is the desync-prevention defense: trip only when VRAM use
stays at/above the ceiling for long enough, never on a transient peak,
and never at all when the ceiling is disabled (None).
"""

from __future__ import annotations

from scripts.watchdog import vram_ceiling_breached, vram_used_pct


def test_vram_used_pct_basic() -> None:
    assert vram_used_pct(4096, 8192) == 50.0
    assert vram_used_pct(7800, 8192) > 95.0


def test_vram_used_pct_zero_total_is_zero() -> None:
    # Guards a division-by-zero if nvidia-smi reports total 0.
    assert vram_used_pct(0, 0) == 0.0


def test_ceiling_disabled_never_breaches() -> None:
    # ~96% use, long elapsed, but ceiling None -> off.
    assert not vram_ceiling_breached(
        7900, 8192, ceiling_pct=None, elapsed_over_s=999.0, ceiling_seconds=120.0
    )


def test_below_ceiling_does_not_breach() -> None:
    # 50% use, even after a long elapsed window -> never trips.
    assert not vram_ceiling_breached(
        4096, 8192, ceiling_pct=90.0, elapsed_over_s=999.0, ceiling_seconds=120.0
    )


def test_over_ceiling_but_not_yet_sustained_does_not_breach() -> None:
    # 96% use but only 30s over a 120s window -> transient, don't kill.
    assert not vram_ceiling_breached(
        7900, 8192, ceiling_pct=90.0, elapsed_over_s=30.0, ceiling_seconds=120.0
    )


def test_over_ceiling_and_sustained_breaches() -> None:
    # 96% use sustained 120s >= 120s window -> desync risk, trip.
    assert vram_ceiling_breached(
        7900, 8192, ceiling_pct=90.0, elapsed_over_s=120.0, ceiling_seconds=120.0
    )


def test_at_ceiling_counts_as_over() -> None:
    # Exactly at the ceiling (>= semantics) with the window satisfied trips.
    # 4096/8192 == 50.0%, so a 50.0 ceiling is met exactly.
    assert vram_used_pct(4096, 8192) == 50.0
    assert vram_ceiling_breached(
        4096, 8192, ceiling_pct=50.0, elapsed_over_s=200.0, ceiling_seconds=120.0
    )
