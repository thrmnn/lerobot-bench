"""VRAM-budget admission gate for running small-VRAM cells concurrently.

Motivation
----------
The overnight sweep historically serialized every cell behind a single
global GPU lock (``scripts/with_gpu_lock.sh`` flock + a strictly
one-cell-at-a-time dispatch loop). The calibration audit found that the
small policies barely touch the 8 GB card -- measured peaks:

    act ~266 MB, smolvla ~920 MB, diffusion ~1.1 GB

so a 1-at-a-time policy leaves ~3-4x of the card idle. This module adds
an **admission gate** that lets several small cells run at once as long
as the *sum of their calibrated peaks* stays under a configurable
budget (default ~6000 MB of the 8 GB card -- >=25% headroom; lowered
from 7000 after the 2026-06-09 near-OOM GPU-PV desync).

OOM-safety (read before changing anything)
------------------------------------------
A careless concurrency change here froze the WSL2 host once. This gate
is **additive**, not a replacement for the existing OOM stack:

* Every cell is STILL launched through ``scripts/run_capped.sh``
  (cgroup ``MemoryMax`` + ``MemorySwapMax=0``). That caps *system RAM*
  and is the host-freeze defense -- it is untouched by this module.
* The existing ``with_gpu_lock.sh`` flock remains the cross-process
  fallback for anything that does NOT go through this gate.
* This gate only bounds *concurrent GPU memory*: a cell is admitted to
  start iff::

      (sum of vram_peak_mb of currently-running cells)
          + this cell's vram_peak_mb  <=  budget_mb

  Otherwise it WAITS until a running cell finishes and frees room.

* **Fail safe, not open.** A cell whose VRAM is *unknown* (no
  calibration entry) is treated as **exclusive**: it runs alone, with
  no other cell concurrent, and no other cell may start while it runs.
  We never admit an unknown policy alongside anything. If calibration
  data is missing entirely, the caller falls back to the legacy
  1-at-a-time behavior (``max_concurrent=1``), which this gate also
  expresses naturally.

The gate reserves on admit and releases on finish; because reservations
are sized by the *calibrated peak* (the worst case the policy hit during
the 20-step probe), the live sum can never exceed the budget even when
every admitted cell is simultaneously at its peak.

This module is GPU-free and pure: it reserves *numbers*, not memory. It
is unit-tested by simulating cells with known peaks; see
``tests/test_vram_scheduler.py``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("vram-scheduler")

# Default budget: ~6 GB of the 8 GB card, leaving >=25% (~2 GB) headroom.
#
# WHY 6000, not 7000 (lowered 2026-06-11): the 2026-06-09 incident was a
# WSL2 GPU-PV desync (host-side ``dxgkio ... Ioctl -22``) triggered by
# sustained ~96% VRAM with allocator thrash on this 8 GB card. Near-OOM
# VRAM is a known TDR / GPU-PV-desync trigger on WSL2, and 7000 MB of an
# 8192 MB card is ~85% reserved BEFORE the CUDA context (~300-600 MB) and
# allocator fragmentation are counted -- comfortably into the danger band.
# A >=25% headroom keeps live use off the TDR cliff. The calibrated peak
# is also only the 20-step probe worst case, so a fragmenting long run can
# overshoot it; the bigger margin absorbs that too. Configurable via
# ``--vram-budget-mb`` for the rare case you truly need the extra GB.
DEFAULT_VRAM_BUDGET_MB = 6000.0

# A hard ceiling on the fraction of total VRAM that should be in use at
# any instant. Sustained use above this is the desync trigger; the
# optional VRAM-ceiling monitor in ``scripts/watchdog.py`` aborts a run
# that stays above it for too long. 0.90 == 90% of the card.
VRAM_CEILING_PCT = 90.0

# Backstop cap on how many cells may run at once regardless of how small
# they are. Bounds CPU / RAM / file-descriptor pressure from many
# concurrent subprocesses even when their summed VRAM is tiny.
DEFAULT_MAX_CONCURRENT = 3


# --------------------------------------------------------------------- #
# Calibration VRAM lookup                                               #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationVramTable:
    """Calibrated ``vram_peak_mb`` keyed by ``(policy, env)``.

    Built from a ``results/calibration-YYYYMMDD.json`` report (the
    :class:`scripts.calibrate.CalibrationReport` shape). Only ``ok``
    cells contribute a number; ``oom`` / ``error`` / ``skipped`` cells
    are deliberately omitted so :meth:`vram_for` returns ``None`` for
    them -- an un-measured cell is *unknown*, and unknown means
    exclusive (fail safe).
    """

    by_cell: dict[tuple[str, str], float]
    by_policy_max: dict[str, float]

    @classmethod
    def from_report(cls, data: dict[str, Any]) -> CalibrationVramTable:
        """Build from a parsed calibration-report dict.

        VRAM is policy-dominated (it is the weights + activations, not
        the env), so we also record the per-policy maximum across envs.
        :meth:`vram_for` prefers the exact ``(policy, env)`` peak and
        falls back to that per-policy max -- always the *larger* number,
        never an optimistic under-estimate.
        """
        by_cell: dict[tuple[str, str], float] = {}
        by_policy_max: dict[str, float] = {}
        for cell in data.get("cells", []):
            if cell.get("status") != "ok":
                continue
            peak = cell.get("vram_peak_mb")
            if peak is None:
                continue
            policy = str(cell["policy"])
            env = str(cell["env"])
            peak_f = float(peak)
            by_cell[(policy, env)] = peak_f
            prev = by_policy_max.get(policy)
            if prev is None or peak_f > prev:
                by_policy_max[policy] = peak_f
        return cls(by_cell=by_cell, by_policy_max=by_policy_max)

    @classmethod
    def from_json_path(cls, path: Path) -> CalibrationVramTable:
        """Load a calibration JSON file. Raises on missing/garbled file."""
        data = json.loads(path.read_text())
        return cls.from_report(data)

    @classmethod
    def empty(cls) -> CalibrationVramTable:
        """A table with no entries -- every lookup is unknown (exclusive)."""
        return cls(by_cell={}, by_policy_max={})

    def is_empty(self) -> bool:
        return not self.by_cell

    def vram_for(self, policy: str, env: str) -> float | None:
        """Return the calibrated peak (MB) for a cell, or ``None`` if unknown.

        Prefers the exact ``(policy, env)`` measurement; falls back to
        the per-policy max across envs (VRAM is policy-dominated). Returns
        ``None`` only when the policy was never successfully calibrated --
        the caller MUST treat ``None`` as exclusive.
        """
        exact = self.by_cell.get((policy, env))
        if exact is not None:
            return exact
        return self.by_policy_max.get(policy)


# --------------------------------------------------------------------- #
# Admission gate                                                        #
# --------------------------------------------------------------------- #


class BudgetExceededError(ValueError):
    """A single cell's calibrated peak alone exceeds the whole budget.

    This is a config error, not a runtime wait condition: the cell can
    never be admitted because even running alone it would blow the
    budget. The caller should drop the cell (or raise the budget) rather
    than deadlock waiting for room that can never exist.
    """


@dataclass(frozen=True)
class Reservation:
    """A granted admission. Returned by :meth:`VramBudgetScheduler.acquire`."""

    policy: str
    env: str
    seed: int
    vram_mb: float | None  # None == admitted as exclusive (unknown VRAM)
    exclusive: bool


class VramBudgetScheduler:
    """Thread-safe VRAM-budget semaphore for concurrent cell dispatch.

    Each worker thread calls :meth:`acquire` before launching its cell's
    subprocess and :meth:`release` (or uses :meth:`admission`) when the
    subprocess exits. Admission blocks until BOTH hold:

    * ``reserved_mb + cell_vram_mb <= budget_mb``  (VRAM fits), and
    * ``running < max_concurrent``                 (backstop cap), and
    * no exclusive cell is running, and
    * if THIS cell is exclusive (unknown VRAM), nothing else is running.

    The scheduler reserves the calibrated *peak*, so the live reserved
    sum is an upper bound on real VRAM use even when all admitted cells
    are at their worst simultaneously. It never touches the GPU.

    A ``max_concurrent`` of 1 reproduces the legacy strictly-serial
    behavior -- that is the safe fallback when calibration data is
    absent.
    """

    def __init__(
        self,
        *,
        budget_mb: float = DEFAULT_VRAM_BUDGET_MB,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        if budget_mb <= 0:
            raise ValueError(f"budget_mb must be positive, got {budget_mb}")
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        self.budget_mb = float(budget_mb)
        self.max_concurrent = int(max_concurrent)
        self._cond = threading.Condition()
        self._reserved_mb = 0.0
        self._running = 0
        self._exclusive_running = False
        # High-water mark for post-sweep reporting / assertions.
        self._peak_reserved_mb = 0.0
        self._peak_running = 0

    # -- introspection (lock-free reads are fine for logging) ---------- #

    @property
    def reserved_mb(self) -> float:
        with self._cond:
            return self._reserved_mb

    @property
    def running(self) -> int:
        with self._cond:
            return self._running

    @property
    def peak_reserved_mb(self) -> float:
        with self._cond:
            return self._peak_reserved_mb

    @property
    def peak_running(self) -> int:
        with self._cond:
            return self._peak_running

    # -- admission ----------------------------------------------------- #

    def _can_admit(self, vram_mb: float | None, exclusive: bool) -> bool:
        """Pure predicate, called only while holding ``self._cond``."""
        if self._exclusive_running:
            return False
        if self._running >= self.max_concurrent:
            return False
        if exclusive:
            # An unknown-VRAM cell may only run when the card is idle.
            return self._running == 0
        assert vram_mb is not None  # exclusive is False -> vram known
        return self._reserved_mb + vram_mb <= self.budget_mb

    def acquire(
        self,
        *,
        policy: str,
        env: str,
        seed: int,
        vram_mb: float | None,
        timeout: float | None = None,
    ) -> Reservation:
        """Block until the cell can be admitted, then reserve its budget.

        ``vram_mb is None`` means the cell's VRAM is unknown -> admitted
        as **exclusive** (runs alone). Returns a :class:`Reservation` to
        pass to :meth:`release`.

        Raises :class:`BudgetExceededError` if a *known* cell's peak alone
        exceeds ``budget_mb`` -- it could never be admitted and waiting
        would deadlock. Raises :class:`TimeoutError` if ``timeout`` elapses
        before room frees.
        """
        exclusive = vram_mb is None
        if not exclusive:
            assert vram_mb is not None
            if vram_mb > self.budget_mb:
                raise BudgetExceededError(
                    f"cell {policy}/{env}/seed{seed} calibrated peak "
                    f"{vram_mb:.0f} MB exceeds budget {self.budget_mb:.0f} MB; "
                    "drop the cell or raise --vram-budget-mb"
                )

        with self._cond:
            admitted = self._cond.wait_for(
                lambda: self._can_admit(vram_mb, exclusive),
                timeout=timeout,
            )
            if not admitted:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting to admit "
                    f"{policy}/{env}/seed{seed} (vram={vram_mb}, "
                    f"reserved={self._reserved_mb:.0f}/{self.budget_mb:.0f} MB, "
                    f"running={self._running})"
                )
            self._running += 1
            if exclusive:
                self._exclusive_running = True
            else:
                assert vram_mb is not None
                self._reserved_mb += vram_mb
            self._peak_reserved_mb = max(self._peak_reserved_mb, self._reserved_mb)
            self._peak_running = max(self._peak_running, self._running)
            logger.debug(
                "admit %s/%s/seed%d vram=%s exclusive=%s -> reserved=%.0f/%.0f MB running=%d",
                policy,
                env,
                seed,
                vram_mb,
                exclusive,
                self._reserved_mb,
                self.budget_mb,
                self._running,
            )
        return Reservation(
            policy=policy,
            env=env,
            seed=seed,
            vram_mb=vram_mb,
            exclusive=exclusive,
        )

    def release(self, reservation: Reservation) -> None:
        """Return a reservation's budget and wake any waiting threads."""
        with self._cond:
            self._running -= 1
            if reservation.exclusive:
                self._exclusive_running = False
            else:
                assert reservation.vram_mb is not None
                self._reserved_mb -= reservation.vram_mb
                # Float drift guard: never let the accumulator go negative.
                if self._reserved_mb < 0:
                    self._reserved_mb = 0.0
            logger.debug(
                "release %s/%s/seed%d -> reserved=%.0f/%.0f MB running=%d",
                reservation.policy,
                reservation.env,
                reservation.seed,
                self._reserved_mb,
                self.budget_mb,
                self._running,
            )
            self._cond.notify_all()

    @contextmanager
    def admission(
        self,
        *,
        policy: str,
        env: str,
        seed: int,
        vram_mb: float | None,
        timeout: float | None = None,
    ) -> Iterator[Reservation]:
        """Context manager wrapper around acquire/release.

        Releases the reservation even if the cell's body raises -- a
        crashed cell must not leak its budget and starve the rest of the
        sweep.
        """
        reservation = self.acquire(
            policy=policy,
            env=env,
            seed=seed,
            vram_mb=vram_mb,
            timeout=timeout,
        )
        try:
            yield reservation
        finally:
            self.release(reservation)
