"""Fail-fast GPU-health preflight for the 8 GB WSL2 RTX 4060.

Why this exists
---------------
On 2026-06-09 the WSL2 GPU passthrough wedged mid-session (host-side
GPU-PV desync, ``dxgkio ... Ioctl failed: -22``) after a JEPA-WM run
pinned the card to ~96% VRAM with allocator thrash -- a classic trigger
for a WSL2 GPU TDR / GPU-PV desync. Two things hurt:

1. We let GPU work run at near-OOM VRAM (the *cause*).
2. When the GPU went down, our tooling SIGSEGV'd cryptically -- a raw
   ``cuInit(0)`` / ``torch.cuda.is_available()`` segfaults *in process*
   on a desynced adapter, so the crash took out the caller and the real
   diagnosis ("the virtual GPU adapter lost host sync, restart the VM")
   took ~30 minutes to reach.

This module is the **phase-0** check to run BEFORE any CUDA dispatch. It
converts that cryptic 30-minute diagnosis into a 2-second clear error.

Design: the dangerous probe runs in a SUBPROCESS
------------------------------------------------
``torch.cuda.is_available()`` can SIGSEGV on a dead adapter. We never
call it in the orchestrator process. :func:`check_torch_cuda` spawns a
throwaway ``python -c`` child; a segfault there is observed by the parent
as a *nonzero exit code* (negative return code == killed by signal), not
a crash of the caller. Fail-fast, never fail-cryptic.

This module touches the GPU only to *read* its health (``nvidia-smi`` and
a child-process import). It allocates nothing. The exit-code contract and
each check are unit-tested with mocked subprocess results -- no real GPU
needed.

Exit-code contract (also returned by :func:`run_preflight`)
-----------------------------------------------------------
* ``0``  -- healthy: ``nvidia-smi`` reachable, torch sees CUDA in a
            child, and free VRAM >= the required headroom.
* ``10`` -- ``nvidia-smi`` unreachable / errored (driver channel down).
            Most likely a WSL2 GPU-PV desync; the fix is ``wsl --shutdown``.
* ``11`` -- torch could not see CUDA in the child (or the child crashed /
            SIGSEGV'd). The adapter is up enough for nvidia-smi but the
            CUDA runtime is unusable -- treat as desync, restart the VM.
* ``12`` -- free VRAM is below the required headroom. Not a desync: some
            other process is holding the card. Wait / free it before
            dispatching, or you risk the near-OOM TDR this guards against.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger("gpu-preflight")


class _CompletedLike(Protocol):
    """Minimal view of ``subprocess.CompletedProcess`` the checks need."""

    returncode: int
    stdout: str


class _Runner(Protocol):
    """A ``subprocess.run``-shaped callable (injectable for tests)."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool = ...,
        text: bool = ...,
        timeout: float = ...,
    ) -> _CompletedLike: ...


# Exit codes (see module docstring). Distinct per failure so callers and
# the runbook can branch on the cause.
EXIT_OK = 0
EXIT_NVIDIA_SMI_UNREACHABLE = 10
EXIT_TORCH_CUDA_UNAVAILABLE = 11
EXIT_INSUFFICIENT_VRAM = 12

# Default headroom a dispatch must see free before it is allowed to run.
# The 2026-06-09 desync followed sustained ~96% VRAM on the 8 GB card;
# we refuse to add load when less than this is free. Tunable via CLI.
DEFAULT_REQUIRED_HEADROOM_MB = 1500

# How long to allow the torch-in-subprocess probe before giving up. A
# desynced adapter can hang inside ``import torch`` / ``cuInit``; a
# timeout is treated the same as a crash (CUDA unavailable).
_TORCH_PROBE_TIMEOUT_S = 60

# Child program: import torch, assert CUDA visible, print free VRAM (MB).
# Runs in its own process so a CUDA SIGSEGV here cannot take down the
# caller -- the parent observes only this child's (negative) returncode.
_TORCH_PROBE_SRC = (
    "import torch,sys\n"
    "if not torch.cuda.is_available():\n"
    "    sys.exit(11)\n"
    "free,_=torch.cuda.mem_get_info()\n"
    "print(int(free//(1024*1024)))\n"
)

_DESYNC_HINT = (
    "GPU unreachable -- likely WSL2 GPU-PV desync (host-side adapter lost "
    "sync; symptom: nvidia-smi fails and dmesg spams 'dxgkio ... Ioctl "
    "failed: -22'). FIX: run `wsl --shutdown` from Windows, wait ~8s, then "
    "restart WSL and reconnect. See docs/RUNBOOK.md -> "
    "'GPU health & WSL2 GPU-PV desync'."
)


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of :func:`run_preflight`.

    ``exit_code`` is the process exit code (see the module docstring);
    ``message`` is a single actionable line suitable for stderr.
    ``free_vram_mb`` is the free VRAM the torch child reported, or
    ``None`` if the run failed before that point.
    """

    exit_code: int
    message: str
    free_vram_mb: int | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK


def check_nvidia_smi(
    *,
    runner: _Runner = subprocess.run,
    timeout: float = 10.0,
) -> bool:
    """Return True iff ``nvidia-smi -L`` succeeds (driver channel is up).

    ``runner`` is injectable for tests; defaults to :func:`subprocess.run`.
    Any failure -- nonzero exit, missing binary, or timeout -- returns
    False (the channel is down; do not dispatch).
    """
    try:
        proc = runner(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.debug("nvidia-smi -L timed out after %.0fs", timeout)
        return False
    return proc.returncode == 0


def check_torch_cuda(
    *,
    runner: _Runner = subprocess.run,
    timeout: float = _TORCH_PROBE_TIMEOUT_S,
) -> tuple[bool, int | None]:
    """Probe ``torch.cuda`` IN A SUBPROCESS; return ``(ok, free_vram_mb)``.

    The child imports torch, asserts CUDA is available, and prints free
    VRAM in MB. We run it out-of-process precisely because a desynced
    adapter can SIGSEGV inside the CUDA runtime: a crash there surfaces as
    a *negative* returncode here (killed by signal), which we report as
    ``(False, None)`` -- never as a crash of this process.

    ``ok`` is True only on a clean exit (returncode 0) with a parseable
    free-VRAM integer on stdout.
    """
    try:
        proc = runner(
            [sys.executable, "-c", _TORCH_PROBE_SRC],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.debug("python executable not found for torch probe")
        return (False, None)
    except subprocess.TimeoutExpired:
        logger.debug("torch CUDA probe timed out after %.0fs (adapter hung?)", timeout)
        return (False, None)

    if proc.returncode != 0:
        logger.debug(
            "torch CUDA probe failed: returncode=%d (negative == killed by signal)",
            proc.returncode,
        )
        return (False, None)
    try:
        free_mb = int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        logger.debug("torch CUDA probe returned unparseable stdout: %r", proc.stdout)
        return (False, None)
    return (True, free_mb)


def run_preflight(
    *,
    required_headroom_mb: int = DEFAULT_REQUIRED_HEADROOM_MB,
    smi_runner: _Runner = subprocess.run,
    torch_runner: _Runner = subprocess.run,
) -> PreflightResult:
    """Run all three checks in order and return a :class:`PreflightResult`.

    Order matters: cheapest/most-diagnostic first. ``nvidia-smi`` failing
    is the clearest desync signal, so it is checked first; only if the
    driver channel is up do we pay for the torch subprocess probe; only if
    CUDA is usable do we gate on free VRAM.
    """
    if not check_nvidia_smi(runner=smi_runner):
        return PreflightResult(EXIT_NVIDIA_SMI_UNREACHABLE, _DESYNC_HINT)

    ok, free_mb = check_torch_cuda(runner=torch_runner)
    if not ok:
        return PreflightResult(
            EXIT_TORCH_CUDA_UNAVAILABLE,
            "nvidia-smi is up but torch.cuda is unusable (the CUDA runtime "
            "probe crashed or saw no device). " + _DESYNC_HINT,
        )

    assert free_mb is not None  # ok is True -> free_mb parsed
    if free_mb < required_headroom_mb:
        return PreflightResult(
            EXIT_INSUFFICIENT_VRAM,
            (
                f"Insufficient free VRAM: {free_mb} MB free < "
                f"{required_headroom_mb} MB required headroom. Another process "
                "is holding the card. Do NOT dispatch -- running near-OOM is "
                "the exact condition that triggered the 2026-06-09 GPU-PV "
                "desync. Wait for it to free or stop the holder, then retry."
            ),
            free_vram_mb=free_mb,
        )

    return PreflightResult(
        EXIT_OK,
        f"GPU healthy: nvidia-smi up, torch.cuda OK, {free_mb} MB free "
        f"(>= {required_headroom_mb} MB headroom).",
        free_vram_mb=free_mb,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu_preflight",
        description=(
            "Fail-fast GPU-health check to run BEFORE any CUDA dispatch. "
            "Verifies nvidia-smi is reachable, torch.cuda works (probed in a "
            "subprocess so a dead-GPU SIGSEGV is a nonzero exit, not a crash "
            "here), and free VRAM meets a required headroom. Exit 0 healthy; "
            "10 nvidia-smi down; 11 torch.cuda unusable; 12 insufficient VRAM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--required-headroom-mb",
        type=int,
        default=DEFAULT_REQUIRED_HEADROOM_MB,
        metavar="MB",
        help=(
            "Minimum free VRAM (MB) required to pass (default: "
            f"{DEFAULT_REQUIRED_HEADROOM_MB}). Guards against dispatching onto "
            "a near-OOM card -- the 2026-06-09 desync followed sustained ~96%% "
            "VRAM."
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="On success print nothing (failures always print to stderr).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_preflight(required_headroom_mb=args.required_headroom_mb)
    if result.ok:
        if not args.quiet:
            print(result.message)
    else:
        print(f"gpu_preflight: {result.message}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
