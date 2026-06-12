"""Unit tests for the fail-fast GPU-health preflight (no real GPU).

Every check is driven through the injectable ``runner`` seam with fake
``subprocess.run`` results, so the exit-code contract is proven without
ever touching CUDA. The cases mirror the four real outcomes:

* healthy        -> exit 0
* nvidia-smi down -> exit 10 (likely WSL2 GPU-PV desync)
* torch.cuda bad  -> exit 11 (incl. a SIGSEGV'd child == negative rc)
* low VRAM        -> exit 12
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import pytest

from embodimetry import gpu_preflight as gp


@dataclass
class _FakeProc:
    """Stand-in for subprocess.CompletedProcess."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


def _runner(proc: _FakeProc | Exception):
    """Build a fake ``subprocess.run`` that returns ``proc`` (or raises it)."""

    def run(_cmd: object, **_kw: object) -> _FakeProc:
        if isinstance(proc, Exception):
            raise proc
        return proc

    return run


# --------------------------------------------------------------------- #
# check_nvidia_smi                                                      #
# --------------------------------------------------------------------- #


def test_nvidia_smi_ok() -> None:
    assert gp.check_nvidia_smi(runner=_runner(_FakeProc(0, "GPU 0: RTX 4060")))


def test_nvidia_smi_nonzero_returncode() -> None:
    assert not gp.check_nvidia_smi(runner=_runner(_FakeProc(255)))


def test_nvidia_smi_missing_binary() -> None:
    assert not gp.check_nvidia_smi(runner=_runner(FileNotFoundError()))


def test_nvidia_smi_timeout() -> None:
    err = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
    assert not gp.check_nvidia_smi(runner=_runner(err))


# --------------------------------------------------------------------- #
# check_torch_cuda                                                      #
# --------------------------------------------------------------------- #


def test_torch_cuda_ok_parses_free_mb() -> None:
    ok, free = gp.check_torch_cuda(runner=_runner(_FakeProc(0, "4096\n")))
    assert ok
    assert free == 4096


def test_torch_cuda_child_reports_unavailable() -> None:
    # child sys.exit(11) -> nonzero rc -> not ok
    ok, free = gp.check_torch_cuda(runner=_runner(_FakeProc(11)))
    assert not ok
    assert free is None


def test_torch_cuda_child_sigsegv_is_negative_rc_not_a_crash() -> None:
    # A SIGSEGV in the child surfaces as returncode -11. The whole point of
    # the subprocess design: the parent observes it as not-ok, never crashes.
    ok, free = gp.check_torch_cuda(runner=_runner(_FakeProc(-11)))
    assert not ok
    assert free is None


def test_torch_cuda_unparseable_stdout() -> None:
    ok, free = gp.check_torch_cuda(runner=_runner(_FakeProc(0, "not-a-number\n")))
    assert not ok
    assert free is None


def test_torch_cuda_timeout_treated_as_unavailable() -> None:
    err = subprocess.TimeoutExpired(cmd="python", timeout=60)
    ok, free = gp.check_torch_cuda(runner=_runner(err))
    assert not ok
    assert free is None


# --------------------------------------------------------------------- #
# run_preflight -- the exit-code contract                              #
# --------------------------------------------------------------------- #


def test_preflight_healthy_exit_0() -> None:
    res = gp.run_preflight(
        required_headroom_mb=1500,
        smi_runner=_runner(_FakeProc(0, "GPU 0")),
        torch_runner=_runner(_FakeProc(0, "4096\n")),
    )
    assert res.exit_code == gp.EXIT_OK
    assert res.ok
    assert res.free_vram_mb == 4096


def test_preflight_nvidia_smi_down_exit_10() -> None:
    res = gp.run_preflight(
        smi_runner=_runner(_FakeProc(255)),
        torch_runner=_runner(_FakeProc(0, "4096\n")),  # never reached
    )
    assert res.exit_code == gp.EXIT_NVIDIA_SMI_UNREACHABLE
    assert "desync" in res.message
    assert "wsl --shutdown" in res.message


def test_preflight_torch_unavailable_exit_11() -> None:
    res = gp.run_preflight(
        smi_runner=_runner(_FakeProc(0, "GPU 0")),
        torch_runner=_runner(_FakeProc(-11)),  # SIGSEGV child
    )
    assert res.exit_code == gp.EXIT_TORCH_CUDA_UNAVAILABLE
    assert "torch.cuda" in res.message


def test_preflight_low_vram_exit_12() -> None:
    res = gp.run_preflight(
        required_headroom_mb=2000,
        smi_runner=_runner(_FakeProc(0, "GPU 0")),
        torch_runner=_runner(_FakeProc(0, "500\n")),  # only 500 MB free
    )
    assert res.exit_code == gp.EXIT_INSUFFICIENT_VRAM
    assert res.free_vram_mb == 500
    assert "Insufficient free VRAM" in res.message


def test_preflight_exactly_at_headroom_passes() -> None:
    res = gp.run_preflight(
        required_headroom_mb=1500,
        smi_runner=_runner(_FakeProc(0, "GPU 0")),
        torch_runner=_runner(_FakeProc(0, "1500\n")),
    )
    assert res.exit_code == gp.EXIT_OK


def test_exit_codes_are_distinct() -> None:
    codes = {
        gp.EXIT_OK,
        gp.EXIT_NVIDIA_SMI_UNREACHABLE,
        gp.EXIT_TORCH_CUDA_UNAVAILABLE,
        gp.EXIT_INSUFFICIENT_VRAM,
    }
    assert len(codes) == 4


# --------------------------------------------------------------------- #
# main() CLI surface                                                    #
# --------------------------------------------------------------------- #


def test_main_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gp,
        "run_preflight",
        lambda **_kw: gp.PreflightResult(gp.EXIT_NVIDIA_SMI_UNREACHABLE, "down"),
    )
    assert gp.main([]) == gp.EXIT_NVIDIA_SMI_UNREACHABLE


def test_main_healthy_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gp,
        "run_preflight",
        lambda **_kw: gp.PreflightResult(gp.EXIT_OK, "healthy", free_vram_mb=4096),
    )
    assert gp.main(["-q"]) == gp.EXIT_OK
    assert capsys.readouterr().out == ""
