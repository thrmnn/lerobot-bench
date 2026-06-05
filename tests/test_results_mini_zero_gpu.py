"""Zero-GPU onboarding guard: the committed mini-parquet + read path.

A fresh ``git clone`` must be able to read a real leaderboard number with
no GPU and no Hub download. This suite pins that contract:

  * ``examples/results-mini.parquet`` exists, has the aggregated schema, the
    expected published cells, the CORRECTED ``act × aloha`` 0.824 cell, and
    NO ``xvla`` / pi0 rows.
  * ``scripts/make_results_mini.py --check`` confirms the committed file
    reproduces a fresh deterministic build (it is not a hand-built blob).
  * ``examples/read_results.py`` runs with no args and prints the table.
  * ``scripts/run_one.py``'s GPU precheck refuses a torch policy with no
    usable CUDA and points the user at the zero-GPU read path — while
    baselines and the dry-run path stay GPU-free.
"""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts import make_results_mini as mk
from scripts import run_one as ro

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_PARQUET = REPO_ROOT / "examples" / "results-mini.parquet"
READ_RESULTS = REPO_ROOT / "examples" / "read_results.py"

_EXPECTED_COLUMNS = {
    "policy",
    "env",
    "n_episodes",
    "n_success",
    "success_rate",
    "wilson_lo",
    "wilson_hi",
}
# The six published v1 leaderboard headline cells the mini must carry.
_EXPECTED_CELLS = {
    ("act", "aloha_transfer_cube"),
    ("diffusion_policy", "pusht"),
    ("smolvla_libero", "libero_spatial"),
    ("smolvla_libero", "libero_object"),
    ("smolvla_libero", "libero_goal"),
    ("smolvla_libero", "libero_10"),
}


@pytest.fixture(scope="module")
def mini() -> pd.DataFrame:
    assert MINI_PARQUET.exists(), (
        f"{MINI_PARQUET} is missing — a fresh clone could not read any number. "
        "Regenerate with `python scripts/make_results_mini.py`."
    )
    return pd.read_parquet(MINI_PARQUET)


def test_mini_schema_and_cells(mini: pd.DataFrame) -> None:
    assert set(mini.columns) == _EXPECTED_COLUMNS
    assert set(zip(mini["policy"], mini["env"], strict=True)) == _EXPECTED_CELLS
    assert len(mini) == len(_EXPECTED_CELLS)


def test_act_aloha_cell_is_corrected_0824(mini: pd.DataFrame) -> None:
    """The act × aloha cell carries the corrected 0.824, NOT the stale 0.016."""
    row = mini[(mini["policy"] == "act") & (mini["env"] == "aloha_transfer_cube")]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["n_episodes"] == 250
    assert round(float(r["success_rate"]), 3) == 0.824
    assert round(float(r["wilson_lo"]), 3) == 0.772
    assert round(float(r["wilson_hi"]), 3) == 0.866
    # The stale pre-#51 reading must be nowhere near this cell.
    assert round(float(r["success_rate"]), 3) != 0.016


def test_no_xvla_or_pi0(mini: pd.DataFrame) -> None:
    """xvla pollutes the canonical parquet at 0.000 — it must not appear here."""
    policies = set(mini["policy"])
    assert not any("xvla" in p for p in policies)
    assert not any(p.startswith("pi0") for p in policies)


def test_make_results_mini_check_passes() -> None:
    """The committed file reproduces a fresh deterministic build."""
    assert mk.main(["--check"]) == 0


def test_read_results_runs_with_no_args(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python examples/read_results.py` prints the leaderboard table, exit 0."""
    monkeypatch.setattr("sys.argv", ["read_results.py"])
    runpy.run_path(str(READ_RESULTS), run_name="__main__")
    out = capsys.readouterr().out
    assert "v1 leaderboard" in out
    assert "0.824" in out  # the headline act cell
    assert "aloha_transfer_cube" in out
    assert "xvla" not in out


def test_read_results_module_imports() -> None:
    """The example imports cleanly (no GPU / lerobot deps in its import graph)."""
    spec = importlib.util.spec_from_file_location("_read_results_probe", READ_RESULTS)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "print_leaderboard")


# --------------------------------------------------------------------- #
# GPU precheck                                                           #
# --------------------------------------------------------------------- #


def _fake_torch(*, cuda: bool, free_gb: float):
    """A minimal torch stand-in for _check_gpu_available."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            mem_get_info=lambda: (int(free_gb * 1024**3), (16 * 1024**3)),
        )
    )


def test_gpu_precheck_skips_non_cuda_device() -> None:
    assert ro._check_gpu_available("cpu") is None


def test_gpu_precheck_flags_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=False, free_gb=0.0))
    msg = ro._check_gpu_available("cuda")
    assert msg is not None and "no CUDA device" in msg


def test_gpu_precheck_flags_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=True, free_gb=0.5))
    msg = ro._check_gpu_available("cuda")
    assert msg is not None and "free VRAM" in msg


def test_gpu_precheck_passes_with_ample_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(cuda=True, free_gb=12.0))
    assert ro._check_gpu_available("cuda") is None


def test_run_one_torch_policy_aborts_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torch policy with no CUDA exits 4 and points at the zero-GPU path."""
    monkeypatch.setattr(ro, "_check_lerobot_available", lambda: None)
    monkeypatch.setattr(ro, "_check_gpu_available", lambda device: "no CUDA device available")

    def _boom(*_a, **_k):  # eval must NOT be reached
        raise AssertionError("eval should not run when the GPU precheck fails")

    monkeypatch.setattr("embodimetry.eval.run_cell_from_specs", _boom)

    outcome = ro.run_one(
        policy_name="act",
        env_name="aloha_transfer_cube",
        seed=0,
        n_episodes=5,
        out_parquet=Path("results/results.parquet"),
        videos_dir=Path("results/videos"),
        record_video=False,
        device="cuda",
        policies_yaml=REPO_ROOT / "configs" / "policies.yaml",
        envs_yaml=REPO_ROOT / "configs" / "envs.yaml",
        dry_run=False,
    )
    assert outcome.exit_code == 4
    assert "examples/read_results.py" in outcome.log_message
    assert outcome.n_rows_appended == 0


def test_run_one_baseline_skips_gpu_precheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Baselines (no_op) never hit the GPU precheck — they run GPU-less."""
    called = {"gpu": False}

    def _track_gpu(device):
        called["gpu"] = True
        return "no CUDA device available"

    monkeypatch.setattr(ro, "_check_lerobot_available", lambda: None)
    monkeypatch.setattr(ro, "_check_gpu_available", _track_gpu)

    # A sentinel raised from eval proves control flowed PAST the precheck
    # (the baseline did not abort on the simulated no-GPU). We don't need a
    # real CellResult — reaching the eval call is the assertion.
    class _ReachedEvalError(Exception):
        pass

    def _fake_eval(*_a, **_k):
        raise _ReachedEvalError

    monkeypatch.setattr("embodimetry.eval.run_cell_from_specs", _fake_eval)

    with pytest.raises(_ReachedEvalError):
        ro.run_one(
            policy_name="no_op",
            env_name="pusht",
            seed=0,
            n_episodes=1,
            out_parquet=Path("/tmp/_zerogpu_test_noop.parquet"),
            videos_dir=Path("/tmp/_zerogpu_test_videos"),
            record_video=False,
            device="cuda",
            policies_yaml=REPO_ROOT / "configs" / "policies.yaml",
            envs_yaml=REPO_ROOT / "configs" / "envs.yaml",
            dry_run=False,
        )
    assert called["gpu"] is False
