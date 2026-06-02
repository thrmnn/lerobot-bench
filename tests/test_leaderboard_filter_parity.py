"""Cross-surface guard: xvla must be filtered out on BOTH public surfaces.

``V1_POLICIES`` + ``filter_to_v1_policies`` live in a single source of
truth (``src/lerobot_bench/leaderboard_filter.py``) re-exported by both
``space/_helpers.py`` and ``dashboard/_helpers.py``. Sharing the
definition is necessary but not sufficient: a future edit could still
drop the *call* to ``filter_to_v1_policies`` from one surface's load
path while leaving the other intact, silently re-introducing the
deferred ``xvla_libero`` rows to one leaderboard only.

``tests/test_dashboard.py`` already pins symbol identity (the two
surfaces import the *same* function object). This test is the
behavioural complement: it feeds one synthetic parquet carrying an
``xvla_libero`` row through *both* surfaces' top-level load path and the
leaderboard aggregator built on top of it, and asserts xvla is absent
from both outputs. Drop the filter call on either surface and this
fails in CI.

Both surfaces' helper modules are gradio-free by design, so this test
imports them without any heavy dep (no gradio / torch / GPU). Each
module ships as a top-level ``_helpers`` on its own deploy target, so
we load them under distinct names via ``importlib.util`` to avoid the
``_helpers`` import collision the other two suites already work around.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lerobot_bench.checkpointing import RESULT_SCHEMA

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_helpers(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"could not build spec for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the helper modules define frozen dataclasses,
    # and dataclasses resolves ``cls.__module__`` via ``sys.modules`` at
    # class-creation time -- an unregistered module makes that lookup
    # ``None`` and the import explodes. A unique name avoids colliding
    # with the ``_helpers`` modules the space/dashboard suites load.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_space_helpers = _load_helpers("space_helpers_parity", _REPO_ROOT / "space" / "_helpers.py")
_dashboard_helpers = _load_helpers(
    "dashboard_helpers_parity", _REPO_ROOT / "dashboard" / "_helpers.py"
)


def _row(policy: str, env: str, seed: int, ep: int, *, success: bool) -> dict[str, Any]:
    return {
        "policy": policy,
        "env": env,
        "seed": seed,
        "episode_index": ep,
        "success": success,
        "return_": 1.0 if success else 0.0,
        "n_steps": 10,
        "wallclock_s": 0.5,
        "video_sha256": "deadbeef",
        "code_sha": "cafef00d",
        "lerobot_version": "0.5.1",
        "timestamp_utc": "2026-05-03T00:00:00Z",
    }


@pytest.fixture
def xvla_parquet(tmp_path: Path) -> Path:
    """Synthetic parquet with one v1 policy (``act``) + one deferred ``xvla_libero``.

    The on-disk file deliberately *keeps* the xvla rows (the published
    parquet does too, for reproducibility); the surfaces' load paths are
    what must drop them.
    """
    rows = [_row("act", "pusht", seed=0, ep=e, success=e % 2 == 0) for e in range(6)]
    rows += [_row("xvla_libero", "libero_10", seed=0, ep=e, success=True) for e in range(6)]
    df = pd.DataFrame(rows, columns=list(RESULT_SCHEMA))
    parquet = tmp_path / "results.parquet"
    df.to_parquet(parquet, index=False)
    return parquet


def test_parquet_on_disk_still_carries_xvla(xvla_parquet: Path) -> None:
    """Sanity: the filter is what drops xvla, not the fixture."""
    raw = pd.read_parquet(xvla_parquet)
    assert "xvla_libero" in set(raw["policy"])


def test_space_load_path_excludes_xvla(xvla_parquet: Path) -> None:
    """Space ``load_results_df`` -> ``compute_leaderboard_table`` drops xvla."""
    _space_helpers.clear_results_cache()
    loaded = _space_helpers.load_results_df(xvla_parquet)
    assert "xvla_libero" not in set(loaded["policy"])

    board = _space_helpers.compute_leaderboard_table(loaded)
    assert "xvla_libero" not in set(board["policy"])


def test_dashboard_load_path_excludes_xvla(xvla_parquet: Path) -> None:
    """Dashboard ``load_results_parquet`` -> ``build_live_leaderboard`` drops xvla."""
    _dashboard_helpers.clear_results_cache()
    loaded = _dashboard_helpers.load_results_parquet(xvla_parquet)
    assert loaded is not None
    assert "xvla_libero" not in set(loaded["policy"])

    board = _dashboard_helpers.build_live_leaderboard(loaded)
    assert "xvla_libero" not in {r.policy for r in board}


def test_both_surfaces_agree_on_v1_policy_set(xvla_parquet: Path) -> None:
    """The two surfaces' load paths surface the *same* policy set.

    Beyond "neither shows xvla", the surviving policy set must match
    between surfaces -- a future edit dropping the filter on one side
    would diverge the sets and fail here even if some other guard hid
    the literal xvla assertion.
    """
    _space_helpers.clear_results_cache()
    _dashboard_helpers.clear_results_cache()

    space_loaded = _space_helpers.load_results_df(xvla_parquet)
    dash_loaded = _dashboard_helpers.load_results_parquet(xvla_parquet)
    assert dash_loaded is not None

    assert set(space_loaded["policy"]) == set(dash_loaded["policy"]) == {"act"}
