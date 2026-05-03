"""Asserts the MDE table in ``docs/MDE_TABLE.md`` matches what
``lerobot_bench.stats`` actually computes.

The doc is the source of truth for every "is this comparison powered?"
claim in ``paper/main.tex`` § Methods and the
``notebooks/01-write-finding.ipynb`` leaderboard gate. If anyone edits
the doc table by hand without re-running the math, these tests fail
and CI blocks the PR.

The simulation test uses a fixed master seed (``42``) so the
Monte-Carlo MDE is byte-reproducible; the asserted range is loose
([0.10, 0.20]) so a future bootstrap implementation tweak that shifts
the MDE by one Δ-grid step does not flake the test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lerobot_bench.stats import wilson_ci

REPO_ROOT = Path(__file__).resolve().parent.parent
MDE_DOC = REPO_ROOT / "docs" / "MDE_TABLE.md"
CALIBRATE_MDE_SCRIPT = REPO_ROOT / "scripts" / "calibrate_mde.py"

# Quoted in docs/MDE_TABLE.md § TL;DR. If you change one, change both.
DOC_HW_AT_P050_N250 = 0.0615
DOC_INCONCLUSIVE_BAND_AT_P050_N250 = 0.1230


def _wilson_half_width(p: float, n: int) -> float:
    """Wilson 95% half-width at the integer ``successes = round(p · n)``."""
    successes = round(p * n)
    lo, hi = wilson_ci(successes, n)
    return (hi - lo) / 2.0


# --------------------------------------------------------------------- #
# 1. The two numbers the paper quotes match the function.               #
# --------------------------------------------------------------------- #


def test_wilson_hw_at_p050_n250_matches_doc() -> None:
    """``wilson_ci(125, 250)`` half-width matches the 0.0615 quoted in the doc.

    Source of truth: ``docs/MDE_TABLE.md`` § TL;DR row 1.
    """
    hw = _wilson_half_width(0.5, 250)
    assert hw == pytest.approx(DOC_HW_AT_P050_N250, abs=1e-4), (
        f"Wilson HW at p=0.5, N=250 = {hw:.6f}; "
        f"docs/MDE_TABLE.md claims {DOC_HW_AT_P050_N250}. "
        "If you changed wilson_ci, update the doc and the constant in this test."
    )


def test_inconclusive_band_at_p050_n250_matches_doc() -> None:
    """``2 × HW`` matches the 0.1230 inconclusive band quoted in the doc.

    Source of truth: ``docs/MDE_TABLE.md`` § TL;DR row 2 + § 4.
    The paper text says ``≈ 0.124`` — that rounds correctly from 0.1230
    but we hold the precise value here because the paper has been
    updated to the precise number with a citation back to this doc.
    """
    hw = _wilson_half_width(0.5, 250)
    band = 2.0 * hw
    assert band == pytest.approx(DOC_INCONCLUSIVE_BAND_AT_P050_N250, abs=1e-4)


# --------------------------------------------------------------------- #
# 2. Wilson CI is symmetric around p=0.5 (within rounding).             #
# --------------------------------------------------------------------- #


def test_wilson_hw_symmetric_around_p050() -> None:
    """``HW(p, N) == HW(1-p, N)`` for the doc grid, modulo integer rounding.

    The Wilson interval is symmetric in ``(p, 1-p)`` analytically; the
    ``round(p · N)`` discretisation introduces at most a 1-success
    asymmetry when ``p · N`` is not an integer. Asserting equality to
    1e-9 is safe because the chosen ``p`` values produce integer
    ``successes`` at N=250.
    """
    for p in (0.05, 0.10, 0.25):
        hw_lo = _wilson_half_width(p, 250)
        hw_hi = _wilson_half_width(1 - p, 250)
        assert hw_lo == pytest.approx(hw_hi, abs=1e-9), (
            f"HW({p}) = {hw_lo:.6f} != HW({1 - p}) = {hw_hi:.6f}"
        )


# --------------------------------------------------------------------- #
# 3. The simulated paired MDE at the seed-42 contract is in range.      #
# --------------------------------------------------------------------- #


def test_simulated_paired_mde_at_p050_in_expected_range() -> None:
    """At ``(p=0.5, N=250, ρ=0, seed=42)`` the bootstrap-CI-based MDE is in [0.10, 0.20].

    Loose range — the simulation has Monte-Carlo noise even under fixed
    seed if the Δ-grid step happens to straddle the 80% power line.
    The doc reports MDE = 0.15 at this cell; we assert membership in
    the wider [0.10, 0.20] band so a small bootstrap-implementation
    tweak does not flake CI.

    Uses a small ``n_outer`` (200) and ``n_boot`` (1000) for test speed
    — the full doc table uses 1000 × 2000 — but the same seed contract.
    """
    from scripts.calibrate_mde import paired_bernoulli

    from lerobot_bench.stats import paired_delta_bootstrap

    rng = np.random.default_rng(42)
    n_pairs = 250
    p = 0.5
    rho = 0.0
    n_outer = 200
    n_boot = 1000

    mde: float | None = None
    for delta in (0.05, 0.10, 0.15, 0.20):
        rejections = 0
        for _ in range(n_outer):
            a, b = paired_bernoulli(n_pairs, p + delta, p, rho, rng)
            res = paired_delta_bootstrap(a, b, rng=rng, n_resamples=n_boot)
            if res.lo > 0.0 or res.hi < 0.0:
                rejections += 1
        power = rejections / n_outer
        if power >= 0.80:
            mde = delta
            break

    assert mde is not None, "no Δ in {0.05..0.20} achieved 80% power — simulation broken"
    assert 0.10 <= mde <= 0.20, (
        f"simulated MDE at (p=0.5, N=250, ρ=0, seed=42) = {mde}; "
        "expected in [0.10, 0.20] per docs/MDE_TABLE.md § 2b. "
        "If the paired_delta_bootstrap implementation changed, re-run "
        "scripts/calibrate_mde.py and update the doc."
    )


# --------------------------------------------------------------------- #
# 4. The doc's "inconclusive band" column is exactly 2 × half-width.    #
# --------------------------------------------------------------------- #


def test_doc_inconclusive_band_equals_twice_half_width() -> None:
    """Parse ``docs/MDE_TABLE.md`` § 1 and verify each row's band == 2·HW.

    Catches manual-edit drift on the doc — if someone bumps the
    half-width column without updating the band column (or vice
    versa), this fails.
    """
    text = MDE_DOC.read_text()

    # Find the table block under the "## 1. Per-cell Wilson..." heading.
    section = re.search(
        r"## 1\. Per-cell Wilson 95% CI half-width.*?(?=\n## )",
        text,
        flags=re.DOTALL,
    )
    assert section is not None, "Could not locate § 1 table block in MDE_TABLE.md"

    row_re = re.compile(
        r"^\|\s*(?P<p>[\d.]+)\s*\|\s*(?P<lo>[\d.]+)\s*\|\s*(?P<hi>[\d.]+)\s*"
        r"\|\s*(?P<hw>[\d.]+)\s*\|\s*(?P<band>[\d.]+)\s*\|$",
        re.MULTILINE,
    )

    rows = list(row_re.finditer(section.group(0)))
    assert len(rows) >= 9, f"expected ≥ 9 rows in § 1 table, got {len(rows)}"

    for row in rows:
        p = float(row.group("p"))
        hw_doc = float(row.group("hw"))
        band_doc = float(row.group("band"))

        # Internal consistency on the doc row itself.
        assert band_doc == pytest.approx(2 * hw_doc, abs=1e-3), (
            f"row p={p}: band {band_doc} != 2 × HW {hw_doc}"
        )

        # External consistency: the doc value matches what wilson_ci computes.
        hw_real = _wilson_half_width(p, 250)
        assert hw_doc == pytest.approx(hw_real, abs=5e-4), (
            f"row p={p}: doc HW={hw_doc} != computed HW={hw_real:.6f}"
        )


# --------------------------------------------------------------------- #
# 5. AST guard — the simulation script does not import torch / lerobot.  #
# --------------------------------------------------------------------- #


def test_calibrate_mde_script_has_no_heavy_top_level_imports() -> None:
    """``scripts/calibrate_mde.py`` must not import torch / lerobot / gymnasium at module load.

    The doc generation must run on a clean CI box without GPU runtimes.
    Mirrors the AST guards in ``test_calibrate.py`` / ``test_run_one.py``
    / ``test_publish_results.py``.
    """
    import ast

    tree = ast.parse(CALIBRATE_MDE_SCRIPT.read_text())
    forbidden = {"torch", "lerobot", "gymnasium", "gym"}
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in forbidden:
                bad.append(f"from {node.module} import ...")
    assert not bad, f"calibrate_mde.py imports {bad} at top level; must be lazy or removed"


def test_calibrate_mde_script_runs_pilot_cleanly() -> None:
    """End-to-end smoke test that ``calibrate_mde.py --pilot`` exits 0."""
    result = subprocess.run(
        ["python", str(CALIBRATE_MDE_SCRIPT), "--pilot", "--n-outer", "20"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"calibrate_mde.py --pilot failed:\n{result.stderr}"
    assert "MDE table" in result.stdout
