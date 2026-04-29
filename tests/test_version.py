"""Smoke test that the package imports and exposes a version string."""

from __future__ import annotations

import re

import lerobot_bench


def test_version_string_is_semver() -> None:
    assert isinstance(lerobot_bench.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-+].*)?", lerobot_bench.__version__)


def test_version_matches_version_file() -> None:
    """The VERSION file at the repo root must match the package version."""
    from pathlib import Path

    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    on_disk = version_file.read_text().strip()
    assert on_disk == lerobot_bench.__version__
