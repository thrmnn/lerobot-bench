"""Shared pytest fixtures.

Mark conventions:
- ``@pytest.mark.slow``: tests that take more than ~5 seconds.
- ``@pytest.mark.gpu``: tests requiring CUDA.
- ``@pytest.mark.sim``: tests requiring a sim env (mujoco, gym-pusht, etc.).

Default CI runs ``pytest -m "not slow and not gpu and not sim"``.
"""

from __future__ import annotations

import pytest

# Pre-existing test modules that call ``publish_results._preflight`` with
# intentionally coverage-incomplete parquets (built to exercise the OTHER
# gates: auth, dry-run, oversize, idempotence, upload failure, and the
# act×aloha stale-rows floor). The #165 coverage gate would trip all of
# them. We scope the neutralizer to exactly these modules so the new
# coverage suite still drives the real gate.
_LEGACY_PREFLIGHT_TEST_MODULES = frozenset(
    {"test_publish_results", "test_merge_corrected_act_rows"}
)


@pytest.fixture(autouse=True)
def _disable_publish_coverage_gate_for_legacy_tests(
    request: pytest.FixtureRequest,
) -> None:
    """Neutralize the #165 publish coverage gate for the pre-existing publish suites.

    ``scripts/publish_results._preflight`` gained a REQUIRED (policy, env)
    coverage gate in #165. The legacy modules in
    :data:`_LEGACY_PREFLIGHT_TEST_MODULES` deliberately build minimal /
    single-policy parquets to exercise the *other* gates; those parquets
    are coverage-incomplete by design and would now trip the new gate.
    This autouse fixture is scoped to those modules and swaps the coverage
    provider for an empty REQUIRED set so they keep asserting the behavior
    they were written for. The new
    ``tests/test_publish_preflight_coverage.py`` is NOT in the set -- it
    drives the gate directly.
    """
    short_name = request.module.__name__.rsplit(".", 1)[-1]
    if short_name not in _LEGACY_PREFLIGHT_TEST_MODULES:
        return
    from scripts import publish_results

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(
        publish_results,
        "_required_coverage_pairs_for_preflight",
        lambda: frozenset(),
    )
