"""Shared pytest fixtures.

Mark conventions:
- ``@pytest.mark.slow``: tests that take more than ~5 seconds.
- ``@pytest.mark.gpu``: tests requiring CUDA.
- ``@pytest.mark.sim``: tests requiring a sim env (mujoco, gym-pusht, etc.).

Default CI runs ``pytest -m "not slow and not gpu and not sim"``.
"""

from __future__ import annotations
