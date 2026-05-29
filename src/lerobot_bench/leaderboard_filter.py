"""Single source of truth for the v1 leaderboard policy gate.

Both public surfaces -- the Hub-hosted Gradio Space (``space/_helpers.py``)
and the local-first sweep dashboard (``dashboard/_helpers.py``) -- must
expose the *same* v1 policy set and apply the *same* xvla-exclusion
filter. They used to redefine both independently, which let the two
surfaces drift apart silently (PR #82 added the filter to both at once,
but nothing pinned them to stay equal). This module is the canonical
definition; the two helpers re-export from it.

Kept deliberately dependency-light: ``pandas`` only, no gradio / torch /
``lerobot_bench`` package imports. That lets the Space's slim pytest job
and the dashboard's importlib-from-file loader both pull it in without
dragging the heavy benchmark deps into their import graph.
"""

from __future__ import annotations

import pandas as pd

# Single source of truth for the v1 leaderboard policy set. The published
# parquet still carries xvla_libero rows for reproducibility (PR #76
# deferred xvla to v1.1 after two patched + one unresolved Hub-JSON
# processor bugs), but the public surfaces must not include them -- xvla
# biases the headline numbers downward and confuses reviewers. Both
# surfaces drop non-v1 policies right after parquet load, so every
# downstream aggregate (Wilson CIs, paired bootstrap, MDE) is computed on
# the v1-only frame.
V1_POLICIES: tuple[str, ...] = (
    "act",
    "diffusion_policy",
    "smolvla_libero",
    "no_op",
    "random",
)


def filter_to_v1_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``policy`` is not in :data:`V1_POLICIES`.

    Applied right after the schema check on every parquet read so every
    downstream leaderboard aggregate (Wilson CIs, paired bootstrap, MDE,
    rollout dropdowns, failure counts) sees the v1 policy set only. The
    published parquet still carries xvla_libero rows for reproducibility;
    this filter is the public-surface gate.

    An empty frame or one missing the ``policy`` column is returned
    unchanged -- cold-start empty frames and partially-written parquets
    pass through rather than raising.
    """
    if df.empty or "policy" not in df.columns:
        return df
    return df[df["policy"].isin(V1_POLICIES)].reset_index(drop=True)
