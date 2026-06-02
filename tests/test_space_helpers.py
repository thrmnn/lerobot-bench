"""Pin the Space's video-URL builder to the publisher's flat naming.

The on-disk / on-Hub MP4 corpus is FLAT:
``{policy}__{env}__seed{seed}__ep{episode:03d}.mp4`` (see
``scripts/publish_results.py::_video_filename``). The Space's
``format_video_url`` must produce ``.../resolve/main/videos/<that
filename>`` or every ``gr.Video`` panel 404s.

We import the publisher's ``_video_filename`` directly so the two
naming schemes cannot drift apart silently -- if the publisher changes
its format string, this test fails until the Space helper is updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scripts.publish_results import _video_filename

# space/ is a sibling of tests/, not on sys.path by default.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPACE_DIR = _REPO_ROOT / "space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from _helpers import (  # noqa: E402  (sys.path mutation must precede import)
    HUB_RAW_PREFIX,
    format_video_url,
    video_filename,
)

_CASES = [
    ("act", "pusht", 0, 0),
    ("diffusion", "aloha", 7, 5),
    ("smolvla", "libero", 42, 123),
    ("no_op", "pusht", 100, 9),
]


@pytest.mark.parametrize(("policy", "env", "seed", "episode"), _CASES)
def test_video_filename_matches_publisher(policy, env, seed, episode):
    """The Space's filename must equal the publisher's, byte-for-byte."""
    expected = _video_filename(policy=policy, env=env, seed=seed, episode_index=episode)
    assert video_filename(policy, env, seed, episode) == expected


@pytest.mark.parametrize(("policy", "env", "seed", "episode"), _CASES)
def test_format_video_url_is_flat_and_resolves(policy, env, seed, episode):
    """URL is ``<resolve>/videos/<flat-name>`` with no nested dirs."""
    url = format_video_url(policy, env, seed, episode)
    flat = _video_filename(policy=policy, env=env, seed=seed, episode_index=episode)
    assert url == f"{HUB_RAW_PREFIX}/videos/{flat}"
    # The old nested layout (videos/<policy>/<env>/seed<N>/...) must be gone.
    assert f"/videos/{policy}/{env}/" not in url
    assert "/resolve/main/" in url


def test_episode_is_zero_padded_to_three():
    """Episode index is zero-padded to 3 digits; seed is not padded."""
    assert format_video_url("act", "pusht", 5, 3).endswith("act__pusht__seed5__ep003.mp4")
    assert format_video_url("act", "pusht", 5, 100).endswith("act__pusht__seed5__ep100.mp4")


@pytest.mark.parametrize(("seed", "episode"), [(-1, 0), (0, -1)])
def test_negative_indices_rejected(seed, episode):
    with pytest.raises(ValueError):
        format_video_url("act", "pusht", seed, episode)
