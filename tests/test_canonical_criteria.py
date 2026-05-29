"""Tests for the v1.1 canonical-criterion path.

Drives :func:`lerobot_bench.eval._run_one_episode` through synthetic
``ScriptedEnv`` rollouts so we can assert that:

* ``criterion='v1_legacy'`` reproduces v1.0 behaviour bit-for-bit
  (success rule = ``final_reward >= success_threshold``, no sticky
  accumulation, env's own ``max_steps`` is the cap).
* ``criterion='canonical'`` applies each env's overlay:
  PushT swaps to ``sticky_is_success`` (reads ``info["is_success"]``);
  Aloha swaps to ``sticky_reward_eq`` with ``strict_reward_value=4.0``;
  LIBERO suites keep the same scoring rule but raise the cap to 600.
* Backward-compat: existing v1.0 parquets do not need to be re-scored
  to remain valid -- :class:`EpisodeResult.success` is computed at
  write-time, not at read-time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lerobot_bench.checkpointing import RESULT_SCHEMA, append_cell_rows, load_results
from lerobot_bench.envs import (
    CRITERION_LABELS,
    CanonicalOverlay,
    EnvRegistry,
    EnvSpec,
)
from lerobot_bench.eval import EpisodeResult, _run_one_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"


# --------------------------------------------------------------------- #
# Synthetic env                                                         #
# --------------------------------------------------------------------- #


class ScriptedEnv:
    """Replays a hand-crafted per-step ``(reward, info, terminated, truncated)`` tape.

    Tape entries are consumed left-to-right by :meth:`step`. When the
    tape is exhausted the env behaves as if truncated -- but in practice
    the eval loop's ``max_steps`` cap should be the boundary we hit first
    in tests that exercise the cap.
    """

    def __init__(
        self,
        *,
        tape: list[tuple[float, dict[str, Any], bool, bool]],
    ) -> None:
        self._tape = tape
        self._cursor = 0

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._cursor = 0
        return {"obs": np.zeros(2, dtype=np.float32)}, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._cursor >= len(self._tape):
            # Tape exhausted -- emit a benign truncation. Tests that need
            # the cap to be the boundary set max_steps <= len(tape).
            return {"obs": np.zeros(2, dtype=np.float32)}, 0.0, False, True, {}
        reward, info, terminated, truncated = self._tape[self._cursor]
        self._cursor += 1
        return {"obs": np.zeros(2, dtype=np.float32)}, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _ZeroPolicy:
    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------- #
# Fixtures: build pusht / aloha / libero specs from the shipped YAML    #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def registry() -> EnvRegistry:
    return EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)


# --------------------------------------------------------------------- #
# 1. Overlay machinery                                                  #
# --------------------------------------------------------------------- #


def test_default_criterion_label_is_v1_legacy(registry: EnvRegistry) -> None:
    """``v1_legacy`` is one of the accepted labels and matches the YAML default."""
    assert "v1_legacy" in CRITERION_LABELS
    assert "canonical" in CRITERION_LABELS


def test_with_criterion_v1_legacy_returns_self(registry: EnvRegistry) -> None:
    pusht = registry.get("pusht")
    assert pusht.with_criterion("v1_legacy") is pusht


def test_with_criterion_unknown_raises(registry: EnvRegistry) -> None:
    pusht = registry.get("pusht")
    with pytest.raises(ValueError, match="unknown criterion"):
        pusht.with_criterion("paper")


def test_with_criterion_canonical_clears_overlay(registry: EnvRegistry) -> None:
    """Second application is a no-op (overlay was already consumed)."""
    libero = registry.get("libero_10").with_criterion("canonical")
    assert libero.max_steps == 600
    libero_again = libero.with_criterion("canonical")
    assert libero == libero_again


def test_registry_with_criterion_applies_to_every_spec(registry: EnvRegistry) -> None:
    canonical_registry = registry.with_criterion("canonical")
    # libero_object: 280 -> 600 (cap moves), success_metric unchanged.
    spec = canonical_registry.get("libero_object")
    assert spec.max_steps == 600
    assert spec.success_metric == "final_reward_threshold"
    # pusht: metric flips, cap stays.
    pusht = canonical_registry.get("pusht")
    assert pusht.success_metric == "sticky_is_success"
    assert pusht.max_steps == 300
    # aloha: metric flips, strict_reward_value populated, cap stays.
    aloha = canonical_registry.get("aloha_transfer_cube")
    assert aloha.success_metric == "sticky_reward_eq"
    assert aloha.strict_reward_value == pytest.approx(4.0)
    assert aloha.max_steps == 400


# --------------------------------------------------------------------- #
# 2. PushT — v1_legacy vs canonical                                     #
# --------------------------------------------------------------------- #


def _run(env: ScriptedEnv, spec: EnvSpec) -> EpisodeResult:
    return _run_one_episode(
        policy=_ZeroPolicy(),
        env=env,  # type: ignore[arg-type]
        episode_index=0,
        episode_seed=0,
        max_steps=spec.max_steps,
        success_threshold=spec.success_threshold,
        success_metric=spec.success_metric,
        strict_reward_value=spec.strict_reward_value,
        record_video=False,
    )


def test_pusht_v1_legacy_counts_lax_window_truncation(registry: EnvRegistry) -> None:
    """``coverage = 0.93`` => ``final_reward = 0.93/0.95 ≈ 0.979`` >= 0.95 => v1 counts as success."""
    pusht = registry.get("pusht")
    tape = [
        (0.5, {"is_success": False}, False, False),
        (0.93 / 0.95, {"is_success": False}, False, True),  # truncation at lax-window
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, pusht)
    assert result.success is True
    assert result.final_reward == pytest.approx(0.93 / 0.95)


def test_pusht_canonical_rejects_lax_window_truncation(registry: EnvRegistry) -> None:
    """Same tape under canonical: ``is_success`` never fired => no sticky success."""
    pusht = registry.get("pusht").with_criterion("canonical")
    tape = [
        (0.5, {"is_success": False}, False, False),
        (0.93 / 0.95, {"is_success": False}, False, True),
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, pusht)
    assert result.success is False


def test_pusht_canonical_accepts_sticky_is_success(registry: EnvRegistry) -> None:
    """Canonical fires when ``is_success`` is True at the terminating step."""
    pusht = registry.get("pusht").with_criterion("canonical")
    tape = [
        (0.5, {"is_success": False}, False, False),
        (
            1.0,
            {"is_success": True},
            True,
            False,
        ),  # gym-pusht terminates the moment is_success fires
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, pusht)
    assert result.success is True


def test_pusht_v1_legacy_and_canonical_both_count_clean_terminate(registry: EnvRegistry) -> None:
    pusht_v1 = registry.get("pusht")
    pusht_canonical = registry.get("pusht").with_criterion("canonical")
    tape = [(1.0, {"is_success": True}, True, False)]
    env_v1 = ScriptedEnv(tape=list(tape))
    env_can = ScriptedEnv(tape=list(tape))
    assert _run(env_v1, pusht_v1).success is True
    assert _run(env_can, pusht_canonical).success is True


# --------------------------------------------------------------------- #
# 3. Aloha — v1_legacy vs canonical                                     #
# --------------------------------------------------------------------- #


def test_aloha_v1_legacy_counts_reward_1_subgoal(registry: EnvRegistry) -> None:
    """v1 accepts ``final_reward >= 1.0`` => reward 1 ("touched") counts."""
    aloha = registry.get("aloha_transfer_cube")
    tape = [
        (0.0, {}, False, False),
        (1.0, {"is_success": False}, False, True),  # truncation while touching
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, aloha)
    assert result.success is True


def test_aloha_canonical_rejects_reward_1_subgoal(registry: EnvRegistry) -> None:
    """Canonical only counts ``reward == 4`` (Transfer subtask)."""
    aloha = registry.get("aloha_transfer_cube").with_criterion("canonical")
    tape = [
        (0.0, {}, False, False),
        (1.0, {"is_success": False}, False, True),
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, aloha)
    assert result.success is False


def test_aloha_canonical_accepts_reward_4_transfer(registry: EnvRegistry) -> None:
    aloha = registry.get("aloha_transfer_cube").with_criterion("canonical")
    tape = [
        (0.0, {}, False, False),
        (1.0, {}, False, False),
        (4.0, {"is_success": True}, True, False),  # gym-aloha terminates on reward == 4
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, aloha)
    assert result.success is True


def test_aloha_canonical_sticky_persists_across_decay(registry: EnvRegistry) -> None:
    """sticky_reward_eq OR-accumulates: a transient ``reward == 4`` counts even if the
    final step's reward decays. (Defensive — gym-aloha auto-terminates on 4, but the
    sticky reduction is the contract.)"""
    aloha = registry.get("aloha_transfer_cube").with_criterion("canonical")
    tape = [
        (4.0, {}, False, False),  # transfer
        (1.0, {}, False, True),  # decayed at truncation
    ]
    env = ScriptedEnv(tape=tape)
    result = _run(env, aloha)
    assert result.success is True


# --------------------------------------------------------------------- #
# 4. LIBERO — same rule, raised cap                                     #
# --------------------------------------------------------------------- #


def test_libero_canonical_raises_cap_to_600(registry: EnvRegistry) -> None:
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        spec = registry.get(suite).with_criterion("canonical")
        assert spec.max_steps == 600, suite
        assert spec.success_metric == "final_reward_threshold"


def test_libero_v1_truncates_before_late_success(registry: EnvRegistry) -> None:
    """v1 cap = 280 for libero_spatial: a success at step 350 is unreachable."""
    spec = registry.get("libero_spatial")
    assert spec.max_steps == 280
    tape = [(0.0, {}, False, False)] * 349 + [(1.0, {"is_success": True}, True, False)]
    env = ScriptedEnv(tape=tape)
    result = _run(env, spec)
    # Loop ran 280 zero-reward steps and never saw the success entry.
    assert result.success is False
    assert result.n_steps == 280


def test_libero_canonical_reaches_late_success(registry: EnvRegistry) -> None:
    """Same tape under canonical's 600 cap: the success at step 350 is reached."""
    spec = registry.get("libero_spatial").with_criterion("canonical")
    assert spec.max_steps == 600
    tape = [(0.0, {}, False, False)] * 349 + [(1.0, {"is_success": True}, True, False)]
    env = ScriptedEnv(tape=tape)
    result = _run(env, spec)
    assert result.success is True
    assert result.n_steps == 350


def test_libero_scoring_rule_is_bit_equivalent(registry: EnvRegistry) -> None:
    """Identical tape => identical success column under both criteria for LIBERO."""
    v1_spec = registry.get("libero_object")
    can_spec = registry.get("libero_object").with_criterion("canonical")
    tape = [(0.0, {}, False, False)] * 100 + [(1.0, {"is_success": True}, True, False)]
    env_v1 = ScriptedEnv(tape=list(tape))
    env_can = ScriptedEnv(tape=list(tape))
    assert _run(env_v1, v1_spec).success is True
    assert _run(env_can, can_spec).success is True


# --------------------------------------------------------------------- #
# 5. Validation of the spec API                                         #
# --------------------------------------------------------------------- #


def test_envspec_rejects_unknown_success_metric() -> None:
    with pytest.raises(ValueError, match="success_metric must be one of"):
        EnvSpec(
            name="bad",
            family="bad",
            max_steps=10,
            success_threshold=1.0,
            lerobot_module="bad",
            gym_id="x/y",
            success_metric="not_a_metric",
        )


def test_envspec_sticky_reward_eq_requires_strict_value() -> None:
    with pytest.raises(ValueError, match="requires strict_reward_value"):
        EnvSpec(
            name="bad",
            family="bad",
            max_steps=10,
            success_threshold=1.0,
            lerobot_module="bad",
            gym_id="x/y",
            success_metric="sticky_reward_eq",
        )


def test_run_one_episode_sticky_reward_eq_requires_strict_value(
    registry: EnvRegistry,
) -> None:
    """Belt-and-braces: the inner loop also rejects a missing strict_reward_value."""
    pusht = registry.get("pusht")
    env = ScriptedEnv(tape=[(0.0, {}, False, True)])
    with pytest.raises(ValueError, match="requires strict_reward_value"):
        _run_one_episode(
            policy=_ZeroPolicy(),
            env=env,  # type: ignore[arg-type]
            episode_index=0,
            episode_seed=0,
            max_steps=pusht.max_steps,
            success_threshold=pusht.success_threshold,
            success_metric="sticky_reward_eq",
            strict_reward_value=None,
            record_video=False,
        )


def test_canonical_overlay_unknown_field_rejected(tmp_path: Path) -> None:
    yaml_path = tmp_path / "envs.yaml"
    yaml_path.write_text(
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 300
    success_threshold: 0.95
    lerobot_module: lerobot.envs.pusht
    canonical:
      typo_field: 600
"""
    )
    with pytest.raises(ValueError, match="canonical has unknown fields"):
        EnvRegistry.from_yaml(yaml_path)


def test_canonical_overlay_bad_max_steps_rejected(tmp_path: Path) -> None:
    yaml_path = tmp_path / "envs.yaml"
    yaml_path.write_text(
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 300
    success_threshold: 0.95
    lerobot_module: lerobot.envs.pusht
    canonical:
      max_steps: -1
"""
    )
    with pytest.raises(ValueError, match=r"canonical\.max_steps must be a positive int"):
        EnvRegistry.from_yaml(yaml_path)


# --------------------------------------------------------------------- #
# 6. Backward-compat: v1.0 parquets remain readable                     #
# --------------------------------------------------------------------- #


def test_v1_parquet_success_column_is_not_recomputed_on_read(tmp_path: Path) -> None:
    """Library code does not re-score on read -- a v1.0 parquet's
    ``success`` column survives a roundtrip under v1.1 even if the
    canonical rule would have produced a different value."""
    out = tmp_path / "results.parquet"
    df = pd.DataFrame(
        [
            {
                "policy": "diffusion_policy",
                "env": "pusht",
                "seed": 0,
                "episode_index": 0,
                # v1 rule: success because final_reward >= 0.95 (lax-window truncation).
                "success": True,
                "return_": 12.5,
                "n_steps": 300,
                "wallclock_s": 5.0,
                "video_sha256": "",
                "code_sha": "abc",
                "lerobot_version": "0.5.1",
                "timestamp_utc": "2026-05-26T00:00:00+00:00",
            }
        ],
        columns=list(RESULT_SCHEMA),
    )
    append_cell_rows(out, df)
    roundtrip = load_results(out)
    assert roundtrip["success"].tolist() == [True]


def test_canonical_overlay_default_is_empty(registry: EnvRegistry) -> None:
    """An env without a YAML ``canonical:`` block has an all-None overlay -- so
    ``with_criterion('canonical')`` is a no-op for it."""
    spec = EnvSpec(
        name="extra",
        family="extra",
        max_steps=50,
        success_threshold=0.5,
        lerobot_module="extra",
        gym_id="x/y",
    )
    assert spec.canonical == CanonicalOverlay()
    canonical = spec.with_criterion("canonical")
    assert canonical.max_steps == spec.max_steps
    assert canonical.success_metric == spec.success_metric
    assert canonical.success_threshold == spec.success_threshold
