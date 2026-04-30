"""Tests for ``lerobot_bench.envs``."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_bench.envs import EnvRegistry, EnvSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"


# --------------------------------------------------------------------- #
# Loading the shipped configs/envs.yaml                                 #
# --------------------------------------------------------------------- #


def test_default_envs_yaml_loads() -> None:
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    assert "pusht" in registry
    assert "aloha_transfer_cube" in registry


def test_default_pusht_spec() -> None:
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    pusht = registry.get("pusht")
    assert pusht.family == "pusht"
    assert pusht.max_steps == 300
    assert pusht.success_threshold == pytest.approx(0.95)
    assert pusht.lerobot_module.startswith("lerobot.envs.")


def test_by_family_groups_correctly() -> None:
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    pusht_family = registry.by_family("pusht")
    assert {s.name for s in pusht_family} == {"pusht"}
    aloha_family = registry.by_family("aloha")
    assert all(s.family == "aloha" for s in aloha_family)


def test_get_unknown_env_lists_available(tmp_path: Path) -> None:
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    with pytest.raises(KeyError, match="unknown env 'totally_fake'"):
        registry.get("totally_fake")


def test_iter_and_len(tmp_path: Path) -> None:
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    specs = list(registry)
    assert len(specs) == len(registry)
    assert all(isinstance(s, EnvSpec) for s in specs)


# --------------------------------------------------------------------- #
# Schema validation                                                     #
# --------------------------------------------------------------------- #


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "envs.yaml"
    p.write_text(body)
    return p


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 300
    success_threshold: 0.95
    # missing lerobot_module
""",
    )
    with pytest.raises(ValueError, match=r"missing required fields.*lerobot_module"):
        EnvRegistry.from_yaml(yaml)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 300
    success_threshold: 0.95
    lerobot_module: lerobot.envs.pusht
    extra_typo_field: oops
""",
    )
    with pytest.raises(ValueError, match=r"unknown fields.*extra_typo_field"):
        EnvRegistry.from_yaml(yaml)


def test_negative_max_steps_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: -1
    success_threshold: 0.95
    lerobot_module: lerobot.envs.pusht
""",
    )
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        EnvRegistry.from_yaml(yaml)


def test_duplicate_env_name_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 300
    success_threshold: 0.95
    lerobot_module: lerobot.envs.pusht
  - name: pusht
    family: pusht
    gym_id: gym_pusht/PushT-v0
    max_steps: 400
    success_threshold: 0.99
    lerobot_module: lerobot.envs.pusht
""",
    )
    with pytest.raises(ValueError, match="duplicate env name 'pusht'"):
        EnvRegistry.from_yaml(yaml)


def test_top_level_must_be_mapping_with_envs_key(tmp_path: Path) -> None:
    yaml = _write(tmp_path, "- not_a_mapping\n")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        EnvRegistry.from_yaml(yaml)
