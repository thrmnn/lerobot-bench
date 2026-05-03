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


# --------------------------------------------------------------------- #
# Factory path validation                                               #
# --------------------------------------------------------------------- #


def test_envspec_accepts_factory_path() -> None:
    spec = EnvSpec(
        name="libero_spatial",
        family="libero",
        max_steps=280,
        success_threshold=1.0,
        lerobot_module="lerobot.envs.libero",
        factory="lerobot.envs.factory",
        factory_kwargs=(("env_type", "libero"), ("task", "libero_spatial")),
    )
    assert spec.uses_factory is True
    assert spec.gym_id is None
    assert spec.factory_kwargs_dict() == {"env_type": "libero", "task": "libero_spatial"}


def test_envspec_accepts_gym_path_default() -> None:
    spec = EnvSpec(
        name="pusht",
        family="pusht",
        max_steps=300,
        success_threshold=0.95,
        lerobot_module="lerobot.envs.pusht",
        gym_id="gym_pusht/PushT-v0",
    )
    assert spec.uses_factory is False
    assert spec.factory is None


def test_envspec_rejects_neither_gym_id_nor_factory() -> None:
    with pytest.raises(ValueError, match="exactly one of 'gym_id' or 'factory'"):
        EnvSpec(
            name="bad",
            family="bad",
            max_steps=10,
            success_threshold=0.5,
            lerobot_module="bad",
        )


def test_envspec_rejects_both_gym_id_and_factory() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        EnvSpec(
            name="bad",
            family="bad",
            max_steps=10,
            success_threshold=0.5,
            lerobot_module="bad",
            gym_id="x/y",
            factory="some.module",
        )


def test_yaml_loader_rejects_neither_set(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: bad
    family: bad
    max_steps: 10
    success_threshold: 0.5
    lerobot_module: bad
""",
    )
    with pytest.raises(ValueError, match="exactly one of 'gym_id' or 'factory'"):
        EnvRegistry.from_yaml(yaml)


def test_yaml_loader_rejects_both_set(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: bad
    family: bad
    gym_id: x/y
    factory: some.module
    max_steps: 10
    success_threshold: 0.5
    lerobot_module: bad
""",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        EnvRegistry.from_yaml(yaml)


def test_yaml_loader_roundtrip_preserves_factory_fields(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: libero_spatial
    family: libero
    factory: lerobot.envs.factory
    factory_kwargs:
      env_type: libero
      task: libero_spatial
      task_ids: [0]
    max_steps: 280
    success_threshold: 1.0
    lerobot_module: lerobot.envs.libero
""",
    )
    registry = EnvRegistry.from_yaml(yaml)
    spec = registry.get("libero_spatial")
    assert spec.uses_factory is True
    assert spec.factory == "lerobot.envs.factory"
    # Lists are coerced to tuples so the spec stays hashable.
    assert spec.factory_kwargs_dict() == {
        "env_type": "libero",
        "task": "libero_spatial",
        "task_ids": (0,),
    }
    assert spec.gym_id is None
    assert spec.gym_kwargs == ()
    # Hashable + equality holds across re-load.
    spec2 = EnvRegistry.from_yaml(yaml).get("libero_spatial")
    assert spec == spec2
    assert hash(spec) == hash(spec2)


def test_yaml_loader_factory_kwargs_must_be_mapping(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
envs:
  - name: bad
    family: bad
    factory: some.module
    factory_kwargs: not_a_mapping
    max_steps: 10
    success_threshold: 0.5
    lerobot_module: bad
""",
    )
    with pytest.raises(ValueError, match="factory_kwargs must be a mapping"):
        EnvRegistry.from_yaml(yaml)


def test_default_libero_envs_load() -> None:
    """The 4 LIBERO suites land in the shipped configs/envs.yaml as factory specs."""
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        spec = registry.get(suite)
        assert spec.family == "libero"
        assert spec.uses_factory is True
        assert spec.factory == "lerobot.envs.factory"
        kwargs = spec.factory_kwargs_dict()
        assert kwargs["env_type"] == "libero"
        assert kwargs["task"] == suite
        assert kwargs["task_ids"] == (0,)
        assert kwargs["obs_type"] == "pixels_agent_pos"
        assert spec.success_threshold == pytest.approx(1.0)
        assert spec.max_steps > 0


def test_pusht_aloha_still_use_gym_path() -> None:
    """Backward-compat: PushT and Aloha specs are unchanged by the factory addition."""
    registry = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    pusht = registry.get("pusht")
    aloha = registry.get("aloha_transfer_cube")
    assert pusht.uses_factory is False
    assert pusht.gym_id == "gym_pusht/PushT-v0"
    assert aloha.uses_factory is False
    assert aloha.gym_id == "gym_aloha/AlohaTransferCube-v0"
