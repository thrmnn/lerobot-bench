"""Tests for ``lerobot_bench.policies``."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_bench.policies import PolicyRegistry, PolicySpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"


# --------------------------------------------------------------------- #
# Loading the shipped configs/policies.yaml                             #
# --------------------------------------------------------------------- #


def test_default_policies_yaml_loads() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    # Baselines must always be present.
    assert "no_op" in registry
    assert "random" in registry


def test_baselines_are_runnable_out_of_the_box() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    no_op = registry.get("no_op")
    assert no_op.is_baseline is True
    assert no_op.is_runnable() is True
    no_op.assert_runnable()  # does not raise


def test_pretrained_policies_with_unlocked_sha_are_not_runnable() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    diff = registry.get("diffusion_policy")
    assert diff.is_baseline is False
    # revision_sha is intentionally null in the shipped YAML — Day 0a TODO.
    assert diff.revision_sha is None
    assert diff.is_runnable() is False
    with pytest.raises(ValueError, match=r"not runnable.*revision_sha"):
        diff.assert_runnable()


def test_supporting_filters_by_env_compat() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    pusht_policies = {p.name for p in registry.supporting("pusht")}
    assert {"no_op", "random", "diffusion_policy", "act"} <= pusht_policies


def test_runnable_excludes_unlocked_pretrained_policies() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    runnable_names = {p.name for p in registry.runnable()}
    # Only the baselines are runnable until Day 0a fills in revision_sha.
    assert runnable_names == {"no_op", "random"}


def test_get_unknown_policy_lists_available() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    with pytest.raises(KeyError, match="unknown policy 'fakenet'"):
        registry.get("fakenet")


# --------------------------------------------------------------------- #
# Schema validation                                                     #
# --------------------------------------------------------------------- #


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "policies.yaml"
    p.write_text(body)
    return p


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: foo
    is_baseline: true
    # missing env_compat
""",
    )
    with pytest.raises(ValueError, match=r"missing required fields.*env_compat"):
        PolicyRegistry.from_yaml(yaml)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: foo
    is_baseline: true
    env_compat: [pusht]
    bogus: field
""",
    )
    with pytest.raises(ValueError, match=r"unknown fields.*bogus"):
        PolicyRegistry.from_yaml(yaml)


def test_baseline_with_weights_metadata_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: weird_baseline
    is_baseline: true
    env_compat: [pusht]
    repo_id: some/repo
""",
    )
    with pytest.raises(ValueError, match="baseline policies must not set"):
        PolicyRegistry.from_yaml(yaml)


def test_invalid_fp_precision_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: fancy
    is_baseline: false
    env_compat: [pusht]
    repo_id: some/repo
    revision_sha: abc123
    fp_precision: int4  # not supported
""",
    )
    with pytest.raises(ValueError, match="fp_precision must be one of"):
        PolicyRegistry.from_yaml(yaml)


def test_env_compat_must_be_list_of_strings(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: foo
    is_baseline: true
    env_compat: "not_a_list"
""",
    )
    with pytest.raises(ValueError, match="env_compat must be a list of strings"):
        PolicyRegistry.from_yaml(yaml)


def test_duplicate_policy_name_rejected(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: same
    is_baseline: true
    env_compat: [pusht]
  - name: same
    is_baseline: true
    env_compat: [aloha_transfer_cube]
""",
    )
    with pytest.raises(ValueError, match="duplicate policy name 'same'"):
        PolicyRegistry.from_yaml(yaml)


def test_iter_and_len() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    specs = list(registry)
    assert len(specs) == len(registry)
    assert all(isinstance(s, PolicySpec) for s in specs)
