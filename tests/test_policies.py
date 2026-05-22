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


def test_pretrained_policies_in_default_yaml_are_locked() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    diff = registry.get("diffusion_policy")
    assert diff.is_baseline is False
    # SHA was locked Day 0a (2026-05-03); see docs/MODEL_CARDS.md.
    assert diff.revision_sha == "84a7c23178445c6bbf7e1a884ff497017910f653"
    assert diff.is_runnable() is True
    diff.assert_runnable()  # does not raise


def test_pretrained_spec_with_null_sha_is_not_runnable(tmp_path: Path) -> None:
    yaml_path = tmp_path / "policies.yaml"
    yaml_path.write_text(
        """
policies:
  - name: not_yet_locked
    is_baseline: false
    env_compat: [pusht]
    repo_id: lerobot/some_future_policy
    revision_sha: null
    fp_precision: fp32
"""
    )
    spec = PolicyRegistry.from_yaml(yaml_path).get("not_yet_locked")
    assert spec.is_runnable() is False
    with pytest.raises(ValueError, match=r"not runnable.*revision_sha"):
        spec.assert_runnable()


def test_supporting_filters_by_env_compat() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    pusht_policies = {p.name for p in registry.supporting("pusht")}
    # diffusion_pusht is PushT-only; act is Aloha-only (Day 0a env_compat fix).
    assert {"no_op", "random", "diffusion_policy"} <= pusht_policies
    assert "act" not in pusht_policies
    aloha_policies = {p.name for p in registry.supporting("aloha_transfer_cube")}
    assert {"no_op", "random", "act"} <= aloha_policies
    assert "diffusion_policy" not in aloha_policies


def test_runnable_includes_locked_pretrained_policies() -> None:
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    runnable_names = {p.name for p in registry.runnable()}
    # Day 0a: diffusion_policy + act SHAs locked.
    # Libero v2 integration: 5 VLA libero finetunes locked.
    assert runnable_names == {
        "no_op",
        "random",
        "diffusion_policy",
        "act",
        "pi05_libero_finetuned_v044",
        "pi0_libero_finetuned_v044",
        "pi0fast_libero",
        "xvla_libero",
        "smolvla_libero",
    }


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


# --------------------------------------------------------------------- #
# Libero v2 VLA policies                                                #
# --------------------------------------------------------------------- #


_VLA_LIBERO_LOCKED = {
    "pi05_libero_finetuned_v044": (
        "lerobot/pi05_libero_finetuned_v044",
        "dbf8a3f794a9c4297b44f40b752712f50073d945",
    ),
    "pi0_libero_finetuned_v044": (
        "lerobot/pi0_libero_finetuned_v044",
        "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
    ),
    "pi0fast_libero": (
        "lerobot/pi0fast-libero",
        "840f4b503f4c09110421c33c810a85b6684fd658",
    ),
    "xvla_libero": (
        "lerobot/xvla-libero",
        "12e8783e996944f5c97e490d37d4c145484ed70a",
    ),
    "smolvla_libero": (
        "lerobot/smolvla_libero",
        "31d453f7edd78c839a8bbc39744a292686daf0de",
    ),
}

_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@pytest.mark.parametrize("policy_name", sorted(_VLA_LIBERO_LOCKED))
def test_vla_libero_policy_loads_with_locked_sha(policy_name: str) -> None:
    """Each VLA libero policy entry has its locked SHA + repo_id and is runnable."""
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    spec = registry.get(policy_name)
    repo_id, sha = _VLA_LIBERO_LOCKED[policy_name]
    assert spec.is_baseline is False
    assert spec.repo_id == repo_id
    assert spec.revision_sha == sha
    assert spec.is_runnable() is True
    spec.assert_runnable()  # does not raise


@pytest.mark.parametrize("policy_name", sorted(_VLA_LIBERO_LOCKED))
def test_vla_libero_policy_env_compat_covers_all_4_suites(policy_name: str) -> None:
    """Every VLA libero entry advertises compat with all 4 LIBERO suites."""
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    spec = registry.get(policy_name)
    assert set(spec.env_compat) >= set(_LIBERO_SUITES)


def test_libero_suites_have_at_least_one_vla_policy() -> None:
    """Every LIBERO suite picks up at least the 5 VLA policies via supporting()."""
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    for suite in _LIBERO_SUITES:
        names = {p.name for p in registry.supporting(suite)}
        # Baselines + 5 VLAs minimum.
        assert {"no_op", "random"} <= names
        assert set(_VLA_LIBERO_LOCKED) <= names


def test_baselines_now_cover_libero_suites() -> None:
    """The libero v2 PR extends no_op + random env_compat to the 4 LIBERO suites."""
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    for baseline in ("no_op", "random"):
        spec = registry.get(baseline)
        assert set(spec.env_compat) >= set(_LIBERO_SUITES)


# --------------------------------------------------------------------- #
# Paper-reported success rates (added 2026-05-12).                      #
# --------------------------------------------------------------------- #


def test_baselines_have_no_paper_reported_success() -> None:
    """no_op / random carry no paper number — the field stays None."""
    registry = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    for name in ("no_op", "random"):
        spec = registry.get(name)
        assert spec.paper_reported_success is None


def test_diffusion_policy_has_pusht_paper_number() -> None:
    """LeRobot Hub model card: 65.4% success on PushT (500 episodes, binary)."""
    spec = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML).get("diffusion_policy")
    assert spec.paper_reported_success == {"pusht": 0.654}
    assert "65.4" in spec.paper_reported_notes


def test_act_has_aloha_paper_number() -> None:
    """Zhao et al. 2023 Table I, ACT row, Transfer (sim, human data) = 50%."""
    spec = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML).get("act")
    assert spec.paper_reported_success == {"aloha_transfer_cube": 0.50}


def test_smolvla_libero_has_all_four_suites() -> None:
    """SmolVLA paper Table 2, 0.45B row: 90/96/92/71 across the 4 LIBERO suites."""
    spec = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML).get("smolvla_libero")
    assert spec.paper_reported_success == {
        "libero_spatial": 0.90,
        "libero_object": 0.96,
        "libero_goal": 0.92,
        "libero_10": 0.71,
    }


def test_xvla_libero_has_all_four_suites() -> None:
    """X-VLA paper Table 2, 0.9B row: 98.2/98.6/97.8/97.6 across the 4 LIBERO suites."""
    spec = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML).get("xvla_libero")
    assert spec.paper_reported_success == {
        "libero_spatial": 0.982,
        "libero_object": 0.986,
        "libero_goal": 0.978,
        "libero_10": 0.976,
    }


def test_paper_reported_success_must_be_in_unit_interval(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: badnum
    is_baseline: false
    env_compat: [pusht]
    repo_id: some/repo
    revision_sha: abc123
    fp_precision: fp32
    paper_reported_success:
      pusht: 90.0   # forgot to divide by 100
""",
    )
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        PolicyRegistry.from_yaml(yaml)


def test_paper_reported_success_rejects_unknown_env(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: typo
    is_baseline: false
    env_compat: [pusht]
    repo_id: some/repo
    revision_sha: abc123
    fp_precision: fp32
    paper_reported_success:
      pushtt: 0.5   # typo, not in env_compat
""",
    )
    with pytest.raises(ValueError, match=r"not in env_compat"):
        PolicyRegistry.from_yaml(yaml)


def test_paper_reported_success_accepts_null_for_unreported_env(tmp_path: Path) -> None:
    """A null entry means 'paper does not report this env'. Preserve it."""
    yaml = _write(
        tmp_path,
        """
policies:
  - name: partial
    is_baseline: false
    env_compat: [pusht, aloha_transfer_cube]
    repo_id: some/repo
    revision_sha: abc123
    fp_precision: fp32
    paper_reported_success:
      pusht: 0.6
      aloha_transfer_cube: null
    paper_reported_notes: "Paper only reports PushT."
""",
    )
    spec = PolicyRegistry.from_yaml(yaml).get("partial")
    assert spec.paper_reported_success == {"pusht": 0.6, "aloha_transfer_cube": None}
    assert spec.paper_reported_notes == "Paper only reports PushT."


def test_paper_reported_success_must_be_mapping(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """
policies:
  - name: wrong
    is_baseline: false
    env_compat: [pusht]
    repo_id: some/repo
    revision_sha: abc123
    fp_precision: fp32
    paper_reported_success: 0.5   # should be a mapping, not a scalar
""",
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        PolicyRegistry.from_yaml(yaml)
