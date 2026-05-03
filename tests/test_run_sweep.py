"""Unit tests for ``scripts/run_sweep.py`` -- YAML schema, expansion, CLI surface.

The end-to-end resume / dispatch / OOM drills live in
``tests/test_resume_drill.py`` (the headline file). This sibling file
covers the lower layers: SweepConfig validation, expand_cells,
overrides merge, --max-cells / --shuffle flags, and the main() exit
codes (config invalid -> 3, no cells -> 4, results path conflict -> 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from scripts import run_sweep as rs

from lerobot_bench.checkpointing import RESULT_SCHEMA
from lerobot_bench.envs import EnvRegistry
from lerobot_bench.policies import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs" / "policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs" / "envs.yaml"
SWEEP_FULL_YAML = REPO_ROOT / "configs" / "sweep_full.yaml"
SWEEP_MINI_YAML = REPO_ROOT / "configs" / "sweep_mini.yaml"


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _minimal_yaml(tmp_path: Path, **overrides: Any) -> Path:
    """Write a minimal-valid sweep YAML; return its path. ``overrides`` deep-merge."""
    base: dict[str, Any] = {
        "policies": ["no_op", "random"],
        "envs": ["pusht"],
        "seeds": [0, 1],
        "episodes_per_seed": 5,
        "results_path": str(tmp_path / "results.parquet"),
        "videos_dir": None,
        "record_video": False,
        "device": "cpu",
        "policies_yaml": str(DEFAULT_POLICIES_YAML),
        "envs_yaml": str(DEFAULT_ENVS_YAML),
    }
    base.update(overrides)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


# --------------------------------------------------------------------- #
# 1. SweepConfig.from_dict -- happy path                                #
# --------------------------------------------------------------------- #


def test_from_dict_happy_path(tmp_path: Path) -> None:
    """Minimal valid YAML loads cleanly with defaults filled in."""
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["no_op"],
            "envs": ["pusht"],
            "seeds": [0],
            "episodes_per_seed": 50,
            "results_path": str(tmp_path / "out.parquet"),
        }
    )
    assert cfg.policies == ("no_op",)
    assert cfg.envs == ("pusht",)
    assert cfg.seeds == (0,)
    assert cfg.episodes_per_seed == 50
    assert cfg.record_video is True  # default
    assert cfg.device == "cuda"  # default
    assert cfg.cell_timeout_s is None
    assert cfg.max_parallel == 1
    assert cfg.overrides == {}
    assert cfg.policies_yaml == rs.DEFAULT_POLICIES_YAML
    assert cfg.envs_yaml == rs.DEFAULT_ENVS_YAML


def test_load_sweep_config_reads_yaml(tmp_path: Path) -> None:
    """The bundled minimal YAML round-trips through load_sweep_config."""
    path = _minimal_yaml(tmp_path)
    cfg = rs.load_sweep_config(path)
    assert cfg.policies == ("no_op", "random")
    assert cfg.episodes_per_seed == 5


# --------------------------------------------------------------------- #
# 2. SweepConfig.from_dict -- validation failures                       #
# --------------------------------------------------------------------- #


def test_from_dict_missing_required_raises() -> None:
    with pytest.raises(ValueError) as exc:
        rs.SweepConfig.from_dict({"policies": ["no_op"]})
    msg = str(exc.value)
    assert "missing required fields" in msg
    assert "envs" in msg
    assert "seeds" in msg


def test_from_dict_unknown_field_raises() -> None:
    with pytest.raises(ValueError) as exc:
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "extra_typo_field": True,
            }
        )
    assert "unknown fields" in str(exc.value)
    assert "extra_typo_field" in str(exc.value)


def test_from_dict_rejects_non_positive_episodes() -> None:
    with pytest.raises(ValueError, match="episodes_per_seed"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 0,
                "results_path": "x.parquet",
            }
        )


def test_from_dict_rejects_max_parallel_gt_1() -> None:
    """v1 contract: serial only."""
    with pytest.raises(ValueError, match="max_parallel must be 1"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "max_parallel": 4,
            }
        )


def test_from_dict_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [-1],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
            }
        )


def test_from_dict_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0, 0, 1],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
            }
        )


def test_from_dict_rejects_bad_cell_timeout() -> None:
    with pytest.raises(ValueError, match="cell_timeout_s"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "cell_timeout_s": -1,
            }
        )


# --------------------------------------------------------------------- #
# 3. Override validation                                                #
# --------------------------------------------------------------------- #


def test_overrides_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="not in sweep config"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "overrides": {"random": {"pusht": {"n_episodes": 10}}},
            }
        )


def test_overrides_unknown_env_raises() -> None:
    with pytest.raises(ValueError, match="overrides unknown env"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "overrides": {"no_op": {"libero": {"n_episodes": 10}}},
            }
        )


def test_overrides_unknown_inner_field_raises() -> None:
    with pytest.raises(ValueError, match="unknown override fields"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "overrides": {"no_op": {"pusht": {"n_episodes": 10, "extra": 1}}},
            }
        )


def test_overrides_n_episodes_must_be_positive_int() -> None:
    with pytest.raises(ValueError, match="must be a positive int"):
        rs.SweepConfig.from_dict(
            {
                "policies": ["no_op"],
                "envs": ["pusht"],
                "seeds": [0],
                "episodes_per_seed": 5,
                "results_path": "x.parquet",
                "overrides": {"no_op": {"pusht": {"n_episodes": 0}}},
            }
        )


# --------------------------------------------------------------------- #
# 4. expand_cells -- cartesian product + sort + overrides               #
# --------------------------------------------------------------------- #


def test_expand_cells_sorted_cartesian_product() -> None:
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["random", "no_op"],
            "envs": ["aloha_transfer_cube", "pusht"],
            "seeds": [1, 0],
            "episodes_per_seed": 7,
            "results_path": "x.parquet",
        }
    )
    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)

    cells = rs.expand_cells(cfg, policy_registry=policies, env_registry=envs)
    # 2 policies x 2 envs x 2 seeds = 8 cells
    assert len(cells) == 8
    # Sort order: (policy, env, seed). First cell should be no_op/aloha/seed0.
    assert cells[0].policy == "no_op"
    assert cells[0].env == "aloha_transfer_cube"
    assert cells[0].seed_idx == 0
    assert cells[-1].policy == "random"
    assert cells[-1].env == "pusht"
    assert cells[-1].seed_idx == 1
    # All cells use the base episodes_per_seed.
    assert all(c.n_episodes == 7 for c in cells)


def test_expand_cells_applies_per_cell_override() -> None:
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["no_op", "random"],
            "envs": ["pusht"],
            "seeds": [0, 1],
            "episodes_per_seed": 50,
            "results_path": "x.parquet",
            "overrides": {"random": {"pusht": {"n_episodes": 25}}},
        }
    )
    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    cells = rs.expand_cells(cfg, policy_registry=policies, env_registry=envs)

    by_policy = {c.policy: c.n_episodes for c in cells}
    assert by_policy["no_op"] == 50  # base
    assert by_policy["random"] == 25  # override


def test_expand_cells_marks_incompat() -> None:
    """A baseline restricted to pusht only -> aloha cell is compatible=False."""
    # Use the shipped baselines (env_compat = [pusht, aloha_transfer_cube])
    # and override with a tmp policies.yaml that drops aloha.
    pass  # covered in test_resume_drill::test_drill10


def test_expand_cells_marks_unrunnable(tmp_path: Path) -> None:
    """A pretrained policy with revision_sha=null -> runnable=False on every cell.

    Uses a tmp policies.yaml since the shipped diffusion_policy/act now ship
    with locked revision SHAs (Day 0a).
    """
    policies_yaml = tmp_path / "policies.yaml"
    policies_yaml.write_text(
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
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["not_yet_locked"],
            "envs": ["pusht"],
            "seeds": [0],
            "episodes_per_seed": 5,
            "results_path": "x.parquet",
            "policies_yaml": str(policies_yaml),
        }
    )
    policies = PolicyRegistry.from_yaml(policies_yaml)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    cells = rs.expand_cells(cfg, policy_registry=policies, env_registry=envs)
    assert len(cells) == 1
    assert cells[0].runnable is False


def test_expand_cells_unknown_policy_raises() -> None:
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["fake_policy"],
            "envs": ["pusht"],
            "seeds": [0],
            "episodes_per_seed": 5,
            "results_path": "x.parquet",
        }
    )
    policies = PolicyRegistry.from_yaml(DEFAULT_POLICIES_YAML)
    envs = EnvRegistry.from_yaml(DEFAULT_ENVS_YAML)
    with pytest.raises(ValueError, match="unknown policy"):
        rs.expand_cells(cfg, policy_registry=policies, env_registry=envs)


# --------------------------------------------------------------------- #
# 5. _build_run_one_argv: shape matches scripts/run_one.py CLI          #
# --------------------------------------------------------------------- #


def test_build_run_one_argv_shape(tmp_path: Path) -> None:
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["no_op"],
            "envs": ["pusht"],
            "seeds": [0],
            "episodes_per_seed": 3,
            "results_path": str(tmp_path / "r.parquet"),
            "videos_dir": str(tmp_path / "v"),
            "record_video": True,
            "device": "cpu",
        }
    )
    cell = rs.PlannedCell(
        policy="no_op",
        env="pusht",
        seed_idx=0,
        n_episodes=3,
        compatible=True,
        runnable=True,
    )
    argv = rs._build_run_one_argv(cell, config=cfg, python_executable="/usr/bin/python3")
    # Argv starts with python + run_one.py
    assert argv[0] == "/usr/bin/python3"
    assert argv[1].endswith("scripts/run_one.py")
    # Required flags all present.
    flat = dict(zip(argv[2::2], argv[3::2], strict=False))
    assert flat["--policy"] == "no_op"
    assert flat["--env"] == "pusht"
    assert flat["--seed"] == "0"
    assert flat["--n-episodes"] == "3"
    assert flat["--out-parquet"] == str(tmp_path / "r.parquet")
    assert flat["--videos-dir"] == str(tmp_path / "v")
    assert flat["--device"] == "cpu"
    assert "--no-record-video" not in argv  # record_video=True


def test_build_run_one_argv_no_record_video(tmp_path: Path) -> None:
    cfg = rs.SweepConfig.from_dict(
        {
            "policies": ["no_op"],
            "envs": ["pusht"],
            "seeds": [0],
            "episodes_per_seed": 3,
            "results_path": str(tmp_path / "r.parquet"),
            "record_video": False,
        }
    )
    cell = rs.PlannedCell(
        policy="no_op",
        env="pusht",
        seed_idx=0,
        n_episodes=3,
        compatible=True,
        runnable=True,
    )
    argv = rs._build_run_one_argv(cell, config=cfg)
    assert "--no-record-video" in argv


# --------------------------------------------------------------------- #
# 6. Manifest helpers                                                   #
# --------------------------------------------------------------------- #


def test_manifest_path_derived_from_results_path(tmp_path: Path) -> None:
    p = tmp_path / "sweep-x" / "results.parquet"
    assert rs.manifest_path_for(p) == tmp_path / "sweep-x" / "sweep_manifest.json"


def test_write_manifest_atomic(tmp_path: Path) -> None:
    """write_manifest leaves no .tmp.json behind on success."""
    manifest = rs.SweepManifest(
        started_utc="2026-05-03T00:00:00+00:00",
        code_sha="abc",
        lerobot_version="unknown",
        config_path="cfg.yaml",
        cells=[
            rs.CellManifestEntry(
                policy="no_op",
                env="pusht",
                seed_idx=0,
                n_episodes=5,
                status=rs.STATUS_PENDING,
            )
        ],
    )
    out_path = tmp_path / "sweep_manifest.json"
    rs.write_manifest(manifest, out_path)
    assert out_path.exists()
    leftovers = list(tmp_path.glob("sweep_manifest.tmp*"))
    assert leftovers == []
    # Roundtrip: the file is valid JSON with the expected shape.
    import json as _json

    parsed = _json.loads(out_path.read_text())
    assert parsed["code_sha"] == "abc"
    assert parsed["cells"][0]["status"] == "pending"


def test_tail_lines_caps_long_stderr() -> None:
    text = "\n".join(f"line{i}" for i in range(500))
    tail = rs._tail_lines(text, 50)
    assert tail.count("\n") == 49  # 50 lines -> 49 newlines
    assert tail.endswith("line499")


def test_tail_lines_short_passthrough() -> None:
    text = "a\nb\nc"
    assert rs._tail_lines(text, 100) == "a\nb\nc"


def test_tail_lines_empty() -> None:
    assert rs._tail_lines("", 10) == ""


# --------------------------------------------------------------------- #
# 7. main() exit codes                                                  #
# --------------------------------------------------------------------- #


def test_main_invalid_yaml_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing required field -> exit 3, stderr explains."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("policies: [no_op]\n")  # missing envs/seeds/...
    rc = rs.main(["--config", str(bad_yaml)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "invalid config" in err
    assert "missing required fields" in err


def test_main_missing_yaml_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = rs.main(["--config", str(tmp_path / "nope.yaml")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "config not found" in err


def test_main_unknown_policy_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A policy name not in policies.yaml -> exit 3 with helpful message."""
    cfg_path = _minimal_yaml(tmp_path, policies=["fake_policy", "no_op"])
    rc = rs.main(["--config", str(cfg_path)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "fake_policy" in err
    assert "unknown policy" in err


def test_main_dry_run_exits_0_no_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run writes manifest, exits 0, never calls subprocess."""
    cfg_path = _minimal_yaml(tmp_path)

    def boom(*_args: Any, **_kwargs: Any) -> rs.SubprocessOutcome:
        raise AssertionError("dry-run must not dispatch")

    monkeypatch.setattr(rs, "_run_subprocess", boom)
    rc = rs.main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    # Manifest landed on disk; parquet did not.
    sweep_dir = tmp_path
    assert (sweep_dir / "sweep_manifest.json").exists()
    assert not (sweep_dir / "results.parquet").exists()


def test_main_results_path_schema_conflict_exits_5(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pre-existing parquet with wrong columns -> exit 5."""
    cfg_path = _minimal_yaml(tmp_path)
    bad_parquet = tmp_path / "results.parquet"
    pd.DataFrame({"unrelated_column": [1, 2, 3]}).to_parquet(bad_parquet, index=False)
    rc = rs.main(["--config", str(cfg_path)])
    assert rc == 5
    err = capsys.readouterr().err
    assert "results path conflict" in err


def test_main_results_path_override_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--results-path overrides the YAML's results_path."""
    cfg_path = _minimal_yaml(tmp_path)
    new_path = tmp_path / "elsewhere.parquet"

    captured: dict[str, Any] = {}

    def fake(argv: list[str], *, timeout_s: float | None = None) -> rs.SubprocessOutcome:
        # Record where run_one was told to append.
        flat = dict(zip(argv[2::2], argv[3::2], strict=False))
        captured["out_parquet"] = flat["--out-parquet"]
        # Pretend it succeeded (no rows; orchestrator only cares about exit).
        return rs.SubprocessOutcome(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rs, "_run_subprocess", fake)
    rc = rs.main(["--config", str(cfg_path), "--results-path", str(new_path)])
    # Sweep returned exit 0 (every fake call returned 0). The manifest
    # lands next to the *override* path, not the YAML's path.
    assert rc == 0
    assert captured["out_parquet"] == str(new_path)
    assert (new_path.parent / "sweep_manifest.json").exists()


def test_main_max_parallel_gt_1_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """v1: --max-parallel > 1 is rejected at the CLI."""
    cfg_path = _minimal_yaml(tmp_path)
    rc = rs.main(["--config", str(cfg_path), "--max-parallel", "4"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "max-parallel" in err


# --------------------------------------------------------------------- #
# 8. Bundled YAMLs are valid                                            #
# --------------------------------------------------------------------- #


def test_bundled_sweep_full_yaml_loads() -> None:
    cfg = rs.load_sweep_config(SWEEP_FULL_YAML)
    assert cfg.episodes_per_seed > 0
    assert len(cfg.policies) >= 1
    assert len(cfg.envs) >= 1


def test_bundled_sweep_mini_yaml_loads() -> None:
    cfg = rs.load_sweep_config(SWEEP_MINI_YAML)
    assert cfg.episodes_per_seed > 0
    # mini intentionally excludes pretrained policies.
    assert "diffusion_policy" not in cfg.policies


def test_bundled_sweep_full_expands_cleanly() -> None:
    """Cells expand without error against the shipped registries.

    Some cells are env-incompat (e.g., diffusion_policy on aloha after the
    Day 0a env_compat correction): the expander still emits them with
    compatible=False so the sweep can mark them skipped explicitly.
    """
    cfg = rs.load_sweep_config(SWEEP_FULL_YAML)
    policies = PolicyRegistry.from_yaml(cfg.policies_yaml)
    envs = EnvRegistry.from_yaml(cfg.envs_yaml)
    cells = rs.expand_cells(cfg, policy_registry=policies, env_registry=envs)
    # All policy x env x seed combinations are emitted; env_compat filtering
    # happens at dispatch time via cell.compatible.
    assert len(cells) == len(cfg.policies) * len(cfg.envs) * len(cfg.seeds)


# --------------------------------------------------------------------- #
# 9. Subprocess outcome dataclass                                       #
# --------------------------------------------------------------------- #


def test_subprocess_outcome_is_frozen() -> None:
    """Sanity: SubprocessOutcome is hashable + immutable so tests can compare."""
    o = rs.SubprocessOutcome(returncode=0, stdout="", stderr="")
    with pytest.raises((AttributeError, TypeError)):
        o.returncode = 1  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 10. RESULT_SCHEMA wire-through                                        #
# --------------------------------------------------------------------- #


def test_result_schema_used_in_preflight(tmp_path: Path) -> None:
    """A parquet matching RESULT_SCHEMA passes pre-flight; mismatched fails."""
    good = tmp_path / "good.parquet"
    pd.DataFrame(columns=list(RESULT_SCHEMA)).to_parquet(good, index=False)
    assert rs._preflight_results_path(good) is None

    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"oops": [1]}).to_parquet(bad, index=False)
    err = rs._preflight_results_path(bad)
    assert err is not None
    assert "missing" in err or "extra" in err
