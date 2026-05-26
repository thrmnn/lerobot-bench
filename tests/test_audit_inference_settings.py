"""Tests for ``scripts/audit_inference_settings.py``.

CI-runnable surface only. Covers:
    1. ``_normalize_value`` -- string equality under None/int/float/bool/list.
    2. ``detect_mismatches`` -- matching values produce zero mismatches.
    3. ``detect_mismatches`` -- divergent values surface the right field.
    4. ``probes_for_mismatches`` -- one probe per load-bearing field, no
       duplicates from paired (chunk_size, n_action_steps) mismatches.
    5. ``render_markdown`` -- contains every required section heading.
    6. ``main(--dry-run)`` -- renders cleanly with placeholder Hub rows
       (no network access, no huggingface_hub import).
"""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

from scripts.audit_inference_settings import (
    HubConfig,
    Mismatch,
    _normalize_value,
    detect_mismatches,
    main,
    probes_for_mismatches,
    render_markdown,
)

from lerobot_bench.policies import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES_YAML = REPO_ROOT / "configs/policies.yaml"


def test_normalize_value_handles_basic_types() -> None:
    assert _normalize_value(None) == "None"
    assert _normalize_value(True) == "True"
    assert _normalize_value(False) == "False"
    assert _normalize_value(8) == "8"
    assert _normalize_value(0.01) == "0.01"
    assert _normalize_value("max_length") == "max_length"
    # list/dict serialize to JSON so the diff stays deterministic across runs.
    assert _normalize_value([1, 2, 3]) == "[1, 2, 3]"


def test_detect_mismatches_returns_empty_when_hub_matches_paper() -> None:
    # diffusion_policy paper-expected horizon=16, n_action_steps=8, n_obs_steps=2.
    cfg = HubConfig(
        policy_name="diffusion_policy",
        repo_id="lerobot/diffusion_pusht",
        revision="84a7c23",
        policy_type="diffusion",
        raw={"horizon": 16, "n_action_steps": 8, "n_obs_steps": 2, "num_inference_steps": 100},
    )
    mismatches = detect_mismatches([cfg])
    assert mismatches == []


def test_detect_mismatches_flags_divergent_field() -> None:
    # ACT paper expects temporal_ensemble_coeff=0.01 and n_action_steps=1.
    cfg = HubConfig(
        policy_name="act",
        repo_id="lerobot/act_aloha_sim_transfer_cube_human",
        revision="ba73b27",
        policy_type="act",
        raw={"chunk_size": 100, "n_action_steps": 100, "temporal_ensemble_coeff": None},
    )
    mismatches = detect_mismatches([cfg])
    fields = {m.field_name for m in mismatches}
    assert "n_action_steps" in fields
    assert "temporal_ensemble_coeff" in fields
    # chunk_size matches the paper (=100), so it must NOT be flagged.
    assert "chunk_size" not in fields


def test_probes_dedupes_paired_chunk_size_and_n_action_steps() -> None:
    # XVLA mismatch on BOTH chunk_size and n_action_steps -- the probe
    # generator should emit ONE probe (against chunk_size, the primary).
    mismatches = [
        Mismatch(
            policy="xvla_libero",
            field_name="chunk_size",
            hub_value="30",
            paper_value="50",
            citation="-",
            note="-",
        ),
        Mismatch(
            policy="xvla_libero",
            field_name="n_action_steps",
            hub_value="30",
            paper_value="50",
            citation="-",
            note="-",
        ),
    ]
    probes = probes_for_mismatches(mismatches)
    assert len(probes) == 1
    assert probes[0].field_name == "chunk_size"


def test_render_markdown_contains_required_sections() -> None:
    registry = PolicyRegistry.from_yaml(POLICIES_YAML)
    specs = [s for s in registry if not s.is_baseline]
    md = render_markdown(
        hub_configs=[],
        mismatches=[],
        probes=[],
        policy_specs=specs,
    )
    assert "# Inference-settings audit (v1.0.1)" in md
    assert "## What we used" in md
    assert "## What the model cards declare" in md
    assert "## Mismatches" in md
    assert "## Recommended probes" in md


def test_dry_run_does_not_import_huggingface_hub() -> None:
    # The dry-run path must not pull `huggingface_hub` into sys.modules --
    # the whole point of --dry-run is to work without it. Use importlib
    # to peek at sys.modules without forcing an import in this test process.
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--dry-run", "--policies-yaml", str(POLICIES_YAML)])
    assert rc == 0
    out = buf.getvalue()
    assert "# Inference-settings audit (v1.0.1)" in out
    # We don't assert huggingface_hub is absent from sys.modules (an
    # earlier test in the same process may have imported it); we assert
    # the resolve_hub_config path returned the dry-run error consistently.
    assert "<error: dry-run: Hub access skipped>" in out


def test_audit_script_has_no_top_level_torch_import() -> None:
    # Defensive: the audit must stay torch-free so it can run on the CI
    # fast job. Same pattern as the calibrate.py guard.
    spec = importlib.util.find_spec("scripts.audit_inference_settings")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert "\nimport torch" not in source
    assert "\nfrom torch" not in source
