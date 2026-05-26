#!/usr/bin/env python3
"""Inference-settings audit for the v1.0 sweep.

Read-only static audit: enumerate every inference-time hyperparameter
that actually affected v1, cross-reference the values declared in each
policy's Hub-pinned ``config.json`` (the file lerobot's
``PreTrainedConfig.from_pretrained`` reads) against the values published
in the policy's source paper, and flag mismatches.

This script does NOT load model weights, does NOT touch a GPU, does NOT
run any episodes. It reads two things:

1. ``configs/policies.yaml`` + ``configs/envs.yaml`` -- the registry
   files the v1 sweep used.
2. ``config.json`` from each policy's pinned Hub revision -- the
   effective inference config the eval loop saw.

Hub configs are resolved via :func:`huggingface_hub.hf_hub_download` so
the script works whether the user already has the snapshots cached or
not (the call is a no-op when the SHA-pinned blob is on disk; with
``--no-network`` it falls back to a cache-only lookup and fails clean
if a snapshot is missing).

Usage::

    python scripts/audit_inference_settings.py                 # write to stdout
    python scripts/audit_inference_settings.py --output docs/INFERENCE_AUDIT.md
    python scripts/audit_inference_settings.py --dry-run       # no Hub access at all
    python scripts/audit_inference_settings.py --no-network    # cache-only Hub lookup

Exit codes:
    0  audit produced cleanly (mismatches still possible -- they are
       data, not errors).
    2  one or more policy configs could not be resolved (e.g. cache
       miss with --no-network). Markdown is still emitted with the
       resolution errors recorded inline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot_bench.policies import PolicyRegistry, PolicySpec

logger = logging.getLogger("audit_inference_settings")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICIES_YAML = REPO_ROOT / "configs/policies.yaml"
DEFAULT_ENVS_YAML = REPO_ROOT / "configs/envs.yaml"


# --------------------------------------------------------------------- #
# Paper-reported inference settings (sourced manually from each paper). #
# Only fields that materially affect inference are captured -- training #
# hyperparameters (LR, weight decay, etc.) are out of scope.            #
#                                                                       #
# Citations live alongside the values so the audit table is auditable.  #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PaperSetting:
    """One paper-reported inference hyperparameter.

    ``value`` is a string so we can carry things like ``"None"`` /
    ``"unspecified"`` / ``"DDIM-100"`` without forcing every field into
    a numeric type. The diff vs the Hub config is a string-equality
    check after both sides are normalized.
    """

    value: str
    citation: str
    note: str = ""


# Per-policy expected inference settings as the source paper describes
# them. Keys are field names that match the Hub ``config.json`` schema
# wherever the upstream paper uses a comparable knob; for paper-specific
# knobs (e.g. "temporal ensembling") we use the canonical knob name.
#
# Conservative rule: only fill an entry when the paper explicitly states
# the value. Otherwise leave it out -- a missing entry is "paper does
# not report this setting" rather than "matches default".
PAPER_SETTINGS: dict[str, dict[str, PaperSetting]] = {
    "diffusion_policy": {
        "horizon": PaperSetting(
            value="16",
            citation="Chi et al. 2023 (Diffusion Policy, RSS), Appendix B.1 Table V "
            "'Hyperparameters of CNN-based DP', T_p (prediction horizon) = 16 "
            "for Push-T (image-based, DDIM-100).",
            note="Hub config matches.",
        ),
        "n_action_steps": PaperSetting(
            value="8",
            citation="Chi et al. 2023, Appendix B.1 Table V, T_a (action execution "
            "horizon) = 8 for Push-T (image-based).",
            note="Hub config matches.",
        ),
        "n_obs_steps": PaperSetting(
            value="2",
            citation="Chi et al. 2023, Appendix B.1 Table V, T_o (observation horizon) "
            "= 2 for Push-T.",
            note="Hub config matches.",
        ),
        "num_inference_steps": PaperSetting(
            value="100",
            citation="Chi et al. 2023 §IV.D 'Implementation details', DDIM-100 "
            "denoising steps at inference for the CNN image policy.",
            note="Hub config sets num_inference_steps=None which means lerobot "
            "falls back to the noise scheduler's training-time num_train_timesteps "
            "(100 in the Hub config's noise_scheduler block) -- so effectively "
            "100 denoising steps. Matches.",
        ),
    },
    "act": {
        "chunk_size": PaperSetting(
            value="100",
            citation="Zhao et al. 2023 (ACT, RSS), Appendix B 'Implementation details', "
            "chunk size k = 100 for simulated tasks.",
            note="Hub config matches.",
        ),
        "n_action_steps": PaperSetting(
            value="1",
            citation="Zhao et al. 2023, §IV.A 'Action chunking with temporal "
            "ensembling', the ensembling rule consumes one action per env step "
            "by aggregating the predicted chunk-prefix overlaps with weighted mean "
            "(equivalent to n_action_steps=1 in lerobot's chunk-execution model).",
            note="MISMATCH: Hub ships n_action_steps=100 (re-plan every 100 steps, "
            "no overlap aggregation). See temporal_ensemble_coeff below.",
        ),
        "temporal_ensemble_coeff": PaperSetting(
            value="0.01",
            citation="Zhao et al. 2023, §IV.A, ensembling weight w_i = exp(-m * i) "
            "with m = 0.01 'works well in practice' (paper's recommended value).",
            note="MISMATCH: Hub ships temporal_ensemble_coeff=None, so ACT runs "
            "in plain chunk-execution mode. Paper's Table I numbers were produced "
            "WITH temporal ensembling enabled.",
        ),
    },
    "smolvla_libero": {
        "chunk_size": PaperSetting(
            value="50",
            citation="Shukor et al. 2025 (SmolVLA, arXiv:2506.01844), §3.2 "
            "'Architecture and training', flow-matching with action chunks of "
            "size 50 (matches the openVLA-style horizon used in the LIBERO eval).",
            note="Hub config matches.",
        ),
        "n_action_steps": PaperSetting(
            value="50",
            citation="Shukor et al. 2025, §4.1 'Evaluation protocol', the policy "
            "executes the full predicted chunk before re-planning (open-loop "
            "chunk execution, n_action_steps == chunk_size).",
            note="Hub config matches.",
        ),
        "num_steps": PaperSetting(
            value="10",
            citation="Shukor et al. 2025, §3.2, 10 denoising / flow-matching "
            "steps at inference (Euler integration).",
            note="Hub config matches.",
        ),
    },
    "xvla_libero": {
        "chunk_size": PaperSetting(
            value="50",
            citation="Bu et al. 2025 (X-VLA, arXiv:2510.10274), Table 12 "
            "'Hyperparameters of X-VLA' lists action chunk H = 50 for LIBERO eval.",
            note="MISMATCH: Hub ships chunk_size=30 (and n_action_steps=30). "
            "The X-VLA LIBERO Hub checkpoint was trained with the shorter "
            "horizon -- our paper-reference 98.1% avg came from the 50-step "
            "chunk variant.",
        ),
        "n_action_steps": PaperSetting(
            value="50",
            citation="Bu et al. 2025, §4 'Experiments', LIBERO eval executes "
            "the full chunk (open-loop, n_action_steps == chunk_size).",
            note="MISMATCH: Hub ships n_action_steps=30.",
        ),
        "num_denoising_steps": PaperSetting(
            value="10",
            citation="Bu et al. 2025, Table 12, T_inference = 10 flow-matching Euler steps.",
            note="Hub config matches.",
        ),
    },
    "pi0_libero_finetuned_v044": {
        "chunk_size": PaperSetting(
            value="50",
            citation="Black et al. 2024 (Pi0, arXiv:2410.24164), §5 'Implementation', "
            "openpi 'action_horizon' = 50 (== lerobot chunk_size). LIBERO eval "
            "uses the same horizon.",
            note="Hub config matches.",
        ),
        "n_action_steps": PaperSetting(
            value="50",
            citation="Black et al. 2024, §5, open-loop chunk execution; the "
            "openpi runtime executes the full action horizon between re-plans.",
            note="Hub config matches.",
        ),
        "num_inference_steps": PaperSetting(
            value="10",
            citation="Black et al. 2024, §3.2 'Flow matching action expert', "
            "10 Euler integration steps at inference.",
            note="Hub config matches.",
        ),
    },
    "pi05_libero_finetuned_v044": {
        "chunk_size": PaperSetting(
            value="50",
            citation="Pi0.5 inherits Pi0's openpi runtime; action_horizon = 50.",
            note="Hub config matches (Pi0.5 paper not separately published as of "
            "2026-05; treat as continuous with Pi0).",
        ),
        "n_action_steps": PaperSetting(
            value="50",
            citation="Open-loop chunk execution (inherits Pi0 protocol).",
            note="Hub config matches.",
        ),
        "num_inference_steps": PaperSetting(
            value="10",
            citation="10 Euler integration steps (inherits Pi0 protocol).",
            note="Hub config matches.",
        ),
    },
    "pi0fast_libero": {
        "chunk_size": PaperSetting(
            value="50",
            citation="Pertsch et al. 2025 (Pi0-FAST, arXiv:2501.09747), §4.1 "
            "'Pi0-FAST architecture', the FAST tokenizer is trained over chunks "
            "of 50 actions (same horizon as Pi0).",
            note="MISMATCH: Hub ships chunk_size=10 (and n_action_steps=10). "
            "Hub finetune shortened the executable horizon -- not the FAST "
            "tokenizer's compression window. Note this also means Pi0-FAST "
            "re-plans 5x more often than Pi0 / Pi0.5 in our v1 sweep.",
        ),
        "n_action_steps": PaperSetting(
            value="50",
            citation="Pertsch et al. 2025, open-loop chunk execution at "
            "horizon 50 (inherits Pi0 runtime).",
            note="MISMATCH: Hub ships n_action_steps=10.",
        ),
        "temperature": PaperSetting(
            value="0.0",
            citation="Pertsch et al. 2025, §3.3 'FAST decoding', greedy "
            "(argmax) decoding at inference.",
            note="Hub config matches (temperature=0.0 == greedy).",
        ),
        "max_decoding_steps": PaperSetting(
            value="256",
            citation="Pertsch et al. 2025, §3.3, autoregressive decoding capped "
            "at the FAST sequence length budget (~256 tokens for a 50-step "
            "chunk).",
            note="Hub config matches.",
        ),
    },
}


# --------------------------------------------------------------------- #
# Hub config resolution                                                 #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class HubConfig:
    """Inference-relevant subset of a Hub ``config.json``.

    ``raw`` carries the full dict so the audit table can pull any
    additional field on demand (the cherry-picked attributes are just
    the most-commonly-referenced ones, not a closed enumeration).

    ``error`` is set when the config could not be loaded (network off
    AND not cached, or HTTP failure). The downstream audit table prints
    the error in the policy's row rather than crashing.
    """

    policy_name: str
    repo_id: str
    revision: str
    policy_type: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# Fields we cherry-pick into the per-policy table. Order matters --
# this is the column order in the rendered markdown row.
INFERENCE_FIELDS: tuple[str, ...] = (
    "chunk_size",
    "n_action_steps",
    "n_obs_steps",
    "horizon",
    "num_inference_steps",
    "num_steps",
    "num_denoising_steps",
    "temporal_ensemble_coeff",
    "temperature",
    "max_decoding_steps",
    "pad_language_to",
    "tokenizer_max_length",
    "max_action_dim",
    "max_state_dim",
    "use_cache",
    "compile_model",
)


def resolve_hub_config(
    spec: PolicySpec,
    *,
    cache_only: bool,
    dry_run: bool,
) -> HubConfig:
    """Read ``config.json`` for ``spec`` from the Hub or local cache.

    With ``dry_run=True`` we never touch the Hub or the cache -- the
    returned record has ``error`` populated. Used by the unit tests and
    CI smoke. With ``cache_only=True`` we ask huggingface_hub to refuse
    a network download (``local_files_only=True``) -- useful in air-
    gapped contexts. Default behaviour fetches missing snapshots.
    """
    assert spec.repo_id is not None
    assert spec.revision_sha is not None

    if dry_run:
        return HubConfig(
            policy_name=spec.name,
            repo_id=spec.repo_id,
            revision=spec.revision_sha,
            policy_type="?",
            error="dry-run: Hub access skipped",
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return HubConfig(
            policy_name=spec.name,
            repo_id=spec.repo_id,
            revision=spec.revision_sha,
            policy_type="?",
            error=f"huggingface_hub not installed: {exc}",
        )

    try:
        path = hf_hub_download(
            spec.repo_id,
            "config.json",
            revision=spec.revision_sha,
            local_files_only=cache_only,
        )
    except Exception as exc:
        return HubConfig(
            policy_name=spec.name,
            repo_id=spec.repo_id,
            revision=spec.revision_sha,
            policy_type="?",
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return HubConfig(
            policy_name=spec.name,
            repo_id=spec.repo_id,
            revision=spec.revision_sha,
            policy_type="?",
            error=f"failed to read {path}: {exc}",
        )

    return HubConfig(
        policy_name=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision_sha,
        policy_type=str(raw.get("type", "?")),
        raw=raw,
    )


# --------------------------------------------------------------------- #
# Mismatch detection                                                    #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mismatch:
    """One field where Hub config disagrees with the paper."""

    policy: str
    field_name: str
    hub_value: str
    paper_value: str
    citation: str
    note: str


def detect_mismatches(hub_configs: list[HubConfig]) -> list[Mismatch]:
    """Compare each policy's Hub config against PAPER_SETTINGS."""
    mismatches: list[Mismatch] = []
    for cfg in hub_configs:
        expected = PAPER_SETTINGS.get(cfg.policy_name, {})
        if not expected or not cfg.ok:
            continue
        for field_name, paper in expected.items():
            hub_raw = cfg.raw.get(field_name, "<absent>")
            hub_str = _normalize_value(hub_raw)
            if hub_str == _normalize_value(paper.value):
                continue
            mismatches.append(
                Mismatch(
                    policy=cfg.policy_name,
                    field_name=field_name,
                    hub_value=hub_str,
                    paper_value=paper.value,
                    citation=paper.citation,
                    note=paper.note,
                )
            )
    return mismatches


def _normalize_value(value: Any) -> str:
    """Stringify a config value for diff-friendly comparison.

    ``None`` -> ``"None"``, floats -> repr (so ``0.01`` and ``"0.01"``
    match), ints -> str, lists/dicts -> JSON. Everything else falls
    back to ``str(value)``.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


# --------------------------------------------------------------------- #
# Probe recommendations                                                 #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Probe:
    """One recommended 1-seed x 50-ep probe to test a mismatch's impact."""

    policy: str
    field_name: str
    hub_value: str
    proposed_value: str
    env: str
    expected_signal: str


# Map mismatch -> recommended probe. Conservative: only enumerate probes
# for mismatches that plausibly move success rate by >= 5 percentage
# points (chunk size, action aggregation, denoising steps). Image-
# normalization and tokenizer-padding mismatches are not in this list
# because they would already be caught by the failing-output sanity
# check at load time.
def probes_for_mismatches(mismatches: list[Mismatch]) -> list[Probe]:
    """Translate each mismatch into a recommended probe (where applicable)."""
    out: list[Probe] = []
    for m in mismatches:
        if m.policy == "act" and m.field_name == "temporal_ensemble_coeff":
            out.append(
                Probe(
                    policy="act",
                    field_name="temporal_ensemble_coeff",
                    hub_value="None",
                    proposed_value="0.01",
                    env="aloha_transfer_cube",
                    expected_signal="If temporal ensembling closes the bench-vs-"
                    "paper gap, our ACT cell will move from current success rate "
                    "toward the paper's 50% (human-teleop training data, Zhao "
                    "et al. 2023 Table I). Pair with n_action_steps=1.",
                )
            )
        elif m.policy == "act" and m.field_name == "n_action_steps":
            # Paired with the temporal_ensemble_coeff probe above; do not
            # emit a separate probe.
            continue
        elif m.policy == "xvla_libero" and m.field_name == "chunk_size":
            out.append(
                Probe(
                    policy="xvla_libero",
                    field_name="chunk_size",
                    hub_value="30",
                    proposed_value="50",
                    env="libero_10",  # The hardest LIBERO suite -- biggest signal.
                    expected_signal="Paper reports 97.6% on LIBERO-Long with "
                    "chunk_size=50. If our xvla cell improves materially at the "
                    "longer horizon, the Hub checkpoint's shorter chunk explains "
                    "the gap. NOTE: weights were trained at chunk_size=30, so "
                    "extending it at inference may not generalize -- this probe "
                    "is mostly diagnostic, not a fix.",
                )
            )
        elif m.policy == "xvla_libero" and m.field_name == "n_action_steps":
            continue  # Paired with chunk_size probe above.
        elif m.policy == "pi0fast_libero" and m.field_name == "chunk_size":
            out.append(
                Probe(
                    policy="pi0fast_libero",
                    field_name="chunk_size",
                    hub_value="10",
                    proposed_value="50",
                    env="libero_10",
                    expected_signal="Pi0-FAST Hub finetune ships a 10-step chunk "
                    "(re-plans 5x more often than Pi0 / Pi0.5). A 50-step probe "
                    "tests whether the shorter chunk drags Pi0-FAST below its "
                    "paper-reported peer. Same caveat as xvla: weights may not "
                    "generalize to a longer chunk at inference.",
                )
            )
        elif m.policy == "pi0fast_libero" and m.field_name == "n_action_steps":
            continue
    return out


# --------------------------------------------------------------------- #
# Markdown rendering                                                    #
# --------------------------------------------------------------------- #


def render_markdown(
    *,
    hub_configs: list[HubConfig],
    mismatches: list[Mismatch],
    probes: list[Probe],
    policy_specs: list[PolicySpec],
) -> str:
    """Produce the full audit markdown.

    Sections (matches the brief):
        # Inference-settings audit (v1.0.1)
        ## What we used
        ## What the model cards declare
        ## Mismatches
        ## Recommended probes
    """
    lines: list[str] = []
    lines.append("# Inference-settings audit (v1.0.1)")
    lines.append("")
    lines.append(
        "Static, read-only audit of every inference-time hyperparameter "
        "that affected the v1 sweep. Generated by "
        "`scripts/audit_inference_settings.py`."
    )
    lines.append("")
    lines.append(
        "**Scope.** Compares each policy's pinned Hub `config.json` (what "
        "`lerobot.PreTrainedConfig.from_pretrained` actually loaded) "
        "against the values published in the policy's source paper. "
        "Training-time knobs (LR, weight decay, batch size) are out of "
        "scope."
    )
    lines.append("")
    lines.append(
        "**What this audit does NOT do.** No new sweep is run. The "
        "audit only reads `config.json` from the pinned SHA; weights "
        "are not loaded. Probe recommendations in the final section "
        "are for follow-up after review."
    )
    lines.append("")

    lines.extend(_render_what_we_used(hub_configs))
    lines.extend(_render_model_cards(policy_specs))
    lines.extend(_render_mismatches(mismatches))
    lines.extend(_render_probes(probes))

    return "\n".join(lines) + "\n"


def _render_what_we_used(hub_configs: list[HubConfig]) -> list[str]:
    """## What we used -- per-policy table of effective inference config."""
    lines = ["## What we used", ""]
    lines.append(
        "Effective inference configuration loaded into the v1 eval loop "
        "(read from each pinned `config.json`). Baselines (`no_op`, "
        "`random`) are excluded -- they carry no inference knobs."
    )
    lines.append("")

    header = ["policy", "type", "revision (short)", *INFERENCE_FIELDS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for cfg in hub_configs:
        if cfg.error is not None:
            row = [cfg.policy_name, "?", cfg.revision[:7], f"<error: {cfg.error}>"]
            row.extend([""] * (len(INFERENCE_FIELDS) - 1))
        else:
            row = [cfg.policy_name, cfg.policy_type, cfg.revision[:7]]
            for f in INFERENCE_FIELDS:
                if f in cfg.raw:
                    row.append(_normalize_value(cfg.raw[f]))
                else:
                    row.append("-")
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")

    lines.append("")
    lines.append(
        "**Reading the table.** A `-` means the field is not present in "
        "the policy's `config.json` (the corresponding lerobot config "
        "class either does not declare the knob or falls back to its "
        "dataclass default). `None` means the field is present and set "
        "to null -- typically that lerobot interprets null as 'fall back "
        "to a related value' (see e.g. diffusion `num_inference_steps`)."
    )
    lines.append("")
    return lines


def _render_model_cards(specs: list[PolicySpec]) -> list[str]:
    """## What the model cards declare -- paper-reported numbers (success only)."""
    lines = ["## What the model cards declare", ""]
    lines.append(
        "Paper-reported success rates from each policy's primary "
        "reference (sourced from `configs/policies.yaml`). These are "
        "not inference knobs -- they are the *outcome* the paper's "
        "inference settings produced -- but they are the comparison "
        "point the mismatch column below pivots on."
    )
    lines.append("")
    lines.append("| policy | env | paper success | source |")
    lines.append("|---|---|---|---|")

    for spec in specs:
        if spec.is_baseline:
            continue
        if not spec.paper_reported_success:
            lines.append(f"| {spec.name} | -- | -- | _no paper success reported in registry_ |")
            continue
        for env_name, value in sorted(spec.paper_reported_success.items()):
            value_str = "n/a" if value is None else f"{value:.3f}"
            cite = _shorten_note(spec.paper_reported_notes)
            lines.append(f"| {spec.name} | {env_name} | {value_str} | {cite} |")
    lines.append("")
    return lines


def _render_mismatches(mismatches: list[Mismatch]) -> list[str]:
    """## Mismatches -- the inference knobs where Hub disagrees with paper."""
    lines = ["## Mismatches", ""]
    if not mismatches:
        lines.append("None detected. Every paper-reported inference knob matches the Hub config.")
        lines.append("")
        return lines

    lines.append(
        f"WARNING: {len(mismatches)} field(s) below disagree between the Hub-pinned "
        "config and the paper. Each entry is annotated with the citation "
        "and a one-line note. `note` indicates whether the mismatch is "
        "load-bearing (likely to move success rate) or cosmetic."
    )
    lines.append("")
    lines.append("| policy | field | hub value | paper value | note |")
    lines.append("|---|---|---|---|---|")
    for m in mismatches:
        lines.append(
            "| "
            + " | ".join(
                _md_cell(c)
                for c in (
                    m.policy,
                    m.field_name,
                    m.hub_value,
                    m.paper_value,
                    m.note,
                )
            )
            + " |"
        )
    lines.append("")
    lines.append("### Citations")
    lines.append("")
    for m in mismatches:
        lines.append(f"- **{m.policy} / {m.field_name}**: {m.citation}")
    lines.append("")
    return lines


def _render_probes(probes: list[Probe]) -> list[str]:
    """## Recommended probes -- 1-seed x 50-ep probes for the load-bearing mismatches."""
    lines = ["## Recommended probes", ""]
    if not probes:
        lines.append(
            "No probes recommended -- either no mismatches were detected, "
            "or all mismatches are cosmetic."
        )
        lines.append("")
        return lines

    lines.append(
        "For each load-bearing mismatch, the table below proposes a "
        "minimal probe: **1 seed x 50 episodes**, single env, exactly "
        "the proposed value swapped in. The expected signal column "
        "says what we'd learn from the result."
    )
    lines.append("")
    lines.append("| policy | field | hub | probe value | env | expected signal |")
    lines.append("|---|---|---|---|---|---|")
    for p in probes:
        lines.append(
            "| "
            + " | ".join(
                _md_cell(c)
                for c in (
                    p.policy,
                    p.field_name,
                    p.hub_value,
                    p.proposed_value,
                    p.env,
                    p.expected_signal,
                )
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "**How to run a probe.** Add a one-off override to "
        "`configs/policies.yaml` (or a sweep-config override), then "
        "`python scripts/run_one.py --policy <name> --env <env> --seed 0 "
        "--episodes 50`. Write the probe's parquet to "
        "`results/probe-YYYYMMDD/` so it is not confused with v1 sweep "
        "data."
    )
    lines.append("")
    return lines


def _md_cell(value: str) -> str:
    """Escape pipe characters in a markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ")


def _shorten_note(text: str, max_chars: int = 120) -> str:
    """Trim a long citation note to one line for the model-card table."""
    flattened = " ".join(text.split())
    if len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1].rstrip() + "..."


# --------------------------------------------------------------------- #
# Entry point                                                           #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-settings audit for the v1.0 sweep.",
    )
    parser.add_argument(
        "--policies-yaml",
        type=Path,
        default=DEFAULT_POLICIES_YAML,
        help="Path to configs/policies.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown report. Default: stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not touch the Hub or local cache. Markdown still renders, "
        "with '<error>' in the Hub-value columns.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Read only from the local Hub cache (huggingface_hub "
        "local_files_only=True). Fails clean if a snapshot is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    registry = PolicyRegistry.from_yaml(args.policies_yaml)
    specs = [s for s in registry if not s.is_baseline]

    hub_configs: list[HubConfig] = []
    for spec in specs:
        if not spec.is_runnable():
            logger.warning("skipping %s -- not runnable (no revision pinned)", spec.name)
            continue
        cfg = resolve_hub_config(spec, cache_only=args.no_network, dry_run=args.dry_run)
        if cfg.error is not None:
            logger.warning("%s: %s", spec.name, cfg.error)
        hub_configs.append(cfg)

    mismatches = detect_mismatches(hub_configs)
    probes = probes_for_mismatches(mismatches)

    markdown = render_markdown(
        hub_configs=hub_configs,
        mismatches=mismatches,
        probes=probes,
        policy_specs=specs,
    )

    if args.output is None:
        sys.stdout.write(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        logger.info("wrote %s", args.output)

    # Exit code 2 if any policy failed to resolve (and we are not in dry-run).
    if not args.dry_run and any(not c.ok for c in hub_configs):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
