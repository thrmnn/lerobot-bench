"""Policy registry for lerobot-bench.

Pure data — no policy loading. The actual ``policy.from_pretrained``
call lives in ``lerobot_bench.eval``; this module describes the
policies we plan to evaluate (HF Hub repo IDs, pinned revision SHAs,
env compat, fp precision) so the eval loop can resolve a policy name
to a runnable spec.

A spec is "runnable" when it is either a baseline (no weights) OR has
both ``repo_id`` and ``revision_sha`` filled in. Pre-Day-0a entries can
ship with ``revision_sha: null`` — those entries load fine into the
registry but are rejected by :meth:`PolicySpec.assert_runnable` and the
eval loop will refuse to run them. This is the explicit-not-silent
substitution rule from the bench-eval-engineer brief.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

FpPrecision = Literal["fp32", "fp16", "bf16"]

# Required for every entry. Optional fields validated conditionally.
_REQUIRED_FIELDS = frozenset({"name", "is_baseline", "env_compat"})
_OPTIONAL_FIELDS = frozenset(
    {
        "repo_id",
        "revision_sha",
        "fp_precision",
        "license",
        "notes",
        # Paper-reported success rates per (policy, env) for the
        # "delta-vs-published-claim" panel on the Space. Sourced from
        # the original paper or HF Hub model card — see policies.yaml
        # comments for per-entry citations. Optional: baseline / unseen-
        # env cells should be left absent or set to null.
        "paper_reported_success",
        "paper_reported_notes",
    }
)
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS

_VALID_FP = frozenset({"fp32", "fp16", "bf16"})


@dataclass(frozen=True)
class PolicySpec:
    """One policy in the benchmark matrix.

    Baselines (``no_op``, ``random``) carry no weights and have
    ``repo_id`` / ``revision_sha`` / ``fp_precision`` set to ``None``.
    Pretrained policies must have ``repo_id`` and ``revision_sha`` once
    locked at Day 0a — until then the entry can ship with
    ``revision_sha=None`` and :meth:`is_runnable` returns ``False``.
    """

    name: str
    is_baseline: bool
    env_compat: tuple[str, ...]
    repo_id: str | None = None
    revision_sha: str | None = None
    fp_precision: FpPrecision | None = None
    license: str | None = None
    notes: str = ""
    # Per-env paper-reported success rates (fractions in [0, 1]) keyed
    # by the same env-name strings used in ``env_compat``. Absent / None
    # for an env means "the paper does not report on this exact env, or
    # we couldn't find a matching number." See policies.yaml comments
    # for citations. The dashboard uses these to render
    # "delta-vs-published" alongside our re-run success rates.
    paper_reported_success: dict[str, float | None] | None = None
    paper_reported_notes: str = ""
    # Internal: the source location for nicer error messages.
    _source: str = field(default="<in-memory>", repr=False, compare=False)

    def is_runnable(self) -> bool:
        """True iff the spec is sufficient to run an eval cell."""
        if self.is_baseline:
            return True
        return bool(self.repo_id) and bool(self.revision_sha)

    def assert_runnable(self) -> None:
        """Raise ``ValueError`` with a user-actionable message if not runnable."""
        if self.is_runnable():
            return
        missing: list[str] = []
        if not self.repo_id:
            missing.append("repo_id")
        if not self.revision_sha:
            missing.append("revision_sha")
        raise ValueError(
            f"policy '{self.name}' is not runnable — missing {missing} "
            f"(source: {self._source}). Lock these at Day 0a per "
            "docs/NEXT_STEPS.md, then update configs/policies.yaml."
        )


class PolicyRegistry:
    """Indexed collection of :class:`PolicySpec`, loaded from YAML."""

    def __init__(self, specs: dict[str, PolicySpec]) -> None:
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: Path | str) -> PolicyRegistry:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "policies" not in data:
            raise ValueError(f"{path}: top-level YAML must be a mapping with key 'policies'")
        entries = data["policies"]
        if not isinstance(entries, list):
            raise ValueError(f"{path}: 'policies' must be a list")

        specs: dict[str, PolicySpec] = {}
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}: policies[{i}] must be a mapping, got {type(entry).__name__}"
                )
            spec = _spec_from_dict(entry, source=f"{path}: policies[{i}]")
            if spec.name in specs:
                raise ValueError(f"{path}: duplicate policy name '{spec.name}'")
            specs[spec.name] = spec
        return cls(specs)

    def get(self, name: str) -> PolicySpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs)) or "<empty>"
            raise KeyError(f"unknown policy '{name}'; available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._specs)

    def supporting(self, env_name: str) -> list[PolicySpec]:
        """Policies whose ``env_compat`` includes ``env_name``."""
        return [s for s in self._specs.values() if env_name in s.env_compat]

    def runnable(self) -> list[PolicySpec]:
        """All policies that pass :meth:`PolicySpec.is_runnable`."""
        return [s for s in self._specs.values() if s.is_runnable()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[PolicySpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def _spec_from_dict(entry: dict[str, Any], *, source: str) -> PolicySpec:
    keys = set(entry)
    missing = _REQUIRED_FIELDS - keys
    if missing:
        raise ValueError(f"{source}: missing required fields: {sorted(missing)}")
    extras = keys - _ALL_FIELDS
    if extras:
        raise ValueError(f"{source}: unknown fields: {sorted(extras)}")

    is_baseline = entry["is_baseline"]
    if not isinstance(is_baseline, bool):
        raise ValueError(f"{source}: is_baseline must be a bool, got {type(is_baseline).__name__}")

    env_compat = entry["env_compat"]
    if not isinstance(env_compat, list) or not all(isinstance(x, str) for x in env_compat):
        raise ValueError(f"{source}: env_compat must be a list of strings")

    fp_precision = entry.get("fp_precision")
    if fp_precision is not None and fp_precision not in _VALID_FP:
        raise ValueError(
            f"{source}: fp_precision must be one of {sorted(_VALID_FP)}, got {fp_precision!r}"
        )

    repo_id = entry.get("repo_id")
    revision_sha = entry.get("revision_sha")

    # Baselines must NOT carry weights metadata.
    if is_baseline and (repo_id or revision_sha or fp_precision):
        raise ValueError(
            f"{source}: baseline policies must not set repo_id / revision_sha / fp_precision"
        )

    paper_reported_success = _parse_paper_reported_success(
        entry.get("paper_reported_success"), env_compat=env_compat, source=source
    )

    return PolicySpec(
        name=str(entry["name"]),
        is_baseline=is_baseline,
        env_compat=tuple(env_compat),
        repo_id=str(repo_id) if repo_id else None,
        revision_sha=str(revision_sha) if revision_sha else None,
        fp_precision=fp_precision,
        license=str(entry["license"]) if entry.get("license") else None,
        notes=str(entry.get("notes", "")),
        paper_reported_success=paper_reported_success,
        paper_reported_notes=str(entry.get("paper_reported_notes", "")),
        _source=source,
    )


def _parse_paper_reported_success(
    raw: Any, *, env_compat: list[str], source: str
) -> dict[str, float | None] | None:
    """Validate ``paper_reported_success`` and return a normalized dict.

    Returns ``None`` if absent (the common case for baselines). Otherwise
    returns a ``{env_name: float | None}`` mapping where each key must
    appear in ``env_compat`` and each non-null value is a fraction in
    ``[0.0, 1.0]``. ``null`` entries are preserved (they mean "the paper
    does not report this exact env-task combination").
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{source}: paper_reported_success must be a mapping of env-name -> "
            f"float (or null), got {type(raw).__name__}"
        )
    env_compat_set = set(env_compat)
    parsed: dict[str, float | None] = {}
    for env_name, value in raw.items():
        if not isinstance(env_name, str):
            raise ValueError(
                f"{source}: paper_reported_success keys must be strings (env names), "
                f"got {type(env_name).__name__}"
            )
        if env_name not in env_compat_set:
            raise ValueError(
                f"{source}: paper_reported_success references env {env_name!r} "
                f"which is not in env_compat {sorted(env_compat_set)}"
            )
        if value is None:
            parsed[env_name] = None
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{source}: paper_reported_success[{env_name!r}] must be a float in "
                f"[0, 1] or null, got {type(value).__name__}"
            )
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"{source}: paper_reported_success[{env_name!r}] = {value} "
                f"is outside [0, 1]; pass success rates as fractions, not percentages"
            )
        parsed[env_name] = float(value)
    return parsed
