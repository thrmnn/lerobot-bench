"""Sim env registry for lerobot-bench.

Pure data — no env construction. The actual ``gym.make`` call lives in
``lerobot_bench.eval``; this module just describes the envs we know
about (gym ids, per-episode step caps, success thresholds, lerobot
module paths) so the eval loop can resolve names to specs without
hardcoding.

Source-of-truth for human edits is ``configs/envs.yaml``. Mirror in
``docs/MODEL_CARDS.md`` (the doc) — but that doc is descriptive; the
YAML is what the runtime reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Required keys every env entry must have. Validated at registry load.
_REQUIRED_FIELDS = frozenset(
    {"name", "family", "gym_id", "max_steps", "success_threshold", "lerobot_module"}
)
_OPTIONAL_FIELDS = frozenset({"gym_kwargs"})
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class EnvSpec:
    """One sim env in the benchmark matrix.

    ``family`` groups envs in the leaderboard ("pusht", "aloha", "libero").
    Multiple sim tasks can share a family (e.g. several Aloha task
    variants). ``lerobot_module`` is the dotted path to the env config
    inside lerobot — used to resolve ``SUCCESS_REWARD`` if available.

    ``gym_kwargs`` carries extra keyword arguments forwarded to
    :func:`gymnasium.make` beyond ``gym_id`` and ``max_episode_steps``.
    Stored as a tuple of ``(key, value)`` pairs so the dataclass remains
    hashable; reconstruct the dict via :meth:`gym_kwargs_dict`. Pretrained
    policies (e.g. ``diffusion_policy``) need ``obs_type='pixels_agent_pos'``
    to receive the image+state observations they were trained on; the
    baseline ``random``/``no_op`` policies are obs-shape-agnostic, so the
    same env spec works for both.
    """

    name: str
    family: str
    gym_id: str
    max_steps: int
    success_threshold: float
    lerobot_module: str
    gym_kwargs: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def gym_kwargs_dict(self) -> dict[str, Any]:
        """Materialize :attr:`gym_kwargs` as a fresh ``dict`` for ``gym.make``."""
        return dict(self.gym_kwargs)


class EnvRegistry:
    """Indexed collection of :class:`EnvSpec`, loaded from YAML."""

    def __init__(self, specs: dict[str, EnvSpec]) -> None:
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: Path | str) -> EnvRegistry:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "envs" not in data:
            raise ValueError(f"{path}: top-level YAML must be a mapping with key 'envs'")
        entries = data["envs"]
        if not isinstance(entries, list):
            raise ValueError(f"{path}: 'envs' must be a list")

        specs: dict[str, EnvSpec] = {}
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: envs[{i}] must be a mapping, got {type(entry).__name__}")
            spec = _spec_from_dict(entry, source=f"{path}: envs[{i}]")
            if spec.name in specs:
                raise ValueError(f"{path}: duplicate env name '{spec.name}'")
            specs[spec.name] = spec
        return cls(specs)

    def get(self, name: str) -> EnvSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs)) or "<empty>"
            raise KeyError(f"unknown env '{name}'; available: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._specs)

    def by_family(self, family: str) -> list[EnvSpec]:
        return [s for s in self._specs.values() if s.family == family]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __iter__(self) -> Iterator[EnvSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def _spec_from_dict(entry: dict[str, Any], *, source: str) -> EnvSpec:
    keys = set(entry)
    missing = _REQUIRED_FIELDS - keys
    if missing:
        raise ValueError(f"{source}: missing required fields: {sorted(missing)}")
    extras = keys - _ALL_FIELDS
    if extras:
        raise ValueError(f"{source}: unknown fields: {sorted(extras)}")

    max_steps = entry["max_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"{source}: max_steps must be a positive int, got {max_steps!r}")

    success_threshold = entry["success_threshold"]
    if not isinstance(success_threshold, int | float):
        raise ValueError(
            f"{source}: success_threshold must be a number, got {type(success_threshold).__name__}"
        )

    gym_kwargs_raw = entry.get("gym_kwargs", {})
    if not isinstance(gym_kwargs_raw, dict):
        raise ValueError(
            f"{source}: gym_kwargs must be a mapping, got {type(gym_kwargs_raw).__name__}"
        )
    if not all(isinstance(k, str) for k in gym_kwargs_raw):
        raise ValueError(f"{source}: gym_kwargs keys must all be strings")
    # Sort for deterministic ordering -- the dataclass is hashable and we want
    # equal mappings to compare equal regardless of YAML key order.
    gym_kwargs = tuple(sorted(gym_kwargs_raw.items(), key=lambda kv: kv[0]))

    return EnvSpec(
        name=str(entry["name"]),
        family=str(entry["family"]),
        gym_id=str(entry["gym_id"]),
        max_steps=int(max_steps),
        success_threshold=float(success_threshold),
        lerobot_module=str(entry["lerobot_module"]),
        gym_kwargs=gym_kwargs,
    )
