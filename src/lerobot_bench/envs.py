"""Sim env registry for lerobot-bench.

Pure data — no env construction. The actual ``gym.make`` call lives in
``lerobot_bench.eval``; this module just describes the envs we know
about (gym ids, per-episode step caps, success thresholds, lerobot
module paths) so the eval loop can resolve names to specs without
hardcoding.

Source-of-truth for human edits is ``configs/envs.yaml``. Mirror in
``docs/MODEL_CARDS.md`` (the doc) — but that doc is descriptive; the
YAML is what the runtime reads.

Two construction paths are supported:

* **gym** — set ``gym_id`` (and optional ``gym_kwargs``). The eval
  loop calls ``gymnasium.make(gym_id, max_episode_steps=max_steps,
  **gym_kwargs)``. Used for PushT, Aloha — anything gym-registered.
* **factory** — set ``factory`` (a dotted module path that exposes
  ``make_env``) and optional ``factory_kwargs``. The eval loop
  ``importlib.import_module(factory).make_env(**factory_kwargs)``.
  Used for LIBERO task suites that lerobot wraps via
  ``lerobot.envs.factory.make_env(env_type="libero", ...)`` because
  the underlying ``libero.libero.envs.OffScreenRenderEnv`` is not
  gym-registered.

Exactly one of ``gym_id`` / ``factory`` must be set; both-set or
neither-set is a YAML schema error caught at registry load.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Required keys every env entry must have. Validated at registry load.
_REQUIRED_FIELDS = frozenset({"name", "family", "max_steps", "success_threshold", "lerobot_module"})
_OPTIONAL_FIELDS = frozenset({"gym_id", "gym_kwargs", "factory", "factory_kwargs"})
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class EnvSpec:
    """One sim env in the benchmark matrix.

    ``family`` groups envs in the leaderboard ("pusht", "aloha", "libero").
    Multiple sim tasks can share a family (e.g. several Aloha task
    variants, or all four LIBERO suites). ``lerobot_module`` is the
    dotted path to the env config inside lerobot — used to resolve
    ``SUCCESS_REWARD`` if available.

    **Construction path** is chosen by which of ``gym_id`` / ``factory``
    is set:

    * ``gym_id`` set → eval loop calls ``gymnasium.make(gym_id,
      max_episode_steps=max_steps, **gym_kwargs_dict())``. PushT, Aloha.
    * ``factory`` set → eval loop calls ``importlib.import_module(factory)
      .make_env(**factory_kwargs_dict())``. LIBERO. The factory must
      return either a gymnasium env or a ``{suite_name: {task_id:
      vec_env}}`` mapping that ``load_env`` can pick a single env from.

    ``gym_kwargs`` and ``factory_kwargs`` are stored as tuples of
    ``(key, value)`` pairs so the dataclass remains hashable;
    reconstruct the dict via :meth:`gym_kwargs_dict` /
    :meth:`factory_kwargs_dict`. Pretrained policies (e.g.
    ``diffusion_policy``) need ``obs_type='pixels_agent_pos'`` to
    receive the image+state observations they were trained on; the
    baseline ``random``/``no_op`` policies are obs-shape-agnostic, so
    the same env spec works for both.
    """

    name: str
    family: str
    max_steps: int
    success_threshold: float
    lerobot_module: str
    gym_id: str | None = None
    gym_kwargs: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    factory: str | None = None
    factory_kwargs: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Mutual exclusion + at-least-one. Loader catches this too with a
        # nicer source-line error message; this is the in-memory belt
        # for direct dataclass construction (used widely in tests).
        if self.gym_id is None and self.factory is None:
            raise ValueError(f"env '{self.name}': exactly one of 'gym_id' or 'factory' must be set")
        if self.gym_id is not None and self.factory is not None:
            raise ValueError(f"env '{self.name}': 'gym_id' and 'factory' are mutually exclusive")

    def gym_kwargs_dict(self) -> dict[str, Any]:
        """Materialize :attr:`gym_kwargs` as a fresh ``dict`` for ``gym.make``."""
        return dict(self.gym_kwargs)

    def factory_kwargs_dict(self) -> dict[str, Any]:
        """Materialize :attr:`factory_kwargs` as a fresh ``dict`` for the factory call."""
        return dict(self.factory_kwargs)

    @property
    def uses_factory(self) -> bool:
        """True iff this spec is constructed via the factory path."""
        return self.factory is not None


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


def _coerce_kwargs(raw: Any, *, source: str, field_name: str) -> tuple[tuple[str, Any], ...]:
    """Validate ``raw`` is a ``dict[str, Any]`` and convert to sorted tuple-of-pairs.

    YAML lists (e.g. ``task_ids: [0]``) are converted to tuples so the
    resulting :class:`EnvSpec` remains hashable. Nested dicts are
    rejected — the kwargs surface for ``gym.make`` / ``make_env`` does
    not need them, and allowing them would force a deep recursion that
    makes equality checks brittle.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: {field_name} must be a mapping, got {type(raw).__name__}")
    if not all(isinstance(k, str) for k in raw):
        raise ValueError(f"{source}: {field_name} keys must all be strings")
    coerced: dict[str, Any] = {}
    for k, v in raw.items():
        coerced[k] = _freeze_value(v, source=source, field_name=f"{field_name}.{k}")
    # Sort for deterministic ordering -- the dataclass is hashable and we want
    # equal mappings to compare equal regardless of YAML key order.
    return tuple(sorted(coerced.items(), key=lambda kv: kv[0]))


def _freeze_value(value: Any, *, source: str, field_name: str) -> Any:
    """Convert YAML containers to hashable equivalents (list -> tuple).

    Recurses one level into lists for the common ``[int, int, ...]`` case
    (``task_ids``); dicts and deeper nesting are rejected with a clear
    message so we don't end up with an unhashable spec at runtime.
    """
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if isinstance(item, dict | list):
                raise ValueError(
                    f"{source}: {field_name} cannot contain nested containers "
                    f"(got {type(item).__name__} inside a list)"
                )
            out.append(item)
        return tuple(out)
    if isinstance(value, dict):
        raise ValueError(
            f"{source}: {field_name} cannot be a nested mapping (use scalars or flat lists only)"
        )
    return value


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

    gym_id_raw = entry.get("gym_id")
    factory_raw = entry.get("factory")
    if gym_id_raw is None and factory_raw is None:
        raise ValueError(f"{source}: exactly one of 'gym_id' or 'factory' must be set")
    if gym_id_raw is not None and factory_raw is not None:
        raise ValueError(
            f"{source}: 'gym_id' and 'factory' are mutually exclusive (set one, not both)"
        )

    gym_kwargs = _coerce_kwargs(entry.get("gym_kwargs", {}), source=source, field_name="gym_kwargs")
    factory_kwargs = _coerce_kwargs(
        entry.get("factory_kwargs", {}), source=source, field_name="factory_kwargs"
    )

    return EnvSpec(
        name=str(entry["name"]),
        family=str(entry["family"]),
        gym_id=str(gym_id_raw) if gym_id_raw is not None else None,
        gym_kwargs=gym_kwargs,
        factory=str(factory_raw) if factory_raw is not None else None,
        factory_kwargs=factory_kwargs,
        max_steps=int(max_steps),
        success_threshold=float(success_threshold),
        lerobot_module=str(entry["lerobot_module"]),
    )
