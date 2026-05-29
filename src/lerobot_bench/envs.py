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

**Success criterion (v1.1).** v1 shipped a single per-env scoring
rule: ``success := final_reward >= success_threshold``. The v1.0.1
audit (``docs/SUCCESS_CRITERION_AUDIT.md``, ``docs/CANONICAL_CRITERIA.md``)
identified three places where the v1 rule diverges from the canonical
paper / Hub eval (PushT sticky any-window, Aloha strict ``reward == 4``,
LIBERO 600-step cap). Rather than break replay of v1.0 parquets, v1.1
makes the rule *selectable*: every spec carries the v1 fields plus an
optional ``canonical`` overlay (``canonical_max_steps``,
``canonical_success_metric``, ``canonical_strict_reward_value``).
:meth:`EnvSpec.with_criterion` returns a fresh spec with the overlay
applied; ``v1_legacy`` is the default and produces bit-identical
behaviour to v1.0.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# Required keys every env entry must have. Validated at registry load.
_REQUIRED_FIELDS = frozenset({"name", "family", "max_steps", "success_threshold", "lerobot_module"})
_OPTIONAL_FIELDS = frozenset(
    {
        "gym_id",
        "gym_kwargs",
        "factory",
        "factory_kwargs",
        "canonical",  # v1.1 overlay; see _coerce_canonical_overlay
    }
)
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS

# Per-env scoring rule names. ``final_reward_threshold`` matches the v1
# behaviour: ``success := final_reward >= success_threshold``.
# ``sticky_is_success`` reads ``info["is_success"]`` each step and
# OR-accumulates -- the lerobot canonical eval's ``any(is_success)``.
# ``sticky_reward_eq`` OR-accumulates ``reward == strict_reward_value``
# (used by the ACT paper's Aloha ``reward == 4`` Transfer rule).
SUCCESS_METRICS = frozenset({"final_reward_threshold", "sticky_is_success", "sticky_reward_eq"})

# Selectable criterion labels. ``v1_legacy`` is the back-compat default;
# ``canonical`` opts into the paper / Hub-card rule per env.
CRITERION_LABELS = frozenset({"v1_legacy", "canonical"})


@dataclass(frozen=True)
class CanonicalOverlay:
    """Per-env overrides applied when ``criterion == 'canonical'``.

    Every field is optional. ``None`` means "keep the v1 value". The
    overlay is the source of truth for what *changes* when the operator
    opts into the canonical rule; the v1 fields on :class:`EnvSpec` are
    untouched so a default-criterion run still replays v1.0 bit-identically.

    See ``docs/CANONICAL_CRITERIA.md`` for the per-env table and the
    paper citations behind each field.
    """

    max_steps: int | None = None
    success_metric: str | None = None
    success_threshold: float | None = None
    strict_reward_value: float | None = None


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

    **Success rule fields.** ``success_metric`` selects how the eval
    loop turns a rollout into a boolean; defaults to
    ``"final_reward_threshold"`` (v1 behaviour). When the metric is
    ``"sticky_reward_eq"``, ``strict_reward_value`` is the integer-valued
    target (e.g. ``4.0`` for Aloha Transfer). The ``canonical`` overlay
    is the v1.1 mechanism for opting into the paper rule per env; see
    :meth:`with_criterion`.
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
    success_metric: str = "final_reward_threshold"
    strict_reward_value: float | None = None
    canonical: CanonicalOverlay = field(default_factory=CanonicalOverlay)

    def __post_init__(self) -> None:
        # Mutual exclusion + at-least-one. Loader catches this too with a
        # nicer source-line error message; this is the in-memory belt
        # for direct dataclass construction (used widely in tests).
        if self.gym_id is None and self.factory is None:
            raise ValueError(f"env '{self.name}': exactly one of 'gym_id' or 'factory' must be set")
        if self.gym_id is not None and self.factory is not None:
            raise ValueError(f"env '{self.name}': 'gym_id' and 'factory' are mutually exclusive")
        if self.success_metric not in SUCCESS_METRICS:
            raise ValueError(
                f"env '{self.name}': success_metric must be one of "
                f"{sorted(SUCCESS_METRICS)}, got {self.success_metric!r}"
            )
        if self.success_metric == "sticky_reward_eq" and self.strict_reward_value is None:
            raise ValueError(
                f"env '{self.name}': success_metric='sticky_reward_eq' requires "
                "strict_reward_value to be set"
            )
        # Validate the canonical overlay's metric (if it overrides one).
        if (
            self.canonical.success_metric is not None
            and self.canonical.success_metric not in SUCCESS_METRICS
        ):
            raise ValueError(
                f"env '{self.name}': canonical.success_metric must be one of "
                f"{sorted(SUCCESS_METRICS)}, got {self.canonical.success_metric!r}"
            )

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

    def with_criterion(self, criterion: str) -> EnvSpec:
        """Return a copy of this spec under the requested scoring criterion.

        ``criterion='v1_legacy'`` returns ``self`` unchanged -- the
        spec's primary fields already encode the v1 behaviour.
        ``criterion='canonical'`` applies the ``canonical`` overlay
        (any ``None`` field falls through to the v1 value).

        Used by ``scripts/run_one.py`` and ``scripts/run_sweep.py`` at
        the boundary between config load and eval dispatch, so the eval
        loop itself never needs to know which criterion is active --
        the spec it receives already carries the right ``max_steps``,
        ``success_metric``, ``success_threshold`` and
        ``strict_reward_value``.
        """
        if criterion not in CRITERION_LABELS:
            raise ValueError(
                f"unknown criterion '{criterion}'; expected one of {sorted(CRITERION_LABELS)}"
            )
        if criterion == "v1_legacy":
            return self
        ov = self.canonical
        return replace(
            self,
            max_steps=ov.max_steps if ov.max_steps is not None else self.max_steps,
            success_metric=(
                ov.success_metric if ov.success_metric is not None else self.success_metric
            ),
            success_threshold=(
                ov.success_threshold if ov.success_threshold is not None else self.success_threshold
            ),
            strict_reward_value=(
                ov.strict_reward_value
                if ov.strict_reward_value is not None
                else self.strict_reward_value
            ),
            # Clear the overlay on the returned spec so a second
            # with_criterion('canonical') call is a no-op rather than
            # double-applying.
            canonical=CanonicalOverlay(),
        )


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

    def with_criterion(self, criterion: str) -> EnvRegistry:
        """Return a fresh registry with :meth:`EnvSpec.with_criterion` applied to every spec."""
        return EnvRegistry(
            {name: spec.with_criterion(criterion) for name, spec in self._specs.items()}
        )

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


_CANONICAL_FIELDS = frozenset(
    {"max_steps", "success_metric", "success_threshold", "strict_reward_value"}
)


def _coerce_canonical_overlay(raw: Any, *, source: str) -> CanonicalOverlay:
    """Validate + build a :class:`CanonicalOverlay` from a YAML sub-mapping.

    Missing keys fall through to ``None``; unknown keys raise. Type
    checking is per-field (positive int for ``max_steps``, string for
    ``success_metric``, number for the two reward fields).
    """
    if raw is None:
        return CanonicalOverlay()
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: canonical must be a mapping, got {type(raw).__name__}")
    extras = set(raw) - _CANONICAL_FIELDS
    if extras:
        raise ValueError(f"{source}: canonical has unknown fields: {sorted(extras)}")

    max_steps = raw.get("max_steps")
    if max_steps is not None and (
        not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
    ):
        raise ValueError(f"{source}: canonical.max_steps must be a positive int, got {max_steps!r}")

    success_metric = raw.get("success_metric")
    if success_metric is not None and not isinstance(success_metric, str):
        raise ValueError(
            f"{source}: canonical.success_metric must be a string, "
            f"got {type(success_metric).__name__}"
        )

    success_threshold = raw.get("success_threshold")
    if success_threshold is not None and not isinstance(success_threshold, int | float):
        raise ValueError(
            f"{source}: canonical.success_threshold must be a number, "
            f"got {type(success_threshold).__name__}"
        )

    strict_reward_value = raw.get("strict_reward_value")
    if strict_reward_value is not None and not isinstance(strict_reward_value, int | float):
        raise ValueError(
            f"{source}: canonical.strict_reward_value must be a number, "
            f"got {type(strict_reward_value).__name__}"
        )

    return CanonicalOverlay(
        max_steps=int(max_steps) if max_steps is not None else None,
        success_metric=str(success_metric) if success_metric is not None else None,
        success_threshold=float(success_threshold) if success_threshold is not None else None,
        strict_reward_value=(
            float(strict_reward_value) if strict_reward_value is not None else None
        ),
    )


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

    canonical = _coerce_canonical_overlay(entry.get("canonical"), source=source)

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
        canonical=canonical,
    )
