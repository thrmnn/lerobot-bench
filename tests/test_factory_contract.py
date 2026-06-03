"""BYO-env factory loader contract — bench side, no heavy sim deps.

These tests exercise ``eval._load_factory_env`` /
``eval._materialize_factory_result`` (the bench's loader for an external,
"bring your own env" factory module such as
``lerobot-env-so100-pickplace``) against a lightweight *stub* factory
that mimics the so100 ``make_env`` contract. The point is to catch drift
in the loader-vs-env contract in the fast CI job, where neither MuJoCo
nor lerobot is installed.

What the contract requires of a BYO factory module (mirrored by the
stubs below):

* exposes ``make_env(...)`` (bare call when no ``env_type`` in
  ``factory_kwargs``);
* honours ``image_size`` / ``max_steps`` / ``reward_mode`` (we assert the
  stub recorded the kwargs the loader forwarded);
* rejects ``n_envs != 1`` — actually enforced *bench-side* by the loader,
  which pops ``n_envs`` and raises before calling the factory;
* may return either a bare ``gym.Env``-like object, a size-1 vector env,
  or a ``{suite: {task: vec_env}}`` mapping — all three collapse to one
  ``GymLikeEnv``.

Implementation notes:

* The stub factory is registered in ``sys.modules`` under a synthetic
  dotted name so the loader's ``importlib.import_module(spec.factory)``
  resolves it without a real package on disk.
* These tests do **not** import gymnasium; the loader duck-types vec
  envs on ``num_envs`` + ``step`` (see ``_wrap_leaf``), so a plain object
  with those attrs stands in for a real ``VectorEnv``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from lerobot_bench import eval as bench_eval
from lerobot_bench.envs import EnvSpec


# --------------------------------------------------------------------------
# Stub envs + factory module mimicking the so100 BYO contract
# --------------------------------------------------------------------------
class _StubEnv:
    """Minimal stand-in for a bare ``gym.Env`` returned by ``make_env``.

    Records the kwargs the factory was built with so the test can assert
    the loader forwarded ``image_size`` / ``max_steps`` / ``reward_mode``.
    No ``num_envs`` attr, so ``_wrap_leaf`` treats it as a scalar env and
    returns it as-is.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.build_kwargs = kwargs
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return {}, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _StubVecEnv:
    """Size-1 vector env — duck-typed by the loader on ``num_envs`` + ``step``."""

    num_envs = 1

    def __init__(self, inner: _StubEnv) -> None:
        self.inner = inner

    def reset(self, *, seed: Any = None) -> tuple[Any, dict[str, Any]]:
        return None, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        return None, None, None, None, {}

    def close(self) -> None:
        self.inner.close()


def _make_stub_factory(*, return_shape: str) -> types.ModuleType:
    """Build a synthetic factory module honouring the bench BYO contract.

    ``return_shape`` selects what ``make_env`` returns:
    ``"bare"`` (a ``_StubEnv``), ``"vec"`` (a size-1 ``_StubVecEnv``), or
    ``"mapping"`` (``{suite: {task: vec_env}}``).
    """
    mod = types.ModuleType("stub_factory")

    def make_env(
        *,
        image_size: int = 240,
        max_steps: int = 400,
        reward_mode: str = "dense",
        **_ignored: Any,
    ) -> Any:
        if reward_mode not in ("dense", "sparse"):
            raise ValueError(f"reward_mode must be 'dense'|'sparse', got {reward_mode!r}")
        env = _StubEnv(image_size=image_size, max_steps=max_steps, reward_mode=reward_mode)
        if return_shape == "bare":
            return env
        if return_shape == "vec":
            return _StubVecEnv(env)
        if return_shape == "mapping":
            return {"so100": {"pickplace": _StubVecEnv(env)}}
        raise AssertionError(f"unknown return_shape {return_shape!r}")

    mod.make_env = make_env  # type: ignore[attr-defined]
    return mod


def _factory_spec(module_name: str, **factory_kwargs: Any) -> EnvSpec:
    """An ``EnvSpec`` on the factory path pointing at ``module_name``."""
    return EnvSpec(
        name="so100_stub",
        family="so100",
        max_steps=400,
        success_threshold=1.0,
        lerobot_module="lerobot_env_so100_pickplace",
        factory=module_name,
        factory_kwargs=tuple(factory_kwargs.items()),
    )


@pytest.fixture
def register_factory(monkeypatch: pytest.MonkeyPatch):
    """Register a stub factory module in ``sys.modules`` and return its name."""

    def _register(*, return_shape: str = "bare") -> str:
        name = f"stub_factory_{return_shape}"
        monkeypatch.setitem(sys.modules, name, _make_stub_factory(return_shape=return_shape))
        return name

    return _register


# --------------------------------------------------------------------------
# Return-shape collapsing: bare / vec / mapping all yield one GymLikeEnv
# --------------------------------------------------------------------------
def test_bare_env_returned_as_is(register_factory):
    name = register_factory(return_shape="bare")
    env = bench_eval._load_factory_env(_factory_spec(name, image_size=64, max_steps=10))
    assert isinstance(env, _StubEnv)


def test_vec_env_is_debatched(register_factory):
    name = register_factory(return_shape="vec")
    env = bench_eval._load_factory_env(_factory_spec(name, image_size=64))
    # A size-1 vec env is wrapped so the eval loop sees a scalar gym API.
    assert isinstance(env, bench_eval._DebatchedVecEnvAdapter)


def test_mapping_single_suite_single_task_unwrapped(register_factory):
    name = register_factory(return_shape="mapping")
    env = bench_eval._load_factory_env(_factory_spec(name, image_size=64))
    assert isinstance(env, bench_eval._DebatchedVecEnvAdapter)


# --------------------------------------------------------------------------
# make_env must honour image_size / max_steps / reward_mode
# --------------------------------------------------------------------------
def test_loader_forwards_make_env_kwargs(register_factory):
    name = register_factory(return_shape="bare")
    spec = _factory_spec(name, image_size=96, max_steps=33, reward_mode="sparse")
    env = bench_eval._load_factory_env(spec)
    assert env.build_kwargs == {
        "image_size": 96,
        "max_steps": 33,
        "reward_mode": "sparse",
    }


def test_invalid_reward_mode_rejected_by_factory(register_factory):
    name = register_factory(return_shape="bare")
    spec = _factory_spec(name, reward_mode="bogus")
    with pytest.raises(ValueError, match="reward_mode"):
        bench_eval._load_factory_env(spec)


# --------------------------------------------------------------------------
# n_envs guard — enforced bench-side before the factory is even called
# --------------------------------------------------------------------------
def test_n_envs_not_one_rejected(register_factory):
    name = register_factory(return_shape="bare")
    spec = _factory_spec(name, n_envs=4)
    with pytest.raises(ValueError, match="n_envs must be 1"):
        bench_eval._load_factory_env(spec)


def test_n_envs_one_is_accepted_and_stripped(register_factory):
    """``n_envs=1`` is valid and must NOT leak into the make_env call."""
    name = register_factory(return_shape="bare")
    spec = _factory_spec(name, n_envs=1, image_size=64)
    env = bench_eval._load_factory_env(spec)
    assert "n_envs" not in env.build_kwargs


# --------------------------------------------------------------------------
# Module-shape errors: missing make_env, multi-suite, multi-task
# --------------------------------------------------------------------------
def test_missing_make_env_raises(monkeypatch: pytest.MonkeyPatch):
    mod = types.ModuleType("stub_no_make_env")
    monkeypatch.setitem(sys.modules, "stub_no_make_env", mod)
    spec = _factory_spec("stub_no_make_env")
    with pytest.raises(RuntimeError, match="does not expose"):
        bench_eval._load_factory_env(spec)


def test_unimportable_factory_module_raises():
    spec = _factory_spec("definitely.not.a.real.module")
    with pytest.raises(ImportError, match="not importable"):
        bench_eval._load_factory_env(spec)


def test_mapping_multi_suite_rejected():
    spec = _factory_spec("stub_factory")
    bad = {"a": {"t": _StubVecEnv(_StubEnv())}, "b": {"t": _StubVecEnv(_StubEnv())}}
    with pytest.raises(RuntimeError, match="expected 1 suite"):
        bench_eval._materialize_factory_result(bad, spec=spec)


def test_mapping_multi_task_rejected():
    spec = _factory_spec("stub_factory")
    bad = {"a": {"t1": _StubVecEnv(_StubEnv()), "t2": _StubVecEnv(_StubEnv())}}
    with pytest.raises(RuntimeError, match=r"expected suite .* to have 1 task"):
        bench_eval._materialize_factory_result(bad, spec=spec)
