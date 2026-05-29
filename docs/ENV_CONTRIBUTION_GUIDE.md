# Env contribution guide

Companion to [`POLICY_DIAGRAM_GUIDE.md`](POLICY_DIAGRAM_GUIDE.md) (policy onboarding) and [`PIPELINE_ROADMAP.md`](PIPELINE_ROADMAP.md). **Use this when adding a new environment to the lerobot-bench leaderboard.**

The bench is designed to host community-contributed envs via the same factory mechanism that hosts LIBERO and the upstream lerobot envs. This doc is the operator-facing walkthrough — the canonical worked example is [`thrmnn/lerobot-env-so100-pickplace`](https://github.com/thrmnn/lerobot-env-so100-pickplace), built as part of v1.1.

## What an env contribution is

A bench-compatible env is **any Python package that exposes a `make_env(**kwargs) -> gymnasium.Env` callable** at its top-level namespace and satisfies the bench's observation contract. The env can live anywhere — your own GitHub repo, a Hugging Face Hub-hosted module, vendored in-tree. The bench treats them all the same way: a registry entry in `configs/envs.yaml` points at the factory, and the eval loop calls it like any other.

## The two integration paths

### Path A — pip-installable package (recommended)

Your env ships as a standalone repo with a `pyproject.toml`. The bench depends on it via a git ref + commit SHA. This is the path `lerobot-env-so100-pickplace` uses.

| Pros | Cons |
|---|---|
| Independent versioning, own release cadence | Two-repo coordination on breaking changes |
| Reusable outside the bench (anyone can `pip install` and run) | Pinned SHA must be bumped in the bench when env updates |
| Clean dependency boundary (bench has no env-specific imports) | First-time setup is one extra file (the pyproject) |

### Path B — Hub-hosted module via lerobot 0.5.1's `HubEnvConfig`

Push your env's `env.py` to a Hugging Face Hub model repo. The bench loads it via `HubEnvConfig(hub_path="user/repo", trust_remote_code=True)`. Lowest friction for the contributor but requires the bench user to opt-in to remote-code execution.

Choose Path A unless you have a specific reason to use Path B. Path B exists for upstream lerobot ecosystem compatibility; the bench prefers explicit pip deps for reproducibility audit trails.

## What the bench expects from your env

### Factory signature

```python
def make_env(*, image_size: int = 240, max_steps: int = 400, render_mode: str = "rgb_array", **kwargs) -> gymnasium.Env:
    """Bench-facing factory.

    Must accept **kwargs so the bench's factory_kwargs dict flows through
    without requiring every env to declare every possible knob.
    """
    ...
```

The bench's loader (`src/lerobot_bench/eval.py:959-1020`) calls this with the `factory_kwargs` from `configs/envs.yaml` plus any defaults. `n_envs` is intentionally not a parameter — the bench runs single-env cells.

### Observation contract (`obs_type='pixels_agent_pos'`)

Return a dict from `reset()` and `step()` with these keys:

```python
{
    "pixels": {                          # multi-camera dict
        "top": np.ndarray,               # shape (image_size, image_size, 3), uint8
        "wrist": np.ndarray,             # shape (image_size, image_size, 3), uint8
        # ... additional cameras as needed
    },
    "agent_pos": np.ndarray,             # shape (n_dof,), float32 — robot joint state
}
```

If your env has only one camera, use a single-key dict (`{"top": ...}`) — the pretrained-policy adapters handle both. If your robot's state isn't joint angles (e.g. end-effector pose), still call it `agent_pos` — the bench is consistent on the key name, the meaning is per-policy.

### Action contract

`gym.spaces.Box` of shape `(n_dof,)`, dtype `float32`, bounded `[-1, 1]` (recommended) or `[low, high]` per joint. Policies expect normalized actions and will rescale; the bench doesn't post-process.

### Success contract

`reward` is real-valued. The bench reads the per-episode terminal reward and compares against `success_threshold` in `configs/envs.yaml`. Two valid patterns:

- **Binary** (`success_threshold=1.0`): emit `reward=1.0` only on success terminal, `reward=0.0` everywhere else. LIBERO uses this.
- **Dense + threshold** (`success_threshold=0.95`): emit a continuous reward (e.g. target-area coverage), bench thresholds at 0.95. PushT uses this.

`terminated=True` should fire when success is reached so the eval loop short-circuits the remaining steps.

### Seeding contract

`reset(seed=s)` must produce a deterministic initial state when given the same `s`. The bench's per-episode seed is `seed_cell * 1000 + episode_index` (see `docs/DESIGN.md` § Methodology); your env's randomization (object poses, distractors, etc.) must be seeded by the same `s` you receive.

## Five-step integration walkthrough

The canonical worked example is `lerobot-env-so100-pickplace`. Mechanically, you do this:

### 1. Build your env repo

Follow the layout in [`thrmnn/lerobot-env-so100-pickplace`](https://github.com/thrmnn/lerobot-env-so100-pickplace) — it's the template:

```
your-env-repo/
├── src/your_env_package/
│   ├── __init__.py             # exports make_env (the bench-facing entry)
│   ├── env.py                  # gym.Env subclass
│   ├── assets/                 # URDFs, MJCFs, meshes
│   └── tasks/                  # task-specific logic
├── tests/
│   ├── test_make_env.py        # factory contract checks
│   └── test_env_smoke.py       # observation shapes, step/reset contracts
├── examples/
│   └── run_random_policy.py    # 5-episode smoke test
├── pyproject.toml              # name=your-env-package
└── README.md
```

Your tests should include:
- Factory returns a `gym.Env` instance
- `make_env(image_size=120)` produces obs with the requested shape
- `make_env(max_steps=10)` truncates at step 10
- `reset()` returns the bench's expected dict structure
- `step(action)` returns the gymnasium 5-tuple

### 2. Vendor responsibly

If you're using a third-party asset (MJCF, URDF, mesh), check the license:
- Apache-2.0 / MIT / BSD: vendor freely, add `NOTICE` or `ATTRIBUTION.md` with original source + commit SHA + license
- GPL / proprietary: don't vendor — link to install instructions instead
- Unlicensed: skip — legal blocker

`lerobot-env-so100-pickplace` vendors the SO-100 MJCF from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) (Apache-2.0) and records the source commit SHA in `src/.../assets/ATTRIBUTION.md`.

### 3. Add an `EnvSpec` entry to `configs/envs.yaml`

In `lerobot-bench`:

```yaml
- name: your_env_name
  family: your_family             # informal grouping for the leaderboard
  factory: your_env_package       # the module that exposes make_env
  factory_kwargs:
    image_size: 240               # passed through to make_env(**kwargs)
    # ... any other knobs
  max_steps: 400
  success_threshold: 1.0          # see Success contract above
  lerobot_module: null            # null for non-lerobot envs
```

The `factory:` field is the dotted import path to a module with a `make_env` function. `factory_kwargs:` is a dict (loaded as `tuple-of-pairs` internally so the spec stays hashable) of arguments passed through to `make_env(**kwargs)`.

### 4. Add a pip dep to `lerobot-bench/pyproject.toml`

```toml
[project.optional-dependencies]
your-env = [
    "your-env-package @ git+https://github.com/yourorg/your-env-package.git@<COMMIT_SHA>",
]
```

Pin the SHA, not a tag or branch — the bench's reproducibility contract requires deterministic dep resolution.

### 5. PR + run a sweep cell

Open a PR against `lerobot-bench` with both changes (`configs/envs.yaml` + `pyproject.toml`). The CI runs `validate-configs.yml` which loads the registry through `EnvRegistry.from_yaml` and catches schema errors. Once green, run:

```bash
python scripts/run_one.py --policy random --env your_env_name --seed 0 --n-episodes 5
```

If this prints success/failure rows to `results/results.parquet`, your env is bench-compatible. Add a row to `docs/MODEL_CARDS.md` describing the task + a sentence on what a successful run looks like.

## Common pitfalls

- **Observation shape mismatch**: if your env returns `(H, W, 3)` but a policy expects `(3, H, W)`, the bench's adapter handles the HWC→CHW conversion. You do NOT need to pre-transpose. But you DO need uint8 for pixels and float32 for `agent_pos`.
- **Action space drift across resets**: `action_space` must be a constant `gym.spaces.Box`, not generated per-reset. Pretrained policies cache action statistics from this object at init time.
- **Render performance**: `render()` is called on every step when video recording is on (the bench's default). If your render is slow, the per-cell wall-clock balloons; expect 20-100 ms/step as the realistic envelope. Profile early.
- **Goal-conditioning**: if your env is a goal-env (GoalEnv API), expose the goal in `info["desired_goal"]` and `info["achieved_goal"]`. The bench's adapters can route this to policies that accept it, but only if the keys are standard.

## Worked example

[`thrmnn/lerobot-env-so100-pickplace`](https://github.com/thrmnn/lerobot-env-so100-pickplace) walks the full path end-to-end: MJCF vendoring with attribution, single-arm SO-100 env class porting (~200 LoC against the gym-aloha template), pick-place task definition, the factory wiring, the test suite, and the bench-side `configs/envs.yaml` PR. Open issues / discussions there for env-design questions; this guide is the meta-doc on top.

## When NOT to contribute an env

- Your task is "[X] but with a different reward function" — that's a probe (see [`examples/write_a_probe.md`](../examples/write_a_probe.md)), not a new env. The bench's probe mechanism handles this without a new registry entry.
- Your env is identical to an existing one with a different policy — that's a policy contribution (see [`POLICY_DIAGRAM_GUIDE.md`](POLICY_DIAGRAM_GUIDE.md)).
- Your env requires custom CUDA kernels / specialised compute — the bench targets reproducible commodity hardware; raise an issue first before building.

---

*Maintainer: bump this doc when the env factory contract changes (`src/lerobot_bench/eval.py:959-1020`) or when `configs/envs.yaml` schema gains/loses fields.*
