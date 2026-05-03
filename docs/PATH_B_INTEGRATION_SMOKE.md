# Path B integration smoke

The shortest test that proves the synthetic mocks in `tests/` match real
lerobot/gym shapes. Run this **once**, in order, the first time the
lerobot conda env has `lerobot==0.5.1` plus a sim extra installed.

The whole thing should take ~15 minutes if everything cooperates and
catch the high-impact "everything builds, nothing runs" failure mode
before PR #14 (the analysis notebook) starts treating synthetic shapes
as truth.

Owner when reviewing failures: `bench-eval-engineer`.

## Prerequisites

- `cd /home/theo/projects/lerobot-bench` and the `lerobot` conda env active.
- `make all` green on the current commit.
- `python -c "import lerobot; print(lerobot.__version__)"` prints `0.5.1`.
  - If not: `pip install -e /home/theo/projects/lerobot` until upstream
    publishes 0.5.1 to PyPI.
- One of `gym-pusht` or `gym-aloha` importable:
  - `python -c "import gymnasium as gym; gym.make('gym_pusht/PushT-v0')"`
- `huggingface-cli whoami` returns the expected account (read-scope is
  enough for this smoke; write scope is needed later for publish).

## Step 1 — baseline cell still works

Sanity check: nothing in the new install path broke the runnable code.

```bash
mkdir -p results/smoke
python scripts/run_one.py \
  --policy random \
  --env pusht \
  --seed 0 \
  --n-episodes 2 \
  --results-path results/smoke/baseline.parquet
```

**Expected:** exit 0, `results/smoke/baseline.parquet` exists,
`python -c "import pandas as pd; print(pd.read_parquet('results/smoke/baseline.parquet'))"`
shows 2 rows with `policy=random`, `env=pusht`, `seed=0`,
`episode_index in {0,1}`, `success` is bool, `n_steps > 0`.

**If this fails:** sim install is broken. Stop. Do not proceed to step 2.

## Step 2 — lock one revision_sha

Pick the policy that's most likely to load cleanly (`diffusion_policy`
on PushT — well-trodden path). Find the current Hub revision SHA:

```bash
huggingface-cli scan-cache | grep diffusion_pusht  # if previously cached
# OR fetch the latest:
python -c "
from huggingface_hub import HfApi
info = HfApi().model_info('lerobot/diffusion_pusht')
print('revision_sha:', info.sha)
print('lastModified:', info.lastModified)
"
```

Record the SHA. Edit `configs/policies.yaml`:

```yaml
- name: diffusion_policy
  is_baseline: false
  env_compat: [pusht, aloha_transfer_cube]
  repo_id: lerobot/diffusion_pusht
  revision_sha: <the-40-char-SHA-from-above>   # was null
  fp_precision: fp32
  license: <whatever the model card says>      # was null
```

Do NOT commit yet — this is a smoke probe.

## Step 3 — implement the Day 0b TODO in `eval.load_policy`

**STATUS: completed (PR landed 2026-05-03).** This step describes
what the work entailed; you do NOT need to redo it on a fresh checkout.

The historical `lerobot.common.policies.factory.make_policy` import
path was wrong for `lerobot==0.5.1`: the `common.` namespace is gone
in 0.5.x, and the surviving `lerobot.policies.factory.make_policy`
no longer takes `(repo_id, revision=..., fp_precision=...)` — its
signature is now `make_policy(cfg: PreTrainedConfig, ds_meta=None,
env_cfg=None, rename_map=None) -> PreTrainedPolicy`. Shape inference
needs either dataset metadata or env config, neither of which we
have at eval time, so we go through `from_pretrained` directly.

What we actually do (see `src/lerobot_bench/eval._load_pretrained_policy`):

```python
import lerobot.policies.factory as _lerobot_factory  # registers all PreTrainedConfig choices
from lerobot.configs.policies import PreTrainedConfig

cfg = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
policy_cls = _lerobot_factory.get_policy_class(cfg.type)
preprocessor, postprocessor = _lerobot_factory.make_pre_post_processors(
    cfg, pretrained_path=repo_id, revision=revision
)  # FileNotFoundError -> recover from legacy safetensors stats
model = policy_cls.from_pretrained(repo_id, revision=revision, config=cfg)
return _LerobotPolicyAdapter(model.to(device).eval(), preprocessor=..., postprocessor=...)
```

Two non-obvious gotchas surfaced during implementation:

1. **The factory module import is load-bearing.** `import
   lerobot.policies.factory` is a side-effect import that registers
   every `PreTrainedConfig` subclass (act, diffusion, pi0, ...) in
   draccus's choice registry. Without it, `PreTrainedConfig.from_pretrained`
   on an `act` checkpoint raises `DecodingError: Couldn't find a
   choice class for 'act'`.

2. **Legacy checkpoints lack `policy_preprocessor.json`.**
   `lerobot/diffusion_pusht` (locked at SHA `84a7c23...`) pre-dates
   the pre/post-processor pipeline split. Its normalization stats
   live inside `model.safetensors` as `normalize_inputs.buffer_*`
   and `normalize_targets.buffer_*` — lerobot 0.5.1 silently drops
   them on load (only a `WARNING` fires) and the policy then outputs
   actions in normalized space, useless on the env.
   `_recover_dataset_stats_from_safetensors` reads the safetensors
   directly, reshapes the buffer keys back to feature-key form, and
   feeds them as `dataset_stats=` to `make_pre_post_processors`.

The adapter (`_LerobotPolicyAdapter`) translates the gym obs dict
`{pixels, agent_pos}` (or Aloha-style `{pixels.<view>, agent_pos}`)
→ the lerobot batch dict `{observation.image, observation.state}`
(or `{observation.images.<view>, observation.state}`), wraps the
inference call in `torch.no_grad()`, and casts the post-processed
torch action back to `numpy.float32` of `env.action_space.shape`.

The env spec also gained a `gym_kwargs` field (forwarded verbatim to
`gym.make`); both shipped envs now set `obs_type: pixels_agent_pos`
so pretrained policies receive the image+state obs they were trained
on. The baseline policies (`random` / `no_op`) are obs-shape-agnostic
and continue to work against the same env spec.

## Step 4 — one real cell, one episode

The actual smoke. One pretrained policy, one env, one seed, one
episode. If this returns a row, the integration is sound.

```bash
python scripts/run_one.py \
  --policy diffusion_policy \
  --env pusht \
  --seed 0 \
  --n-episodes 1 \
  --results-path results/smoke/integration.parquet
```

**Expected:** exit 0. Inspect the row:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('results/smoke/integration.parquet')
print(df.to_string())
print('n_steps:', df.n_steps.iloc[0])
print('wallclock_s:', df.wallclock_s.iloc[0])
"
```

Sanity bounds for PushT + diffusion_policy:
- `n_steps`: between 1 and ~300 (PushT's `max_steps`)
- `wallclock_s`: 5-60s on the dev box GPU
- `success`: bool (likely True at least sometimes for diffusion_pusht)
- `return_`: float
- `video_sha256`: nullable string (None unless render is also exercised)
- `code_sha`: 40-char hex matching `git rev-parse HEAD`
- `lerobot_version`: `0.5.1`
- `timestamp_utc`: ISO 8601 string

**If this fails with a shape mismatch:** the `_LerobotPolicyAdapter`
needs work. Read the traceback, fix the obs/action transform, retry.

**If this fails with `RenderSizeError`:** premortem mitigation #5 just
fired — the render cap is too tight for real episode lengths. File a
follow-up to `render-pipeline-engineer` and rerun with `--no-render`
(if such a flag exists; if not, that's the fix to add first).

**If this fails with `CUDA out of memory`:** calibration thresholds
are about to need updating. Note the policy + env + observed VRAM and
hand off to the next sweep planning step.

## Step 5 — render one episode end-to-end

If step 4 succeeded without rendering a video, exercise the render
path on one episode now. Look at the rendered MP4 in a video player
(VS Code has a preview, or `ffplay results/smoke/<the_mp4>.mp4`).

**Watch for:**
- File ≤ 2 MiB (size cap).
- 256x256, ~10 fps (visible motion is roughly real-time slowed).
- Visually plausible — the agent is *attempting* the task, not flailing.
- No "stuck on frame 0" rendering bug from observation-vs-frame
  confusion.

## Step 6 — record outcomes, commit only what's real

If everything passed: commit the `revision_sha` + `license` edit to
`configs/policies.yaml` AND any adapter code AND the import-path patch
in a single PR titled `feat(eval): wire pretrained policy loading via
lerobot factory + lock diffusion_policy SHA`. Update CHANGELOG and
`docs/MODEL_CARDS.md` with the SHA.

If steps 4 or 5 failed: do NOT commit `configs/policies.yaml`. Open
an issue or hand off to `bench-eval-engineer` with the traceback and
the exact command that failed. The synthetic test suite will keep
green; we now know what reality looks like.

## What this catches

Failure modes this smoke is specifically designed to surface, in
order of how much pain they cause if discovered later:

1. `lerobot.common.policies.factory.make_policy` signature drift
   between the install and our planned call.
2. Obs dict shape (`{image, state}` vs nested vs flat) not matching
   what the adapter assumes.
3. Action dtype (`float32` vs `float64`) or shape mismatch with
   `env.action_space`.
4. `torch.no_grad()` forgotten — discovered as VRAM growth across
   episodes within the cell.
5. Render cap blowing up on episodes longer than the synthetic
   30-frame test inputs.
6. `revision_sha` garbage-collected from the Hub between when we
   pinned it and when the sweep runs (rare but real).
7. `SUCCESS_REWARD` semantics differ from the YAML threshold
   (relevant if/when we switch envs.py to read from
   `lerobot.envs.<env>.config.SUCCESS_REWARD`).

## What this does NOT catch

- Per-cell VRAM peak (calibration's job).
- Cross-policy fairness — a finding-quality comparison needs the
  full sweep with multiple seeds, not one episode.
- Hub upload throttling (PR #12's job).
- Space cold-start latency (PR #13's job).

This is a probe, not a benchmark. One episode, one cell, one finding:
"the wires touch."
