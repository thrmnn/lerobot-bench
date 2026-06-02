# World-Model / JEPA Research Track (PLANNED / PROPOSED)

**Status: PROPOSED — not shipped.** No code in this wave. This document is
the design contract for an exploratory track that evaluates *world-model
(WM) planners* — V-JEPA-class latent forward models driving a planner —
as ordinary bench policies. Nothing here lands on the v1 leaderboard.

## 0. The keystone

A world-model planner is not a new kind of bench primitive. It is a
`PolicyCallable`. The bench's eval contract is

```
(policy, env, seed, n_eps) -> CellResult   # success rate + Wilson/bootstrap CIs
```

and a policy is *anything* that satisfies the `PolicyCallable` protocol in
`src/lerobot_bench/eval.py:82`:

```python
def __call__(self, obs: dict[str, Any]) -> NDArray[np.floating[Any]]: ...
def reset(self) -> None: ...
```

`__call__` is `act(obs) -> action`. `reset()` is called once per episode
(after `env.reset`, before the first `policy(obs)`). A WM planner hides an
entire latent-MPC loop *inside* `__call__`: it encodes the observation into
the WM's latent space, rolls candidate action sequences through the learned
forward model, scores them against a goal/cost, and returns the first action
of the best plan. From the rollout loop's perspective it is indistinguishable
from ACT or diffusion — same dict in, same `float32` action ndarray out,
same per-episode `reset()`.

**Consequence:** a WM planner flows through the existing
`(policy, env, seed, n_eps) -> CellResult` pipeline *unchanged*. No new
rollout code, no new statistics code, no new CellResult shape. The only
bench-side code gap is a single dispatch branch in `load_policy` (below).
Everything else — sweep, Wilson/bootstrap CIs, paired bootstrap, failure
taxonomy, parquet schema — is reused verbatim.

## 1. The WM-as-policy adapter contract

### 1.1 The one bench-code gap: a `wm` dispatch kind in `load_policy`

`load_policy` (`src/lerobot_bench/eval.py:626`) currently branches twice:

- `spec.is_baseline` → `_NoOpPolicy` / `_RandomPolicy` (needs `action_shape`).
- otherwise → `_load_pretrained_policy(...)`, which lazy-imports lerobot and
  wraps a `PreTrainedPolicy` in `_LerobotPolicyAdapter`.

A WM planner is neither a baseline nor a lerobot `PreTrainedPolicy`. It needs
a **third branch** — a future `wm` *kind* — that constructs a
`_WorldModelPolicyAdapter` (name TBD). The proposed dispatch key is an
explicit field on the spec (e.g. `kind: wm`) rather than inferring from
`repo_id`, so the branch is unambiguous and the existing two branches stay
untouched. **This document describes that interface; it does not implement
it.** Sketch of the intended branch (for reference only — not to be written
in this wave):

```python
# load_policy, after the is_baseline branch, before the pretrained branch:
if spec.kind == "wm":
    return _load_wm_policy(
        repo_id=spec.repo_id,
        revision=spec.revision_sha,
        planner_cfg=spec.wm_planner,   # MPC horizon, n_samples, cost spec
        action_shape=action_shape,
        device=device,
    )
```

The adapter `__call__` keeps the same belt-and-braces discipline as
`_LerobotPolicyAdapter`: run under `torch.no_grad`, return CPU
`numpy.float32` reshaped to `action_shape`. `reset()` clears any internal
planner state (warm-start buffers, receding-horizon plan cache) at episode
boundaries — the WM analogue of clearing ACT/diffusion's action queue.

### 1.2 Adapter inputs / outputs

| | contract |
|---|---|
| input to `__call__` | the gym observation dict the env already produces (images + state), identical to what `_LerobotPolicyAdapter` receives |
| internal | encode obs → WM latent; sample/optimize action sequences; roll through learned dynamics; score vs goal latent / reward; pick first action |
| output of `__call__` | `np.float32` action of `action_shape` (env-supplied), exactly like every other policy |
| `reset()` | clear receding-horizon plan cache / warm-start state |
| state ownership | the WM model, planner config, and goal spec are owned by the adapter; the bench never sees them |

The WM checkpoint, the planner (MPC/CEM hyperparameters), and the goal/cost
specification are **internal** to the adapter. The bench contract surface is
exactly `__call__` + `reset()`. No new parquet columns are required; a WM
cell is just another `(policy, env, seed, n_eps)` row.

### 1.3 Where it registers

Same machinery as every other policy, no new registry code:

- **`configs/policies.yaml`** — a new entry with `is_baseline: false`, an
  `env_compat` list (initially `[pusht]`), `repo_id` + `revision_sha`
  pinning the public WM checkpoint, and the proposed `kind: wm` discriminator
  plus a `wm_planner` block (MPC horizon, sample count, cost). Adding the
  `kind` / `wm_planner` fields to the `PolicySpec` schema in
  `src/lerobot_bench/policies.py` is part of the future implementation wave,
  not this one.
- **`src/lerobot_bench/policies.py`** — `PolicySpec` already carries
  `repo_id` / `revision_sha` / `env_compat` / runnable-gating; the WM entry
  reuses all of it. Only the `kind` / `wm_planner` optional fields are new.

### 1.4 Gated OFF the v1 board, exactly like xvla

WM policies are **exploratory** and must not bias the headline v1 numbers.
They are gated off the public board the same way `xvla_libero` is:

- They are **NOT** added to `V1_POLICIES` in
  `src/lerobot_bench/leaderboard_filter.py`.
- `filter_to_v1_policies` drops every non-v1 policy right after parquet load,
  so the Space, the dashboard, and every downstream aggregate (Wilson CIs,
  paired bootstrap, MDE) never see WM rows.
- The published parquet *may* still carry WM rows for reproducibility (the
  xvla precedent), but the public surfaces stay v1-only.

This gives the track a real number computed by the real pipeline while
keeping it off the leaderboard until it has earned a place.

## 2. Prior-art landscape

| approach | what it is | bench fit | checkpoint availability |
|---|---|---|---|
| **DINO-WM** (arXiv:2411.04983) | DINOv2-feature latent world model + visual MPC planner; plans in a frozen patch-feature latent toward a goal image | **best first fit** — public PushT checkpoint + MPC planner, zero training | public, includes PushT |
| **V-JEPA 2 / V-JEPA-2-AC** | large self-supervised video JEPA; the **-AC** (action-conditioned) head enables planning/control | strong long-term target | **no** PushT/LIBERO checkpoint — slow lane |
| latent MPC (generic) | the planning pattern both of the above instantiate: sample/optimize action seqs, roll through learned latent dynamics, score vs goal | the adapter's internal loop | n/a |

### 2.1 Recommendation: DINO-WM PushT as the cheapest first target

**Recommend DINO-WM's public PushT checkpoint + its MPC planner as the
zero-training first target.** Rationale:

- It ships a public PushT checkpoint, so milestone 1 needs **no training** —
  load weights, wrap the MPC loop in the adapter, run one cell.
- PushT is already a first-class bench env (`gym-pusht`, `diffusion_pusht`
  lives on it), so the env, success metric, and seeding contract are reused
  unchanged — a WM PushT number is directly comparable to the existing
  `diffusion_policy` PushT cell.
- DINO-WM's frozen-DINOv2-feature latent + visual MPC is exactly the
  `encode → roll → score → first-action` pattern the adapter wraps.

**V-JEPA-2-AC is the slow lane.** It has no PushT or LIBERO checkpoint, so
it implies either training/finetuning a WM or building an env it already has
a checkpoint for — out of scope for a first number. Track it as the
ambitious follow-on once the adapter contract is proven on DINO-WM.

## 3. Recommendation: a separate `lerobot-wm-research` repo

**Recommendation (pending user confirmation — not a settled decision):**
host the WM track in a **separate repo, `lerobot-wm-research`**, mirroring
the SO-100 split, rather than vendoring WM code into this benchmark.

Why a separate repo:

- **Toolchain freeze.** This bench is pinned to **lerobot 0.5.1** for
  reproducibility. DINO-WM / V-JEPA pull a different, faster-moving stack
  (DINOv2, their own planner deps, possibly a newer torch). A separate repo
  keeps the WM toolchain *moving* without ever threatening the bench's
  reproducibility freeze.
- **Clean dependency boundary.** The WM repo depends on the bench's *eval
  contract* (the `PolicyCallable` protocol + the `(policy, env, seed,
  n_eps) -> CellResult` shape), not on the bench's internals. It can expose a
  checkpoint + adapter the bench loads via `repo_id`/`revision_sha`, exactly
  like any Hub policy.
- **Exploratory blast radius.** Failed WM experiments, training runs, and
  heavy deps stay out of the bench's CI and git history.

This mirrors the existing SO-100 split precedent. **It is a recommendation,
not a decision** — confirm with the user before creating the repo. If the
user prefers a single repo, the alternative is a `wm/` subpackage with an
isolated optional-dependency group, accepting the toolchain-coupling risk.

## 4. Smallest first milestone

**One WM checkpoint, one env, no training, one number on the (filtered)
board.**

1. DINO-WM public **PushT** checkpoint — no training.
2. Wrap its MPC planner in the WM adapter so `act(obs) -> action` satisfies
   `PolicyCallable`.
3. Add the `wm` dispatch branch to `load_policy` (the one bench-code gap).
4. Register one `policies.yaml` entry (`env_compat: [pusht]`, `kind: wm`,
   pinned SHA), kept OFF `V1_POLICIES`.
5. Run **one cell**: `(dino_wm, pusht, seeds, n_eps)` through the unchanged
   sweep → one `CellResult` with Wilson/bootstrap CIs, sitting next to
   `diffusion_policy` on PushT but gated off the public board.

Success criterion for milestone 1 is **plumbing, not SOTA**: a defensible
PushT success number from a WM planner, produced by the real pipeline. The
comparison story ("can a zero-training latent-MPC planner approach a trained
diffusion policy on PushT?") is the first interesting question the track can
answer — but only after the number exists.

## 5. Fast-lane / slow-lane boundary

- **Fast lane (do first):** DINO-WM + PushT, zero training, one cell. Reuses
  an existing bench env, an existing success metric, and a public checkpoint.
  The only new code is the `wm` dispatch branch + one config entry. This is
  the milestone in §4.
- **Slow lane (later):** V-JEPA-2-AC; WM training/finetuning; LIBERO
  (no public WM checkpoint, so a training run or an env-specific WM is
  required); CEM/MPC planner tuning; multi-env WM comparisons. Everything
  that needs training, a missing checkpoint, or new env wiring is slow lane.

The boundary is **"is there a public checkpoint for an env the bench already
runs?"** Yes → fast lane (DINO-WM/PushT). No → slow lane.

---

### Scope guardrails (this wave)

- **Design only.** No code. The `wm` dispatch branch in `load_policy`
  (`src/lerobot_bench/eval.py:~662`), the `kind`/`wm_planner` schema fields,
  the adapter, the config entry, and the separate repo are all **future
  work** described here, not implemented.
- WM policies stay **off `V1_POLICIES`** and are dropped by
  `leaderboard_filter.py` — the xvla precedent.
- The whole track is **PLANNED / PROPOSED** until the user confirms the
  separate-repo recommendation and greenlights the milestone-1 build.
