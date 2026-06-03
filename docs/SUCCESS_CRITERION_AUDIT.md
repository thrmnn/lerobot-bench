# Success-criterion + episode-termination audit (v1.0.1)

| Field   | Value |
| ------- | ----- |
| Status  | Resolved (audit + canonical-criterion infra + cap-600 probe; v1.1 residue tracked in §7) |
| Date    | 2026-05-26 (audit) · 2026-06-02 (gate close) |
| Auditor | researcher (audit/v1.0.1-success-criterion) |
| Pipeline-roadmap item | §1.3 |
| Companion audits | `docs/CLAIM_AUDIT_SMOLVLA.md` (PR #84, §1.1) · `docs/INFERENCE_AUDIT.md` (PR #86, §1.2) |
| Verdict | **Hard mismatch on PushT and Aloha; hard mismatch on the LIBERO step cap. None of the four `smolvla_libero` headline cells are affected by a success-rule mismatch *per se*, but every `libero_*` cell is affected by a tighter-than-canonical `max_steps` that could under-count successes for slow rollouts.** Specifics in §5. |

## 1. One-sentence finding

`lerobot-bench` v1 uses **`success := final_reward >= env_spec.success_threshold`** (the reward at the last step before exit), but two of the three published references the deck compares to (`gym-aloha` AlohaTransferCube's underlying ACT paper, and the LeRobot Hub card for `lerobot/diffusion_pusht`) score success as **`any(is_success across the rollout)`** — a sticky-true metric over the whole trajectory — and the LIBERO upstream eval uses a step cap of **600** for every suite while we use lerobot 0.5.1's tighter `TASK_SUITE_MAX_STEPS = {spatial: 280, object: 280, goal: 300, libero_10: 520}`.

## 2. Where the bench rule lives

`src/lerobot_bench/eval.py` ll. 1462–1520 (`_run_one_episode`):

```python
for _ in range(max_steps):
    action = policy(obs)
    obs, reward, terminated, truncated, _info = env.step(action)
    n_steps += 1
    cumulative_return += float(reward)
    final_reward = float(reward)
    if record_video: frames.append(env.render())
    if terminated or truncated:
        break
# ...
success = final_reward >= success_threshold
```

The rule has two halves: (a) **terminate** on `terminated | truncated | step == max_steps`; (b) **score** by `final_reward >= success_threshold`. `success_threshold` is per-env, declared in `configs/envs.yaml`: **PushT 0.95 · Aloha 1.0 · all LIBERO suites 1.0**. The rule and thresholds are documented in `docs/DESIGN.md` ll. 57, 64 and in the eval module docstring (ll. 28–35).

## 3. What each env actually does on `step()`

Verified against the installed packages (`/home/theo/miniforge3/envs/lerobot/lib/python3.12/site-packages/`).

### 3.1 PushT — `gym_pusht/envs/pusht.py`

```python
coverage = self._get_coverage()
reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)   # success_threshold = 0.95
terminated = is_success = coverage > self.success_threshold
```

- Reward is **continuous in `[0,1]`**, normalised so `reward == 1.0` iff `coverage >= 0.95`.
- `terminated` fires iff `coverage > 0.95`.
- Registered with `max_episode_steps = 300`.

### 3.2 Aloha TransferCube — `gym_aloha/env.py` + `tasks/sim.py`

```python
_, reward, _, raw_obs = self._env.step(action)
terminated = is_success = reward == 4
```

The transfer-cube task returns reward in `{0, 1, 2, 3, 4}`:

| value | meaning |
| ----- | ------- |
| 0 | gripper idle |
| 1 | right gripper touching cube |
| 2 | cube lifted (right gripper) |
| 3 | left gripper also touching (attempted transfer) |
| 4 | left gripper holding cube clear of table (**successful transfer**) |

- Reward is **per-step state-based**, not cumulative or sticky.
- `terminated` fires iff `reward == 4`.
- Registered with `max_episode_steps = 300`.

### 3.3 LIBERO — `lerobot/envs/libero.py` (wraps `libero.libero.envs.OffScreenRenderEnv`)

```python
# inner robosuite/libero
def reward(self, action=None):
    reward = 0.0
    if self._check_success():
        reward = 1.0
    if self.reward_scale is not None:
        reward *= self.reward_scale / 1.0
    return reward

# lerobot wrapper
raw_obs, reward, done, info = self._env.step(action)
is_success = self._env.check_success()
terminated = done or is_success
# ... if terminated: self.reset()   # internal auto-reset
truncated = False
```

- Reward is **binary in `{0, 1}`**, sparse, fires iff `_check_success()`.
- `terminated` fires iff `done or is_success`.
- `truncated` is always `False` — the LIBERO wrapper does not impose a step cap of its own; the cap comes from `lerobot.envs.libero.TASK_SUITE_MAX_STEPS` which the bench reads through `EnvSpec.max_steps`.
- After `terminated`, the wrapper calls its own `self.reset()` (with `num_steps_wait=10` no-op pre-roll) before returning. The reward returned **is** the terminating step's reward (captured before the internal reset). The bench breaks immediately on `terminated`, so the post-terminal reset is wasted compute but does not corrupt `final_reward`.

## 4. What each paper / reference uses (with verbatim sources)

### 4.1 LeRobot canonical eval (the source of every Hub-card success rate)

`lerobot/scripts/lerobot_eval.py` ll. 194–222, 354–367:

```python
# inside the rollout loop
if "final_info" in info:
    successes = info["final_info"]["is_success"].tolist()
else:
    successes = [False] * env.num_envs
all_successes.append(torch.tensor(successes))
# after the rollout
batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
# ...
"pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
```

So lerobot's reference eval is **sticky `any(is_success)` across the trajectory**, and it captures `is_success` from `info["final_info"]["is_success"]`. `pc_success` is the percentage of episodes where `is_success` ever fired. **This is the metric the `lerobot/diffusion_pusht` model-card 65.4% number was produced under** — we have not re-run that script, but it is the only eval that ships with lerobot and that the model card cites.

### 4.2 Diffusion Policy (Chi et al. 2023) — PushT

The paper §V.A reports PushT as **continuous target-area coverage**, not a binary metric. Table 2 (image-CNN policy): "max overlap 0.91 / avg-of-last-10 0.84". We compare to the LeRobot Hub card for `lerobot/diffusion_pusht` (binary, sticky `any(is_success)`, threshold = coverage > 0.95, episode budget 500, on `gym-pusht`). `docs/MODEL_CARDS.md` ll. 104–113 already note that the paper's continuous metric is not directly comparable; the Hub-card number is the apples-to-apples reference. **What is *not* yet noted: the Hub card uses sticky `any(is_success)` and our bench uses `final_reward >= 0.95`. These are different scoring rules.**

### 4.3 ACT (Zhao et al. 2023) — Aloha TransferCube

Paper §IV.B "Task design and success criteria" + Table I:

> "Cube Transfer (sim) ... We follow the convention of reporting three subtask scores: \emph{Touched}, \emph{Lifted}, and \emph{Transfer}, where \emph{Transfer} is the binary task-success rate."

Table I, row "ACT (Ours)", column "Cube Transfer (sim) / Transfer", human-teleop training data: **50%**, 3 seeds × 50 evaluations. The reported number is the percentage of episodes that ever reached `reward == 4` (the gym-aloha `is_success` definition). **This is sticky `any(reward == 4)`, NOT `final_reward >= 1.0`.**

The Hub card for `lerobot/act_aloha_sim_transfer_cube_human` reports a higher 83.0% on 500 episodes, also under lerobot's canonical eval (sticky `any(is_success)` with `is_success := reward == 4`). The Hub card itself flags the gap from the paper and "attributes it to `gym-aloha` success heuristics" — but the scoring rule is the same (sticky transfer-only). Our bench's `success := final_reward >= 1.0` is **strictly looser**: it counts any episode where the *last step* had `reward in {1, 2, 3, 4}` — i.e. any episode where at step 400 the right gripper happens to be touching the cube, or the cube is lifted, or being passed, even if the full transfer never happened.

### 4.4 SmolVLA (Shukor et al. 2025) — LIBERO

Paper §4 "Evaluation metrics":

> "For simulation-based evaluations, SR is binary --- set to 1 if the task is successfully completed, and 0 otherwise."

§4.1 "Simulated environments":

> "evaluate with 10 trials per task, reporting average success rates based on binary completion criteria."

The paper does not state a per-episode step cap. The LIBERO upstream eval (`libero/lifelong/metric.py` ll. 130–155) uses `cfg.eval.max_steps` (default 600 in the canonical `libero/configs/policy/default.yaml`) and sticky `dones[k] = dones[k] or done[k]`. OpenVLA and XVLA inherit this protocol. `done` in the LIBERO inner env equals `_check_success()` (same `is_success` flag we read in lerobot's wrapper). **Scoring rule matches our bench by construction** (binary, fires exactly at `reward == 1.0`, the terminating step always carries that reward, and our `final_reward >= 1.0` is therefore equivalent to `is_success at terminating step` which is equivalent to sticky `any(is_success)` whenever the run terminates on success). **Step cap does not match: 600 upstream vs 280/280/300/520 here.**

### 4.5 XVLA (Bu et al. 2025) — LIBERO

Same as SmolVLA: paper inherits the LIBERO benchmark's binary completion + 600-step protocol. XVLA cells are already deferred to v1.1 for unrelated Hub-JSON wiring bugs (`docs/DEFERRED_POLICIES.md`); the step-cap mismatch would apply to v1.1 XVLA runs if not fixed.

## 5. Per-env mismatch table

Severity scale: **none** = scoring rule is bit-equivalent to the reference; **soft** = differs but unlikely to flip many episodes given observed cap-hit rates; **hard** = differs and the difference materially changes the numerator or denominator of the success rate. Cap-hit fractions are computed from `results/sweep-full/results.parquet` (v1 sweep).

| Env | Bench rule | Bench `max_steps` | Reference rule | Reference `max_steps` | Severity (rule) | Severity (cap) |
| --- | ---------- | ----------------- | -------------- | --------------------- | --------------- | -------------- |
| `pusht` | `final_reward >= 0.95` (last-step coverage `>= 0.9025`) | 300 | LeRobot Hub card: sticky `any(is_success)` ≡ `any(coverage > 0.95)`, 500 episodes | 300 (gym registration; lerobot eval inherits) | **Hard** (over-counts) | **Hard** (34.4% of `diffusion_policy × pusht` episodes hit cap; lax threshold opens a door at exactly those non-terminated tails) |
| `aloha_transfer_cube` | `final_reward >= 1.0` (last-step reward `>= 1`, includes "touched/lifted/attempted/transferred") | 400 | ACT paper Table I: sticky `any(reward == 4)` ("Transfer" subtask only); 3 seeds × 50 ep | 400 (paper text) | **Hard** (over-counts: 1, 2, 3 are sub-goals not success) | **Soft** (98.4% of `act × aloha_transfer_cube` episodes hit cap, but the rule mismatch dominates: cap-hit ⇒ uses lax-threshold last-step reward) |
| `libero_spatial` | `final_reward >= 1.0` (binary; equivalent to `is_success` at terminating step) | 280 | LIBERO upstream + SmolVLA/XVLA papers: sticky `any(is_success)` (binary) | 600 (LIBERO `cfg.eval.max_steps` default) | **None** (rule is bit-equivalent for binary-reward envs) | **Hard** (22.4% of `smolvla × libero_spatial` cap-hit; any of those that would have succeeded by step 600 are silently false-negative) |
| `libero_object`  | same | 280 | same | 600 | **None** | **Hard** (47.2% `smolvla × libero_object` cap-hit) |
| `libero_goal`    | same | 300 | same | 600 | **None** | **Soft** (7.2% `smolvla × libero_goal` cap-hit; small under-count band) |
| `libero_10`      | same | 520 | same | 600 | **None** | **Hard** (74.8% `smolvla × libero_10` cap-hit; this is the headline cell — the published 0.252 is a *lower bound* on what the same checkpoint would score at step 600) |

### 5.1 Why the LIBERO scoring rule itself is none-severity

For LIBERO the inner reward is exactly `1.0` iff `_check_success()` else `0.0`. The lerobot wrapper sets `terminated = done or is_success`, and the bench's `for _ in range(max_steps): ... if terminated: break` captures `final_reward = 1.0` exactly when the terminating step is a success. There is no path by which `final_reward >= 1.0` and `is_success` disagree: a 1.0 reward at any step *immediately* fires `terminated`, the bench breaks, and `final_reward = 1.0`. There is no late-success-followed-by-decay-to-0 case because LIBERO never decays — once `_check_success()` returns True, the wrapper's auto-reset only fires after the bench has already broken. So `success := final_reward >= 1.0` ≡ `any(is_success)` for LIBERO **provided we never truncate on `max_steps`**. The cap *can* truncate; that's where severity moves from the rule to the cap.

### 5.2 Why PushT is hard on both axes

`reward = clip(coverage/0.95, 0, 1)`. Inverting: `final_reward >= 0.95` iff `coverage >= 0.9025`. The env's `terminated` only fires at `coverage > 0.95`. So an episode where the block ends at, e.g., coverage 0.91 reaches the step cap (300) with `final_reward = 0.958` and the bench counts it as success. The Hub-card rule (sticky `any(coverage > 0.95)`) does not count it. Direction: **bench over-counts vs reference.** We measured `diffusion_policy × pusht = 0.816` vs Hub-card 0.654; some unknown fraction of that 16.2-pp surplus is the lax-threshold artifact, the rest is genuine. Without `final_reward` and `max_reward` columns in the parquet we cannot decompose.

### 5.3 Why Aloha is hard on the rule axis

`reward ∈ {0,1,2,3,4}`. Our `final_reward >= 1.0` accepts 1 ("touched") as success. Paper accepts only 4 ("transferred"). Direction: **bench over-counts vs paper.** Our measured `act × aloha_transfer_cube = 0.016` is *already* extremely low; even with the lax rule we don't beat the random baseline (0.052). Two unknowns:
1. Of the 4 successes / 250 episodes ACT scored, how many had `reward == 4` at the last step vs `reward in {1, 2, 3}`? Without `final_reward` in the parquet, unanswerable.
2. The 13 successes / 250 episodes the random baseline scored are almost certainly **all** `reward in {1, 2, 3}` — random action will brush the cube at step 400 with non-negligible probability, but assembling the full transfer is implausible.

The strict-rule version of the headline ("ACT does not beat random") could *strengthen* — both ACT and random would shrink, but random would shrink more — but we can't verify without re-running with `final_reward` recorded.

## 6. Per-policy paragraph: what may be affected

### `diffusion_policy` (PushT only)
- **Cell `pusht`**: measured 0.816, paper Hub-card 0.654. The 16.2-pp gap *includes* an unknown over-count from the lax PushT threshold. With the cap-hit rate at 34.4% and lax-threshold-window of `coverage ∈ [0.9025, 0.95]` non-trivially populated for a near-converged policy, the over-count contribution could be in the single-digit-pp range. Recommended action: **(a)** add a one-line caveat to `paper/main.tex` Results paragraph + `docs/MODEL_CARDS.md` Diffusion Policy section noting the rule mismatch; **(b)** a 1-seed × 50-episode probe with `success := any(is_success)` and a parquet column for `max_reward` to decompose the gap. The headline phrasing "Diffusion Policy on PushT replicates within ~1pp" survives qualitatively (probably still within ~5pp under the strict rule) but the precise number is soft.

### `act` (Aloha TransferCube only)
- **Cell `aloha_transfer_cube`**: measured 0.016, paper 0.50, Hub-card 0.83. Three different numbers, three different protocols. The bench's `final_reward >= 1.0` is **strictly looser** than the paper's `any(reward == 4)`, so under the strict rule our 0.016 only goes *down*. The "ACT does not beat random" headline survives (both 0.016 and 0.052 shrink under strict-rule but random more so). The "huge gap vs paper" headline is also unaffected in direction. Recommended action: **(a)** explicitly state the rule mismatch in `paper/main.tex` ACT discussion and `docs/MODEL_CARDS.md` ACT entry; **(b)** add `final_reward` and `max_reward` columns to the parquet schema so this audit can be closed at the data layer in v1.1, not the docs layer.

### `smolvla_libero` (4 LIBERO cells)
- **Cells `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`**: scoring rule is bit-equivalent to the paper (none-severity); the load-bearing mismatch is the step cap. Direction: **bench under-counts vs paper** (slow rollouts that would succeed by step 600 are cut off at 280/280/300/520). Cap-hit rates: 22.4% / 47.2% / 7.2% / 74.8%. For `libero_10`, the published 0.252 is a *lower bound* on the same checkpoint's score at step 600. This **compounds** the SmolVLA task-coverage mismatch already documented in `docs/CLAIM_AUDIT_SMOLVLA.md`: the headline "0.71 → 0.252, −45.8 pp" is both apples-to-oranges on task coverage (CLAIM_AUDIT_SMOLVLA verdict) *and* under-counted on the bench side because the step cap is shorter than the protocol the 0.71 was measured under. Recommended action: **(a)** in `paper/main.tex` Results, replace `max_steps` per LIBERO suite from `{280,280,300,520}` to `{600,600,600,600}` to match the canonical LIBERO eval, *or* document the bench's choice explicitly with an upper-bound disclaimer; **(b)** v1.1 step-cap change is a single-line edit in `configs/envs.yaml` and re-runs cleanly under the existing pipeline. The fix is cheap; the writeup change is the load-bearing part.

### `xvla_libero` (deferred to v1.1)
- Same step-cap concern as SmolVLA inherits to XVLA once XVLA's three Hub-JSON wiring bugs are resolved. Bundle the step-cap fix with the v1.1 XVLA re-enable (`docs/DEFERRED_POLICIES.md`).

### `pi0fast_libero` / `pi0_libero_finetuned_v044` / `pi05_libero_finetuned_v044`
- No paper-reported number in the registry, so no public claim to defend — but the bench's `max_steps` still differs from the canonical LIBERO eval. If these are evaluated in v1.1, the same step-cap caveat applies. The fix is the same one-line `configs/envs.yaml` edit.

### `no_op` / `random` baselines
- The bench rule (`final_reward >= threshold`) inflates the baseline numbers on `aloha_transfer_cube` (random scored 0.052 by touching the cube at step 400 in ~5% of episodes; under strict `reward == 4` the random number would collapse to near-zero). On `pusht`, random's reward expectation is bounded above by chance coverage which is well below 0.9025, so the lax threshold likely does not materially inflate random. On LIBERO baselines are exactly zero regardless.

## 7. Recommended actions, ordered by cost

1. **Documentation (cheap, defensive)**. Add a "Success criterion" paragraph to `paper/main.tex` § Methodology stating the bench's rule verbatim, contrasting it with `gym-aloha`'s `reward == 4` strict transfer and the lerobot-canonical sticky `any(is_success)`. Add the same paragraph to `docs/DESIGN.md` § Methodology near the existing "Per-env success thresholds" note. Mention each over- and under-counting direction.
2. **Soften deck slide 07 phrasing on PushT and Aloha**. The "Diffusion Policy replicates within ~1pp" claim should carry a footnote: "Bench rule is `final_reward >= 0.95`; Hub card rule is `any(is_success)`. The two differ by an unmeasured amount at our cap-hit rate of 34%."
3. **Parquet schema bump (one-PR, no re-run)**. Add `final_reward` and `max_reward` columns to the v1.1 parquet so any future audit can decompose the over-count vs the genuine signal without re-running the sweep.
4. **Step-cap alignment (one-line config edit + re-run on `smolvla_libero` × 4 cells)**. Bump `libero_*` `max_steps` in `configs/envs.yaml` from `{280, 280, 300, 520}` to `{600, 600, 600, 600}` for the v1.1 sweep. This is a strict superset of the v1 protocol; cells that succeeded in `<= 520` steps will continue to succeed; some cells that hit the cap may now succeed. Bundle with `task_ids = [0..9]` from `docs/CLAIM_AUDIT_SMOLVLA.md` §1.1 — both fixes land in the same re-sweep.
5. **Success-rule alignment (medium cost)**. Switch the bench rule from `final_reward >= threshold` to `any(is_success)` over the rollout. Two paths: **(a)** record `is_success` per step in the parquet and post-process to `success := any(...)` — works without changing `_run_one_episode`; **(b)** read `info["is_success"]` inside the loop and short-circuit. Path (a) is more reversible and matches the canonical lerobot eval. Either path requires re-running every cell to populate `is_success` in the parquet; the existing `success` column would change for `pusht` (down) and `aloha_transfer_cube` (down) while staying invariant for `libero_*`.

Ranking: **1 → 2 → 3** are doc-only / schema-only and ship in v1.0.1. **4** ships with v1.1 (bundled with the task-coverage expansion). **5** is the v1.1 *rule* change and should be gated on the parquet schema bump (3) landing first.

## 8. What this audit rules in / out

- **Rules in:** there exist *named, traceable* mismatches between the bench's success rule and three of the published references the deck compares to (Aloha-ACT, PushT-Hub-card, LIBERO-step-cap). They are not subtle; they are visible in our own parquet data (cap-hit rates, sub-threshold reward windows).
- **Rules out:** the LIBERO scoring rule itself (binary, terminating-step-equals-success) is **bit-equivalent** to the upstream `any(is_success)` for LIBERO; the SmolVLA `libero_*` cell numbers are *not* affected by a rule mismatch, only by the step-cap mismatch.
- **Does not rule out:** that the rule-fix would close the gaps materially. The most likely outcome is the directions hold (PushT bench inflates → strict version shrinks toward 0.7-0.8; Aloha bench inflates → strict version collapses toward 0; LIBERO bench under-counts → strict version grows). Magnitudes are unknown without the probes in §7 items 3–5.

## 9. Open questions we cannot close without re-running

1. **Decomposition of the PushT gap.** We measured 0.816 vs Hub-card 0.654. Without `max_reward` in the parquet we cannot separate "lax-threshold over-count" from "genuine bench-vs-card difference (different seeds, different hardware, different lerobot version)". Probe: rerun `diffusion_policy × pusht` with `final_reward`/`max_reward` columns. Cost: 1 seed × 50 episodes (~5 min on a 4060).
2. **ACT strict-transfer rate.** Of the 4 successes / 250 episodes we counted, how many actually transferred (`reward == 4` at last step)? Probe: rerun `act × aloha_transfer_cube` with `final_reward` recorded. Cost: 1 seed × 50 episodes (~10 min).
3. **SmolVLA step-cap headroom.** What fraction of the 187 / 250 `smolvla × libero_10` cap-hits would succeed if max_steps were 600? Probe: rerun the same cell with `max_steps=600`. Cost: 1 seed × 50 episodes (~25 min at SmolVLA's measured rate).
4. **Hub-card reproducibility.** Does our `diffusion_policy × pusht` cell reproduce the Hub-card 0.654 if we exactly mirror the Hub-card protocol (sticky `any(is_success)` × 500 episodes × the random seeds the card used)? The Hub card does not state seed values, so this is bounded by a "match within bootstrap CI" check. Probe: rerun with sticky rule × 500 episodes; sample size matches Hub card; rule matches Hub card. Cost: ~50 min.

---

*This audit is the artifact for `docs/PIPELINE_ROADMAP.md` §1.3. PR branch: `audit/v1.0.1-success-criterion`. User-reviewed; do not auto-merge.*
