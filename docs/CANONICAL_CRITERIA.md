# Canonical success criteria (v1.1)

Companion to [`SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md)
(audit) and the per-policy notes in [`MODEL_CARDS.md`](MODEL_CARDS.md).
This file is the operator-facing reference for what changes when
`--canonical` is passed to `scripts/run_sweep.py` (or
`scripts/run_one.py`), with the paper / Hub citations behind each row.

| Field | Value |
| ----- | ----- |
| Status | Implemented (v1.1) |
| Default | `v1_legacy` — bit-identical replay of v1.0 |
| Opt-in  | `--canonical` CLI flag on `scripts/run_sweep.py` / `scripts/run_one.py`, or `criterion: canonical` in the sweep YAML |
| Backward compat | Existing `results.parquet` files require no rewrite; `success` column stays valid for the criterion the sweep used |

## What "criterion" controls

Three knobs per env, encoded on `EnvSpec`:

1. **`max_steps`** — per-episode step cap before forced truncation.
2. **`success_metric`** — how a rollout is reduced to a boolean:
   * `final_reward_threshold` — `success := final_reward >= success_threshold`. v1 default.
   * `sticky_is_success` — `success := any(info["is_success"])`. Lerobot canonical reduction.
   * `sticky_reward_eq` — `success := any(reward == strict_reward_value)`. ACT Aloha Transfer rule.
3. **`success_threshold` / `strict_reward_value`** — threshold or target value the metric reads.

The v1 fields on each env spec are the `v1_legacy` values. The optional
`canonical:` overlay declares the deltas. `EnvSpec.with_criterion('canonical')`
applies the overlay; the eval loop sees a single resolved spec and never
needs to branch.

## Per-env table

| Env | v1_legacy `max_steps` | canonical `max_steps` | v1_legacy `success_metric` | canonical `success_metric` | Direction of bias under v1_legacy | Source for canonical |
| --- | ---------------------- | --------------------- | -------------------------- | -------------------------- | ----------------------------------- | -------------------- |
| `pusht`                | 300 | 300 | `final_reward_threshold` (>= 0.95 ≡ coverage >= 0.9025) | `sticky_is_success` (`any(coverage > 0.95)`) | **Over-counts** at the lax-window cap-hit tails | `gym_pusht.envs.pusht.PushTEnv.step` sets `info["is_success"] = coverage > 0.95`; lerobot canonical eval (`lerobot/scripts/lerobot_eval.py` ll. 354–367) reduces `is_success` with `any` |
| `aloha_transfer_cube`  | 400 | 400 | `final_reward_threshold` (>= 1.0; reward ∈ {1, 2, 3, 4} all count) | `sticky_reward_eq` (`any(reward == 4)`) | **Over-counts** (touched / lifted / attempted shouldn't be Transfer) | Zhao et al. 2023, "ACT", Table I "Cube Transfer (sim) / Transfer" subtask — only `reward == 4` is a transfer; reward levels 1-3 are sub-goals |
| `libero_spatial`       | 280 | 600 | `final_reward_threshold` (binary; bit-equivalent to `any(is_success)` for LIBERO) | same | **Under-counts** truncated-but-would-succeed rollouts | `libero/configs/policy/default.yaml` ships `cfg.eval.max_steps = 600`; `libero/lifelong/metric.py` uses `dones[k] = dones[k] or done[k]` to OR-reduce |
| `libero_object`        | 280 | 600 | same as `libero_spatial` | same | same | same |
| `libero_goal`          | 300 | 600 | same | same | same | same |
| `libero_10`            | 520 | 600 | same | same | same | same |

(Source-of-truth verbatim quotes for each row live in `docs/SUCCESS_CRITERION_AUDIT.md` §3-§4.)

## Sticky-vs-final mechanics

Both PushT and Aloha (gym-aloha) **terminate the step immediately when
`is_success` fires**, so under v1_legacy the `final_reward` already reflects
the terminating step's value — but only when the env terminated by success.
The two divergent cases:

* **PushT cap-hit**: episode runs to step 300 without `coverage > 0.95` ever
  firing. v1_legacy still scores a success if the final-step coverage is in
  `[0.9025, 0.95]`; canonical does not. v1 sweep cap-hit rate for
  `diffusion_policy × pusht` is 34.4%.
* **Aloha sub-goal-at-cap**: episode runs to step 400 with the right gripper
  touching the cube but no transfer completed (`reward == 1` at last step).
  v1_legacy scores a success (`>= 1.0`); canonical requires `reward == 4`.
  v1 sweep cap-hit rate for `act × aloha_transfer_cube` is 98.4%.

For LIBERO the scoring rule is bit-equivalent (binary reward, env terminates
on `_check_success()`); only the cap matters. The 4 SmolVLA × LIBERO cells'
cap-hit rates under v1_legacy: `spatial=22.4% · object=47.2% · goal=7.2% ·
libero_10=74.8%`. The libero_10 0.252 headline is therefore a **lower bound**
on what the same checkpoint would score at `max_steps=600`.

## Replayability of v1.0 parquets

v1.0 `results.parquet` files were written under `v1_legacy`. The `success`
column in those files was computed with the v1 rule and is unchanged.
Library code in `lerobot_bench` does not recompute `success` on read, so
mixing v1.0 and v1.1-canonical rows in the same DataFrame requires a
`criterion` column to disambiguate — that column is **not** added to the
parquet schema in v1.1 (the schema bump is gated on the v1.1 re-sweep
ticket, not this implementation PR).

To run a single cell under canonical without disturbing v1.0 results:

```
python scripts/run_one.py --policy smolvla_libero --env libero_10 --seed 0 \
    --canonical \
    --n-episodes 50 \
    --out-parquet results/canonical-probe/results.parquet \
    --videos-dir results/canonical-probe/videos
```

To re-sweep the smolvla × libero_10 cell across all 5 seeds under canonical:

```
python scripts/run_sweep.py --config configs/sweep_full.yaml --canonical \
    --results-path results/sweep-canonical/results.parquet
```

(Both commands keep the original `results/sweep-full/results.parquet` intact
and write the canonical results to a separate path — there is no in-place
mutation of the v1.0 sweep.)

## What this PR explicitly does NOT do

* Does not re-run any cells. The audit's open-questions probes (§9 of
  `SUCCESS_CRITERION_AUDIT.md`) remain open; this is the implementation
  PR, not the data PR.
* Does not bump the parquet schema. Adding a `criterion` column lands
  with the canonical re-sweep ticket so the schema bump and the data
  population land together.
* Does not change the v1 leaderboard. `v1_legacy` remains the default
  and every published v1 number is bit-reproducible by replaying without
  `--canonical`.
