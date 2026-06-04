# Write your own probe

A **probe** is a one-off script that re-runs a specific cell or cells under modified conditions — different inference settings, a different env step cap, a different success criterion — to test a methodology question the main sweep doesn't address.

The v1.0.1 audit produced two reference probes that this guide walks through. Use them as templates; the pattern generalizes.

| Probe | What it changes | Result |
|---|---|---|
| [`scripts/probes/probe_act_temporal_ensemble.py`](../scripts/probes/probe_act_temporal_ensemble.py) | ACT `temporal_ensemble_coeff` Hub default → paper setting | 0.016 → **0.764** on `aloha_transfer_cube` (PR #97; +74.8 pp, Wilson CIs disjoint by an order of magnitude) |
| [`scripts/probes/probe_smolvla_libero_canonical_cap.py`](../scripts/probes/probe_smolvla_libero_canonical_cap.py) | LIBERO env `max_steps` v1 default (520) → canonical (600) | running at time of writing; see [`docs/PROBE_RESULTS_V1.0.1.md`](../docs/PROBE_RESULTS_V1.0.1.md) |

Both probes write standard `RESULT_SCHEMA`-compatible parquet rows to `results/probes/<probe-name>/`, with a `summary.json` for the headline number — so the output slots into the same analysis tooling as the main sweep.

## When to write a probe

A probe is the right tool when:

- A cell measurement looks suspicious and you want to test an alternative hypothesis (the ACT case: "is this an inference-config artifact, not architecture failure?")
- A methodology audit identifies a setting the v1 sweep didn't vary (the LIBERO step-cap case: "the canonical protocol uses 600 steps; how much of our 'lower bound' is cap-truncation?")
- You're evaluating a hypothesis from a paper or upstream issue that doesn't justify a full sweep ($N{=}250{\times}{\sim}{N_\text{cells}}$ episodes of compute) but does justify N=250 episodes on one cell.

A probe is **not** the right tool for:
- Adding a new policy to the leaderboard (use the [policy contribution path](../CONTRIBUTING.md)).
- Running a hyperparameter sweep (use the canonical `scripts/run_sweep.py` with a custom YAML).
- A bug-fix re-run of an existing cell (just re-run `scripts/run_one.py` and merge the rows in).

## The two override patterns

The reference probes demonstrate two ways to inject the modified condition.

### Pattern A — monkey-patch the policy config

When the variable you want to change is **on the policy config object** (anything `PreTrainedConfig` carries — inference settings, head dimensions, action-chunk size, dropout, etc.), monkey-patch `lerobot.configs.policies.PreTrainedConfig.from_pretrained` before calling `embodimetry.eval.run_cell_from_specs`.

```python
from lerobot.configs.policies import PreTrainedConfig

original = PreTrainedConfig.from_pretrained

def patched(cls, *args, **kwargs):
    cfg = original(*args, **kwargs)
    if cfg.type == "act":  # only patch the policy type you care about
        cfg.temporal_ensemble_coeff = 0.01
        cfg.n_action_steps = 1
    return cfg

PreTrainedConfig.from_pretrained = classmethod(patched)
```

The rest of the bench pipeline (`load_policy` → `_load_pretrained_policy` → `policy_cls.from_pretrained(config=cfg)`) reads the patched config transparently. The `if cfg.type == "act"` guard keeps the override scoped — re-using the same probe loop for a non-ACT policy would not silently apply the temporal-ensembling change.

Full implementation: [`scripts/probes/probe_act_temporal_ensemble.py`](../scripts/probes/probe_act_temporal_ensemble.py).

### Pattern B — `dataclasses.replace` on the env spec

When the variable is **on the env spec** (`max_steps`, `success_threshold`, `gym_kwargs`, `task_id`, etc.), `dataclasses.replace` the registry-loaded `EnvSpec` before passing it to `run_cell_from_specs`. The spec is frozen, so you can't mutate it in place — but `replace` gives you a new frozen spec with one field overridden.

```python
import dataclasses
from embodimetry.envs import EnvRegistry

env_reg = EnvRegistry.from_yaml("configs/envs.yaml")
base_env_spec = env_reg.get("libero_10")          # max_steps=520 (v1 default)
env_spec = dataclasses.replace(base_env_spec, max_steps=600)  # canonical LIBERO cap
```

Pass `env_spec` to `run_cell_from_specs`; everything downstream (env construction, episode loop, success check) sees the new cap.

Full implementation: [`scripts/probes/probe_smolvla_libero_canonical_cap.py`](../scripts/probes/probe_smolvla_libero_canonical_cap.py).

### Why not extend the registry?

Both patterns are intentionally one-off scripts, not new config knobs. The bench's `PolicyRegistry` + `EnvRegistry` are the **leaderboard contract** — anything in the YAML registries appears as a cell on the public leaderboard. A probe lives outside the leaderboard contract: its rows go to `results/probes/<probe-name>/` (not `results/sweep-full/`), and its parquet is **not** consumed by `space/_helpers.compute_leaderboard_table` (which loads `results.parquet` only).

When a probe result becomes load-bearing for a future leaderboard read, the override moves into the registry as a new policy or env entry (e.g. a hypothetical `act_paper_settings` policy with `temporal_ensemble_coeff: 0.01, n_action_steps: 1` baked in). For one-off audit work — keep it as a script.

## The probe skeleton

A probe script has six pieces:

1. **Constants** (top of file): policy name, env name, seeds, episodes/seed, output dir.
2. **Override mechanism** (Pattern A or B above): the one thing the probe is changing.
3. **Registry loads**: `PolicyRegistry.from_yaml(...)` + `EnvRegistry.from_yaml(...)`.
4. **Per-seed loop**: call `run_cell_from_specs(...)`, write rows via `append_cell_rows`, accumulate per-seed success rate.
5. **Summary write**: `summary.json` with policy name, env name, probe description, per-seed rates, pooled rate, and the v1 baseline number for comparison.
6. **Headline log line**: `logger.info("PROBE COMPLETE pooled=%.4f (v1 baseline=%.4f)", ...)`.

The two reference probes are ~120 LoC each and follow this exact skeleton — open them side-by-side and the partition is obvious.

## Running a probe

```bash
# Make sure the lerobot conda env is active (or use the absolute path):
python scripts/probes/probe_act_temporal_ensemble.py

# Output goes to:
#   results/probes/act-aloha-temporal-ensemble/results.parquet  (RESULT_SCHEMA, 250 rows)
#   results/probes/act-aloha-temporal-ensemble/summary.json     (pooled + per-seed)
#   results/probes/act-aloha-temporal-ensemble/videos/          (250 MP4s, one per episode)
```

Wall-clock for the reference probes (1× RTX 4060 Laptop): ACT at paper settings ≈ 50 min, SmolVLA at cap=600 ≈ 30-60 min. Both are checkpointable at the cell boundary — if the process dies mid-seed, the partial seed restarts from episode 0 but completed seeds are preserved.

## Reading the probe output

`examples/read_results.py` already loads any `RESULT_SCHEMA`-compatible parquet — point it at your probe's output:

```bash
python examples/read_results.py --results results/probes/act-aloha-temporal-ensemble/results.parquet
```

For a paired comparison against the v1 baseline cell, `examples/compare_two_policies.py` works the same way — pass two parquet paths and the comparison runs on the row union.

For the structured summary, `summary.json` is parseable as standard JSON:

```bash
python -c "import json; print(json.dumps(json.load(open('results/probes/act-aloha-temporal-ensemble/summary.json')), indent=2))"
```

## Promoting a probe finding into the bench

If a probe result is load-bearing for v1.0.2 framing, three things happen:

1. **`docs/PROBE_RESULTS_V1.0.1.md` fill-in** — per-seed table + Wilson CI + interpretation paragraph (the ACT case landed via [PR #97](https://github.com/thrmnn/lerobot-bench/pull/97); follows a 3-bucket "what does Δ mean" structure documented in the scaffold).
2. **README + MODEL_CARDS update** — replace any "probe pending" language with the empirical numbers ([PR #101](https://github.com/thrmnn/lerobot-bench/pull/101) is the ACT-fill template).
3. **Deck + paper updates** — add a row to the relevant table or callout, with a footnote pointing to the probe.

The full v1.0.1 → v1.0.2 audit-cycle handoff is documented in [`docs/PROBE_RESULTS_V1.0.1.md`](../docs/PROBE_RESULTS_V1.0.1.md).

## See also

- [`docs/INFERENCE_AUDIT.md`](../docs/INFERENCE_AUDIT.md) — the audit that identified the ACT inference-settings gap, and how to identify similar gaps for other policies.
- [`docs/SUCCESS_CRITERION_AUDIT.md`](../docs/SUCCESS_CRITERION_AUDIT.md) — the audit that identified the LIBERO step-cap gap.
- [`docs/CANONICAL_CRITERIA.md`](../docs/CANONICAL_CRITERIA.md) — the per-env canonical-vs-v1 table.
- [`docs/PIPELINE_ROADMAP.md`](../docs/PIPELINE_ROADMAP.md) — the v1.0.1 → v1.1 → vNext audit and probe roadmap, including which §1 audits are still open.
