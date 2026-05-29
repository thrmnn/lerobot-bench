# Draft GitHub issue for `huggingface/lerobot`

> **Status:** DRAFT — not yet posted upstream. The user will review this file,
> then open the issue manually on `huggingface/lerobot`. Edit freely before
> posting.
>
> **Suggested title:** `ACT Hub checkpoint (act_aloha_sim_transfer_cube_human) ships with inference defaults that hide ~75pp of published success rate`

---

## Summary

The Hub checkpoint
[`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)
serializes its config with `temporal_ensemble_coeff=None` and
`n_action_steps=100`. With these defaults, `lerobot`'s inference loop re-queries
the transformer every 100 steps and replays a stale 100-step action chunk in
between — which collapses the policy's competence on
`gym-aloha/AlohaTransferCube-v0` to **~1.6%** pooled across 5 seeds × 50
episodes. Re-running the **same checkpoint** with the paper's inference
settings (`temporal_ensemble_coeff=0.01`, `n_action_steps=1`, the original ACT
overlapping-chunk weighted averaging from Zhao et al. 2023, Table I) recovers
**~76.4%** on the same protocol — a +74.8pp swing whose 95% Wilson confidence
intervals are disjoint by an order of magnitude. The architecture and weights
are fine; the pushed inference defaults silently mis-represent the policy.

Filing this so other benchmark consumers don't hit the same trap. We're happy
to send a PR for whichever resolution path the maintainers prefer (see
"Possible resolutions" below).

## Repro

The probe script (5 seeds × 50 episodes on `aloha_transfer_cube`, otherwise
identical to our v1.0.0 sweep) lives in
[`thrmnn/lerobot-bench`](https://github.com/thrmnn/lerobot-bench):

- Script: <https://github.com/thrmnn/lerobot-bench/blob/main/scripts/probes/probe_act_temporal_ensemble.py>
- Results doc: <https://github.com/thrmnn/lerobot-bench/blob/main/docs/PROBE_RESULTS_V1.0.1.md>

The script monkey-patches
`lerobot.configs.policies.PreTrainedConfig.from_pretrained` to override the two
fields on ACT configs only, then runs the standard `lerobot` policy-load
pipeline. The override is the only thing that changes between the v1.0.0 and
v1.0.2 columns in the evidence table below.

Minimal 5-line reproduction of the defaults that ship with the checkpoint:

```python
from lerobot.configs.policies import PreTrainedConfig

cfg = PreTrainedConfig.from_pretrained(
    "lerobot/act_aloha_sim_transfer_cube_human",
    revision="ba73b2766f1371cdc133ca4efb97eb090d744625",
)
print(cfg.type, cfg.temporal_ensemble_coeff, cfg.n_action_steps)
# act None 100
```

## Evidence

5 seeds × 50 episodes per row, `gym-aloha/AlohaTransferCube-v0`, same checkpoint
SHA (`ba73b27…`), same seeding contract, only the two inference fields differ.
Wilson 95% CIs.

| Setting | `temporal_ensemble_coeff` | `n_action_steps` | Pooled success | 95% CI | N |
|---|---|---|---|---|---|
| **Hub default (v1.0.0)** | `None` | `100` | **0.016** | [0.006, 0.040] | 250 |
| **Paper setting (v1.0.2 probe)** | `0.01` | `1` | **0.764** | [0.708, 0.812] | 250 |
| Zhao et al. 2023 Table I, "Cube Transfer (sim) / Transfer", human-teleop column | `0.01` | `1` | 0.50 | — (3 seeds × 50 ep) | 150 |

Per-seed for the v1.0.2 probe row: 0.92 / 0.80 / 0.76 / 0.66 / 0.68. Our probe
exceeds the paper's reported 0.50 by +26.4pp, which is consistent with the
paper's smaller eval (3 seeds × 50 ep vs our 5 × 50) and with normal
seed-to-seed variation. The load-bearing variable is the inference setting,
not the architecture or the checkpoint weights.

## Possible resolutions

Three options, smallest blast radius first. We have no preference between
(a) and (b); (c) is the most principled but affects every ACT user.

**(a) Re-push the checkpoint config with paper settings.**
Update `lerobot/act_aloha_sim_transfer_cube_human`'s pushed config so
`temporal_ensemble_coeff: 0.01` and `n_action_steps: 1` are the defaults
serialized on the Hub. Smallest behavior change for downstream users — the
inference path already supports both modes — but requires a maintainer to
re-push the checkpoint and bump a revision. Existing users who explicitly
loaded the prior SHA are unaffected.

**(b) Update the model card on the Hub.**
Add a prominent "you must override these defaults" section to the
[model card](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)
with a runnable snippet showing the override. Zero behavior change. Lowest
risk, but relies on users reading the card before evaluating. We're happy to
draft this and open a PR on the Hub repo if this is the preferred path.

**(c) Change the lerobot ACT factory's defaults.**
[`src/lerobot/policies/act/configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py#L120)
currently has `temporal_ensemble_coeff: float | None = None` (and
`n_action_steps` defaults that pair with it). Flipping these to the paper
values would make every ACT policy load with the canonical inference protocol
unless the user explicitly opts out. Highest blast radius — affects all ACT
checkpoints on the Hub and every fresh `ACTConfig()` — but it's the
intervention that closes the trap class, not just this instance. Note that
[`modeling_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py#L175)
already documents `0.01` as "the default value used by the original ACT work."

## Why we're filing this now

We're publishing a multi-policy benchmark
([lerobot-bench v1.0.0](https://github.com/thrmnn/lerobot-bench), 5 policies ×
4 envs, 5 seeds × 50 episodes per cell, ~3,800 episodes total) that uses
`lerobot` policies verbatim with their Hub defaults. Our v1.0.0 leaderboard
reads ACT at 0.016 on `aloha_transfer_cube` — which made us doubt the
architecture before we doubted the inference config. PR #86 in our repo
identified the inference-defaults gap as the most plausible explanation; the
probe in PR #92 / #97 confirmed it with disjoint CIs. The audit is documented
in [`docs/PROBE_RESULTS_V1.0.1.md`](https://github.com/thrmnn/lerobot-bench/blob/main/docs/PROBE_RESULTS_V1.0.1.md).

We're surfacing this upstream so that other benchmark consumers (and reviewers
of our paper) don't have to repeat the audit. The Hub default isn't malicious —
`lerobot`'s inference factory works correctly with any value of the two
fields, and the breakage only becomes visible at benchmark-scale evaluation
where the order-of-magnitude success-rate swing is hard to miss.

## References

- Checkpoint: <https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human> (pinned SHA `ba73b2766f1371cdc133ca4efb97eb090d744625`)
- `lerobot` ACT config defaults: [`src/lerobot/policies/act/configuration_act.py#L120`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py#L120) (`temporal_ensemble_coeff: float | None = None`)
- `lerobot` ACT temporal ensembler note: [`src/lerobot/policies/act/modeling_act.py#L175`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py#L175) ("The default value … used by the original ACT work is 0.01.")
- Zhao et al. 2023, _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_, arXiv:[2304.13705](https://arxiv.org/abs/2304.13705) (Table I, "Cube Transfer (sim) / Transfer", human-teleop column = 0.50)
- Our probe script: <https://github.com/thrmnn/lerobot-bench/blob/main/scripts/probes/probe_act_temporal_ensemble.py>
- Our probe results doc: <https://github.com/thrmnn/lerobot-bench/blob/main/docs/PROBE_RESULTS_V1.0.1.md>

## Maintainer ask

Happy to send a PR for option **(b)** — a model card update on
`lerobot/act_aloha_sim_transfer_cube_human` with the override snippet — if
that's the preferred resolution path. Options **(a)** and **(c)** probably
need a maintainer to drive (re-push checkpoint / change library defaults).

Either way, let us know which direction you'd prefer and we'll either send the
PR or stand down.
