# Documentation index

`embodimetry` is a public, reproducible **instrument** that scores every
robot-policy paradigm — pretrained imitation, fine-tuning, classical control,
and a gated world-model-planning rung — as the same `obs → action` callable on
shared LeRobot tasks. The v1 public leaderboard is the L0 rung: a 6×6
policy×env matrix of 22 cells (18 published) × 5 seeds = 110 cell-seed runs
dispatched, 0 failures, with 5 policies published and xvla deferred to v1.1.
Every number is a binary-outcome rate with a Wilson + bootstrap confidence
interval, a minimum-detectable-effect (MDE) bound, paired comparisons, and a
hand-labeled failure taxonomy.

This page is the map of the `docs/` directory. Find the group that matches
why you are here, then follow the link. If you only want to *look* at results,
you do not need any of this — open the
[live leaderboard](https://huggingface.co/spaces/thrmnn/embodimetry).

---

## New here?

Start here to understand what the benchmark is and to get a result on your
own machine.

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — from `git clone` to a real
  benchmark result in under five minutes; runs one policy on one env. Read
  this first if you want to run anything locally.
- [`FAQ.md`](FAQ.md) — short answers to the real questions users, contributors,
  and reviewers ask; each deep answer points at the doc that owns it.
- [`tutorials/interpreting-the-leaderboard.md`](tutorials/interpreting-the-leaderboard.md)
  — how to read a leaderboard cell as a statistical claim, not just a number,
  and the four misreadings the design prevents.
- [`../examples/`](../examples/) — small, self-contained runnable scripts
  ([`run_one_cell.md`](../examples/run_one_cell.md),
  [`compare_two_policies.py`](../examples/compare_two_policies.py),
  [`read_results.py`](../examples/read_results.py)), each doing exactly one
  thing against the real API.

## Using and interpreting results

For anyone consuming the published numbers — verifying a claim, reading the
dataset, or building on the leaderboard.

- [`REPRODUCE.md`](REPRODUCE.md) — how to reproduce a published leaderboard
  cell bit-for-bit; defines the `(policy, env, seed)` reproducibility contract.
- [`MDE_TABLE.md`](MDE_TABLE.md) — minimum detectable difference at the planned
  sweep N; the source of truth for whether any comparison is statistically
  powered.
- [`FAILURE_TAXONOMY.md`](FAILURE_TAXONOMY.md) — the six failure modes and the
  labeling template for classifying *how* a policy fails when it fails.
- [`HUB_DATASET_README.md`](HUB_DATASET_README.md) — the dataset card for the
  results published to the Hugging Face Hub; read before consuming the parquet.
- [`API.md`](API.md) — hand-authored reference for the public `embodimetry`
  Python API (registries, eval core, stats helpers, renderer, checkpointing).

## Audits & probes

The methodology-audit trail behind the numbers: where a measured cell
diverges from its source paper, which divergences were re-run as probes,
and which policies are deferred and why. Read these before quoting any
paper-vs-measured delta.

- [`PROBE_RESULTS_V1.0.1.md`](PROBE_RESULTS_V1.0.1.md) — the re-run probes
  (ACT temporal-ensemble, `libero_10` canonical cap) and what each resolved.
- [`INFERENCE_AUDIT.md`](INFERENCE_AUDIT.md) — Hub-default vs. paper inference
  settings; the source of the ACT default-vs-paper gap.
- [`CLAIM_AUDIT_SMOLVLA.md`](CLAIM_AUDIT_SMOLVLA.md) — the single-task vs.
  suite-average scope mismatch behind the SmolVLA paper-vs-measured deltas.
- [`SUCCESS_CRITERION_AUDIT.md`](SUCCESS_CRITERION_AUDIT.md) — success-rule and
  step-cap mismatches (Aloha reward bands, PushT sticky-coverage, LIBERO 600).
- [`CANONICAL_CRITERIA.md`](CANONICAL_CRITERIA.md) — the canonical per-env
  success and step-cap definitions the audits measure against.
- [`DEFERRED_POLICIES.md`](DEFERRED_POLICIES.md) — why `xvla` and the `pi0`
  family are executed-but-deferred, with the locked SHAs that carry to v1.1.

## Contributing

For anyone adding a policy, fixing a bug, or sending a PR.

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup, conventions,
  and how to submit a PR.
- [`MODEL_CARDS.md`](MODEL_CARDS.md) — one card per v1 policy, with locked repo
  IDs and revision SHAs; the template a new policy entry must follow.
- [`ENV_CONTRIBUTION_GUIDE.md`](ENV_CONTRIBUTION_GUIDE.md) — how to add a new
  simulated environment to the matrix under the eval/seeding contract.
- [`POLICY_DIAGRAM_GUIDE.md`](POLICY_DIAGRAM_GUIDE.md) — conventions for the
  per-policy architecture diagrams used in the cards and the writeup.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — failure modes you are likely to
  hit running the benchmark, each with a copy-paste fix.

## Operating a sweep

For anyone running, resuming, or publishing the full benchmark sweep.

- [`RUNBOOK.md`](RUNBOOK.md) — operational how-to for running, resuming,
  publishing, and rolling back the sweep and the public Space.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — short architecture index: dataflow
  diagrams and links into the design docs.
- [`PATH_B_INTEGRATION_SMOKE.md`](PATH_B_INTEGRATION_SMOKE.md) — the one-time
  smoke test that proves the synthetic test mocks match real lerobot/gym
  shapes; run once when the sim environment is first installed.
- [`ORCHESTRATION.md`](ORCHESTRATION.md) — how the sweep is dispatched and
  scheduled across cells: the cell-seed run units, queueing, and resume logic.
- [`ORCHESTRATION_PLAYBOOK.md`](ORCHESTRATION_PLAYBOOK.md) — the versioned
  catalog of reusable multi-agent workflows (council, repro-audit, gpu-task,
  prepublish-gate, …) grouped by phase of work, each with its proven phases
  and the real failure it exists to prevent.
- [`MONITORING.md`](MONITORING.md) — the operator dashboard (progress, RAM
  watchdog, and the Failures tab that pre-distinguishes timeouts before labels).

## Project and meta

Design rationale, planning, security, and release process — read when you
want the *why* behind the project or are preparing a release.

- [`DESIGN.md`](DESIGN.md) — the technical design doc: scope, methodology, and
  reviewer concerns; the *what* and *why* of the benchmark.
- [`blog/capability-ladder-audit.md`](blog/capability-ladder-audit.md) — the
  narrative walk-through of the ladder (L0–L3) and the honest negatives:
  the self-caught norm bug, the within-noise L1 lift, the SmolVLA collapse,
  the L2 controller ceiling, and the gated L3 open question.
- [`PIPELINE_ROADMAP.md`](PIPELINE_ROADMAP.md) — how the benchmark evolves past
  v1.0: the publish chain, the methodology-audit gate, coverage breadth, the
  sim-to-real bridge, and the world-model research track (§6).
- [`TWO_SPEED.md`](TWO_SPEED.md) — the two-speed operating model: the fast-lane
  production benchmark vs. the slow-lane world-model research track, and the
  single gated adapter PR that is the only write from research into the bench.
- [`WM_RESEARCH_TRACK.md`](WM_RESEARCH_TRACK.md) — the world-model / JEPA planner
  research track: scope, repo split, and how a planner is evaluated as a policy.
- [`SECURITY.md`](SECURITY.md) — security policy: how to report a vulnerability
  and what to expect.
- [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) — the supply-chain and code-level
  security audit of `main`, with findings and hardening recommendations.
- [`../RELEASE.md`](../RELEASE.md) — the ordered v1.0.0 release checklist and
  ship runbook.
- [`../CHANGELOG.md`](../CHANGELOG.md) — notable changes per release, following
  Keep a Changelog.

---

The repo-root [`README.md`](../README.md) is the project's front door — start
there for the leaderboard overview and headline result.
