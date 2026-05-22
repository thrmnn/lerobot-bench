# Documentation index

`lerobot-bench` is a public, reproducible benchmark of pretrained LeRobot
manipulation policies — 6 policies × 6 simulated environments, run under a
multi-seed contract with Wilson and bootstrap confidence intervals, minimum
detectable difference (MDE) bounds, paired comparisons, and a hand-labeled
failure taxonomy.

This page is the map of the `docs/` directory. Find the group that matches
why you are here, then follow the link. If you only want to *look* at results,
you do not need any of this — open the
[live leaderboard](https://huggingface.co/spaces/thrmnn/lerobot-bench).

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
- [`API.md`](API.md) — hand-authored reference for the public `lerobot_bench`
  Python API (registries, eval core, stats helpers, renderer, checkpointing).

## Contributing

For anyone adding a policy, fixing a bug, or sending a PR.

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — development setup, conventions,
  and how to submit a PR.
- [`MODEL_CARDS.md`](MODEL_CARDS.md) — one card per v1 policy, with locked repo
  IDs and revision SHAs; the template a new policy entry must follow.
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

## Project and meta

Design rationale, planning, security, and release process — read when you
want the *why* behind the project or are preparing a release.

- [`DESIGN.md`](DESIGN.md) — the technical design doc: scope, methodology, and
  reviewer concerns; the *what* and *why* of the benchmark.
- [`CEO-PLAN.md`](CEO-PLAN.md) — strategic framing of the project as a triptych
  artifact (benchmark + arxiv writeup + upstream PR).
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — live execution checklist tracking what
  lands in the next PR.
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
