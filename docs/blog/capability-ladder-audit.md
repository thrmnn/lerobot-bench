# The Capability-Ladder Audit: when does planning substitute for learning?

*A builder's note on what falls out when you put pretrained policies, fine-tunes,
classical control, and world-model planning on one measurement contract — and read
the results honestly, including the uncomfortable ones.*

---

Here is the result that made me trust the rest of the numbers: we took a SmolVLA
policy that scored **0.252** on a LIBERO-10 task, fine-tuned it with LoRA to do
*better*, and watched its closed-loop success rate fall to roughly **zero**.

Not because the training diverged. The behavior-cloning loss converged cleanly —
the offline curves looked like a textbook fine-tune. It was the *online* number,
the only one that counts, that collapsed. The policy had quietly learned to flip
the sign of its gripper action. Open-loop it looked fine; closed-loop it never
closed its hand. The collapse reproduced across learning rates. And the thing
everyone reaches for first — "you ran out of VRAM, the run was truncated" — wasn't
it either: the whole thing fit in 8 GB. It was a data-wiring bug in how we plumbed
actions, and the *only* instrument that caught it was a closed-loop success rate
with a confidence interval attached.

That gap — between a loss curve that says "great" and a robot that says "no" — is
the entire reason this project exists.

## One contract, four paradigms

The embodied-AI literature is split into tribes that don't share a number.
Pretrained-VLA people cite suite-averaged success. Classical-control people cite
tracking error. World-model people cite planning cost and rollout MSE. Each tribe
benchmarks against itself, so you can't actually answer the question a practitioner
has at 2am: *for my task, should I fine-tune the big model, write a controller, or
plan through a world model?*

So we collapsed all of it to one interface. Every method — a frozen pretrained
checkpoint, a LoRA fine-tune, a hand-written controller, a world-model planner doing
CEM at inference — is reduced to the same `PolicyCallable`: `obs -> action`. Then
every one of them runs the *same* eval contract:

- **One seeding rule.** A cell is a `(policy, env, seed, n_episodes)` tuple;
  the seed triple determines everything, so any number is bit-reproducible.
- **One uncertainty story.** Every leaderboard entry is a binary-outcome rate with
  a **Wilson 95% interval**, plus a bootstrap CI for derived quantities. No bare
  point estimates.
- **One honesty gate.** Before we claim method A beats method B, the **minimum
  detectable effect** at our sample size has to be smaller than the gap we're
  claiming. If the CIs overlap, we say "within noise" — not "improvement."

The payoff isn't the leaderboard. It's that the *same ruler* now measures a
controller and a transformer, so the cross-paradigm question becomes a subtraction.

## What the ruler said

**L0 — pretrained, zero-shot.** A pretrained Diffusion Policy on PushT scores
**0.816 [0.739, 0.874]** (N=125). A pretrained ACT on the Aloha transfer-cube task
scores **0.824 [0.772, 0.866]** (N=250). These are the strong rungs: large models,
no task-specific tuning, already good.

*(One aside, because it's the kind of thing this contract exists to catch: an early
release briefly read 0.016 for that ACT cell. It was a normalization bug on our
harness side, not the policy's fault — fixed, re-run, 0.824. Plumbing. The point is
the instrument flagged it.)*

**L2 — classical control.** Here's the first uncomfortable one. A competent
hand-written controller on the same precision task gets you to roughly **0.50 mean
coverage** — it reaches the right region, moves the right way, does *most* of the
task — but clears the strict success bar only about **1%** of the time
(**0.012**). Read that as a sentence: *control gets you halfway; learning buys the
last 50% of precision.* The gap between "basically working" and "actually
succeeding" is exactly the gap a learned policy closes and a controller, on its
own, does not. If your instinct was "you could just write a controller for that,"
the instrument says: halfway, yes; over the bar, no.

**L1 — fine-tuning.** The second uncomfortable one, and it cuts the other way.
Fine-tuning ACT on the cube task moved it from **0.824 to 0.864** — *+0.040*. That
looks like a win until you read the intervals: **they overlap.** At our sample size
this is within noise. We do not get to call it an improvement, so we don't. The
contract's whole job is to stop us from banking a 4-point bump that a bootstrap
can't distinguish from luck.

And then SmolVLA, above: a fine-tune that didn't fail to help — it *actively
collapsed*, **0.252 → ~0.0**, while every offline metric smiled. (For the record,
that 0.252 is one LIBERO-10 task, not the suite average; we're careful about that
scope, and the collapse is real within it.)

Three fine-tuning stories, one ruler: a real-but-not-significant nudge, a silent
catastrophe, and — the lesson — *you cannot tell which is which from the loss
curve.* You need the closed-loop number with the interval.

## The thesis this earns

A neutral, reproducible measurement contract is most valuable precisely when it
tells you something you didn't want to hear. It told us fine-tuning made one policy
worse and another no-better-than-noise. It told us a classical controller is closer
to a learned policy than the "VLAs win" narrative assumes — and also that "closer"
stops dead at the precision bar. None of those are flattering to anyone's tribe.
All of them are things you'd want to know before spending a month.

The uncomfortable results are the *product*. A benchmark that only ever confirms
the thing you hoped is a press release with error bars.

## The number we're chasing next

There's one rung we haven't measured: **L3, world-model planning** — does planning
through a learned dynamics model *substitute* for learning a policy, and where does
the trade sit against a fine-tuned VLA on the same contract? We have the planner
wired as a `PolicyCallable` and the eval harness ready. What we don't have is the
compute: a paper-faithful CEM rollout runs ~4 minutes per step on the 8 GB card
this is parked on, so the cells are compute-parked, not done. That number — when
planning beats learning, when it doesn't, and what it costs — is the one we're
chasing next. It is **pending**, and we won't quote it until it clears the same gate
as everything above.

When does planning substitute for learning? We built the ruler to answer it
honestly. Ask us again when the L3 cells are green.

---

**Try it / read the receipts**

- Repo (all numbers reproduce from a seed triple, no GPU needed for the mini-run):
  <https://github.com/thrmnn/embodimetry>
- One-cell Colab try-it that re-derives `act × aloha 0.824 [0.772, 0.866]` and the
  SmolVLA × LIBERO cells from the published parquet — no GPU, no Hub download:
  see the **Try it** link in the repo README.

*Every number above is a binary-outcome rate with a confidence interval, anchored
to a pinned `lerobot` release and per-checkpoint SHAs. If a claim here doesn't trace
to a parquet row, it isn't in the leaderboard — and the L3 question stays open until
it does.*
