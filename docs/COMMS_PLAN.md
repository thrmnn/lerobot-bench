# Launch comms plan — embodimetry v1

The launch sells **two settled, citable contributions** and **one in-flight
question**. Everything below is sequenced so the in-flight question never
reads as a settled result.

- **Citable (we stand behind these):** (1) a single **cross-paradigm
  evaluation contract** — pretrained / fine-tuned / classical-control /
  world-model-planner scored as the same `PolicyCallable` under one
  seed + Wilson/bootstrap-CI + MDE ruler, with floor baselines and honest
  negatives (the SmolVLA-LoRA collapse, the classical `0.012`) kept on the
  board; (2) a **self-auditing methodology** — the instrument caught and
  corrected its *own* headline overclaim (ACT × aloha `0.016 → 0.824`,
  isolated by a clean 2×2 ablation), with a guard test
  (`tests/test_no_ungated_l3_claims.py`) that code-enforces the unproven
  rung stays labeled in-flight.
- **In-flight (the animating question, NOT a third result):** the
  **dynamics-complexity gradient** — *does manipulation success degrade along
  a dynamics-complexity gradient, with world-model-vs-VLA as the contrast?*
  Current signal: Wall-nav `9/24 = 0.375` [0.21, 0.57] vs PushT-contact `0`
  (the contact endpoint of the hypothesized curve), `N=24`, **single
  world-model family, cross-env confounded**, with a second-family
  (`jepa-wms`) de-confound running. It earns its own citation in **v1.1**
  after the de-confound + canonical-600 land.

> **The one rule:** every public mention of the gradient is phrased *we pose /
> we ask / in-flight / the curve we are measuring* — **never** *we show / we
> find / planning substitutes*. Every L3 number travels **with** its CI + the
> single-family caveat + the cross-env confound in the same sentence. The
> guard test fails the build if this regresses.

---

## The ONE thing that travels

If a reader takes away a single sentence and a single figure, make it these.

- **Sentence (the lead, citable):**
  *"We built a single ruler that scores pretrained, fine-tuned, classical, and
  world-model-planner robot policies together — and the first thing it caught
  was our own bug: a normalization fault had pinned ACT at 0.016, below the
  random floor; a clean 2×2 ablation recovered it to 0.824. An eval suite must
  be audited as carefully as the policies it scores."*

- **Figure (the lead, citable):** the **ACT 2×2 normalization ablation**
  (`paper/figures/paper/act_norm_ablation.pdf`) — buggy 0.016 / 0.016 vs fixed
  0.812 / 0.768. It is the self-audit in one image and is fully in-tree
  (traces to `results/probes/act-norm-ablation/`).

- **The gradient is the *hook*, never the *headline*.** Lead a thread or talk
  with the question ("does success degrade along a dynamics-complexity
  gradient?") to earn attention, then immediately land on the contract +
  self-audit as the thing we actually deliver. Never put a bare `0.375` or
  "gradient confirmed" in a headline, alt-text, or social card.

---

## Sequence (gated, in order)

Steps marked **[OWNER]** are authed/irreversible — only the release manager
runs them. Everything else is a local check anyone can run. The ordering is
load-bearing: **the artifact and the guard land before any narrative**, so the
gradient is provably in-flight at the moment of first public contact.

### Phase 0 — pre-flight (local, ungated)
1. Guard green: `PYTHONPATH=$(pwd)/src python -m pytest
   tests/test_no_ungated_l3_claims.py -q`. If red, **stop** — a surface has
   re-inflated the L3 rung; fix before anything ships.
2. Paper builds: `cd paper && make all` → `main.pdf`.
3. Surface parity check: README, `site/index.html`, `paper/deck/index.html`,
   and `docs/blog/capability-ladder-audit.md` all carry the `0.016 → 0.824`
   lead and gate every gradient mention (the guard asserts this).

### Phase 1 — arXiv [OWNER]
4. Submit `paper/main.pdf` to **cs.RO primary, cs.LG secondary**. Title and
   abstract lead with the contract + self-audit; the gradient appears as the
   *in-flight hypothesis the contract is purpose-built to test*, with its CI +
   single-family + confound caveat in the abstract itself.
5. Hold the arXiv ID; it anchors every downstream surface. Do **not** post the
   thread until the ID resolves.

### Phase 2 — living leaderboard [OWNER]
6. After the v1 HF publish gate (`docs/PUBLISH_RUNBOOK.md`), point the
   [HF Space](https://huggingface.co/spaces/thrmnn/embodimetry) at the
   published dataset. The leaderboard *is* the data — the per-cell Wilson CIs,
   the floor baselines, and the retained negatives carry the contract claim
   without any prose.
7. The Space's L3 panel must render the gradient as **in-flight**: the Wall
   `9/24` cell shows its Wilson CI spanning chance and a "single-family,
   confounded — v1.1" badge. No bare success number, ever.

### Phase 3 — the thread (skeleton below) [OWNER to post]
8. Post only after arXiv ID + Space are live, so every claim links to a
   re-runnable artifact. The researcher drafts; the human posts.

### Phase 4 — one-researcher outreach [OWNER]
9. A single targeted note to **one** robotics researcher whose work the
   contract is built to serve (a world-model-planning or eval-methodology
   author), not a broadcast. Lead with the self-audit (it is the credibility
   proof), offer the contract as a neutral ruler for their method, and name
   the gradient as the open question we would collaborate to de-confound.
   Attach the arXiv link and the one figure. No claim of a finding.

---

## The 4-tweet skeleton (researcher drafts, human posts)

The thread *opens on the question* to earn the scroll, then delivers the two
citable contributions. The gradient never appears as a settled result.

1. **Hook (the in-flight question).**
   "Does a robot's success fall along a *dynamics-complexity gradient* — point
   nav → contact → deformable — as you swap a world-model planner for a learned
   policy? We don't answer that yet. To even *ask* it cleanly you need one
   ruler for paradigms that are never measured together. So we built the ruler.
   🧵 [arXiv]"

2. **Contribution 1 — the contract.**
   "embodimetry scores pretrained / fine-tuned / classical / world-model-planner
   policies as the *same* PolicyCallable under one seed + Wilson-CI + MDE ruler.
   Floor baselines and honest negatives stay on the board — a SmolVLA-LoRA
   *collapse*, a classical controller at 0.012. No neighbor keeps the negatives."
   [forest_plot figure]

3. **Contribution 2 — the self-audit (the lead figure).**
   "The first thing the ruler caught was *our own bug*: a normalization fault
   pinned ACT at 0.016, below the random floor. A clean 2×2 ablation recovered
   it to 0.824. An eval suite must be audited as carefully as the policies it
   scores — and a guard test now code-enforces it."
   [act_norm_ablation figure]

4. **The in-flight gradient (gated) + the ask.**
   "At the top of the ladder we *pose*, not answer: a zero-training world-model
   planner hits Wall-nav 9/24=0.375 [0.21,0.57] but ~0 on PushT — the contact
   endpoint of the hypothesized curve. N=24, single WM family, cross-env
   confounded; a 2nd-family de-confound is running. It's the curve we're
   measuring, not one we've measured. v1.1. [Space link]"

---

## Guardrails for every surface (the guard test enforces these)

- Lead with `0.016 → 0.824` and the contract; the gradient is the hook, not
  the headline.
- Verb discipline on the gradient: *pose / ask / in-flight / the curve we are
  measuring*. Never *show / find / planning substitutes for learning*.
- Every L3 number ships with: its Wilson CI, "single world-model family",
  "cross-env confounded", and "de-confound running". Never a bare `0.375`.
- PushT `0` = **the contact endpoint of the hypothesized curve** (a feature),
  never "the planner failed".
- The cross-env confound is stated as an **open limitation**, including in the
  social card / abstract — not buried.
- If any surface drifts, `tests/test_no_ungated_l3_claims.py` fails the build.
  That failure is the signal to fix the copy, not to weaken the test.
