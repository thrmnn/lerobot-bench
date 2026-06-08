# Orchestration playbook

**Version:** 1.0 · **Status:** living · **Scope:** Embodimetry + the WM repo

A versioned catalog of the reusable multi-agent workflows this project
discovered and proved in practice. Each one was earned by a real failure
or a real reputation landmine — the "why it exists" lines below are not
hypotheticals; they tie to a moment this session.

`docs/ORCHESTRATION.md` covers the *mechanics* of a wave (worktree
isolation, disjoint file ownership, serial merge into strict `main`). This
doc is the *catalog*: which named workflow to reach for, in what phase of
work, and the ordered phases each one runs. Treat it as the project's
operating manual.

---

## How to use

These workflows run one of two ways:

- **Via the Workflow tool** (dynamic multi-agent). The proven patterns:
  - **council** — `positions → chair-synthesis → red-team → chair-final`.
    N worktree-isolated position agents each argue one lens from on-disk
    evidence, the chair synthesises a tentative ruling, an adversarial
    red-team tries to falsify it, then the chair rules with dissents
    preserved. **Never drop the red-team phase** — it is the de-risking
    part, the step that stops the research-impact lens from quietly winning
    a fork two other lenses would veto.
  - **review → adversarially-verify** — any produced number or claim is
    handed to a second agent (`redteam-finding`) whose job is to *kill* it:
    enumerate confounds, check N/Wilson/MDE adequacy, hunt the
    un-normalized-proprio bug class, and try to reproduce the *opposite*
    conclusion. A claim that survives ships with its caveats attached.
  - **ground-truth-read** — any GPU/dispatch agent's stdout is **untrusted**.
    The verdict is always the number re-read from the written
    parquet/manifest on disk. This is baked into `gpu-task` and
    `repro-audit`.

- **As a monitoring `/loop`** (standing self-paced cadence). `loop-tick`
  is the heartbeat: one tick reads status, advances `pr-drain` by exactly
  one step, refills the single-GPU queue if idle, and surfaces owner-gated
  blockers. The governing pattern is **loop-until-dry**: keep ticking until
  the PR queue is empty (or only owner-gated PRs remain) and the GPU queue
  is drained.

Shared invariants across every workflow:

- **Worktree isolation** (`isolation: worktree`) for any repo-modifying
  agent — concurrent git ops otherwise race the shared `HEAD`.
- **`PYTHONPATH=$(pwd)/src`**, not `pip install -e .`, inside a worktree —
  the editable `.pth` is static and points at the parent's `src/`.
- **The machine-global GPU lock** (`/tmp/embodimetry-gpu.lock`, honored by
  both repos) gates every CUDA-touching dispatch; 1-at-a-time behind a VRAM
  semaphore.
- **Decisions and verdicts are dated artifacts** written to
  `docs/decisions/` — agent prose is not the record.

---

## Catalog by phase of work

Priority is the codified rank (10 = highest). The lower the phase in a
publish chain, the higher the blast radius if it is skipped.

| Phase | Workflow | Priority | When to use (one line) |
|-------|----------|----------|------------------------|
| Understand | `make-status` | 6 | Every orchestration tick — one-shot fleet snapshot so you stop re-deriving state. |
| Decide | `council` | 8 | Any irreversible scope/strategy/resourcing fork. |
| Decide | `reconcile-narrative` | 8 | Before any distribution — enforce ONE true story across every public surface. |
| Build | `gpu-task` | 9 | Any CUDA-touching dispatch (sweeps, L1 finetune, WM/CEM). |
| Build | `merge-results-artifacts` | 8 | Splice off-main result artifacts onto `main` with their SHAs. |
| Build | `new-env-integration` | 5 | Add a new env to the ladder without contaminating the VLA leaderboard. |
| Verify | `repro-audit` | 9 | Pre-publish and post-merge of any results PR. |
| Verify | `redteam-finding` | 9 | Any single claim about to enter the paper/leaderboard. |
| Verify | `verify-render` | 7 | Any dashboard/Space/GIF/figure change. |
| Ship | `prepublish-gate` | 10 | Immediately before any HF Hub publish or the owner-gated #177 merge. |
| Operate | `pr-drain` | 7 | Any time >1 PR is open against strict `main`. |
| Operate | `loop-tick` | 6 | The standing self-paced background cadence. |

---

## Understand

### `make-status` · priority 6

**When to use:** every orchestration tick, to stop re-deriving fleet state
and burning tokens.

**Why it exists:** the audit flagged *uncommitted* `wall_n24_*` result
artifacts in the WM working tree as the single worst possible state — an
uncommitted headline-candidate number. A one-shot snapshot makes that
condition impossible to miss instead of re-deriving it by hand each tick.

**Phases**
1. **gpu** — `nvidia-smi` util + mem; read `/tmp/embodimetry-gpu.lock` holder.
2. **prs** — `gh pr list` count for both Embodimetry and the WM repo.
3. **worktrees** — count prunable worktrees under `.claude/worktrees/`.
4. **dirty** — `git status --porcelain` across BOTH repos; warn loudly on
   uncommitted result artifacts (e.g. `wall_n24_*`).
5. **print** — emit a single compact status block.

**Inputs:** both repo paths; GPU lock path.
**Outputs:** one status block — GPU util/mem, lock holder, PR counts ×2,
prunable-worktree count, dirty-tree warnings across both repos.

---

## Decide

### `council` · priority 8

**When to use:** any irreversible scope/strategy/resourcing fork (e.g.
"merge L3 as the headline?").

**Why it exists:** this very chair task is one. Without a mandatory
adversarial pass, the research-impact lens quietly wins a fork that the
methodology lens and the on-disk check would both veto — exactly how an
unshipped L3 number nearly became a published headline.

**Phases**
1. **frame** — state the fork, the options, and the reversibility cost;
   define the lenses to staff.
2. **positions** — launch N worktree-isolated agents, each arguing one
   lens from on-disk evidence, in parallel.
3. **chair-synthesis** — chair merges positions into a tentative ruling
   with explicit trade-offs.
4. **red-team** — adversarial pass against the tentative ruling, probing
   JEPA sunk-cost, HF-role conflict, and over-scoping; must try to falsify
   with on-disk evidence. **Do not drop this phase.**
5. **chair-final** — final ruling, dissents preserved verbatim.
6. **record** — write to `docs/decisions/<date>-<slug>.md`.

**Inputs:** the fork/question; option set; lens list; N position agents.
**Outputs:** a `docs/decisions/` record with the final ruling, trade-offs,
and preserved dissents.

### `reconcile-narrative` · priority 8

**When to use:** before any distribution (arXiv, thread, Space) — to
enforce ONE true story across every public surface.

**Why it exists:** the narrative drifted ahead of the artifact. The L3
"planning substitutes for learning" headline did not exist on disk, yet it
had leaked into the self-description; meanwhile the canonical parquet still
shipped the buggy `act×aloha=0.016`. This workflow makes the
normalization-bug + auditable-instrument story the lead everywhere and
gates every L3 claim behind PROPOSED.

**Phases**
1. **inventory** — enumerate all public surfaces (paper/main.tex, README,
   site/index.html, deck, business-case, MEMORY) and their headline claims.
2. **audit-l3** — grep each surface for L3 / Wall / DINO / "planning
   substitutes for learning"; classify each hit as result (forbidden) vs
   PROPOSED/in-flight.
3. **rewrite-lead** — ensure the self-caught normalization-bug + instrument
   story leads every surface; demote it nowhere.
4. **gate-claims** — rewrite every forbidden L3 hit to PROPOSED/in-flight,
   or remove it.
5. **sync-memory** — update `MEMORY.md` so the unshipped L3 number is
   recorded as unshipped.
6. **assert-test** — add a test that fails if any tracked surface
   reintroduces an ungated L3-as-result string or drops the normalization
   lead.

**Inputs:** list of surface paths; the canonical lead-story sentence; the
forbidden-claim regex set.
**Outputs:** edited surfaces + a committed consistency test; diff summary.

---

## Build

### `gpu-task` · priority 9

**When to use:** any CUDA-touching dispatch — sweeps, L1 finetune, WM/CEM.

**Why it exists:** the repeatedly-burned lesson that **GPU agents narrate
unreliably**. The mandatory ground-truth read exists because a dispatch
agent's stdout summary has, more than once, disagreed with the parquet it
wrote. The verdict is the disk number; stdout is discarded.

**Phases**
1. **acquire-lock** — `scripts/with_gpu_lock.sh` on the machine-global
   `/tmp/embodimetry-gpu.lock` (honored by both repos); a VRAM-budget
   semaphore keyed on calibration `vram_peak_mb`, not a blunt
   `max_parallel==1`.
2. **cap** — wrap in `scripts/run_capped.sh` (`systemd-run --user --scope`
   `MemoryMax` / `MemorySwapMax=0`), sized from calibration.
3. **dispatch** — run the CUDA job to completion (background-bash for long
   jobs; `watchdog.py` observing, **no breach-grace**).
4. **ground-truth-read** — MANDATORY re-read of the written
   parquet/manifest/JSONL; agent stdout is discarded.
5. **release-and-report** — release the lock; report disk numbers only +
   artifact path.

**Inputs:** command + args; calibration `vram_peak_mb`; memory cap; lock
timeout.
**Outputs:** disk-confirmed result numbers + committed artifact path; lock
released.

### `merge-results-artifacts` · priority 8

**When to use:** splicing result artifacts that live only in worktrees onto
`main`, and enforcing that every cited ladder rung has a committed path.

**Why it exists:** the reproducibility-asymmetry gap — corrected ACT rows
and the L1/L2 summary JSONs existed only off-`main`, so README/paper cited
rungs whose artifacts were not on `main`. This splices them in with their
code+checkpoint SHAs and adds a test so a citation can never outrun an
artifact again.

**Phases**
1. **locate** — find the off-main artifacts (corrected act rerun parquet,
   `act_aloha_l1` / `smolvla_libero10_l1` / `classical_pusht_l2` summary
   JSONs); fail loudly if the restored parquet is still missing.
2. **merge-canonical** — `scripts/merge_corrected_act_rows.py --staging` to
   splice corrected `act×aloha` rows into canonical + `_publish_staging`,
   within the post-merge band.
3. **splice-ladder** — copy L1/L2 summary JSONs into `results/ladder/`
   carrying their `code_sha` + checkpoint SHA.
4. **guard** — enforce `code_sha.nunique()==1` per cell on the merged
   parquet.
5. **artifact-exists-test** — assert every ladder rung cited in
   README/paper resolves to an existing committed artifact path.
6. **verify** — re-read the merged parquet from disk; confirm `act×aloha`
   pooled `p_hat` is in band.

**Inputs:** worktree paths; ladder JSON names; canonical + staging parquet
paths.
**Outputs:** populated `results/ladder/` on `main`; corrected canonical +
staging parquet; artifact-exists + single-SHA tests.

### `new-env-integration` · priority 5

**When to use:** adding a new env to the ladder (Reacher / OGBench-Cube /
two-room Wall) — gated until publish-blockers land, high-value once
unblocked.

**Why it exists:** new envs belong on the SEPARATE WM/planning axis, not as
VLA-leaderboard rows, and the un-normalized-proprio bug makes `PushT=0` look
like a capability failure when it is a normalization failure. This forces
the z-score fix and an `N >= ladder` run *before* any env is called
"failing".

**Phases**
1. **scaffold** — follow `docs/ENV_CONTRIBUTION_GUIDE.md`; register on the
   WM/planning axis, NOT as leaderboard rows.
2. **normalize** — compute and commit dataset z-score proprio stats; verify
   CEM proprio is in-distribution *before* any run.
3. **calibrate** — `scripts/calibrate.py` for the VRAM/memory budget;
   record `vram_peak_mb` for the semaphore.
4. **run-at-N** — dispatch via `gpu-task` at `N >= 50` (ideally 250),
   checkpoint + CEM budget fixed; a within-env dynamics-complexity sweep,
   never a 2-cell cross-env anecdote.
5. **redteam** — run `redteam-finding` on the resulting claim before it
   touches any surface.
6. **commit-artifact** — commit the per-episode parquet (seeds, CEM budget,
   checkpoint SHA) under `results/` on the WM axis.

**Inputs:** env name; checkpoint SHA; CEM budget; N (>=50); dataset for
z-score stats.
**Outputs:** committed per-episode parquet on the WM/planning axis +
redteam verdict; NOT a leaderboard row.

---

## Verify

### `repro-audit` · priority 9

**When to use:** pre-publish and post-merge of any results PR; gates
`prepublish-gate`.

**Why it exists:** turns "bit-for-bit replayable" from a paper *claim* into
an *enforced, dated artifact*. The verdict table is built ONLY from re-read
parquet/manifest numbers — agent prose is untrusted.

**Phases**
1. **select-cells** — sample K cells stratified across L0/L1/L2 from the
   shipped parquet (record policy×env, seeds, code_sha, checkpoint SHA).
2. **reproduce** — run `scripts/reproduce_cell.py` per cell under
   `scripts/with_gpu_lock.sh` with the `run_capped.sh` memory cap.
3. **ground-truth-read** — read the freshly written per-cell
   parquet/manifest from disk; never trust dispatch stdout.
4. **compare** — diff reproduced vs shipped `p_hat` per cell against the
   MDE / `|Δ| < 0.123` inconclusive band.
5. **emit** — write a dated `docs/REPRO_AUDIT_<date>.md` verdict table from
   disk numbers only.

**Inputs:** shipped parquet path; K (default 3); rung filter; MDE band.
**Outputs:** dated `docs/REPRO_AUDIT_<date>.md`; pass/fail per cell;
non-zero exit if any cell diverges beyond MDE.

### `redteam-finding` · priority 9

**When to use:** any claim about to enter the paper/leaderboard;
auto-trigger when the larger-N Wall rerun lands.

**Why it exists:** the de-risking core the audit stresses must *never* be
dropped. A 5-lens audit caught its own overreach — Finding A was
underpowered (N=6) and confounded — only because a phase like this forces
enumerating confounds and attempting the OPPOSITE conclusion.

**Phases**
1. **restate** — pin the exact claim + its committed artifact (path, seeds,
   N, checkpoint SHA, code_sha); fail if no committed artifact backs it.
2. **alternatives** — enumerate confounds (truncation/step-cap artifact,
   env+checkpoint+dynamics co-variation, pooling/Simpson, single-task scope).
3. **stat-adequacy** — Wilson CI + check N against the ladder's N=250
   contract and the `|Δ| < 0.123` gate; reject any rate below ladder N.
4. **bug-class-hunt** — check the un-normalized-proprio failure class
   (`PushT=0` may be the proprio bug, not capability); inspect z-score
   normalization.
5. **reproduce-opposite** — actively construct the analysis that yields the
   opposite conclusion (re-run cap-limited cells at canonical 600 steps;
   stratified bootstrap on the pooled column).
6. **verdict** — kill-or-survive ruling; if survive, attach the surviving
   Limitations sentence; record in `docs/decisions/`.

**Inputs:** claim text; committed artifact path; ladder N + MDE band;
affected public surfaces.
**Outputs:** kill-or-survive verdict + required caveat text + confound
list; blocks promotion on kill.

### `verify-render` · priority 7

**When to use:** any dashboard/Space/GIF/figure change.

**Why it exists:** the orchestrator **cannot see rendered output**. The
verdict must be a concrete screenshot or ffmpeg-extracted frame artifact,
never prose — and the preview must bind `0.0.0.0` so the owner can review it
over Tailscale from tab-samsung (a `127.0.0.1` URL is unreachable there).

**Phases**
1. **change** — apply the dashboard/Space/figure/GIF edit.
2. **render-artifact** — produce a concrete artifact path (screenshot for
   web, single ffmpeg-extracted frame for video); sanity-check ONE output
   before any batch encode.
3. **diff** — generate a before/after image/frame diff; the verdict is the
   artifact, never prose.
4. **serve** — bind the preview server to `0.0.0.0` and hand back
   `http://100.104.205.62:<port>/` for Tailscale review.
5. **verdict** — attach artifact paths + an explicit "visual not
   self-verified" caveat.

**Inputs:** changed surface; port; before/after baseline path.
**Outputs:** screenshot/frame artifact paths + before/after diff + the
`0.0.0.0` Tailscale URL.

---

## Ship

### `prepublish-gate` · priority 10

**When to use:** immediately before any HF Hub publish or the owner-gated
#177 canonical-dataset merge. The highest-priority workflow.

**Why it exists:** the #1 reputation landmine. The canonical AND staging
parquet still ship the buggy `act×aloha=0.016` — the exact number the
paper's own headline disproves. This single go/no-go composes repro
sampling + parquet preflight + L3-claim red-team + HF-recusal so that
shipping a dataset whose headline cell *refutes the paper* cannot happen.

**Phases**
1. **resolve-target** — locate the parquet about to be published (canonical
   `results/sweep-full/results.parquet` + `_publish_staging` copy);
   hard-fail early if either is missing.
2. **repro-audit-sample** — invoke `repro-audit` on `K >= 3` cells spanning
   L0-L2 under the GPU lock; capture the disk-vs-paper verdict table.
3. **parquet-preflight-guard** — load the target parquet; assert `act×aloha`
   pooled `p_hat >= 0.5` (reuses `publish_results._preflight`), assert
   headline cell == paper Table-1 value, assert `code_sha.nunique()==1` per
   cell and no pre-PR#51 SHAs.
4. **redteam-l3-claims** — scan every staged public surface
   (paper/README/site/deck) for L3 / Wall / "planning substitutes for
   learning"; fail if any appears as a *result* rather than PROPOSED.
5. **hf-recusal-check** — confirm `NEUTRALITY.md` / `GOVERNANCE.md` present
   and current; flag any HF-role conflict surface.
6. **verdict** — one dated GO/NO-GO line per check + overall; NO-GO if any
   single check fails; write to `docs/decisions/`.

**Inputs:** target parquet path(s); paper Table-1 headline value; list of
public-surface paths; K (default 3).
**Outputs:** a single GO/NO-GO verdict + per-check dated table in
`docs/decisions/`; non-zero exit on NO-GO to block the publish/merge.

---

## Operate

### `pr-drain` · priority 7

**When to use:** any time >1 PR is open against strict `main`.

**Why it exists:** ~30 PRs/session against a strict + squash `main` makes
hand-driving the prose runbook expensive and error-prone. Squash merges
serialize, `git branch --merged` misses them, and CHANGELOG is the
recurring conflict — this mechanizes all three.

**Phases**
1. **adopt-fragments** — switch CHANGELOG to `changelog.d/` fragment-per-PR
   so orthogonal PRs land in any order (one-time, then assumed).
2. **pick-next** — choose the next mergeable PR (skip owner-gated like #177).
3. **update-branch** — `gh pr update-branch`; resolve only `changelog.d` if
   needed.
4. **wait-green** — poll CI to green (background-bash poll, not foreground
   sleep).
5. **squash-merge** — merge; re-detect the merged set via
   `gh pr list --state merged` (`git branch --merged` misses squash-merges).
6. **loop** — repeat until the queue is empty or only owner-gated remain.

**Inputs:** repo; PR exclusion list (owner-gated); poll interval.
**Outputs:** drained queue; per-PR merge log; surfaced owner-gated remainder.

### `loop-tick` · priority 6

**When to use:** the standing self-paced orchestration heartbeat (a
monitoring `/loop`).

**Why it exists:** the monitoring loop currently evaporates per session.
One cheap, persistent tick keeps the fleet moving: status → one merge step
→ GPU refill → surface gated blockers.

**Phases**
1. **status** — run `make-status` (fleet snapshot) to avoid re-deriving state.
2. **pr-drain-step** — advance `pr-drain` by exactly one merge step.
3. **gpu-refill** — if the GPU lock is free, dispatch the next queued
   `gpu-task` (1-at-a-time, behind the VRAM semaphore).
4. **surface-gated** — list owner-gated blockers (e.g. #177
   `prepublish-gate`) for the user.
5. **sleep-to-next** — schedule the next tick (self-paced cadence).

**Inputs:** inbox/status source; GPU queue; tick cadence.
**Outputs:** per-tick log — PRs advanced, GPU job dispatched, owner-gated
items surfaced.

---

## Maintaining this catalog

- Bump the **Version** header on any add/remove/phase change; this is the
  operating manual, treat it like one.
- A new workflow earns a row only once it has been *proven* — used to catch
  or prevent a real failure. Theory does not get a row.
- Keep every "why it exists" tied to a concrete moment. If you cannot name
  the moment, the workflow is not ready for the catalog.
