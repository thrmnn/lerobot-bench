# Failure taxonomy — labeling template

Status: TEMPLATE (filled in on Day 7 of `docs/CEO-PLAN.md`, after the
full sweep produces real rollouts).
Owner: researcher-writeup agent + the human.
Source: `docs/DESIGN.md` § Failure modes (six modes), `docs/CEO-PLAN.md`
§ Scope decisions § D1.

## Why this exists

A leaderboard says *what* succeeds and *what* doesn't. A failure
taxonomy says *how* a policy fails when it fails — and that is the
diagnostic depth a HF Robotics reviewer reads for. The headline finding
in the writeup is one sentence; the taxonomy bar chart is what makes
that sentence specific. Without it the artifact is a benchmark; with it
it's a benchmark with a thesis.

The taxonomy folds into the writeup as a **horizontal stacked bar
chart per policy** (one bar per `(policy, env)` cell, bar segments are
the six failure modes proportionally normalized to the cell's failure
count). A reviewer who reads only the chart should be able to say "this
policy fails by overshoot, that one fails by drift" — and that
statement should match the verbal description in the Discussion section.

## The six modes

These are the canonical labels. Add new modes only if a rollout
genuinely cannot be assigned to any of them; resist the urge to invent
sub-categories — the chart legend has to stay readable on the Space.

### 1. Trajectory overshoot

**Definition.** Agent moves past the target object or zone and either
fails to correct or corrects too late to satisfy the success threshold
within `max_steps`. Most often manifests as the end-effector swinging
through the target on a high-velocity motion.

**Heuristic.** Look for the end-effector entering the success region,
then leaving it without the env emitting a terminal reward. In the
rendered MP4 this is visible as the gripper visibly passing through or
beyond the goal position.

**Example timestamp (hypothetical Aloha transfer cube).** Around step
180 of a 400-step episode: the gripper closes on the cube successfully,
moves toward the receptacle, but the lift trajectory carries it 3-5 cm
past the receptacle's centre and the cube falls outside the success
threshold's bounding box.

### 2. Gripper slip

**Definition.** Agent grasps the target successfully but loses contact
mid-trajectory before the success condition is met. Distinct from
**premature release** in that the gripper is still commanded closed —
the slip is a physics outcome, not a control decision.

**Heuristic.** Look for a successful grasp event followed by the object
falling out of the gripper while the gripper state remains "closed."
In Aloha rollouts, watch for the cube dropping vertically while both
arms are in motion.

**Example timestamp (hypothetical Aloha transfer cube).** Around step
220: gripper closed at step 150, lift initiated, but at step 220 the
cube is visible falling toward the table while the gripper jaws are
still in their commanded-closed pose.

### 3. Timeout

**Definition.** Agent runs out of `env.max_steps` (PushT 300, Aloha
400, Libero 600) without ever satisfying the success threshold. Includes
both "still trying" and "stuck" — distinguish from **drift** (mode 6)
by whether the agent is making non-zero progress toward the goal.

**Heuristic.** Episode terminates with `truncated=True` and
`final_reward < env.success_threshold`, AND the action sequence in the
last 50 steps shows non-trivial variance (i.e. the agent is still
issuing meaningful actions, just not reaching the goal in time).

**Example timestamp (hypothetical PushT).** Episode terminates at step
300 with reward 0.78 (below the 0.95 success threshold). The agent has
been pushing the T-shape toward the goal region for the entire episode
but cannot quite get the orientation right; the final 50 steps show the
agent oscillating between two near-goal poses.

### 4. Wrong object

**Definition.** Agent interacts with a non-target object — picks up the
wrong cube, pushes the wrong shape, or attends to a distractor. Most
relevant for envs with multiple objects in the scene (Aloha, Libero);
not applicable to PushT (single object).

**Heuristic.** First grasp / push event in the rollout targets an
object that is not the env's success-threshold target. In the Aloha
transfer cube task, watch for the agent grasping the receptacle's
indicator cube instead of the transfer cube.

**Example timestamp (hypothetical Aloha multi-object task).** Step 90:
the gripper closes on the red distractor block instead of the green
target block specified by the language conditioning. Subsequent steps
attempt to "transfer" the wrong block and the env never registers
success.

### 5. Premature release

**Definition.** Agent opens the gripper or releases the held object
before the success zone is reached. Distinct from **gripper slip** in
that the gripper command itself transitions to "open" while the agent
is mid-trajectory — this is a control failure, not a physics failure.

**Heuristic.** Look for a `gripper_open` action issued while the
end-effector is outside the success region AND a held object is dropped
as a result. In Libero rollouts, this often presents as the policy
"giving up" on a long-horizon task halfway through.

**Example timestamp (hypothetical Libero pick-and-place).** Step 340 of
600: the policy issues `gripper.open=1` while the held mug is still 15
cm above the target shelf; the mug drops to the table and the episode
times out without success.

### 6. Drift

**Definition.** Agent action collapses to a stationary or oscillatory
mode that makes no progress toward the goal. Includes (a) actions
collapsing to near-zero magnitude (the policy "freezes"), (b) actions
oscillating between two poses without net displacement, and (c) the
end-effector wandering away from the workspace entirely.

**Heuristic.** In the last 100 steps of a failed episode, the
end-effector position variance is near zero AND the cumulative reward
delta is < 0.05. Distinguishes from **timeout** by the *quality* of the
final actions: timeout = "trying but not getting there," drift = "no
longer trying meaningfully."

**Example timestamp (hypothetical SmolVLA on Aloha).** From step 150
onward: the policy emits actions whose joint velocity norms collapse to
near-zero. The arms remain in roughly the same pose for the remaining
250 steps, and the episode ends with reward 0.0 and `truncated=True`.

## Labeling protocol

1. **Sample size:** **5 failed rollouts per `(policy, env)` cell** at
   minimum, **10 if there is ambiguity** (i.e. if the first 5 rollouts
   landed in 4 or more different modes — the cell needs more samples to
   stabilise the proportions).
2. **First-fit rule:** the six modes are *not* mutually exclusive — a
   gripper slip can lead to a timeout, a wrong-object grasp can drift
   into a premature release. Label each rollout with the **first mode
   that fits in chronological order**. This keeps the per-cell mode
   counts summing to the total failure count and the bar chart legible.
   The trade-off is that rare composite modes (e.g. "slip then timeout")
   collapse into the earlier label; document any cell where this matters
   in the rollout's Notes column.
3. **Sampling strategy:** stratify across seeds. If a cell has 5 seeds
   with 10 failures each, draw 1 failure from each seed, not 5 failures
   from one seed. This avoids one bad seed dominating the chart.
4. **Render dependency:** labeling requires the per-episode MP4. The
   labeler watches the video, scrubs to the failure event, assigns the
   first-fit mode. The MP4 SHA-256 in the parquet's `video_sha256`
   column is the canonical reference — labels are joined back to the
   parquet on `(policy, env, seed, episode_index)`.
5. **When in doubt, prefer the more diagnostic mode.** "Drift" and
   "Timeout" can both fit a stuck rollout; pick "Drift" if the actions
   are clearly degenerate, "Timeout" if the policy is still making
   sensible attempts.
6. **Two-labeler check (stretch):** for the cells the writeup
   highlights, have a second pass on the same rollouts and report
   inter-labeler agreement (Cohen's kappa) in the methods. This is a
   stretch; v1 is single-labeler.

## When this happens

Day 7 of `docs/CEO-PLAN.md` § Updated timeline. By that point the full
sweep (Day 5-6) has produced real rollouts, the Space (Day 4) is live
with the Browse-Rollouts tab so the labeler can scrub through MP4s
quickly, and the writeup (Day 8) needs the labeled CSV to render the
taxonomy bar chart. Path A (the work that ships before lerobot install)
gets to this template; the actual labels land on Day 7.

## CSV template

Fill this in as labels are produced. Append-only — never overwrite a
row, edit the Notes column instead. The bar chart in the writeup reads
straight from this CSV (no intermediate transformation).

| labeled_by | date_iso | policy | env | seed | episode_index | video_sha256 | mode | notes |
|---|---|---|---|---|---|---|---|---|
| _example_ | 2026-05-09 | diffusion_policy | aloha_transfer_cube | 2 | 17 | _abcdef..._ | trajectory_overshoot | gripper closes around step 150, swings 4cm past receptacle |
| _example_ | 2026-05-09 | diffusion_policy | aloha_transfer_cube | 2 | 23 | _123456..._ | gripper_slip | grasp at step 140, cube drops at step 210 mid-lift |
| _example_ | 2026-05-09 | act | pusht | 0 | 8 | _789abc..._ | timeout | T-shape oscillates around 0.85 reward, never reaches 0.95 |
| _example_ | 2026-05-09 | smolvla | aloha_transfer_cube | 1 | 4 | _def012..._ | drift | actions collapse to near-zero from step 150 onward |
| _example_ | 2026-05-09 | pi0 | libero_object | 3 | 11 | _345678..._ | premature_release | gripper opens at step 340, mug drops 15cm above target |

**Mode labels** (use exactly these strings — the bar chart's legend is
keyed on them):

* `trajectory_overshoot`
* `gripper_slip`
* `timeout`
* `wrong_object`
* `premature_release`
* `drift`

## Cross-references

* Failure modes catalogued in `docs/DESIGN.md` § Failure modes.
* CEO-plan rationale in `docs/CEO-PLAN.md` § Scope decisions § D1.
* Per-policy entries in `docs/MODEL_CARDS.md` § Failure modes — the
  per-policy section is populated from this taxonomy on Day 7.
* The bar chart cell lives in `notebooks/01-write-finding.ipynb`
  (placeholder until labels exist).
* The writeup section that interprets the chart lives in
  `paper/main.tex` § Results.
