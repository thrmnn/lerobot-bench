# FAQ

Real questions a user, contributor, or reviewer asks about
`lerobot-bench`, with short answers. Where a question has a deep answer,
the answer points at the doc that owns it rather than duplicating it.

New to the project? Start with [`docs/GETTING_STARTED.md`](GETTING_STARTED.md).

## Contents

- [Using the benchmark](#using-the-benchmark)
- [Reproducing results](#reproducing-results)
- [Contributing a policy](#contributing-a-policy)
- [Methodology](#methodology)
- [Gotchas](#gotchas)

---

## Using the benchmark

### What does the benchmark actually measure?

Per-episode **task success** for pretrained LeRobot manipulation
policies on simulated environments. An episode runs from `env.reset()`
until the env terminates, truncates, or hits the per-env step cap; it
is scored a success when the final-step reward clears the env's
`success_threshold`. A leaderboard cell aggregates 5 seeds × 50 episodes
(N=250 binary outcomes) into a success rate with a confidence interval.
The benchmark is inference-only — no policy is trained or fine-tuned by
this project, and even the baselines (`no_op`, `random`) are
weight-free. See [`docs/DESIGN.md`](DESIGN.md) § Methodology.

### How do I read a leaderboard cell?

A cell is one `(policy, env)` pair. The headline number is its success
rate — the pooled fraction of successful episodes across all 5 seeds.
The interval next to it is a 95% confidence interval (Wilson for the
point estimate, bootstrap for distributional summaries and paired
deltas). Treat the interval, not the point estimate, as the claim: two
cells whose intervals overlap heavily are not meaningfully ranked.

### What does "n/a" mean in a cell?

It means no public pretrained checkpoint exists for that `(policy, env)`
pair at lock-in time — the cell was never runnable, not silently
skipped. A policy only runs on the envs in its `env_compat` list
(`act` on `aloha_transfer_cube`, `diffusion_policy` on `pusht`, the
LIBERO VLAs on the four `libero_*` suites), so the matrix is sparse by
design. Cross-policy comparisons are only made within shared envs. See
the v1 scope matrix in [`README.md`](../README.md) § v1 scope.

### Why is one cell's confidence interval so much wider than another's?

Two reasons. First, CI width depends on the success rate itself — a
Wilson interval is widest near p=0.5 and tightest near 0 or 1, so a
mid-range cell is intrinsically less precise than a saturated one at the
same N. Second, a few cells were **auto-downscoped** to fewer episodes
(N=50 or N=100 instead of 250) when calibration flagged slow inference;
fewer episodes means a wider interval. The exact half-widths at every
`(p, N)` combination are tabulated in [`docs/MDE_TABLE.md`](MDE_TABLE.md).

### What does an "inconclusive at N=250" pair mean?

It means the observed success-rate delta between two cells is smaller
than what sampling noise alone could produce at this sample size — the
benchmark cannot rank them. The leaderboard renders such pairs in
neutral grey and excludes them from the headline finding rather than
implying an ordering that the data does not support. The threshold is
per-cell (it scales with the success rate); see
[`docs/MDE_TABLE.md`](MDE_TABLE.md) § The "inconclusive at this N" rule.

### Why are the baseline policies (`no_op`, `random`) on the leaderboard?

They are the floor. `no_op` emits a zero action every step; `random`
emits uniform noise. A pretrained policy that does not clearly beat both
baselines on a task has not demonstrated it learned anything useful
there, so the baselines make every other cell interpretable. They are
also weight-free, which keeps the project's no-training constraint
intact without exception.

### Can I look at results without installing anything?

Yes. The [live leaderboard Space](https://huggingface.co/spaces/thrmnn/lerobot-bench)
renders the full leaderboard, paired comparisons, and a browsable
rollout video for every `(policy, env, seed, episode)`. Installation is
only needed to *run* the benchmark locally.

---

## Reproducing results

### How do I verify a published leaderboard number is real?

Run `make reproduce CELL=policy/env/seed`. It re-runs that cell on your
hardware and checks the per-episode `success` and `n_steps` against the
published parquet — **bit-for-bit**, not within error bars. A single
flipped episode is reported as a mismatch. The full walkthrough,
expected wall-clock per cell, and how to read the verdict are in
[`docs/REPRODUCE.md`](REPRODUCE.md).

### What exactly makes a cell reproducible?

The contract: same `lerobot` version + same pinned checkpoint SHA +
same seed → identical per-episode outcomes. The cell seeds NumPy and
Torch once at cell start from `seed_idx * 1000`, and resets each episode
`e` with `seed_idx * 1000 + e`. See [`docs/REPRODUCE.md`](REPRODUCE.md)
§ The contract and [`docs/DESIGN.md`](DESIGN.md) § Methodology.

### My re-run does not match the published cell. What broke?

In order of likelihood: (1) `lerobot` version drift — confirm
`pip show lerobot` reports `0.5.1`; (2) checkpoint SHA drift — the
`revision_sha` in `configs/policies.yaml` must be the exact commit the
sweep pinned; (3) genuine nondeterminism, which is a benchmark bug worth
filing as an issue. The `reproduce_cell.py` verdict names the first
divergent episode and the likely cause. See
[`docs/REPRODUCE.md`](REPRODUCE.md) § What a mismatch implies.

### Why is mid-cell resume not bit-reproducible?

Policy stochasticity inherits the Torch generator, which advances across
episodes within a cell. Resuming at episode `k` would not reproduce the
same `k..n-1` tail as a fresh run. So `checkpointing.py` resumes only at
**cell boundaries** — a cell that dies mid-run restarts from episode 0.
This is a deliberate consequence of the seeding contract, documented in
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) § Reproducibility.

---

## Contributing a policy

### How do I add a pretrained policy to the benchmark?

One PR, one YAML entry, no Python. Add a mapping under `policies:` in
`configs/policies.yaml` with the policy name, `repo_id`, a pinned
40-char `revision_sha`, and the `env_compat` list, then dry-run it with
`scripts/run_one.py --dry-run`. The exact template, every field, and how
to obtain the locked SHA are in [`CONTRIBUTING.md`](../CONTRIBUTING.md)
§ Add a policy.

### Why must `revision_sha` be a commit hash and not a branch?

A floating ref like `main` can change under you — a re-tagged or moved
Hub checkpoint produces different actions and silently breaks the
reproducibility contract. A pinned 40-char SHA is what makes a cell a
verifiable claim. The `validate-configs` CI job rejects entries without
one. See [`CONTRIBUTING.md`](../CONTRIBUTING.md) § Getting the locked
`revision_sha`.

### I do not have a GPU — can I still contribute?

Yes, for the YAML entry. A `--dry-run` resolves the policy and env
through the registries without touching weights, sim, or Torch, so you
can validate a `configs/policies.yaml` change on a CPU box. A maintainer
with a GPU runs calibration and a real cell before the policy is
admitted to a sweep. See [`CONTRIBUTING.md`](../CONTRIBUTING.md)
§ What the maintainer does to admit it.

### I just want to suggest a policy, not do the work. Where do I go?

Open a [Propose a policy](https://github.com/thrmnn/lerobot-bench/issues/new?template=propose-a-policy.yml)
issue. A maintainer or another contributor can pick it up.

---

## Methodology

### How are seeds chosen, and why 5 of them?

Five seeds × 50 episodes is the contracted design — N=250 binary
outcomes per cell. It is the bar, not an aspiration: if compute cannot
support it, scope cuts the matrix, not the seeds. Per cell, NumPy and
Torch are seeded once from `seed_idx * 1000` and each episode resets the
env with `seed_idx * 1000 + e`. See [`docs/DESIGN.md`](DESIGN.md)
§ Methodology → Seeding contract.

### What is the MDE and why does the project care about it?

MDE is the **minimum detectable effect** — the smallest success-rate
delta the benchmark can resolve at N=250 with adequate statistical
power. Headline findings only cite a delta when its magnitude exceeds
the MDE for those cells; everything tighter is reported as inconclusive.
The full per-`(p, N)` table, both the closed-form bound and the
bootstrap-simulated MDE, is [`docs/MDE_TABLE.md`](MDE_TABLE.md).

### How are two policies compared on the same env?

By the **paired** success-rate delta. Episodes are paired by
`(seed_idx, episode_idx)` — same env reset, same scenario — and a paired
bootstrap or paired Wilcoxon / McNemar test is computed over those
pairs. Pairing tightens the interval on the delta relative to treating
the cells as independent. A "significantly better" claim always carries
the paired test and an effect size (Cohen's h) alongside it. See
[`docs/DESIGN.md`](DESIGN.md) § Methodology and the `paired_*` helpers
in [`docs/API.md`](API.md).

### What is the auto-downscope rule?

A 20-step calibration probe measures per-cell step latency and VRAM. If
a cell is too slow (`mean_step_ms > 100`) or VRAM-pressured
(`vram_peak_mb > 5500`), its episode budget is trimmed so the full sweep
fits the compute window — N drops from 250 toward 100 or 50. The
downscope is recorded as a methodology footnote, never hidden; the
down-scoped N variants are tabulated in
[`docs/MDE_TABLE.md`](MDE_TABLE.md) § 3.

### How is "success" defined for an episode?

`success = (final_reward >= env_spec.success_threshold)` — the reward at
the last step before the episode exits. This matches Aloha/LIBERO
task-complete semantics directly; for PushT (whose reward tracks
coverage and is monotonically related to it) it is a defensible, if
slightly conservative, score. The choice is stated in the public
methodology and in the [`eval`](API.md#module-eval) module docstring.

### How is a failure categorised?

Failed rollouts are hand-labelled into a fixed taxonomy (trajectory
overshoot, gripper slip, timeout, wrong-object, premature release,
drift) and rendered as a per-policy bar chart. The labels live in a
`labels.json` next to each MP4. The labeling template and rationale are
in `docs/FAILURE_TAXONOMY.md`.

---

## Gotchas

### Rendering fails on a headless box (`GLFW` / `EGL` / `DISPLAY` errors).

The Aloha and LIBERO envs render through MuJoCo, which needs an OpenGL
context. Pick a backend with `MUJOCO_GL` before launching: `osmesa`
(offscreen software, most portable for servers and CI) or `egl`
(NVIDIA hardware offscreen, no display). Under WSLg, export `DISPLAY=:0`
and let MuJoCo use the default GLX backend. See
[`docs/GETTING_STARTED.md`](GETTING_STARTED.md) § Common issues. For a
deeper render-failure walkthrough see `docs/TROUBLESHOOTING.md`.

### Why is `lerobot` pinned to exactly `0.5.1`?

Because the reproducibility contract depends on it. A different
`lerobot` release can change env dynamics or policy inference and break
bit-for-bit determinism — version drift is the single most common cause
of a non-reproducing cell. The pin is recorded in `pyproject.toml` and
the Space's `requirements.txt`. If `lerobot.__version__` prints anything
other than `0.5.1`, recreate the conda env before running anything.

### Why is the Pi0 family not on the leaderboard?

The Pi0 policies (`pi0_libero`, `pi0fast_libero`,
`pi05_libero_finetuned_v044`) are **deferred to v1.1**. On the reference
hardware (32 GB host RAM, WSL2) they spike to roughly 30 GB of CPU RAM
during `from_pretrained` cold load — HF Transformers' default
weight-conversion path materializes the full model on the host before
moving it to GPU — which overflows the host budget. v1.1 will revisit
them with quantized weights or an `accelerate device_map="auto"`
streaming load. This is stated in [`README.md`](../README.md) § v1 scope
and the paper's Limitations section, not papered over.

### A `run_one.py` cell exits non-zero. What does the code mean?

The exit code is the diagnosis: `0` clean, `2` ran but some episodes
errored (rows still appended), `3` policy not runnable (missing
`revision_sha`), `4` missing runtime (`lerobot` or sim extras not
installed), `5` policy/env not in the registry or env not in the
policy's `env_compat`. The full table is in
[`docs/GETTING_STARTED.md`](GETTING_STARTED.md) § Your first result.

### A full sweep ran out of memory. What is the safety net?

Heavy workloads run under a kernel-enforced 18 GB cgroup memory cap via
`scripts/run_capped.sh`, and a pre-flight gate refuses to launch when
baseline RAM is already above 55% used. For the in-sweep OOM playbook
(fp16 fallback, VRAM-creep handling, dropping a policy) see
[`docs/RUNBOOK.md`](RUNBOOK.md) § OOM playbook.

### Can I run the benchmark on CPU?

You can, but it is not the configuration the leaderboard was measured
on. `scripts/run_one.py` defaults to `--device cuda`; pass `--device
cpu` to force it. CPU runs are much slower and a CPU-only re-run is not
a valid bit-for-bit reproduction of a GPU-measured cell. See
[`docs/GETTING_STARTED.md`](GETTING_STARTED.md) § Common issues.

### Where do results go, and why is `results/` empty in the repo?

`results/` is gitignored. A local run writes `results/results.parquet`
(one row per episode) and `results/videos/*.mp4`. The published sweep's
artifacts live on the HF Hub dataset
[`thrmnn/lerobot-bench-results-v1`](https://huggingface.co/datasets/thrmnn/lerobot-bench-results-v1);
pull them with `huggingface-cli download`. See
[`docs/REPRODUCE.md`](REPRODUCE.md) § Prerequisites.
