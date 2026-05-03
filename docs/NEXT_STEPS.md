# Next steps

Live execution checklist. Updated as work lands. Source of truth for
"what gets touched in the next PR." The strategic plan is in
`docs/CEO-PLAN.md`; the technical spec is in `docs/DESIGN.md`; this is
the runway between them.

## Status as of 2026-05-03

**Merged to main (11 PRs):**
- PR #1 — agent team, hooks, CI evolution, runbook.
- PR #2 — dependabot batch (5 GitHub Actions major bumps, all green).
- PR #3 — `stats.py`: `bootstrap_ci`, `paired_delta_bootstrap`, `paired_wilcoxon`, `cohens_h`, `wilson_ci`. 20 tests against analytical references.
- PR #4 — `envs.py` + `policies.py` with strict YAML loaders. `configs/envs.yaml` ships PushT + Aloha; `configs/policies.yaml` ships baselines + DiffPolicy/ACT (with `revision_sha: null` until Day 0a lockin).
- PR #5 — EOD checkpoint doc.
- PR #6 — `render.py`: episode → MP4 (256px / 10fps / H.264 / ≤2MB) + thumbnail strip; pure imageio.v3 + libx264.
- PR #7 — `checkpointing.py`: cell-boundary parquet resume layer.
- PR #8 — `scripts/calibrate.py` Day 0b calibration spike scaffold (inner measurement loop is a TODO until lerobot install).
- PR #9 — `eval.py` orchestration core. Seeding contract enforced; baselines fully runnable; pretrained loading is a Day 0b TODO.
- PR #10 — `scripts/run_one.py` single-cell CLI shell with atomic parquet append, lazy render import, AST-guarded dry-run.
- PR #11 — `docs/PATH_B_INTEGRATION_SMOKE.md` + NEXT_STEPS sync (Path B integration smoke checklist).

**In flight:**
- PR #15 — `space/app.py` + `space/_helpers.py` + `space/requirements.txt` Gradio Space (this PR).

**Local state:** lerobot conda env has `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, `scipy`, `imageio[ffmpeg]`, `types-PyYAML`, `pandas-stubs` installed. `make all` green. **236 tests passing** with PR #15 in (PR #15 added 17 in `tests/test_space.py` on top of PR #13's 219). lerobot itself is NOT yet installed — Day 0a item; the Space deliberately does not need it.

## Path A queue (committed plan, no human input needed)

These ship without lerobot installed. Everything tests against synthetic data.

| PR | File(s) | Owner agent | Status |
|---|---|---|---|
| #9  | `src/lerobot_bench/eval.py` | bench-eval-engineer | merged |
| #10 | `scripts/run_one.py` | sweep-sre | merged |
| #12 | `scripts/run_sweep.py` — matrix orchestrator + resume drill | sweep-sre | merged |
| #14 | `scripts/publish_results.py` — HF Hub upload | sweep-sre | merged |
| #15 | `space/app.py` + `space/requirements.txt` — Gradio Space | spaces-frontend-engineer | merged |
| #16 | `docs/FAILURE_TAXONOMY.md` — labeling template | researcher-writeup | merged |
| #17 | `notebooks/01-write-finding.ipynb` — analysis scaffold | researcher-writeup (+ stats-rigor-reviewer veto) | merged |
| #18 | `paper/main.tex` + `paper/references.bib` — arxiv template | researcher-writeup | merged |

**Path A is exhausted with PR #18.** Path B (lerobot install +
revision_sha lockin) is now the critical path — pretrained
`eval.load_policy` and the live calibration spike both block on it.
The benchmark sweep + writeup now wait for the human-driven Day 0a
+ Day 0b items below.

### Premortem mitigations status

Tracked individually because each one is a "thing that ships before
the sweep so the sweep does not produce a wrong number nobody catches".

| # | Mitigation | Status | Artifact |
|---|---|---|---|
| 4 | Pre-sweep MDE table at N=250 + auto-downscope variants + per-cell inconclusive gate in the notebook + paper Methods precise-value update | **done** | `docs/MDE_TABLE.md`, `scripts/calibrate_mde.py`, `tests/test_mde_consistency.py`, `notebooks/01-write-finding.ipynb` cell 5b, `paper/main.tex` § Methods MDE-bound paragraph |
| 5 | Render ladder for oversized Aloha episodes | **done** | `src/lerobot_bench/render.py` `RENDER_LADDER` (PR earlier) |

## Resume now

PR #18 (`paper/main.tex` + `paper/references.bib` + `paper/Makefile` +
`paper/.gitignore`) lands the 4-page arxiv writeup scaffold (cs.RO
primary, cs.LG secondary). Standard 11pt article class, no boutique
template. Five sections: Abstract (headline-finding sentence is a
`\todo` so the writer cannot ship a fabricated claim), Introduction
(open data / open eval / open analysis), Methods (full sweep contract
+ seeding equation + bootstrap protocol + MDE bound at $N{=}250$),
Results (leaderboard / forest / paired / taxonomy placeholders each
citing the producing notebook cell), Discussion (5-bullet limitations
+ future work + upstream PR pointer). 10-entry references.bib with
real arxiv IDs / DOIs / ISBNs. Build verified: `make` produces a 6-page
PDF (4 body + abstract + bibliography), no errors.

**Path A is exhausted with PR #18.** Next live work is human-driven:
Day 0a (`huggingface-cli login`, `wandb login`, lock 5 policy repo
IDs + revision SHAs in `configs/policies.yaml` and
`docs/MODEL_CARDS.md`) unblocks pretrained `eval.load_policy`; Day 0b
(`make calibrate` on the dev box) locks the matrix shape; Day 1
(Libero spike) decides the env count; Days 5-6 run the full sweep
overnight; Days 7-8 fill in the notebook + paper placeholders with
real numbers.

---

## Day 0a — auth, lockin, novelty (HUMAN — blocks live policy evaluation)

Owner: human (decisions and credential steps Claude cannot make).

- [x] **2026-05-03**: `huggingface-cli login` done. HF whoami: `Theozinh0` (different from GH owner `thrmnn`). All hardcoded HF Hub paths rebranded `thrmnn/lerobot-bench-results-v1` → `Theozinh0/lerobot-bench-results-v1` in `space/`, `scripts/publish_results.py`, `docs/{DESIGN,ARCHITECTURE,RUNBOOK}.md`. GitHub URLs (`github.com/thrmnn/lerobot-bench`) remain correct since `thrmnn` owns the GH repo.
- [x] **2026-05-03**: wandb API key rotated; `wandb login` done with new key.
- [x] **2026-05-03**: lerobot installed via `pip install -e "/home/theo/projects/lerobot[pusht,aloha]"` in the `lerobot` conda env. `lerobot==0.5.1` confirmed. Torch CUDA initially failed against the system's CUDA 12.1 toolkit — resolved by installing torch 2.10.0 with cu126 wheel from `https://download.pytorch.org/whl/cu126`.
- [x] **2026-05-03**: Locked diffusion_policy + act revision SHAs in `configs/policies.yaml` and `docs/MODEL_CARDS.md`. **Also corrected the env_compat lists**: `diffusion_pusht` is PushT-only and `act_aloha_sim_transfer_cube_human` is Aloha-only (the prior `[pusht, aloha_transfer_cube]` listing on both was incorrect and would have produced nonsensical eval cells).
- [x] **2026-05-03**: Added 5 VLA libero finetune entries to `configs/policies.yaml` with locked Hub revision SHAs: `pi05_libero_finetuned_v044`, `pi0_libero_finetuned_v044`, `pi0fast_libero`, `xvla_libero`, `smolvla_libero`. All 5 carry `env_compat` for the 4 LIBERO suites. SHAs locked via `huggingface_hub.HfApi().model_info(repo_id).sha` (see Libero v2 PR for exact values; mirrored in `docs/MODEL_CARDS.md`). License flags: pi0 / pi0.5 are gemma-licensed (review terms before redistribution); pi0fast license unspecified on the Hub card (treat as all-rights-reserved until clarified upstream); xvla + smolvla are apache-2.0.
- [ ] Verify `lerobot.envs.<env>.config.SUCCESS_REWARD` exists in 0.5.1; if it does, switch `envs.py` to read from there instead of the hardcoded YAML thresholds (or document the choice to keep the YAML authoritative).
- [x] **2026-05-03**: 10-min novelty search done. No competing public lerobot multi-policy leaderboard exists on HF Datasets or Spaces. Two `lerobot/video-benchmark-*` datasets exist but are video-codec benchmarks, unrelated to policy evaluation. Framing holds; no pivot.
- [x] **Resolved 2026-04-30**: GitHub repo owner is `thrmnn`; SSH alias `github-thrmnn` configured.

## Day 0b — calibration spike (Claude scaffolds, human runs)

Owner: `sweep-sre` agent for the script; human runs it on the dev box.

- [x] Scaffold `scripts/calibrate.py` that loads each locked policy, runs 20 steps × 1 episode per (policy, available_env), writes `results/calibration-YYYYMMDD.json` with `{policy, env, mean_ms_per_step, p95_ms, vram_peak_mb}`. Fail loudly on OOM with a one-line resume command. **Done in PR #7 — scaffold only; inner measurement loop is a TODO that completes after Day 0a lerobot install.**
- [x] Scaffold `configs/sweep_full.yaml` — empty matrix, populated from calibration output via the auto-downscope rule. **Done in PR #12 — ships with baselines + pretrained policies enumerated; `overrides: {}` waits for calibration JSON.**
- [x] **2026-05-03**: Pretrained `eval.load_policy` branch wired up in `src/lerobot_bench/eval.py` (`_LerobotPolicyAdapter` + `_load_pretrained_policy` + `_recover_dataset_stats_from_safetensors` + `_gym_obs_to_batch`). Verified end-to-end: `scripts/run_one.py --policy diffusion_policy --env pusht --seed {0,1,2} --n-episodes 1` → 3 valid parquet rows, success rate 2/3 across the 3 sample seeds. `configs/envs.yaml` extended with `gym_kwargs.obs_type: pixels_agent_pos` for both shipped envs (the obs format every pretrained policy was trained on; baselines are obs-shape-agnostic so unaffected). The historical `lerobot.common.policies.factory` import path documented in PATH_B_INTEGRATION_SMOKE.md was wrong — real path is `lerobot.policies.factory`, and the call shape uses `PreTrainedConfig.from_pretrained` + `get_policy_class` + `make_pre_post_processors` rather than the historical one-shot `make_policy(repo_id=..., revision=..., fp_precision=...)` signature. PATH_B_INTEGRATION_SMOKE.md Step 3 updated with the actual path and the two non-obvious gotchas (factory side-effect import; legacy safetensors normalization recovery).
- [ ] Run `make calibrate` on the dev box. Inspect output. Lock matrix shape. **(Path B / Day-0a-blocked: requires `lerobot==0.5.1` installed and pretrained `revision_sha` values locked in `configs/policies.yaml`. Both done; calibration is now unblocked.)**

**Exit criterion:** `results/calibration-YYYYMMDD.json` on disk, final matrix shape committed under `configs/`.

## Day 1 — Libero spike + smoke action

Owner: `bench-eval-engineer` agent, human-supervised.

- [x] **2026-05-03**: Libero install + factory integration done. `hf-libero==0.1.3` was already installed alongside lerobot (transitive dep). `EnvSpec` extended with `factory` / `factory_kwargs` fields and `eval.load_env` learned a factory dispatch path; `_DebatchedVecEnvAdapter` wraps the size-1 vec env that lerobot's libero factory returns into the single-env API the cell loop expects. `_ensure_libero_setup` writes `~/.libero/config.yaml` non-interactively before any libero-touching import (libero's `__init__` calls `input()` on first run otherwise). All 4 LIBERO suites (`libero_spatial`/`libero_object`/`libero_goal`/`libero_10`) added to `configs/envs.yaml` with `task_ids=[0]` (one task per suite for v1) and the canonical `agentview_image,robot0_eye_in_hand_image` 2-camera setup. End-to-end smoke test: smolvla_libero on libero_spatial, seed 0 → success=True, n_steps=79, episode wallclock=12.75s. Libero is **in** for v1.
- [x] Each locked VLA policy x LIBERO suite cell is now factory-eligible (5 × 4 = 20 cells of zero-competition VLA comparison). Calibration on the dev box still pending — `make calibrate` will probe latency now that the loaders work end-to-end.

**Exit criterion:** every locked (policy, env) cell produces a valid action; Libero in/out decision committed to `docs/MODEL_CARDS.md` and `configs/envs.yaml`. **Done 2026-05-03.**

## Days 2-3 — core eval + mini sweep

Owner: `bench-eval-engineer` (lib), `sweep-sre` (orchestration), `render-pipeline-engineer` (videos), `stats-rigor-reviewer` (CI math).

- [ ] `src/lerobot_bench/eval.py`: the `(policy, env, seed, n_eps) → CellResult` core. Seeding contract from DESIGN.md § Methodology.
- [x] ~~`src/lerobot_bench/stats.py`~~ — landed in PR #3 (2026-04-30).
- [ ] `src/lerobot_bench/render.py`: episode → MP4 (256 px / 10 fps / H.264 / ≤2MB). **Next PR.**
- [ ] `src/lerobot_bench/checkpointing.py`: cell-boundary skip on resume.
- [ ] One full cell — 1 policy × 1 env × 1 seed × 5 episodes — produces a row in `results.parquet` and one MP4 ≤ 2MB.
- [ ] Mini sweep: 3 policies × 2 envs × 2 seeds × 25 episodes. Push to HF Hub dataset. Verify Hub read.

## Day 4 — Spaces app + resume drill

Owner: `spaces-frontend-engineer`, with `sweep-sre` for the resume test.

- [x] `space/app.py` with three tabs: Leaderboard, Browse Rollouts, Methodology. Direct Hub URLs on `gr.Video`. **Done in PR #14 — pure-Python helpers split into `space/_helpers.py` (gradio-free, AST-guarded) so `tests/test_space.py` runs in the project's pytest fast job.**
- [x] `space/requirements.txt` pinning `gradio>=5,<6` + project pulled via `git+https://github.com/thrmnn/lerobot-bench@<sha>`. **Done in PR #14. Note: `lerobot==0.5.1` is intentionally NOT installed on the Space — the Space only reads parquet + MP4 URLs from the Hub dataset, no policy inference, no torch needed; pin is documented in the Methodology tab and reproducibility lives in `pyproject.toml`.**
- [x] `make space-deploy` targets the HF Spaces git remote. **Already in Makefile from PR #1; PR #14 ships the directory it pushes.**
- [x] Resume drill: kill `run_sweep.py` mid-cell, restart, confirm the killed cell restarts from episode 0 cleanly. **Done in PR #12 — `tests/test_resume_drill.py` covers cold start, mid-sweep `KeyboardInterrupt` + clean resume, partial-cell drop-and-rerun, OOM continue, dry-run, idempotent re-resume; the live `kill -9` rehearsal still wants a wet run on the dev box.**

## Days 5-10 — full sweep, fine-tune track, writeup, ship

Tracked at the day level in `docs/CEO-PLAN.md` § Updated timeline. Each
day's work is a separate PR; that table is authoritative.

## Standing items (every PR)

- [ ] `make all` passes locally — the git-push hook gates on it.
- [ ] CHANGELOG.md updated under `[Unreleased]` if user-visible.
- [ ] Conventional Commits in PR title.
- [ ] Tests added for new code paths. Sim/GPU tests carry the right pytest mark.
- [ ] No writes to `results/`, `*.parquet`, `*.mp4` from library code (the hook will block; but library code shouldn't try anyway).

## Open questions / blockers

1. ~~**GitHub remote** — resolved, owner is `thrmnn`.~~
2. ~~**lerobot env Python tooling** — resolved.~~
3. ~~**HF username confirmation**~~ — resolved 2026-05-03: HF whoami is `Theozinh0`, all hardcoded paths rebranded.
4. **`lerobot==0.5.1` not on PyPI**: install path on the dev box is `pip install -e /home/theo/projects/lerobot`. CI `smoke.yml` will fail until upstream PyPI release lands; logged as `::warning::` for now. Day 0a unblocks local install.
5. **Auth not yet performed**: `huggingface-cli login` (write scope, needed for publish) and `wandb login` (optional). Day 0a item. **Wandb API key from chat on 2026-04-30 must be rotated before login.**
6. **Sim extras not yet installed** in the lerobot env (`gym-pusht`, `gym-aloha`, optional Libero). Day 1 item.
