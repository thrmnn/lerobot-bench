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
| #15 | `space/app.py` + `space/requirements.txt` — Gradio Space | spaces-frontend-engineer | **this PR** |
| #16 | `notebooks/01-write-finding.ipynb` — analysis scaffold | researcher-writeup (+ stats-rigor-reviewer veto) | pending |
| #17 | `paper/main.tex` + `paper/references.bib` — arxiv template | researcher-writeup | pending |
| #18 | `docs/FAILURE_TAXONOMY.md` — labeling template | researcher-writeup | pending |

After #18: Path A is exhausted; Path B (lerobot install + revision_sha lockin)
becomes critical for any further progress.

## Resume now

PR #15 (`space/app.py` + `space/_helpers.py` + `space/requirements.txt`)
lands the public Gradio surface at the top of the read stack: three tabs
(Leaderboard with Wilson 95% CIs, Browse-Rollouts with direct Hub video
URLs, Methodology), helpers split out of `app.py` so the project's
pytest fast job exercises the data layer without pulling Gradio. The
space-smoke workflow boots the app and curls `/` on every PR that
touches `space/**`. Next: PR #16 (`notebooks/01-write-finding.ipynb`)
— the analysis scaffold. Path B (Day 0a auth + revision_sha lockin) is
independent and unblocks pretrained policies for `eval.load_policy`.

---

## Day 0a — auth, lockin, novelty (HUMAN — blocks live policy evaluation)

Owner: human (decisions and credential steps Claude cannot make).

- [ ] `huggingface-cli login` (write scope) — token saved to `~/.cache/huggingface/token`.
- [ ] `wandb login` (optional; only if W&B will track runs). **Reminder:** rotate the API key shared in chat on 2026-04-30 before logging in.
- [ ] Resume the lerobot install: `pip install -e /home/theo/projects/lerobot` in the `lerobot` conda env (until `lerobot==0.5.1` lands on PyPI). Confirm `python -c "import lerobot; print(lerobot.__version__)"` prints `0.5.1`.
- [ ] Lock the **5 policy repo IDs + revision SHAs** into `docs/MODEL_CARDS.md` AND `configs/policies.yaml`. The `revision_sha: null` entries in the YAML block `PolicySpec.assert_runnable()`.
- [ ] Add SmolVLA + Pi0 entries to `configs/policies.yaml` once their HF Hub repo IDs are picked.
- [ ] Verify `lerobot.envs.<env>.config.SUCCESS_REWARD` exists in 0.5.1; if it does, switch `envs.py` to read from there instead of the hardcoded YAML thresholds (or document the choice to keep the YAML authoritative).
- [ ] **10-min novelty search** on HF Hub + Google Scholar for "lerobot benchmark" / "multi-policy lerobot eval". If something close exists, reposition before any code lands.
- [x] **Resolved 2026-04-30**: GitHub repo owner is `thrmnn`; SSH alias `github-thrmnn` configured.

## Day 0b — calibration spike (Claude scaffolds, human runs)

Owner: `sweep-sre` agent for the script; human runs it on the dev box.

- [x] Scaffold `scripts/calibrate.py` that loads each locked policy, runs 20 steps × 1 episode per (policy, available_env), writes `results/calibration-YYYYMMDD.json` with `{policy, env, mean_ms_per_step, p95_ms, vram_peak_mb}`. Fail loudly on OOM with a one-line resume command. **Done in PR #7 — scaffold only; inner measurement loop is a TODO that completes after Day 0a lerobot install.**
- [x] Scaffold `configs/sweep_full.yaml` — empty matrix, populated from calibration output via the auto-downscope rule. **Done in PR #12 — ships with baselines + pretrained policies enumerated; `overrides: {}` waits for calibration JSON.**
- [ ] Run `make calibrate` on the dev box. Inspect output. Lock matrix shape. **(Path B / Day-0a-blocked: requires `lerobot==0.5.1` installed and pretrained `revision_sha` values locked in `configs/policies.yaml`.)**

**Exit criterion:** `results/calibration-YYYYMMDD.json` on disk, final matrix shape committed under `configs/`.

## Day 1 — Libero spike + smoke action

Owner: `bench-eval-engineer` agent, human-supervised.

- [ ] **4-hour cap** on Libero install on WSL2. If `pip install gym-libero` (or upstream equivalent) plus a single rollout fails inside 4 hours, drop Libero from v1 and proceed with PushT + Aloha. No exceptions.
- [ ] If Libero in: extend `configs/envs.yaml` with the locked Libero task variant(s).
- [ ] Verify each locked policy can produce one action tensor against a fresh env reset for each (policy, env) cell. Assert shape against env action space.

**Exit criterion:** every locked (policy, env) cell produces a valid action; Libero in/out decision committed to `docs/MODEL_CARDS.md` and `configs/envs.yaml`.

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
3. **HF username confirmation**: rebrand assumed HF Hub username = `thrmnn` (matches GH). If HF account is different, run `huggingface-cli whoami` after login and submit a follow-up rebrand PR — only `docs/DESIGN.md`, `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md` reference HF paths (dataset + Space).
4. **`lerobot==0.5.1` not on PyPI**: install path on the dev box is `pip install -e /home/theo/projects/lerobot`. CI `smoke.yml` will fail until upstream PyPI release lands; logged as `::warning::` for now. Day 0a unblocks local install.
5. **Auth not yet performed**: `huggingface-cli login` (write scope, needed for publish) and `wandb login` (optional). Day 0a item. **Wandb API key from chat on 2026-04-30 must be rotated before login.**
6. **Sim extras not yet installed** in the lerobot env (`gym-pusht`, `gym-aloha`, optional Libero). Day 1 item.
