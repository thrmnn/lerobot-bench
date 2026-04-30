# Next steps

Live execution checklist. Updated as work lands. Source of truth for
"what gets touched in the next PR." The strategic plan is in
`docs/CEO-PLAN.md`; the technical spec is in `docs/DESIGN.md`; this is
the runway between them.

## Day 0a — auth, lockin, novelty (next 1-2 hours of focused work)

Owner: human (these are decisions and credential steps Claude cannot make).

- [ ] `huggingface-cli login` (write scope) — token saved to `~/.cache/huggingface/token`.
- [ ] `wandb login` (optional; only if W&B will track runs).
- [ ] Resume the paused `pip install -e ".[all]"` in the `lerobot` conda env. Confirm `pip show lerobot-bench` resolves and `lerobot-bench --version` prints.
- [ ] Install dev tools in the same env: `pip install -e ".[dev]"`. `make typecheck` must work locally.
- [ ] Lock the **5 policy repo IDs + revision SHAs** and write them into `docs/MODEL_CARDS.md`. The fields currently marked `TBD` block the sweep config.
- [ ] Verify `lerobot.envs.<env>.config.SUCCESS_REWARD` exists in 0.5.1; otherwise pin the hardcoded thresholds (PushT 0.95, Aloha 1.0, Libero 1.0) in `envs.py` with a docstring stating the verification source.
- [ ] **10-min novelty search** on HF Hub + Google Scholar for "lerobot benchmark" / "multi-policy lerobot eval". If something close exists, reposition before any code lands.
- [ ] Decide: GitHub repo owner — `theoh-io` (per README) or `thrmnn` (per current `gh auth`)? This affects the remote URL in `pyproject.toml` and badges in `README.md`. **Currently blocking the first PR push.**

## Day 0b — calibration spike (Claude can scaffold, human runs)

Owner: `sweep-sre` agent for the script; human runs it on the dev box.

- [ ] Scaffold `scripts/calibrate.py` that loads each locked policy, runs 20 steps × 1 episode per (policy, available_env), writes `results/calibration-YYYYMMDD.json` with `{policy, env, mean_ms_per_step, p95_ms, vram_peak_mb}`. Fail loudly on OOM with a one-line resume command.
- [ ] Scaffold `configs/sweep_full.yaml` — empty matrix, populated from calibration output via the auto-downscope rule.
- [ ] Run `make calibrate` on the dev box. Inspect output. Lock matrix shape.

**Exit criterion:** `results/calibration-YYYYMMDD.json` on disk, final matrix shape committed under `configs/`.

## Day 1 — Libero spike + smoke action

Owner: `bench-eval-engineer` agent, human-supervised.

- [ ] **4-hour cap** on Libero install on WSL2. If `pip install -e ".[libero]"` plus a single rollout fails inside 4 hours, drop Libero from v1 and proceed with PushT + Aloha. No exceptions.
- [ ] Scaffold `src/lerobot_bench/{envs,policies}.py` registries.
- [ ] Verify each locked policy can `policy.act(obs)` against a fresh env reset for each (policy, env) cell. Output a single action tensor; assert shape against env action space.

**Exit criterion:** every locked (policy, env) cell produces a valid action; Libero in/out decision committed to `docs/MODEL_CARDS.md`.

## Days 2-3 — core eval + mini sweep

Owner: `bench-eval-engineer` (lib), `sweep-sre` (orchestration), `render-pipeline-engineer` (videos), `stats-rigor-reviewer` (CI math).

- [ ] `src/lerobot_bench/eval.py`: the `(policy, env, seed, n_eps) → CellResult` core. Seeding contract from DESIGN.md § Methodology.
- [ ] `src/lerobot_bench/stats.py`: `bootstrap_ci`, `paired_wilcoxon`, `cohens_h`. Unit-tested with synthetic data.
- [ ] `src/lerobot_bench/render.py`: episode → MP4 (256 px / 10 fps / H.264 / ≤2MB).
- [ ] `src/lerobot_bench/checkpointing.py`: cell-boundary skip on resume.
- [ ] One full cell — 1 policy × 1 env × 1 seed × 5 episodes — produces a row in `results.parquet` and one MP4 ≤ 2MB.
- [ ] Mini sweep: 3 policies × 2 envs × 2 seeds × 25 episodes. Push to HF Hub dataset. Verify Hub read.

## Day 4 — Spaces app + resume drill

Owner: `spaces-frontend-engineer`, with `sweep-sre` for the resume test.

- [ ] `space/app.py` with three tabs: Leaderboard, Browse Rollouts, Methodology. Direct Hub URLs on `gr.Video`.
- [ ] `space/requirements.txt` pinning `lerobot==0.5.1`, `gradio>=5`.
- [ ] `make space-deploy` targets the HF Spaces git remote.
- [ ] Resume drill: kill `run_sweep.py` mid-cell, restart, confirm the killed cell restarts from episode 0 cleanly.

## Days 5-10 — full sweep, fine-tune track, writeup, ship

Tracked at the day level in `docs/CEO-PLAN.md` § Updated timeline. Each
day's work is a separate PR; that table is authoritative.

## Standing items (every PR)

- [ ] `make all` passes locally — the git-push hook gates on it.
- [ ] CHANGELOG.md updated under `[Unreleased]` if user-visible.
- [ ] Conventional Commits in PR title.
- [ ] Tests added for new code paths. Sim/GPU tests carry the right pytest mark.
- [ ] No writes to `results/`, `*.parquet`, `*.mp4` from library code (the hook will block; but library code shouldn't try anyway).

## Open questions / explicit blockers

1. **GitHub remote**: owner choice (`theoh-io` vs `thrmnn`) is blocking the first push of `chore/claude-team-and-infra`. Resolve before Day 0a closes.
2. **HF Hub dataset name**: design doc says `theoh-io/lerobot-bench-results-v1`. If the GH owner changes, mirror it here so attribution lines up.
3. **HF Space URL**: same — owner alignment matters. The Space's git remote is set once and rarely changed.
4. **lerobot env Python tooling**: dev extras (`mypy`, `pytest`, `ruff`) not yet installed in the `lerobot` conda env. `make all` will fail until they are. Day 0a item.
