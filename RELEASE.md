# Release checklist — lerobot-bench v1.0.0

The definitive, ordered list of everything that must be true to ship
**lerobot-bench v1.0.0**. This doc is also the project's ship runbook: work it
top to bottom, tick each box only when the referenced command actually
succeeded.

It complements — does not replace — `docs/RUNBOOK.md` (day-to-day sweep ops)
and `docs/REPRODUCE.md` (the reproducibility contract). Where a phase has a
deeper how-to, this doc points at it rather than duplicating it.

**Release manager:** one person owns the cut start to finish. Do not split
phases across people — the version-triple bump and the tag must be done by the
same hand.

**Current state at time of writing:** overnight sweep in flight (~69/107
cells); `VERSION`, `src/lerobot_bench/__version__.py`, and `pyproject.toml`
all read `0.0.1`; `CHANGELOG.md` has a populated `[Unreleased]` section. None
of the phases below are complete yet.

---

## Phase 1 — Data complete

The sweep parquet is the foundation. Nothing downstream is trustworthy until
this phase is fully green.

- [ ] **Full sweep finished — 110/110 cells.** The overnight run launched by
      `scripts/launch_overnight_sweep.sh` (writes to `results/sweep-full/`)
      has every planned cell stamped `completed` in
      `results/sweep-full/sweep_manifest.json`. Confirm with:
      ```bash
      python -c "
      import json
      m = json.load(open('results/sweep-full/sweep_manifest.json'))
      cells = m['cells'] if isinstance(m, dict) else m
      from collections import Counter
      print(Counter(c['status'] for c in cells))
      "
      ```
      Every cell must be `completed` (or an explicitly justified `skipped` for
      an `env_compat` filter). Zero `pending`, zero `failed`.
- [ ] **No partial cells.** Each completed cell has exactly its contracted
      episode count (50, or 25 for the two auto-downscoped cells). Spot-check
      with the `groupby` snippet in `docs/RUNBOOK.md` § Resume drill. A
      mid-cell SIGKILL leaves a partial cell — drop and re-run it per the
      RUNBOOK before declaring the sweep done.
- [ ] **Corrected `act` rows merged into `results.parquet`.** The re-run `act`
      cells are present in `results/sweep-full/results.parquet` and the stale
      rows are gone. Verify the `act` row count and that no `(policy, env,
      seed_idx)` key is duplicated:
      ```bash
      python -c "
      import pandas as pd
      df = pd.read_parquet('results/sweep-full/results.parquet')
      act = df[df.policy_name == 'act']
      print('act rows:', len(act))
      dups = df.groupby(['policy_name','env_name','seed_idx','episode_index']).size()
      print('duplicate keys:', int((dups > 1).sum()))
      "
      ```
      Duplicate keys must be `0` (`checkpointing.append_cell_rows` guards
      against this, but a manual merge can reintroduce it).
- [ ] **`review_results.py` shows no unexplained anomalies.**
      ```bash
      make review-results
      ```
      Exit `0` is the goal. If it exits `1`, every flagged cell in the
      `ANOMALIES` section must have a written explanation (e.g. a known
      baseline floor) — an unexplained flag blocks the release. The four
      anomaly checks are documented in the script header and `docs/DESIGN.md`
      § Methodology.
- [ ] **Provenance is internally consistent.** `lerobot==0.5.1` is the
      installed version (`python -c "import lerobot; print(lerobot.__version__)"`)
      and every policy `revision_sha` in `configs/policies.yaml` matches the
      checkpoint the sweep actually used.

## Phase 2 — Analysis

One real, defensible number — not a notebook full of placeholders.

- [ ] **Analysis notebook run on final data.** `notebooks/01-write-finding.ipynb`
      re-executed top to bottom against `results/sweep-full/results.parquet`
      (not the synthetic fallback parquet). Execute headless to prove it:
      ```bash
      jupyter nbconvert --to notebook --execute --inplace \
          notebooks/01-write-finding.ipynb
      ```
      The notebook reads the real parquet with no code edits — only the data
      path changes.
- [ ] **Reproducibility footer is real.** The notebook's final cell prints a
      non-synthetic row count, the real `git rev-parse HEAD`, and the bootstrap
      config. Confirm the parquet path is `results/sweep-full/results.parquet`.
- [ ] **Per-cell MDE gate applied.** Cell 5b's `paired_df_gated` flags every
      inconclusive comparison (`docs/MDE_TABLE.md` § 4). Anything flagged
      `inconclusive_per_cell` renders neutral and is excluded from the headline.
- [ ] **One headline finding locked.** A single defensible sentence, citing a
      delta only where `|delta| > MDE`. This sentence becomes the paper
      abstract's finding clause and the Space's leaderboard caption — write it
      once, here:

      > _Headline finding: _______________________________________________

## Phase 3 — Paper

- [ ] **Results section filled from real numbers.** The 12-row leaderboard
      table, forest plot, paired-comparison table, and failure-taxonomy chart
      in `paper/main.tex` § Results carry real figures regenerated from
      `notebooks/01-write-finding.ipynb` — no synthetic data.
- [ ] **All `\todo` placeholders resolved.** The `\todo{...}` macro must not
      appear in the final source:
      ```bash
      grep -n 'todo{' paper/main.tex
      ```
      Must return nothing. (At time of writing: 30 occurrences.)
- [ ] **Paper compiles clean.** From `paper/`:
      ```bash
      make -C paper
      ```
      Runs the `pdflatex → bibtex → pdflatex → pdflatex` build. No errors; the
      only acceptable warning is the cosmetic `'h' float specifier changed to
      'ht'`. Output stays at 4 pages of body (≤6 with abstract + bibliography).
- [ ] **References intact.** `paper/references.bib` entries all have real
      arXiv IDs / DOIs / ISBNs — no fabricated citations introduced while
      filling Results.
- [ ] **Limitations current.** The Pi0-family deferral to v1.1 is still
      documented in § Discussion / Future Work (it is — keep it).

## Phase 4 — Publish

- [ ] **Dataset pushed to HF Hub.** Logged in (`huggingface-cli login`, write
      scope), then:
      ```bash
      make publish ARGS="--results-path results/sweep-full/results.parquet \
          --manifest-path results/sweep-full/sweep_manifest.json \
          --videos-dir results/sweep-full/videos"
      ```
      `scripts/publish_results.py` runs pre-flight gates (schema match,
      manifest parse, every `video_sha256` row has its MP4), stages into
      `_publish_staging/`, writes `_provenance.json`, and uploads to
      `thrmnn/lerobot-bench-results-v1`. Exit `0` = clean; exit `2` = some
      videos skipped on the size cap (inspect `_provenance.json#skipped_videos`
      and decide if acceptable); exit `3/4/5` = pre-flight / auth / upload
      failure — fix and re-run (the uploader is idempotent).
- [ ] **Dataset card in place.** The Hub dataset's `README.md` matches the
      in-repo source of truth `docs/HUB_DATASET_README.md` (schema table,
      methodology pointer, BibTeX, MIT license).
- [ ] **Dry-run sanity first (optional but recommended).** Re-run the publish
      command with `--dry-run` appended to `ARGS` before the real push — it
      stages and writes provenance without any network call.
- [ ] **Space deployed and live.** From `space/`, push to the HF Spaces git
      remote:
      ```bash
      make space-deploy
      ```
      Then confirm the Space booted and serves the leaderboard:
      `https://huggingface.co/spaces/thrmnn/lerobot-bench` returns 200 and the
      Leaderboard tab shows real cells. `space-smoke.yml` already boot-tests
      `app.py` on every `space/**` push; a green run there is the CI signal.
      Rollback if broken: `cd space && git push -f hf-space main~1:main`
      (`docs/RUNBOOK.md` § Deploy + roll back the Space).
- [ ] **README hero screenshot captured.** Screenshot the live Space
      leaderboard and commit it to `docs/assets/leaderboard.png` (the path
      `README.md` already references; see `docs/assets/README.md`). The repo
      renders fine while it is absent — but v1 should ship with the hero image.
- [ ] **One published cell verified end-to-end.** Pull the dataset back and
      reproduce a cell to prove the published parquet is sound:
      ```bash
      huggingface-cli download thrmnn/lerobot-bench-results-v1 \
          --repo-type dataset --local-dir results/sweep-full
      make reproduce CELL=act/aloha_transfer_cube/0
      ```
      Must print `REPRODUCED ✓` (exit `0`). See `docs/REPRODUCE.md`.

## Phase 5 — Repo hygiene

- [ ] **CHANGELOG `[Unreleased]` → `[1.0.0]`.** In `CHANGELOG.md`, rename the
      `[Unreleased]` heading to `## [1.0.0] - 2026-05-DD` (real date), add a
      fresh empty `[Unreleased]` above it, and update the link refs at the
      bottom of the file (add a `[1.0.0]` compare/tag line; point
      `[Unreleased]` at `v1.0.0...HEAD`).
- [ ] **Version bumped in all three places, in lock-step.** `release.yml`
      hard-fails the build if they disagree:
      - `VERSION` → `1.0.0`
      - `src/lerobot_bench/__version__.py` → `__version__ = "1.0.0"`
      - `pyproject.toml` `[project].version` → `1.0.0`
- [ ] **Pre-Alpha classifier updated.** Bump
      `Development Status :: 2 - Pre-Alpha` in `pyproject.toml` to
      `5 - Production/Stable` (or `4 - Beta` if preferred) — Pre-Alpha is wrong
      for a v1.
- [ ] **All CI green on the release commit.** `ci.yml` (lint + mypy + fast
      tests), `validate-configs.yml`, `space-smoke.yml`, and the most recent
      `smoke.yml` / `install-smoke.yml` runs are all passing on the commit that
      will be tagged. Locally, `make all` (lint + typecheck + test) passes.
- [ ] **Docs index in place.** `README.md` Quick links and the docs cross-refs
      resolve; `docs/RUNBOOK.md` § Release a new lerobot-bench version is
      accurate; no doc references a file that does not exist.
- [ ] **Citation section filled.** `README.md` § Citation currently says the
      pre-print "lands when the sweep completes" — replace the placeholder with
      the real BibTeX (mirror `docs/HUB_DATASET_README.md`).
- [ ] **lerobot pin untouched.** `lerobot==0.5.1` in `pyproject.toml` is
      unchanged — the pin is the reproducibility anchor and is sacred for v1.

## Phase 6 — Release

- [ ] **Build dry-run via `workflow_dispatch`.** Before tagging, trigger
      `release.yml` manually with `publish_testpypi: false` to confirm the
      version-triple check passes and `python -m build` + `twine check`
      succeed on a clean runner. (Optionally flip `publish_testpypi: true` once
      to validate the package installs from TestPyPI.)
- [ ] **Tag and push.** From `main`, with the version triple and CHANGELOG
      already committed:
      ```bash
      git tag v1.0.0
      git push origin main --tags
      ```
      The `v1.0.0` tag triggers `release.yml`. **Tags are immutable** — a bad
      release is superseded by the next version, never re-tagged.
- [ ] **`release.yml` ran clean.** The `build` job (version check → build →
      twine check) and the `github-release` job (attaches sdist + wheel,
      `generate_release_notes: true`) both succeeded. Artifacts are on the GH
      release.
- [ ] **GitHub release notes finalized.** The auto-generated notes are edited
      into a human v1.0.0 summary: the headline finding, the 6×6 matrix scope,
      the Pi0 deferral, links to the Space and the Hub dataset.
- [ ] **arXiv submission.** Upload the `paper/` LaTeX source (not the PDF —
      arXiv builds it) to arXiv, cs.RO primary / cs.LG secondary. Once the
      arXiv ID is live, add it to `README.md` § Citation and the GH release.

## Phase 7 — Upstream

- [ ] **`lerobot.eval.multi_seed` PR opened.** `src/lerobot_bench/eval.py`'s
      multi-seed eval pipeline is extracted and a PR opened against
      `huggingface/lerobot` (the third of the three v1 artifacts named in
      `README.md` § TL;DR). This can trail the release slightly but must be
      open before v1 is announced as "done". Link the PR from the GH release.

---

## How to cut the release — walkthrough

Once **Phases 1–4 are fully ticked** (data, analysis, paper, publish), the
mechanical cut is short. Do it in one sitting, from `main`, after a
`git pull`.

```bash
# 0. Start clean on main.
git checkout main && git pull origin main
make all                              # lint + typecheck + test must pass

# 1. Bump the version triple — all three, same value.
#    VERSION, src/lerobot_bench/__version__.py, pyproject.toml [project].version
#    -> 1.0.0

# 2. Update CHANGELOG.md: rename [Unreleased] -> [1.0.0] - <date>,
#    add a fresh empty [Unreleased], fix the link refs at the file foot.

# 3. Commit the release prep.
git commit -am "chore(release): v1.0.0"

# 4. (Recommended) Dry-run the release pipeline before tagging.
#    GH Actions UI -> release workflow -> Run workflow
#    -> publish_testpypi: false. Confirm the version check + build pass.

# 5. Tag and push. The tag triggers release.yml.
git tag v1.0.0
git push origin main --tags

# 6. Watch release.yml: build (version check + build + twine check)
#    then github-release (attaches sdist + wheel, generates notes).

# 7. Edit the GitHub release notes into a human v1.0.0 summary.

# 8. Submit paper/ LaTeX source to arXiv; add the arXiv ID back into
#    README.md § Citation and the GH release once it is live.

# 9. Open the lerobot.eval.multi_seed PR against huggingface/lerobot.
```

**If something goes wrong after tagging:** the tag is immutable. Do not
`git tag -f`, do not delete and re-push. Fix the problem, bump to `v1.0.1`,
and cut again — the next version supersedes the bad one. This is the same rule
stated in `docs/RUNBOOK.md` § Release a new lerobot-bench version.

**Non-negotiables, repeated because they bite:**

- Never bypass hooks — no `--no-verify`, no skipping pre-commit. A failing
  hook is a real signal; fix the cause.
- The version triple (`VERSION` / `__version__.py` / `pyproject.toml`) must
  agree with the tag — `release.yml` fails the build otherwise.
- `lerobot==0.5.1` stays pinned. Bumping it invalidates every published cell
  and requires a full sweep re-run, which is a v2 conversation.
