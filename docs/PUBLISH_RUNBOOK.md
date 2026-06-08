# v1 publish runbook — one command per step

The v1 HF publish is the single owner-gated blocker that unlocks the public
Demo (Space) and the Impact story. This doc reduces it to a flat,
copy-pasteable sequence: run each block top to bottom, in order.

- Steps marked **[OWNER]** are authed and/or irreversible — only the release
  manager (HF write token + `thrmnn` git auth) runs them.
- Every other step is a local, read-only or idempotent check; run it freely.
- Run from the **main tree** (`/home/theo/projects/lerobot-bench`), on `main`,
  after `git pull`. `results/` is gitignored, so the data mutation in Step 1
  happens on the real on-disk parquets, not in any worktree.

This is the *publish slice* of the broader cut in `RELEASE.md` (Phase 4) and
the Space deploy/rollback in `docs/RUNBOOK.md` § Deploy + roll back the Space —
it does not replace them.

Pin paths once:

```bash
SWEEP=results/sweep-full
RES=$SWEEP/results.parquet
MAN=$SWEEP/sweep_manifest.json
VID=$SWEEP/videos
HUB=thrmnn/embodimetry-v1
```

---

## Step 0 — Confirm the blocker is real (read-only)

The publish preflight is the gate. Run it before doing anything; it should
abort on the stale `act` rows until Step 1 lands.

```bash
python scripts/publish_results.py \
  --results-path $RES --manifest-path $MAN --videos-dir $VID --dry-run
```

Verifies: the floor guard fires.
Expected now: exit `3`,
`Stale pre-#51 act×aloha rows detected (pooled=0.016 < 0.5)`.
That single line is the whole reason Step 1 exists.

---

## Step 1 [OWNER] — Apply the corrected `act` row merge (#177)

Splices the corrected `act×aloha_transfer_cube` re-run (pooled 0.824,
code_sha `7361d96`) over the stale pre-#51 rows (0.016) in **both** the
canonical parquet and the `_publish_staging/` copy. Row count and cell count
are preserved; only the `act×aloha` rows change.

```bash
# 1a. Inspect the delta first — writes nothing.
python scripts/merge_corrected_act_rows.py --sweep-dir $SWEEP --dry-run

# 1b. Apply to canonical + staging.
python scripts/merge_corrected_act_rows.py --sweep-dir $SWEEP --staging
```

Verifies: dry-run prints `act×aloha pooled 0.016 -> 0.824 (total rows N,
unchanged)`; the apply writes the same atomically. The script self-guards
(refuses to splice if the re-run isn't N=250 @ ~0.824, or if the post-merge
pooled rate leaves `[0.77, 0.87]`, or if the row/cell count changes).

Post-merge verification (the staging copy must now read 0.824, not 0.016):

```bash
python -c "
import pandas as pd
df = pd.read_parquet('$SWEEP/_publish_staging/results.parquet')
m = (df.policy=='act') & df.env.astype(str).str.contains('aloha')
print('staging act×aloha pooled:', round(float(df[m].success.astype(float).mean()), 3))
assert df[m].success.astype(float).mean() > 0.5, 'STILL STALE — re-run 1b with --staging'
print('OK')
"
```

Rollback: `results/` is gitignored, so there is no git revert. If a bad splice
lands, restore the canonical from the untouched re-run inputs — the merge is
idempotent and re-derivable: re-running 1b against the original
`results-act-rerun.parquet` reproduces the same 0.824 rows. Keep a copy of
`results.parquet` before 1b if you want a literal byte-for-byte undo:
`cp $RES $RES.pre177.bak`.

---

## Step 2 — Publish preflight, green (read-only dry-run)

Same command as Step 0; now it must pass. This is the floor guard + schema
gate + coverage gate + xvla strip, all in one.

```bash
python scripts/publish_results.py \
  --results-path $RES --manifest-path $MAN --videos-dir $VID --dry-run
```

Verifies (exit `0` expected): canonical-name guard; REQUIRED/OPTIONAL schema
split; act×aloha floor (now 0.824 ≥ 0.5); every REQUIRED `(policy, env)` cell
present (90 v1 cells); xvla rows + their ~875 MP4s stripped; every referenced
`video_sha256` has its MP4 on disk. Exit `2` here means some MP4s exceeded the
2 MiB cap — inspect the staged `_provenance.json#skipped_videos` and decide if
acceptable before proceeding. Exit `3` means a gate is still red — fix and
re-run; do **not** proceed to Step 3.

No rollback (no writes hit the Hub; staging dir is local scratch).

---

## Step 3 [OWNER] — Authed HF dataset upload

Push parquet + manifest + filtered MP4s to the public dataset. Idempotent:
re-running with identical inputs yields zero or one Hub commit
(content-addressed dedup).

```bash
huggingface-cli login        # write scope, account: thrmnn — one time
make publish ARGS="--results-path $RES --manifest-path $MAN --videos-dir $VID"
```

Target repo: `thrmnn/embodimetry-v1` (dataset).
Verifies: exit `0` = clean upload; exit `2` = some videos skipped on the size
cap (audit `_provenance.json`); exit `4` = auth/repo missing
(`huggingface-cli login` + confirm the dataset exists); exit `5` = mid-upload
failure (network / rate limit) — the uploader is idempotent, just re-run.

Rollback: the Hub keeps commit history. Revert via
`huggingface-cli` / the Hub UI to the prior dataset commit, or re-upload a
corrected bundle (a later commit supersedes the bad one). Bytes are public
once pushed — there is no "unsend"; correct forward.

Optional but recommended: confirm the dataset card matches the in-repo source
of truth `docs/HUB_DATASET_README.md` (schema table, methodology pointer,
BibTeX, MIT license).

---

## Step 4 [OWNER] — Deploy the Space

The Space reads the dataset by URL, so it renders non-empty only **after**
Step 3. `space/` lives inside this monorepo; the deploy is a subtree push that
lands `space/`'s contents at the Space root. Full detail:
`docs/RUNBOOK.md` § Deploy + roll back the Space.

```bash
# One-time, if the remote isn't set (auth as thrmnn):
#   huggingface-cli repo create embodimetry --type space --space_sdk gradio
#   git remote add hf-space https://huggingface.co/spaces/thrmnn/embodimetry

make space-deploy        # == git subtree push --prefix space hf-space main
```

Target: `https://huggingface.co/spaces/thrmnn/embodimetry`.
Verifies: the Space boots and the Leaderboard tab shows real cells; the URL
returns `200`. `space-smoke.yml` boot-tests `app.py` on every `space/**` push —
a green run there is the CI signal. Before deploying, confirm the GitHub SHA
pinned in `space/requirements.txt` is an ancestor of `main`.

Rollback: a subtree push has no plain `main~1:main` ancestor on the remote.
Revert `space/` on `main`, then re-run `make space-deploy` to push the prior
good tree.

---

## Step 5 [OWNER] — Cut the git tag + GitHub release

Final, irreversible step. **Tags are immutable** — a bad release is superseded
by the next version, never re-tagged. Do the version-triple bump +
CHANGELOG + release notes per `RELEASE.md` Phases 5–6 first; this block is only
the tag-and-push.

```bash
git checkout main && git pull origin main
make all                          # lint + typecheck + test must pass
# (version triple + CHANGELOG already committed per RELEASE.md Phase 5)
git tag v1.0.0
git push origin main --tags       # triggers release.yml
```

Verifies: `release.yml` `build` job (version-triple check → `python -m build` →
`twine check`) and `github-release` job both pass; sdist + wheel attach to the
GH release.

Rollback: **none** — the tag is immutable. Do not `git tag -f`, do not delete
and re-push. Fix the problem, bump to `v1.0.1`, and cut again (next version
supersedes). This is the same rule as `RELEASE.md` and `docs/RUNBOOK.md`
§ Release a new embodimetry version.

---

## Owner-gated steps, at a glance

| Step | Action | Gated reason |
| ---- | ------ | ------------ |
| 1 | Merge corrected `act` rows (#177) | mutates gitignored canonical + staging parquet |
| 3 | HF dataset upload | HF write token; bytes go public |
| 4 | Space deploy | `thrmnn` git auth; public Space |
| 5 | git tag + GitHub release | immutable tag; triggers `release.yml` |

Steps 0 and 2 are local read-only dry-runs — run them as often as you like.
