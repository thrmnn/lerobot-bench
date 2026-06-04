# Security & Supply-Chain Audit — embodimetry

**Date:** 2026-05-22
**Auditor:** Council role — Security (CSO methodology, daily mode, 8/10 confidence gate)
**Branch audited:** `origin/main`
**Scope:** Secrets, CI/CD security, dependency supply chain, code-level risk, release security.

## Verdict

**PASS with low-risk hardening recommended.** embodimetry is in good shape for a
public release. No secrets are committed or in git history, all YAML loading uses
`safe_load`, no `eval`/`exec`/`shell=True`, and every subprocess call uses list-form
argv (no shell-injection surface). Workflow `permissions:` are mostly scoped to
`contents: read`. The findings below are hardening items, not active vulnerabilities —
the highest is a MEDIUM (unpinned third-party GitHub Action). Two MEDIUM fixes were
applied in this PR; the rest are recommendations.

**Findings: 0 critical · 0 high · 4 medium · 3 low.**

---

## Findings

### Finding 1 — Unpinned third-party GitHub Action in release.yml

* **Severity:** MEDIUM · **Confidence:** 9/10 · **Status:** VERIFIED · **Phase:** 4 (CI/CD)
* **Category:** CI/CD — supply chain
* **Evidence:** `.github/workflows/release.yml:73` — `uses: softprops/action-gh-release@v3`
* **Description:** `softprops/action-gh-release` is a third-party action pinned to a
  floating major tag (`@v3`). A tag is mutable: if the maintainer's account or the
  repo is compromised, an attacker can repoint `v3` to malicious code. That action
  runs in the `github-release` job with `contents: write` and access to the release
  artifacts.
* **Exploit scenario:** Attacker compromises the `softprops` GitHub account, force-pushes
  a malicious commit and moves the `v3` tag to it. The next `v*.*.*` tag push on
  embodimetry runs the trojanised action with `contents: write`, letting it tamper
  with the published release or exfiltrate the (OIDC-minted) tokens in the job.
* **Impact:** Supply-chain compromise of published GitHub releases.
* **Recommendation:** Pin to a full commit SHA with a version comment, e.g.
  `uses: softprops/action-gh-release@<40-char-sha>  # v3.x`. **Fixed in this PR** —
  see "Fixes applied". Dependabot (already configured for `github-actions`) will keep
  the pin current via PRs.

### Finding 2 — Overly broad workflow-level `permissions:` in release.yml

* **Severity:** MEDIUM · **Confidence:** 8/10 · **Status:** VERIFIED · **Phase:** 4 (CI/CD)
* **Category:** CI/CD — least privilege
* **Evidence:** `.github/workflows/release.yml:14-15` — `permissions: contents: write`
  declared at workflow scope.
* **Description:** `contents: write` is granted to *every* job in `release.yml`,
  including `build` (which only checks out code + builds artifacts) and `testpypi`
  (which publishes via OIDC and needs only `id-token: write`). Only the
  `github-release` job legitimately needs `contents: write`. A bug or compromised
  dependency in the `build` job would inherit write access to the repo it does not need.
* **Exploit scenario:** A malicious transitive dep pulled during `pip install build twine`
  in the `build` job executes with `contents: write` and pushes a commit / tag to the repo.
* **Impact:** Privilege beyond need-to-know; widens blast radius of any build-step compromise.
* **Recommendation:** Set workflow-level `permissions: contents: read` and override
  per-job: add `permissions: contents: write` only to `github-release`. The `testpypi`
  job already scopes its own `id-token: write`. **Fixed in this PR** — see "Fixes applied".

### Finding 3 — Space pulls project code from a mutable branch ref

* **Severity:** MEDIUM · **Confidence:** 9/10 · **Status:** VERIFIED · **Phase:** 3 (Supply Chain)
* **Category:** Supply chain — reproducibility / integrity
* **Evidence:** `space/requirements.txt` last line —
  `git+https://github.com/thrmnn/lerobot-bench.git@feat/space-app`
* **Description:** The deployed HF Space installs embodimetry from a Git **branch**
  ref. A branch is mutable: any future push to `feat/space-app` silently changes the
  code the live Space runs, and the Space is not reproducible from a fixed commit.
  The file's own comment already flags this ("Pinned to a branch ref while the PR is
  open; flip to the merged-main SHA when this PR lands").
* **Exploit scenario:** Anyone with push access to that branch (or an attacker who
  compromises it) changes the code the public Space executes, with no review gate on
  the Space side.
* **Impact:** Non-reproducible Space; uncontrolled code change on a public endpoint.
* **Recommendation:** Pin to a 40-char commit SHA of `main` once the Space PR merges:
  `git+https://github.com/thrmnn/lerobot-bench.git@<sha>`. **Not fixed here** —
  `space/` is outside this PR's file-ownership scope; flagged for the Space owner.

### Finding 4 — No SECURITY.md at repo root / not linked to GitHub advisories

* **Severity:** MEDIUM · **Confidence:** 8/10 · **Status:** VERIFIED · **Phase:** 13
* **Category:** Disclosure process
* **Evidence:** `SECURITY.md` lives only at `docs/SECURITY.md`. GitHub surfaces the
  "Report a vulnerability" / security-policy link from a file named `SECURITY.md` at
  repo root, `.github/`, or `docs/`. `docs/` *is* a recognised location, so the link
  works — but the policy was thin: no supported-versions table, no disclosure
  timeline, no mention of GitHub private vulnerability reporting.
* **Description:** A public benchmark with a HF Space and dataset benefits from a
  clear, complete disclosure policy so researchers know how and where to report.
* **Recommendation:** Expand `docs/SECURITY.md` with a supported-versions table, a
  response-time commitment, a coordinated-disclosure window, and a pointer to GitHub's
  private vulnerability reporting. **Fixed in this PR** — `docs/SECURITY.md` rewritten.
  Also recommend enabling *Private vulnerability reporting* in repo Settings → Security.

### Finding 5 — No CODEOWNERS protection on workflow files

* **Severity:** LOW · **Confidence:** 8/10 · **Status:** VERIFIED · **Phase:** 4 (CI/CD)
* **Category:** CI/CD — change control
* **Evidence:** No `.github/CODEOWNERS` file exists.
* **Description:** Without CODEOWNERS, changes to `.github/workflows/*` (the files that
  hold CI permissions and secret access) are not forced through a designated reviewer.
  For a solo-maintainer research repo this is low impact, but it is the cheap
  guardrail that catches a malicious workflow change in a community PR.
* **Recommendation:** Add `.github/CODEOWNERS` with `/.github/ @thrmnn` and enable
  branch protection requiring code-owner review. **Not fixed here** — creating
  CODEOWNERS is outside this PR's file-ownership scope; flagged as a recommendation.

### Finding 6 — Unpinned first-party actions across all workflows

* **Severity:** LOW · **Confidence:** 9/10 · **Status:** VERIFIED · **Phase:** 4 (CI/CD)
* **Category:** CI/CD — supply chain
* **Evidence:** All five workflows use floating major tags for first-party actions:
  `actions/checkout@v6`, `actions/setup-python@v6`, `actions/upload-artifact@v7`,
  `actions/download-artifact@v8`, `pypa/gh-action-pypi-publish@release/v1`.
* **Description:** Per standard supply-chain guidance, first-party `actions/*` on a
  floating tag is LOW risk (GitHub controls the namespace; the published-tag attack is
  far less likely than for a solo-maintainer third-party action). It is still best
  practice to SHA-pin everything for full reproducibility.
* **Recommendation:** SHA-pin first-party actions too if you want fully reproducible
  CI. Dependabot's `github-actions` updater already keeps pins current. **Not fixed
  here** — deliberately deferred: pinning the *third-party* action (Finding 1) is the
  high-value change; pinning first-party actions is optional polish and would touch
  job steps the file-ownership constraint reserves for other agents.

### Finding 7 — Dependency floors are wide; lerobot pin is exact (by design)

* **Severity:** LOW · **Confidence:** 7/10 · **Status:** VERIFIED · **Phase:** 3 (Supply Chain)
* **Category:** Supply chain — version policy
* **Evidence:** `pyproject.toml:25-37`. `lerobot==0.5.1` is exact; everything else is a
  floor with a major-version ceiling (`torch>=2.7,<3.0`, `numpy>=2.0,<3.0`, etc.).
* **Description:** The exact `lerobot==0.5.1` pin is intentional and correct — it is the
  reproducibility anchor for the sweep, and `dependabot.yml` documents that Python deps
  are deliberately not auto-bumped. The wide floors on the other deps mean a fresh
  `pip install` resolves to whatever is latest within the major range; that is normal
  for a research library but means CI is not bit-reproducible. No committed lockfile
  exists. For a *library* repo (not an app) a missing lockfile is not a finding.
* **Recommendation:** No action required. Optionally, if you want CI reproducibility,
  add a `requirements-ci.lock` produced by `pip freeze` and install from it in `ci.yml`.
  Leave `pyproject.toml` floors as-is so downstream installers stay flexible.

---

## What was checked and found clean

* **Secrets.** No `.env`, `.pem`, `.key`, `.netrc`, or token files are git-tracked.
  Git history (143 commits, all branches) scanned for `hf_`, `sk-`, `AKIA`, `ghp_`
  patterns — zero hits. `.gitignore` covers `.env*`, `*.pem`, `*.key`, and all model
  binary extensions. `.pre-commit-config.yaml` runs `detect-private-key` on every commit.
* **Code-level.** No `shell=True`, no `os.system`/`os.popen`, no `eval()`/`exec()` of
  any kind (the one `.eval()` hit is `torch.nn.Module.eval()`, a false positive). All
  six `subprocess` call sites (`run_sweep.py`, `reproduce_cell.py`, `run_one.py`,
  `calibrate.py`, `watchdog.py`, `publish_results.py`, `eval.py`) use list-form `argv`
  — no shell-injection surface. All YAML loading uses `yaml.safe_load` /
  `yaml.safe_dump`; no `yaml.load` without a safe loader.
* **Network / untrusted input.** The HF Space reads only the project's own hardcoded
  dataset (`HUB_DATASET_REPO = "thrmnn/embodimetry-v1"`); the parquet source
  is a constant, not user-controllable — no SSRF. The Space does no policy inference,
  loads no checkpoints, and renders results data as Gradio dataframes/markdown (not raw
  HTML), so a poisoned dataset cannot trigger XSS through an `gr.HTML` escape hatch.
  Checkpoint loading uses `safetensors.torch.load_file` (not `torch.load`/`pickle`),
  which does not deserialize arbitrary code.
* **Token handling.** No code reads `HF_TOKEN`/`HUGGINGFACE_*`/`WANDB_*` env vars; the
  HF token is used only by `huggingface-cli login` on the operator's dev box, as
  documented. Env-var overrides that exist (`DASHBOARD_RESULTS_DIR`, `LIBERO_CONFIG_PATH`)
  are trusted operator input, not an attack surface.
* **CI/CD baseline.** `ci.yml`, `smoke.yml`, `space-smoke.yml`, `validate-configs.yml`
  all declare `permissions: contents: read`. No `pull_request_target` anywhere — fork
  PRs run with read-only tokens. No `${{ github.event.* }}` interpolation into `run:`
  steps — no script-injection vector. `concurrency` groups are set.
* **Release security.** The PyPI publish path is correct: `release.yml`'s `testpypi`
  job uses **OIDC trusted publishing** (`pypa/gh-action-pypi-publish` with
  `id-token: write` and a GitHub `environment`) — no long-lived PyPI token is stored
  as a secret. This is the recommended pattern.
* **Dependabot.** `.github/dependabot.yml` covers the `github-actions` ecosystem
  weekly with grouped PRs. Python deps are intentionally excluded (documented:
  `lerobot==0.5.1` is the reproducibility anchor) — a deliberate, defensible choice.

---

## Fixes applied in this PR

1. **`release.yml`** — pinned `softprops/action-gh-release` from `@v3` to a commit SHA
   (Finding 1).
2. **`release.yml`** — narrowed workflow-level `permissions:` from `contents: write`
   to `contents: read`, and added a job-scoped `permissions: contents: write` to the
   `github-release` job only (Finding 2).
3. **`docs/SECURITY.md`** — rewritten with a supported-versions table, response-time
   commitment, coordinated-disclosure window, and a pointer to GitHub private
   vulnerability reporting (Finding 4).

## Recommendations not applied (out of scope for this PR)

* **Finding 3** — pin `space/requirements.txt` to a commit SHA once the Space PR
  merges (`space/` is outside this PR's file ownership).
* **Finding 5** — add `.github/CODEOWNERS` and enable code-owner review on `.github/`.
* **Finding 6** — optionally SHA-pin first-party `actions/*` for full CI reproducibility.
* **Finding 7** — optionally add a `requirements-ci.lock` for bit-reproducible CI.
* Enable **Private vulnerability reporting** in repo Settings → Security.

---

*This is an AI-assisted scan that catches common vulnerability patterns. It is not a
substitute for a professional security audit. For a public release handling community
contributions, consider a professional review of the CI/CD pipeline before 1.0.*
