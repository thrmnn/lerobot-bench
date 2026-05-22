# Security Policy

`lerobot-bench` is a public research benchmark, not production infrastructure.
The threat surface is small: there is no user authentication, no PII, and no
payments. This policy documents how to report a vulnerability and what to expect.

## Supported versions

The benchmark is pre-1.0; only the latest release receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| `0.0.x` | :white_check_mark: |
| `< 0.0` | :x:                |

When `1.0` ships, this table will track the current minor series.

## Surface

- **Public artifacts:** this GitHub repo, a Hugging Face Hub dataset
  (`thrmnn/lerobot-bench-results-v1`, read-only public), and a Hugging Face
  Space (read-only public Gradio app). The Space reads only the published
  dataset; it runs no policy inference and loads no model checkpoints.
- **Local secrets:** an HF Hub token (write scope, used only for
  `huggingface-cli login` on the maintainer's dev box) and optionally a W&B API
  key. Both live in standard locations (`~/.cache/huggingface/token`,
  `~/.netrc`) and are never committed. No code in this repo reads token
  environment variables.

## Reporting a vulnerability

**Do not open a public issue for security reports.**

Preferred: use GitHub's **private vulnerability reporting** —
on the repo, go to the *Security* tab → *Report a vulnerability*. This keeps the
report private until a fix is published.

Alternative: email `thermann.ai@gmail.com` with `[lerobot-bench security]` in
the subject line. Include the affected file/version, a description, and a
proof-of-concept or reproduction steps if you have one.

## Disclosure process

- **Acknowledgement:** within 5 working days of your report.
- **Assessment:** an initial severity assessment within 10 working days.
- **Fix & disclosure:** we aim to ship a fix and publish an advisory within
  90 days of acknowledgement. Coordinated disclosure is appreciated — please
  give us this window before any public write-up.
- **Credit:** reporters are credited in the advisory and release notes unless
  you ask to remain anonymous.

Examples of in-scope issues: a malicious or tampered checkpoint reachable
through the benchmark, a dataset-leak path, a supply-chain compromise of the CI
pipeline or release artifacts, or a code-execution / injection vector in the
rendered Space playground.

## Hardening in place

- `.pre-commit-config.yaml` runs `detect-private-key` on every commit, so an
  accidental key paste is blocked before it can land.
- `.gitignore` excludes `.env*`, `*.pem`, `*.key`, and all model-binary
  extensions.
- GitHub Actions workflows declare least-privilege `permissions:` and use no
  `pull_request_target` trigger, so fork pull requests run with read-only
  tokens.
- PyPI releases use OIDC trusted publishing — no long-lived PyPI token is
  stored as a repository secret.
- Dependabot tracks GitHub Actions versions weekly.

A full audit lives at [`docs/SECURITY_AUDIT.md`](SECURITY_AUDIT.md).
