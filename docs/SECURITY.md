# Security

This is a research benchmark, not production infrastructure. The threat surface is small.

## Surface

- **Public artifacts**: this GitHub repo, a HF Hub dataset (read-only public), and a HF Space (read-only public Gradio app). No user authentication, no PII, no payments.
- **Local secrets**: HF Hub token (write scope, used only for `huggingface-cli login` on the dev box) and optionally a W&B API key. Both live in standard locations (`~/.cache/huggingface/token`, `~/.netrc`) and are never committed.

## Reporting

If you find a security issue (e.g., a malicious checkpoint, dataset leak, or prompt-injection vector in the rendered playground), email `thermann.ai@gmail.com` with `[lerobot-bench security]` in the subject. Do not open a public issue.

## Pre-commit guard

`.pre-commit-config.yaml` runs `detect-private-key` on every commit. Any accidental key paste is blocked at commit time.
