#!/usr/bin/env bash
# Fail-fast GPU-health preflight to run BEFORE any CUDA dispatch.
#
# Thin wrapper over `python -m embodimetry.gpu_preflight`. It verifies:
#   1. nvidia-smi -L is reachable (driver channel up),
#   2. torch.cuda works -- probed IN A SUBPROCESS, so a dead-GPU SIGSEGV
#      is observed as a nonzero exit, NOT a crash of this script,
#   3. free VRAM >= a required headroom (guards near-OOM TDR).
#
# On failure it prints one actionable line (e.g. "likely WSL2 GPU-PV
# desync; fix: wsl --shutdown") and exits nonzero. This converts the
# cryptic 2026-06-09 30-min diagnosis into a 2-second clear error.
#
# Usage:
#   scripts/gpu_preflight.sh [--required-headroom-mb MB] [-q]
#
# Exit codes (passed through from the python module):
#   0   healthy
#   10  nvidia-smi unreachable (likely GPU-PV desync -> wsl --shutdown)
#   11  torch.cuda unusable (CUDA runtime crashed/no device -> wsl --shutdown)
#   12  insufficient free VRAM (another process holds the card; wait)
#
# Honors $PYTHON (default: python). If embodimetry is not pip-installed,
# this resolves the repo src/ onto PYTHONPATH automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Make the package importable without a pip install (worktree-friendly).
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m embodimetry.gpu_preflight "$@"
