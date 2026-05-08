#!/usr/bin/env bash
# Wrap a command with WSL2-friendly resource caps.
#
# - ulimit -v (virtual memory) caps process VA at 12 GB by default
# - oom_score_adj=800 makes the kernel kill THIS process before claude/shell
# - ionice c2 n7 = best-effort, low IO priority (don't starve the host)
#
# Usage:
#   scripts/run_with_caps.sh [--vmem-kb N] [--oom-adj N] -- python ...
#
# Env overrides:
#   LEROBOT_VMEM_KB     hard VA cap, default 12000000 (12 GB)
#   LEROBOT_OOM_ADJ     oom_score_adj value, default 800

set -euo pipefail

VMEM_KB="${LEROBOT_VMEM_KB:-12000000}"
OOM_ADJ="${LEROBOT_OOM_ADJ:-800}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vmem-kb)
            VMEM_KB="$2"
            shift 2
            ;;
        --oom-adj)
            OOM_ADJ="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 64
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "usage: $0 [--vmem-kb N] [--oom-adj N] -- <command> [args...]" >&2
    exit 64
fi

ulimit -v "$VMEM_KB"

# Best-effort: WSL2 may not have /proc/self/oom_score_adj writable for all setups.
echo "$OOM_ADJ" >/proc/self/oom_score_adj 2>/dev/null || true

# ionice low priority — best effort, ignore if class not available
exec ionice -c2 -n7 "$@"
