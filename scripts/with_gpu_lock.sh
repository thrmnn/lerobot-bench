#!/usr/bin/env bash
# Acquire the lerobot-bench GPU semaphore, then exec the command.
# Prevents two CUDA-touching processes from racing for the 8 GB VRAM budget.
#
# Usage:
#   scripts/with_gpu_lock.sh [--timeout SECS] -- <command> [args...]
#
# Default timeout is 1 hour. Use --timeout 0 to fail immediately if held.

set -euo pipefail

LOCK_FILE="${LEROBOT_GPU_LOCK:-/tmp/lerobot-bench-gpu.lock}"
TIMEOUT=3600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"
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
    echo "usage: $0 [--timeout SECS] -- <command> [args...]" >&2
    exit 64
fi

# flock -w TIMEOUT acquires exclusive lock or fails after TIMEOUT.
# Using fd 200 so the lock is held for the duration of the exec'd command.
exec flock -w "$TIMEOUT" -E 75 "$LOCK_FILE" "$@"
