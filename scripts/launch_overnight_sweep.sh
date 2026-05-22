#!/usr/bin/env bash
# Launch the overnight sweep under the kernel-enforced memory cap.
#
# Usage:
#   scripts/launch_overnight_sweep.sh              # 16 GB cap, --resume aware
#   MEM_CAP=20G scripts/launch_overnight_sweep.sh  # override cap
#   scripts/launch_overnight_sweep.sh --dry-run    # plan only
#
# The 16 GB default leaves ~15 GB of the 31 GB host free for the
# operator's own activity. A cell that exceeds the cap is OOM-killed
# inside the cgroup -- the sweep continues, the host stays responsive.
#
# Pre-flight is enforced by scripts/run_capped.sh (refuses to launch
# if baseline RAM > LAUNCH_MAX_USED_PCT% used, default 55%).
#
# Writes:
#   logs/sweep-YYYYMMDD-HHMMSS.log         (full stdout/stderr)
#   results/sweep-full/sweep_manifest.json (per-cell plan + status)
#   results/sweep-full/results.parquet     (incremental rows)
#   results/sweep-full/videos/             (one MP4 per episode)

set -euo pipefail

PY="${PY:-/home/theo/miniforge3/envs/lerobot/bin/python}"
MEM_CAP="${MEM_CAP:-16G}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/sweep-${TS}.log"
mkdir -p logs results/sweep-full

# --- Headless rendering --------------------------------------------------
# MuJoCo (aloha / pusht / LIBERO cells) needs a GL context to render the
# camera images. WSLg ships an X server at :0; a fresh post-reboot shell
# has DISPLAY unset, so glfw can't connect and every render aborts with
# SIGABRT (exit -6). Export DISPLAY when the WSLg X socket is present.
if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
    echo "headless: DISPLAY was unset; bound to WSLg X server :0"
fi
if [ -z "${DISPLAY:-}" ]; then
    echo "FATAL: no DISPLAY and no WSLg X socket at /tmp/.X11-unix/X0;" >&2
    echo "       MuJoCo rendering will abort. Start WSLg or set MUJOCO_GL." >&2
    exit 4
fi

# Default args: resume any partial run; honor caller overrides.
ARGS=(--config configs/sweep_full.yaml --resume)
if [ "$#" -gt 0 ]; then
    ARGS=(--config configs/sweep_full.yaml "$@")
fi

echo "launching sweep MEM_CAP=${MEM_CAP} log=${LOG}"
echo "$$" > /tmp/lerobot-bench-sweep.pid
echo "${TS}" > /tmp/lerobot-bench-sweep-ts
echo "${LOG}" > /tmp/lerobot-bench-sweep-log

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
exec scripts/run_capped.sh "${MEM_CAP}" -- \
    "${PY}" scripts/run_sweep.py "${ARGS[@]}" \
    > "${LOG}" 2>&1
