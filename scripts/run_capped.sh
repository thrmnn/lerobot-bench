#!/usr/bin/env bash
# Run a heavy command under a kernel-enforced memory cap.
#
# Defense in depth (in order of strictness):
#   1. systemd-run --user --scope -p MemoryMax=N -p MemorySwapMax=0
#        kernel OOM-kills inside the cgroup when memory hits the cap.
#        Cannot be exceeded — Windows host stays safe.
#   2. Pre-flight gate: refuse to launch if used RAM is already above
#        $LAUNCH_MAX_USED_PCT% of total (default 55%). Protects parallel
#        workloads (browser, IDE, solar pipeline) from starvation.
#   3. (Optional) Observational watchdog set by the caller, attached to
#        the PID returned in /tmp/last-capped-target-pid. Configure with
#        a low threshold + zero grace; cgroup is the primary defense.
#
# Usage:
#   scripts/run_capped.sh <mem-cap-bytes-or-suffixed> -- <cmd> [args...]
#
# Examples:
#   scripts/run_capped.sh 18G -- python scripts/calibrate.py --policy act --env pusht
#   MEM_CAP=12G scripts/run_capped.sh -- python -c "..."
#
# Environment:
#   MEM_CAP                fallback cap if no positional given (e.g. 18G)
#   LAUNCH_MAX_USED_PCT    refuse launch if RAM used > this% (default 55)

set -euo pipefail

print_help() {
    cat <<'EOF'
run_capped.sh -- run a heavy command under a kernel-enforced memory cap.

usage:
  scripts/run_capped.sh <mem-cap> -- <cmd> [args...]
  MEM_CAP=<mem-cap> scripts/run_capped.sh -- <cmd> [args...]

  <mem-cap>  a memory ceiling, e.g. 18G or 12000M. The command is run in a
             systemd cgroup with MemoryMax/MemoryHigh set to this and swap
             disabled; the kernel OOM-kills it inside the cgroup if exceeded.

examples:
  scripts/run_capped.sh 18G -- python scripts/calibrate.py --policy act --env pusht
  MEM_CAP=12G scripts/run_capped.sh -- python scripts/run_one.py --policy act --env pusht --seed 0

environment:
  MEM_CAP              fallback cap when no positional <mem-cap> is given (e.g. 18G)
  LAUNCH_MAX_USED_PCT  refuse to launch if RAM already used > this percent (default 55)

exit codes:
  2  bad invocation (missing cap, missing '--', or no command)
  3  pre-flight refusal -- RAM already above LAUNCH_MAX_USED_PCT
EOF
}

usage() {
    print_help >&2
    exit 2
}

case "${1:-}" in
    -h | --help)
        print_help
        exit 0
        ;;
esac

if [ "${1:-}" = "--" ]; then
    if [ -z "${MEM_CAP:-}" ]; then
        echo "run_capped.sh: no memory cap given." >&2
        echo "  pass it as the first arg (e.g. 18G) or set MEM_CAP=18G." >&2
        echo "  see: scripts/run_capped.sh --help" >&2
        exit 2
    fi
    CAP="${MEM_CAP}"
    shift
elif [ -n "${1:-}" ] && [ "$1" != "--" ]; then
    CAP="$1"
    shift
    [ "${1:-}" = "--" ] || usage
    shift
else
    usage
fi

[ "$#" -ge 1 ] || usage

LAUNCH_MAX_USED_PCT="${LAUNCH_MAX_USED_PCT:-55}"

# --- Pre-flight gate ---------------------------------------------------------
read -r MEM_TOTAL MEM_AVAIL < <(awk '
    /^MemTotal:/   {t=$2}
    /^MemAvailable:/ {a=$2; print t, a; exit}
' /proc/meminfo)

USED_KB=$((MEM_TOTAL - MEM_AVAIL))
USED_PCT=$((100 * USED_KB / MEM_TOTAL))

echo "pre-flight: RAM total=$((MEM_TOTAL/1024))MB used=$((USED_KB/1024))MB (${USED_PCT}%) avail=$((MEM_AVAIL/1024))MB"
echo "pre-flight: cap=${CAP} swap_cap=0"

if [ "${USED_PCT}" -gt "${LAUNCH_MAX_USED_PCT}" ]; then
    echo "pre-flight: REFUSE — RAM used ${USED_PCT}% > LAUNCH_MAX_USED_PCT=${LAUNCH_MAX_USED_PCT}%" >&2
    echo "pre-flight: free up RAM (close apps / pause parallel work) and retry," >&2
    echo "pre-flight: or raise the gate for one launch: LAUNCH_MAX_USED_PCT=70 $0 ..." >&2
    echo "pre-flight: see docs/TROUBLESHOOTING.md -> run_capped.sh pre-flight refusal" >&2
    exit 3
fi

echo "pre-flight: OK — launching under cgroup MemoryMax=${CAP}"
echo "----"

# --- Launch under cgroup -----------------------------------------------------
# --scope: attach to current shell so we keep stdout/stderr.
# -q: quiet systemd-run banner.
# Run in a detached child so we can record its PID and let the caller
# attach a watchdog.
exec systemd-run --user --scope -q \
    -p MemoryMax="${CAP}" \
    -p MemorySwapMax=0 \
    -p MemoryHigh="${CAP}" \
    -- "$@"
