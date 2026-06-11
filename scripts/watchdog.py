"""WSL2-aware resource watchdog.

Samples RAM, VRAM, and load every `--interval` seconds. If usage breaches
thresholds (defaults tuned for 32 GB WSL2 + 8 GB GPU), sends SIGTERM to
`--pid` (or its process group via `--pgid`), waits `--grace` seconds, then
SIGKILL.

Defaults trigger at:
- RAM available < 30% of total (i.e. > 70% used) — protects Windows host
- VRAM free < 200 MB
- Load > 2× CPU count for 3 consecutive samples

Optional (off by default) `--vram-ceiling-pct` adds a SUSTAINED VRAM-use
guard: if VRAM used stays above the ceiling (e.g. 90%) for longer than
`--vram-ceiling-seconds`, the run is aborted. Sustained near-OOM VRAM is
the WSL2 GPU-PV-desync / TDR trigger (2026-06-09 incident: ~96% sustained
froze the host). This is distinct from the instant `--vram-free-mb-min`
floor and is additive to the cgroup RAM cap (the primary RAM defense).

Writes JSONL samples to `--out` for post-mortem.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts:
                out[key] = int(parts[0])  # kB
    return out


def read_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg") as f:
        a, b, c, *_ = f.read().split()
    return float(a), float(b), float(c)


def read_vram_mb() -> tuple[int, int] | None:
    """Returns (used_mb, total_mb) or None if no GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode()
        used, total = (int(x.strip()) for x in out.strip().split("\n")[0].split(","))
        return used, total
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def vram_used_pct(used_mb: int, total_mb: int) -> float:
    """VRAM used as a percent of total. 0.0 if total is non-positive."""
    if total_mb <= 0:
        return 0.0
    return 100.0 * used_mb / total_mb


def vram_ceiling_breached(
    used_mb: int,
    total_mb: int,
    *,
    ceiling_pct: float | None,
    elapsed_over_s: float,
    ceiling_seconds: float,
) -> bool:
    """Sustained-VRAM-ceiling predicate (pure; unit-tested).

    True iff a ceiling is configured AND current VRAM use is above it AND
    it has already been above it for at least ``ceiling_seconds``. This is
    the desync guard: sustained near-OOM VRAM, not a transient peak.
    ``elapsed_over_s`` is how long use has *already* been continuously over
    the ceiling (the caller tracks the streak; 0 on the first sample over).
    """
    if ceiling_pct is None:
        return False
    if vram_used_pct(used_mb, total_mb) < ceiling_pct:
        return False
    return elapsed_over_s >= ceiling_seconds


def signal_target(pid: int | None, pgid: int | None, sig: int) -> None:
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
    elif pid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watchdog",
        description=(
            "WSL2-aware resource watchdog. Samples RAM, VRAM and load every\n"
            "--interval seconds and appends each sample as JSONL to --out. On\n"
            "a sustained threshold breach it SIGTERMs --pid (or --pgid), waits\n"
            "--grace seconds, then SIGKILLs. cgroup MemoryMax is the primary\n"
            "memory defense (see scripts/run_capped.sh) -- this watchdog is\n"
            "the observational second layer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # observe only -- log samples, never kill (recommended default)\n"
            "  python scripts/watchdog.py --out results/watchdog.jsonl --no-kill\n\n"
            "  # guard a sweep process, killing its whole process group on breach\n"
            "  python scripts/watchdog.py --out results/watchdog.jsonl --pgid 12345\n\n"
            "exit codes:\n"
            "  0  received SIGINT/SIGTERM and shut down cleanly\n"
            "  2  a breach was sustained and the target was killed"
        ),
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="JSONL",
        help="Path to append JSONL samples + breach events to (parent dirs created).",
    )
    ap.add_argument(
        "--pid",
        type=int,
        default=None,
        metavar="PID",
        help="Target process to signal on a sustained breach (default: none -- log only).",
    )
    ap.add_argument(
        "--pgid",
        type=int,
        default=None,
        metavar="PGID",
        help="Target process group to signal on breach; preferred over --pid "
        "so child processes are killed too (default: none).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds between resource samples (default: 5.0).",
    )
    ap.add_argument(
        "--grace",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Seconds to wait between SIGTERM and SIGKILL of the target (default: 10.0).",
    )
    ap.add_argument(
        "--ram-used-pct-max",
        type=float,
        default=70.0,
        metavar="PCT",
        help="Breach if used RAM exceeds this percent of total "
        "(default: 70.0 -- protects the Windows host).",
    )
    ap.add_argument(
        "--vram-free-mb-min",
        type=int,
        default=200,
        metavar="MB",
        help="Breach if free VRAM drops below this many MB (default: 200).",
    )
    ap.add_argument(
        "--vram-ceiling-pct",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Optional VRAM ceiling: if VRAM USED stays above this percent of "
            "total for --vram-ceiling-seconds, treat it as a breach and abort "
            "the target. Sustained near-OOM VRAM is the WSL2 GPU-PV-desync / "
            "TDR trigger (2026-06-09 incident: ~96%% sustained). Disabled by "
            "default; try 90. Distinct from --vram-free-mb-min, which is an "
            "INSTANT free-MB floor."
        ),
    )
    ap.add_argument(
        "--vram-ceiling-seconds",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help=(
            "How long VRAM use must stay above --vram-ceiling-pct before it "
            "counts as a breach (default: 120). Tolerates brief peaks; only "
            "SUSTAINED near-OOM trips it."
        ),
    )
    ap.add_argument(
        "--load-mult-max",
        type=float,
        default=2.0,
        metavar="MULT",
        help="Breach if 1-min load average exceeds MULT x CPU count (default: 2.0).",
    )
    ap.add_argument(
        "--load-consec",
        type=int,
        default=3,
        metavar="N",
        help="Require this many consecutive samples over the load threshold "
        "before it counts as a breach (default: 3).",
    )
    ap.add_argument(
        "--breach-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Require any breach (RAM/VRAM/load) to persist this long before kill. "
            "Tolerates transient spikes (e.g. cold model load). 0 = kill on first sample."
        ),
    )
    ap.add_argument(
        "--no-kill",
        action="store_true",
        help="Log only; never signal target (useful for unattended observation)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_count = os.cpu_count() or 1
    load_breach_streak = 0
    breach_start_ts: float | None = None
    vram_over_ceiling_since: float | None = None
    breached = False

    vram_ceiling_desc = (
        f"{args.vram_ceiling_pct}%/{args.vram_ceiling_seconds:.0f}s"
        if args.vram_ceiling_pct is not None
        else "off"
    )
    print(
        f"watchdog: pid={args.pid} pgid={args.pgid} interval={args.interval}s "
        f"ram_max={args.ram_used_pct_max}% vram_min={args.vram_free_mb_min}MB "
        f"vram_ceiling={vram_ceiling_desc} "
        f"load_max={args.load_mult_max}×{cpu_count}={args.load_mult_max * cpu_count:.1f} "
        f"out={out_path}",
        flush=True,
    )

    def shutdown(signum: int, _frame: object) -> None:
        print(f"watchdog: caught signal {signum}, exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    with out_path.open("a") as out_f:
        while True:
            mi = read_meminfo()
            mem_total = mi.get("MemTotal", 1)
            mem_avail = mi.get("MemAvailable", mi.get("MemFree", 0))
            ram_used_pct = 100.0 * (1.0 - mem_avail / mem_total)
            la1, la5, la15 = read_loadavg()
            vram = read_vram_mb()
            vram_free_mb = (vram[1] - vram[0]) if vram else None

            now = time.time()
            sample = {
                "ts": now,
                "ram_used_pct": round(ram_used_pct, 2),
                "ram_avail_kb": mem_avail,
                "swap_used_kb": mi.get("SwapTotal", 0) - mi.get("SwapFree", 0),
                "vram_used_mb": vram[0] if vram else None,
                "vram_total_mb": vram[1] if vram else None,
                "vram_free_mb": vram_free_mb,
                "vram_used_pct": round(vram_used_pct(vram[0], vram[1]), 2) if vram else None,
                "load1": la1,
                "load5": la5,
                "load15": la15,
            }
            out_f.write(json.dumps(sample) + "\n")
            out_f.flush()

            # Evaluate breach
            ram_breach = ram_used_pct > args.ram_used_pct_max
            vram_breach = vram_free_mb is not None and vram_free_mb < args.vram_free_mb_min
            load_breach = la1 > args.load_mult_max * cpu_count
            load_breach_streak = load_breach_streak + 1 if load_breach else 0
            sustained_load = load_breach_streak >= args.load_consec

            # Sustained VRAM-ceiling guard (desync prevention). Track how long
            # VRAM use has been continuously above the ceiling; trip only once
            # that streak reaches --vram-ceiling-seconds.
            over_ceiling = (
                args.vram_ceiling_pct is not None
                and vram is not None
                and vram_used_pct(vram[0], vram[1]) >= args.vram_ceiling_pct
            )
            if over_ceiling:
                if vram_over_ceiling_since is None:
                    vram_over_ceiling_since = now
                elapsed_over = now - vram_over_ceiling_since
            else:
                vram_over_ceiling_since = None
                elapsed_over = 0.0
            sustained_vram_ceiling = vram is not None and vram_ceiling_breached(
                vram[0],
                vram[1],
                ceiling_pct=args.vram_ceiling_pct,
                elapsed_over_s=elapsed_over,
                ceiling_seconds=args.vram_ceiling_seconds,
            )

            reasons = []
            if ram_breach:
                reasons.append(f"RAM used {ram_used_pct:.1f}% > {args.ram_used_pct_max}%")
            if vram_breach:
                reasons.append(f"VRAM free {vram_free_mb}MB < {args.vram_free_mb_min}MB")
            if sustained_vram_ceiling and vram is not None:
                reasons.append(
                    f"VRAM used {vram_used_pct(vram[0], vram[1]):.1f}% >= "
                    f"{args.vram_ceiling_pct}% sustained {elapsed_over:.0f}s "
                    f">= {args.vram_ceiling_seconds:.0f}s (desync risk)"
                )
            if sustained_load:
                reasons.append(f"load1 {la1:.1f} sustained over threshold")

            if reasons:
                if breach_start_ts is None:
                    breach_start_ts = now
                    out_f.write(
                        json.dumps({"event": "breach_start", "reasons": reasons, "ts": now}) + "\n"
                    )
                    out_f.flush()
                    print(
                        f"watchdog: breach detected — {'; '.join(reasons)} "
                        f"(grace {args.breach_seconds}s)",
                        flush=True,
                    )
                sustained = (now - breach_start_ts) >= args.breach_seconds
            else:
                if breach_start_ts is not None:
                    out_f.write(json.dumps({"event": "breach_cleared", "ts": now}) + "\n")
                    out_f.flush()
                    print("watchdog: breach cleared", flush=True)
                breach_start_ts = None
                sustained = False

            if reasons and sustained and not breached:
                breach_msg = f"watchdog: BREACH — {'; '.join(reasons)}"
                print(breach_msg, flush=True)
                out_f.write(json.dumps({"event": "breach", "reasons": reasons}) + "\n")
                out_f.flush()
                breached = True
                if not args.no_kill and (args.pid or args.pgid):
                    signal_target(args.pid, args.pgid, signal.SIGTERM)
                    time.sleep(args.grace)
                    signal_target(args.pid, args.pgid, signal.SIGKILL)
                    out_f.write(json.dumps({"event": "killed_target"}) + "\n")
                    out_f.flush()
                    return 2
                if args.no_kill:
                    # In observation-only mode, keep logging but don't re-emit breach
                    pass

            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
