"""WSL2-aware resource watchdog.

Samples RAM, VRAM, and load every `--interval` seconds. If usage breaches
thresholds (defaults tuned for 32 GB WSL2 + 8 GB GPU), sends SIGTERM to
`--pid` (or its process group via `--pgid`), waits `--grace` seconds, then
SIGKILL.

Defaults trigger at:
- RAM available < 30% of total (i.e. > 70% used) — protects Windows host
- VRAM free < 200 MB
- Load > 2× CPU count for 3 consecutive samples

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


def signal_target(pid: int | None, pgid: int | None, sig: int) -> None:
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
    elif pid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--pid", type=int, default=None, help="Target PID to kill on breach")
    ap.add_argument("--pgid", type=int, default=None, help="Target process group ID")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--grace", type=float, default=10.0, help="SIGTERM→SIGKILL grace seconds")
    ap.add_argument(
        "--ram-used-pct-max",
        type=float,
        default=70.0,
        help="Trigger if used RAM exceeds this percent of total",
    )
    ap.add_argument("--vram-free-mb-min", type=int, default=200)
    ap.add_argument("--load-mult-max", type=float, default=2.0, help="× CPU count")
    ap.add_argument(
        "--load-consec",
        type=int,
        default=3,
        help="Require this many consecutive load breaches",
    )
    ap.add_argument(
        "--breach-seconds",
        type=float,
        default=0.0,
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
    breached = False

    print(
        f"watchdog: pid={args.pid} pgid={args.pgid} interval={args.interval}s "
        f"ram_max={args.ram_used_pct_max}% vram_min={args.vram_free_mb_min}MB "
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

            sample = {
                "ts": time.time(),
                "ram_used_pct": round(ram_used_pct, 2),
                "ram_avail_kb": mem_avail,
                "swap_used_kb": mi.get("SwapTotal", 0) - mi.get("SwapFree", 0),
                "vram_used_mb": vram[0] if vram else None,
                "vram_total_mb": vram[1] if vram else None,
                "vram_free_mb": vram_free_mb,
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

            reasons = []
            if ram_breach:
                reasons.append(f"RAM used {ram_used_pct:.1f}% > {args.ram_used_pct_max}%")
            if vram_breach:
                reasons.append(f"VRAM free {vram_free_mb}MB < {args.vram_free_mb_min}MB")
            if sustained_load:
                reasons.append(f"load1 {la1:.1f} sustained over threshold")

            now = sample["ts"]
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
