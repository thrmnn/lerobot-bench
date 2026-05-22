# Troubleshooting

Failure modes you are likely to hit running `lerobot-bench`, and the fix for
each. Every entry gives a copy-paste command. If a script's own error message
points you here, the matching section is below.

See also: [`RUNBOOK.md`](RUNBOOK.md) for sweep operations,
[`REPRODUCE.md`](REPRODUCE.md) for verifying published cells.

---

## Headless MuJoCo rendering (`DISPLAY` / `MUJOCO_GL`)

**Symptom:** a sim env crashes on import or `reset()` with
`GLFW error`, `Failed to initialize GLEW`, `Could not create GL context`,
or `cannot open display`. Common on a VPS, an HPC node, or WSL2 without an
X server.

**Cause:** MuJoCo defaults to a windowed GL backend (`glfw`) that needs a
display. Headless hosts have none.

**Fix:** force an off-screen backend before launching anything that imports
the sim:

```bash
export MUJOCO_GL=egl     # GPU off-screen; needs a working EGL driver
# or, if EGL is unavailable / flaky:
export MUJOCO_GL=osmesa  # CPU software rendering, slower but always works
```

Put it in the same shell as the sweep, or prefix the command:

```bash
MUJOCO_GL=egl python scripts/run_one.py --policy act --env pusht --seed 0
```

`unset DISPLAY` as well if a stale `DISPLAY` is set but no server is running —
some MuJoCo versions try `glfw` whenever `DISPLAY` is non-empty.

---

## CUDA not found / no GPU

**Symptom:** `torch.cuda.is_available()` is `False`, `nvidia-smi` is missing,
or a re-run with `--device cuda` fails with `CUDA driver version is
insufficient` / `no CUDA-capable device is detected`.

**Checks:**

```bash
nvidia-smi                                   # driver + GPU visible?
python -c "import torch; print(torch.cuda.is_available())"
```

**Fixes:**

- **No GPU on this host** — run on CPU instead. `reproduce_cell.py` takes
  `--device cpu`; `calibrate.py` falls back to CPU timings automatically when
  CUDA is absent (VRAM fields read 0). Note a CPU re-run is much slower and,
  for the reproducibility check, must still match the reference bit-for-bit.

  ```bash
  python scripts/reproduce_cell.py --policy act --env pusht --seed 0 --device cpu
  ```

- **GPU present but `torch` can't see it** — the conda env has a CPU-only
  torch build, or the CUDA driver is older than the torch CUDA runtime.
  Recreate the `lerobot` env (the `lerobot==0.5.1` pin is sacred — do not bump
  it; see [`REPRODUCE.md`](REPRODUCE.md)).

- **WSL2** — CUDA needs a recent Windows NVIDIA driver; the driver is supplied
  by Windows, not installed inside the distro. If `nvidia-smi` fails inside
  WSL2, update the Windows-side driver.

---

## `run_capped.sh` pre-flight refusal

**Symptom:** `scripts/run_capped.sh` exits 3 with:

```
pre-flight: REFUSE — RAM used 61% > LAUNCH_MAX_USED_PCT=55%
```

**Cause:** the pre-flight gate refuses to launch a memory-heavy job when the
host is already busy, so a browser / IDE / parallel workload is not starved.

**Fixes:**

- Free RAM (close apps, pause parallel work) and retry — this is the intended
  response.
- If the other load is expected and acceptable, raise the gate for that one
  launch:

  ```bash
  LAUNCH_MAX_USED_PCT=70 scripts/run_capped.sh 18G -- python scripts/calibrate.py
  ```

  Do **not** raise it blindly: the cgroup `MemoryMax` cap still applies, but a
  host already near full RAM can still thrash before the cgroup bites.

**Other `run_capped.sh` exit-2 errors** (`no memory cap given`, bad usage)
mean the invocation is malformed — run `scripts/run_capped.sh --help`. The cap
goes first, then a literal `--`, then the command:

```bash
scripts/run_capped.sh 18G -- python scripts/run_one.py --policy act --env pusht --seed 0
```

---

## Parquet mid-write read errors

**Symptom:** `review_results.py` exits 2 with `could not read results parquet:
... mid-write`, or `pyarrow` raises `Invalid: ... Parquet magic bytes not
found` / a truncated-footer error.

**Cause:** the running sweep writes `results.parquet` one cell at a time. A
read that lands during a write sees a half-written file. This is **transient,
not corruption** — the sweep is fine.

**Fix:** just re-run the reader a few seconds later:

```bash
python scripts/review_results.py
```

`review_results.py` is strictly read-only and never touches the sweep, so it
is always safe to re-run (or cron) against a live sweep dir. The same applies
to `reproduce_cell.py` reading the reference parquet.

---

## The conda env

**Symptom:** `ModuleNotFoundError: No module named 'lerobot'`,
`calibrate.py` cell status `error: missing runtime`, or
`reproduce_cell.py` exiting 3 because `run_one.py` could not import torch.

**Cause:** the `lerobot` conda env is not activated, or the editable install
is missing.

**Fix:**

```bash
conda activate lerobot                       # env name is `lerobot`
python -c "import lerobot; print(lerobot.__version__)"   # must print 0.5.1
```

`lerobot==0.5.1` is a **sacred pin** — every published number depends on it.
Do not bump it to silence an import error; recreate the env instead. See
[`REPRODUCE.md`](REPRODUCE.md) for the full environment setup.

**Worktree note:** the editable install (`__editable__.*.pth`) points at one
fixed `src/` tree. If you are working inside a git worktree under
`.claude/worktrees/`, prefix checks with `PYTHONPATH`:

```bash
PYTHONPATH=$(pwd)/src python scripts/calibrate.py --dry-run
```

---

## Quick reference: script exit codes

| Script | Non-zero exit means |
| --- | --- |
| `calibrate.py` | `2` partial (some cells failed) · `3` nothing to calibrate · `4` missing runtime/config |
| `review_results.py` | `1` anomalies flagged · `2` could not run (parquet/config) |
| `reproduce_cell.py` | `1` mismatch vs reference · `2` reference/cell missing · `3` re-run failed |
| `merge_calibration.py` / `auto_downscope.py` | `2` input file missing or not a calibration report |
| `run_capped.sh` | `2` bad invocation · `3` pre-flight RAM refusal |
| `watchdog.py` | `2` a breach was sustained and the target was killed |

Run any script with `--help` for a copy-paste example invocation.
