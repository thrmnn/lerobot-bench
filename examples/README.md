# Examples

Small, self-contained, runnable examples for `lerobot-bench`. Each one
does **one** thing and is short enough to read in full. They use the
real public API (`scripts/run_one.py`, `lerobot_bench.stats`) — nothing
here invents a function that does not exist in `src/`.

| File | What it shows | How to run |
|---|---|---|
| [`run_one_cell.md`](run_one_cell.md) | Run a single `(policy, env, seed)` cell with `run_one.py` and inspect the per-episode parquet rows it produces. | Copy-paste the commands. Documents a GPU command — does not run on import. |
| [`read_results.py`](read_results.py) | Load `results.parquet`, pool a cell's seeds, compute success rate + **Wilson 95% CI** with `lerobot_bench.stats`. | `python examples/read_results.py` |
| [`compare_two_policies.py`](compare_two_policies.py) | **Paired comparison** of two policies on a shared env: Δsuccess with a bootstrap CI, Wilcoxon test, Cohen's h, and the **MDE inconclusive check**. | `python examples/compare_two_policies.py` |

## Running the Python examples

The two `.py` examples import `lerobot_bench`. With the package
installed (`pip install -e ".[all]"` from the repo root) just run them:

```bash
python examples/read_results.py
python examples/compare_two_policies.py
```

Both accept `--results <path>` to point at a real sweep parquet (the
full sweep writes `results/sweep-full/results.parquet`; a local
`run_one.py` writes `results/results.parquet`). **If no parquet is on
disk** they fall back to a clearly-labelled *synthetic* sample so the
API call path still runs — replace it with `--results` for real
numbers. Each script's `--help` lists the policy/env flags.

These examples are read-only over a parquet — they are lightweight and
safe to run while a sweep is in progress. They never launch an eval
cell; `run_one_cell.md` *documents* that heavier command instead.

## Where to go next

- **Reading a leaderboard cell properly** — success rate, CI width, the
  MDE band, paired comparisons, "n/a", the no-op/random floor:
  [`docs/tutorials/interpreting-the-leaderboard.md`](../docs/tutorials/interpreting-the-leaderboard.md).
- **The MDE math** — Wilson half-widths and the inconclusive band at
  every N: [`docs/MDE_TABLE.md`](../docs/MDE_TABLE.md).
- **Your first real result** — full install + run walkthrough:
  [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).
- **Verifying a published cell is bit-reproducible**:
  [`docs/REPRODUCE.md`](../docs/REPRODUCE.md).
