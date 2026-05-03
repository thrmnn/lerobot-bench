---
title: lerobot-bench
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Public multi-policy benchmark for pretrained LeRobot policies.
---

# lerobot-bench (HF Space)

Public surface of the **lerobot-bench** project. Three tabs:

1. **Leaderboard** — per-cell success rate with Wilson 95% CIs.
2. **Browse Rollouts** — pick `(policy, env, seed)`, watch the
   pre-recorded MP4s stream straight from the Hub dataset.
3. **Methodology** — seeding contract, CI math, sparse-matrix policy.

Data: [`thrmnn/lerobot-bench-results-v1`](https://huggingface.co/datasets/thrmnn/lerobot-bench-results-v1).

Code: <https://github.com/thrmnn/lerobot-bench>.

This Space runs on the free CPU tier — no policy inference, no GPU.
All numbers come from a pre-computed sweep on `lerobot==0.5.1`.
