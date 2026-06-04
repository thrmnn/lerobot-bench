---
title: embodimetry
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

# embodimetry (HF Space)

Public surface of the **embodimetry** project. Five tabs:

1. **Leaderboard** — per-cell success rate with Wilson 95% CIs.
2. **Paired comparisons** — per-env Δsuccess between two policies with
   pivotal-bootstrap CIs and a per-cell MDE bound.
3. **Rollouts** — pick `(policy, env, seed)`, watch the pre-recorded
   MP4s stream straight from the Hub dataset.
4. **Failures** — the six canonical failure modes; per-cell counts
   populate once the labeling pipeline ships `labels.json`.
5. **About** — seeding contract, CI math, sparse-matrix policy.

Data: [`thrmnn/embodimetry-v1`](https://huggingface.co/datasets/thrmnn/embodimetry-v1).

Code: <https://github.com/thrmnn/lerobot-bench>.

This Space runs on the free CPU tier — no policy inference, no GPU.
All numbers come from a pre-computed sweep on `lerobot==0.5.1`.
