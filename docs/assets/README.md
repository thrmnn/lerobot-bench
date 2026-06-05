# docs/assets

Visual identity and README/social imagery for **embodimetry**.

## Vector assets (committed)

| File | Description |
|---|---|
| `logo.svg` | Primary wordmark — icon mark + `embodimetry` text. Monochrome (`currentColor`), works on light and dark backgrounds. Used in the README hero. |
| `mark.svg` | Compact square icon mark (128×128). For favicons and the Hugging Face Space avatar. Same `currentColor` treatment. |
| `social-card.svg` | 1280×640 social-preview card — project name, value prop, the 6×6 matrix hint, headline stats, MIT badge. Use for the GitHub repo social preview and Twitter/HF cards. Brand palette (graphite `#14171C`, reference blue `#2F6DF0`, calibrated green `#1F9D6B`). |
| `fig-capability-ladder.svg` | 1000×240 strip showing the L0→L4 capability ladder (zero-shot → fine-tune → classical control → world-model MPC → RL/guarantees). Brand palette; embedded in the README "Capability ladder" section. |
| `fig-v1-leaderboard-bars.svg` | 1000×430 horizontal bar chart of the six v1 published policy×env cells with Wilson 95% CIs. Numbers sourced from `results.parquet` / paper Table 2; embedded in the README "v1 leaderboard" section. |

The mark is a benchmark bar-chart whose bars are capped with robotic nodes —
robotics + measurement, restrained and technical.

### Regenerating raster versions

GitHub's social-preview uploader and some HF surfaces want a PNG. Export from
the SVG when needed (requires `rsvg-convert` or `inkscape`):

```bash
rsvg-convert -w 1280 -h 640 docs/assets/social-card.svg -o docs/assets/social-card.png
rsvg-convert -w 256  -h 256 docs/assets/mark.svg        -o docs/assets/mark.png
```

## Screenshots — TODO checklist

These are **not yet captured**. The README references them as image slots that
degrade gracefully while absent. The maintainer captures them in one pass once
both apps run locally — this directory holds the slots and this index.

Two apps, captured separately:

- **Space** — public leaderboard, 5 tabs (Leaderboard, Paired comparisons,
  Rollouts, Failures, About). Run with `python space/app.py`.
- **Dashboard** — local operator view, 7 tabs (Sweep progress, Calibration
  inspector, Rollout preview, Live event log, Policies, Envs, About). Run with
  `make dashboard`.

### Capture list

| File | App / tab | Recommended size | Referenced by |
|---|---|---|---|
| `leaderboard.png` | Space — Leaderboard tab | 1600×1000 | README hero image |
| `space-paired.png` | Space — Paired comparisons tab | 1600×1000 | README "Comparing policies" section |
| `space-rollouts.png` | Space — Rollouts tab (videos loaded) | 1600×1000 | README "Browsing rollouts" section |
| `dashboard-progress.png` | Dashboard — Sweep progress tab | 1600×1000 | README "Operator dashboard" section |
| `dashboard-rollout.png` | Dashboard — Rollout preview tab | 1600×1000 | README "Operator dashboard" section |
| `dashboard-calibration.png` | Dashboard — Calibration inspector tab | 1600×1000 | docs/RUNBOOK.md calibration step |
| `dashboard-events.png` | Dashboard — Live event log tab | 1600×1000 | docs/RUNBOOK.md monitoring step |
| `rollout-still.png` | Single frame from an episode rollout MP4 | native (e.g. 640×480) | README hero / social card backdrop |

### Capture notes

- Capture at 2× device pixel ratio where possible, then trim to content.
- Keep each PNG under ~500 KB (downscale or `pngquant` if needed).
- Use the default Gradio theme — the Space ships `gr.themes.Soft()`, the
  dashboard `gr.themes.Default()`; capture each as it renders, do not restyle.
- For tab screenshots, wait for the data tables / videos to finish loading so
  the slot shows real content, not an empty-state hint.
- The hero image (`leaderboard.png`) is the most visible asset — capture it
  last, with a populated results parquet, so the leaderboard table is full.
