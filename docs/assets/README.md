# docs/assets

Visual identity and README/social imagery for **lerobot-bench**.

## Vector assets (committed)

| File | Description |
|---|---|
| `logo.svg` | Primary wordmark — icon mark + `lerobot-bench` text. Monochrome (`currentColor`), works on light and dark backgrounds. Used in the README hero. |
| `mark.svg` | Compact square icon mark (128×128). For favicons and the Hugging Face Space avatar. Same `currentColor` treatment. |
| `social-card.svg` | 1280×640 social-preview card — project name, value prop, the 6×6 matrix hint, headline stats, MIT badge. Use for the GitHub repo social preview and Twitter/HF cards. |

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
degrade gracefully while absent. The project maintainer captures them once the
apps are running locally — this directory just holds the slots and this index.

- [ ] `leaderboard.png` — HF Space leaderboard tab (`thrmnn/lerobot-bench`). README hero image.
- [ ] `dashboard-progress.png` — local operator dashboard, live sweep-progress tab (`make dashboard`).
- [ ] `dashboard-rollout.png` — local operator dashboard, rollout video-preview tab.
- [ ] `rollout-still.png` — a representative frame from an episode rollout MP4.

Capture at 2× device pixel ratio where possible, trim to content, and keep each
PNG under ~500 KB.
