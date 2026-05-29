# site/

Landing page for **lerobot-bench**. Static HTML+CSS, no build step, no JS framework. Intended to be served at `thrmnn.github.io/lerobot-bench` once GitHub Pages is enabled.

This is the *narrative* surface (recruiter/researcher reads the headline, clicks through). The *interactive* surface is the Gradio Space at `huggingface.co/spaces/thrmnn/lerobot-bench` (separate codebase under `space/`).

## Files

| Path | Purpose |
| --- | --- |
| `index.html` | Single landing page: hero, artifact buttons, methodology badges, results figures, headline-finding card, methodology snapshot, v1.0.1 audit caveats, footer. |
| `style.css` | Palette + typography mirror `paper/deck/index.html`. Responsive at <800px. |
| `assets/*.svg` | Self-contained copies of the web-style figures (`replication_scatter`, `forest_plot`) from `paper/figures/web/`, so the page works when Pages serves `/site` as root. Re-copy after `scripts/render_figures.py --style web` if the source figures change. (The ACT probe comparison is rendered as a CSS chart in the headline card, so `act_probe_bar.svg` is not embedded.) |

## Preview locally

Bind to `0.0.0.0` so it's reachable over Tailscale from the tab:

```bash
python -m http.server 8765 --bind 0.0.0.0 -d site
```

Then open `http://100.104.205.62:8765/` (or `http://localhost:8765/` from the host).

## Deploy

GitHub Pages — **not enabled yet**. When the user is ready:

1. Settings → Pages → Source: `Deploy from a branch`, branch `main`, folder `/site`.
2. Site goes live at `https://thrmnn.github.io/lerobot-bench/`.

No build step, no CI — Pages serves the static files directly.

## Visual reference

The palette, typography hierarchy (`.h-1`, `.h-2`, `.eyebrow`, `.mono`), and card styling are lifted from `paper/deck/index.html`. If the deck's visual language changes, mirror it here too.
