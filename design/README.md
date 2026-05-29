# Visual identity kit

Single source of truth for the lerobot-bench brand: palette, typography, mark, voice.
This directory is the design-system equivalent of [`docs/POLICY_DIAGRAM_GUIDE.md`](../docs/POLICY_DIAGRAM_GUIDE.md) — a reusable visual language for the deck, the live HF Space, the README chrome, and any future surface (slides, posters, blog posts).

If you are adding a slide, a chart, a Space tab, or a README badge: read this first, then propagate changes back to consumers.

## Brand identity

lerobot-bench is a public, reproducible benchmark of pretrained LeRobot manipulation policies — five policies, six sim envs, 5,250 episodes, every per-episode result open. The visual identity carries the same posture as the writing: dark surface, monospaced numerics, evidence-first composition, semantic color reserved for things that mean something (replicated, below paper, deferred, caveated).

**Tagline.** *Does pretrained robot software actually work?* — from the deck's title meta. Use it as a hero subtitle, not a slogan.

## Files in this kit

| File | What |
|---|---|
| [`logo.svg`](logo.svg) | Canonical mark · 128×128 · `currentColor` · monochrome |
| [`logo-wordmark.svg`](logo-wordmark.svg) | Mark + `lerobot-bench` wordmark · monochrome |
| [`palette.svg`](palette.svg) | Color-swatch reference rendered with hex + role + contrast ratio |
| [`palette.json`](palette.json) | Machine-readable palette for build scripts (Tailwind config, theme generator) |
| [`typography-sample.svg`](typography-sample.svg) | Type stack rendered with the live Google Fonts |

## Color palette

Identical to [`paper/deck/index.html`](../paper/deck/index.html) `:root` — do not fork. Contrast ratios computed against `bg` per WCAG 2.x; the relevant text threshold is 4.5:1 (AA body) / 7:1 (AAA body).

### Surfaces

| Name | Hex | Role |
|---|---|---|
| `bg` | `#0a0d12` | page background |
| `panel` | `#11161e` | card / panel surface |
| `panel-2` | `#161c26` | hover / elevated panel |
| `line` | `#1e242e` | hairline divider |
| `line-2` | `#2a323e` | stronger divider / border |

### Text

| Name | Hex | Role | vs `bg` |
|---|---|---|---|
| `ink` | `#f5f7fa` | primary text | 16.5:1 — AAA |
| `dim` | `#c5ccd6` | secondary / body | 11.0:1 — AAA |
| `mute` | `#7d8593` | tertiary / labels / captions | 4.4:1 — AA UI / large text only |

`mute` sits *just* under the 4.5:1 AA body threshold. Use it for labels (mono eyebrows, captions, footer chrome), never for primary body copy.

### Semantic

| Name | Hex | Role | vs `bg` |
|---|---|---|---|
| `accent` | `#7aa3ff` | primary accent · links · highlight | 7.4:1 — AAA |
| `accent-soft` | `#5b8def` | accent pressed / secondary blue | — |
| `ok` | `#34d399` | success · replicated cell | 9.8:1 — AAA |
| `warm` | `#fbbf24` | warning · methodology caveat · scope flag | 11.3:1 — AAA |
| `fail` | `#f87171` | failure · measured below paper | 6.6:1 — AA |
| `defer` | `#a78bfa` | deferred / experimental (xvla, pi0 family) | 6.7:1 — AA |

Semantic colors are load-bearing. `ok` means a cell replicated within CI of the paper number; `fail` means it didn't; `warm` flags a methodology caveat (e.g. single-task probe vs 10-task suite average); `defer` flags a policy that was executed but excluded from the v1 leaderboard (see [`docs/DEFERRED_POLICIES.md`](../docs/DEFERRED_POLICIES.md)). Don't repurpose them for decoration.

### Expanding the palette

If a new surface needs a color outside this set: open a PR to **this file first**, justify the addition, then propagate to `paper/deck/index.html`'s `:root` block and any other consumers in the same PR. Keep this file and the deck CSS in lockstep.

## Typography

Three families, loaded together from Google Fonts:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400..700;1,400..700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
```

| Family | Use | Fallback chain |
|---|---|---|
| **Instrument Sans** | Body, UI, headings | `'Helvetica Neue', Arial, sans-serif` |
| **Instrument Serif** (italic) | Display moments, "voice" phrases, mega heads | `'Cambria', Georgia, serif` |
| **JetBrains Mono** | Numbers, code, eyebrows, captions, chrome labels | `'SFMono-Regular', Menlo, Consolas, monospace` |

Numerics always run with `font-feature-settings: "tnum"` so columns align in tables and stat blocks. Italic serif is reserved — use it for one phrase per slide, not for prose.

See [`typography-sample.svg`](typography-sample.svg) for a rendered specimen.

## Visual primitives

The deck ships a reusable SVG `<symbol>` library inline in [`paper/deck/index.html`](../paper/deck/index.html) (cameras, arms, neural-net glyphs, etc.) and a documented diagram language for policy-as-function illustrations. **Do not duplicate either here.**

- SVG symbol library — `paper/deck/index.html`, `<defs>` block near the top
- Policy diagram conventions — `docs/POLICY_DIAGRAM_GUIDE.md` *(if present; see [`docs/`](../docs/))*

When you need a glyph, reach for the deck's symbol library before drawing a new one.

## Logo + monogram

The canonical mark is the chart-bar icon used in the deck chrome (`.chrome .brand svg` in `paper/deck/index.html`): a rounded square frame containing three rising bars. It reads as "benchmark" without literal labels and stays legible at 16×16 favicon scale.

- [`logo.svg`](logo.svg) — bare mark, 128×128, `currentColor`, `stroke-width="10"`. Use as favicon, app icon, slide chrome.
- [`logo-wordmark.svg`](logo-wordmark.svg) — mark + `lerobot-bench` wordmark in JetBrains Mono Bold; `-bench` rendered at 55% opacity to echo the deck's two-tone wordmark treatment. Use for headers, social cards, README hero.

Both files are monochrome and inherit color via `currentColor` — drop them into any surface and they pick up the host's text color. Don't recolor.

There are also older 6-stroke variants at [`docs/assets/logo.svg`](../docs/assets/logo.svg) and [`docs/assets/mark.svg`](../docs/assets/mark.svg) with dot-capped bars and a baseline. Those predate the deck and are kept for the existing README hero and HF social-card; the deck-chrome geometry in this directory is the **canonical** mark going forward. Don't introduce a third variant.

## Voice + tone

Operator docs, not marketing copy. The writing is evidence-first, citation-dense, and careful with strong claims — every headline number is footnoted with the scope it was measured under, every caveat is surfaced inline rather than buried in a footnote. The deck's framing — "the replication gap, in one table" — is the canonical tone: state the gap, show the data, name the scope. Use semantic color (`warm` for caveat, `fail` for below-paper, `ok` for replicated) so the reader can scan a table and see the story before they read it.

Don't write copy that the data hasn't earned. If a number is a lower bound, say it's a lower bound. If a row compares apples to oranges, flag the row with `warm` and explain in a `mute` footer.

## Where this gets consumed

- `paper/deck/index.html` — `:root` CSS block, inline SVG icons, chrome wordmark
- `README.md` — hero image, badge strip, `docs/assets/social-card.svg`
- `space/app.py` — Gradio leaderboard (color-coded cells)
- `dashboard/` — local operator dashboard (log-tail coloring, calibration inspector)
- `paper/main.tex` — figures regenerated from `notebooks/01-write-finding.ipynb`

Future visual-language additions land via a PR to this README first, then propagate to consumers in the same PR. Keep the kit and its consumers in lockstep.
