# Policy diagram guide

Companion to `paper/deck/index.html` and `docs/PIPELINE_ROADMAP.md`. **Use this when adding a new policy to the matrix** (v1.1: pi-family, future: anything else).

## What it is

The deck has a shared SVG `<defs>` block at the top of `<body>` with named **primitives** for the common pieces of a robot policy: cameras, language inputs, model backbones, action outputs, proprioception. Each policy's architecture diagram (slide C2 — "the 4 learned policies") is composed by `<use>`-ing these primitives, not by hand-drawing the rectangles each time.

This means **adding a new policy = picking primitives + positioning them**, not redoing the visual language from scratch. The deck stays consistent automatically.

## Where the primitives live

`paper/deck/index.html` — search for the comment `REUSABLE POLICY-DIAGRAM PRIMITIVES`. The block looks like:

```html
<svg style="position:absolute;width:0;height:0;overflow:hidden" aria-hidden="true">
  <defs>
    <marker id="pa" .../>      <!-- shared arrow marker -->
    <symbol id="prim-camera" viewBox="0 0 50 50">...</symbol>
    <symbol id="prim-camera-stack" viewBox="0 0 50 50">...</symbol>
    <symbol id="prim-language" viewBox="0 0 100 24">...</symbol>
    <symbol id="prim-transformer" viewBox="0 0 120 60">...</symbol>
    <symbol id="prim-vit-lm" viewBox="0 0 160 70">...</symbol>
    <symbol id="prim-vit-lm-lg" viewBox="0 0 180 80">...</symbol>
    <symbol id="prim-diffusion" viewBox="0 0 110 64">...</symbol>
    <symbol id="prim-action-chunk" viewBox="0 0 70 36">...</symbol>
    <symbol id="prim-action-traj" viewBox="0 0 50 36">...</symbol>
    <symbol id="prim-proprio" viewBox="0 0 60 30">...</symbol>
  </defs>
</svg>
```

## How to compose a diagram

Each policy's architecture lives inside a `<div class="arch">` block, with an SVG `viewBox="0 0 320 90"` (so x-coords range 0–320 and y-coords 0–90). Inside, you compose with `<use>`:

```html
<svg viewBox="0 0 320 90">
  <use href="#prim-camera"       x="4"   y="10" width="46" height="46"/>
  <use href="#prim-language"     x="0"   y="62" width="92" height="22"/>
  <line x1="52" y1="34" x2="106" y2="40" stroke="#5b8def" stroke-width="1.2"/>
  <line x1="92" y1="73" x2="106" y2="54" stroke="#5b8def" stroke-width="1.2"/>
  <use href="#prim-vit-lm"       x="106" y="14" width="146" height="64"/>
  <line x1="252" y1="46" x2="270" y2="46" stroke="#7aa3ff" stroke-width="1.4" marker-end="url(#pa)"/>
  <use href="#prim-action-chunk" x="268" y="22" width="52" height="34"/>
</svg>
```

Each `<use>` reads the symbol's internal `viewBox` and renders it into the rectangle you specify with `x/y/width/height`. The colours, strokes, labels are baked into the symbol — you don't need to repeat them.

Connecting lines, arrows, and per-policy text annotations stay inline because they're position-specific.

## Existing examples in the deck

| policy | uses these primitives |
|---|---|
| `act` | inline (bespoke — 2 cameras + transformer + chunk; pre-symbol-library) |
| `diffusion_policy` | inline (bespoke — image stack + vision encoder + diffusion U-Net) |
| `smolvla_libero` | `prim-camera` + `prim-language` + `prim-vit-lm` + `prim-action-chunk` ✓ |
| `xvla_libero` | `prim-camera` + `prim-language` + `prim-vit-lm-lg` + raw text for action ✓ |

ACT and Diffusion Policy will be migrated to the symbol system in a follow-up PR once we add a 5th policy that re-uses their primitives (i.e. another transformer-imitation policy or another diffusion policy).

## Adding a new policy — checklist

1. **Pick the primitives.** Look at the policy paper's architecture diagram and identify the input modalities, the backbone, and the output format. Map each to a primitive from the list above.
2. **If a primitive is missing**, add it to the `<defs>` block. Give it a stable `viewBox` (the published symbols are 50×50 for inputs, ~120×60 for blocks, ~70×36 for outputs — match the conventions). Use the deck palette only:
   - `--ink #f5f7fa` for text on backbones
   - `--accent #7aa3ff` for outputs and key boundaries
   - `--warm #fbbf24` for language / generative loops
   - `--defer #a78bfa` for deferred / experimental
   - `--c5ccd6` strokes for inputs (cameras, proprio)
3. **Copy the closest existing `.pcard` block** in slide C2 (currently SmolVLA is the cleanest template). Update `<span class="name">`, `<span class="sub">`, the `<span class="tag">`, the description `<p>`, and the `<div class="specs">` row.
4. **Wire the `<svg>`** by composing the primitives with `<use>` and connecting `<line>`s. Keep the 320×90 viewBox so the layout stays consistent across cards.
5. **Add a column to the tradeoff slide C3** (`.tradeoff-table`). Copy the column structure from an existing policy; fill in params, modality, inference cost, training data, best cell, spread, takeaway.
6. **Update the matrix slide 04** (`.matrix`) by adding a row label and the 6 cell `<div class="c">` entries with the appropriate `.ok` / `.warm` / `.fail` / `.def` class.
7. **Update the overview meta strip** (`#overview .ov-meta`) — bump the "policies" count from 5.
8. **Update `MODEL_CARDS.md`** with the new policy's row, `paper_rates` per env, training source, and a one-line description.
9. **Update `configs/policies.yaml`** with the new policy entry (factory, repo_id, pinned revision SHA, envs list, paper_rates).
10. **Run calibration + sweep**: `python scripts/calibrate.py --policy <name>`, then a full sweep cell.
11. **PR template**: title `feat(policy): add <name> to v1.x`; body must include before/after of slide C2 + C3 (screenshots), MDE table impact, sweep wall-clock.

## Why the symbol library exists

Three reasons:

- **Consistency.** All policy diagrams use the same camera glyph, the same arrow style, the same colour for "accent" vs "warm". Future viewers recognise the visual vocabulary across slides.
- **Reusability.** Adding a new VLA, a new diffusion variant, or a new imitation policy is a 30-second SVG composition, not a 30-minute hand-drawing exercise.
- **Auditable accuracy.** When a primitive (say, the language-input bubble) needs to change — for example, to highlight that one policy ingests tokens vs raw text — you change it in `<defs>` once and every policy diagram updates. No drift between cards.

## When to ignore the system

If a policy's architecture is genuinely unique (e.g. a hybrid hierarchical planner + low-level policy with multiple visible stages), inline-draw the SVG bespoke and add a comment `<!-- bespoke: doesn't fit primitive library -->`. Don't force a primitive that misrepresents the architecture.

---

*Last updated 2026-05-26. Maintainer: bump this doc when adding/removing primitives in `<defs>`.*
