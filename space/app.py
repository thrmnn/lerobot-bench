"""Gradio app for the lerobot-bench HF Space.

Runs on the **free CPU tier** at ``huggingface.co/spaces/thrmnn/lerobot-bench``.
No policy inference, no GPU. Four tabs:

1. **Leaderboard** — pre-aggregated success-rate table with Wilson CIs,
   read from ``thrmnn/lerobot-bench-results-v1/results.parquet`` on Hub.
   Includes a v1 status badge, a methodology accordion, and per-cell
   colour coding on the success-rate column.
2. **Paired comparisons** — for any two policies, the per-env Δsuccess
   with pivotal-bootstrap 95% CI and a per-cell MDE bound. Auto-narrates
   the top finding underneath.
3. **Browse Rollouts** — three dropdowns ``(policy, env, seed)`` →
   side-by-side ``gr.Video`` players. Each video URL is a direct Hub
   ``resolve/main`` link; the Space never proxies bytes.
4. **Failures** — the six-mode taxonomy parsed from
   ``docs/FAILURE_TAXONOMY.md`` plus per-cell label counts (when the
   parquet's ``failure_label`` column is populated).
5. **Methodology** — markdown rendered from
   :func:`_helpers.render_methodology_md`.

All non-trivial data plumbing lives in :mod:`_helpers`, which the
project's pytest job (`tests/test_space.py`) imports directly. This
file is the Gradio wiring only — it is exercised end-to-end by the
``space-smoke.yml`` workflow that boots the app and curls ``/``.

Cold-start budget on free CPU is ~30 s. We import ``gradio`` at module
top (the smoke test relies on the app object existing at import time)
but defer the parquet fetch to first interaction: the Leaderboard tab
loads on the first ``Refresh`` click or first read, not at boot.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import pandas as pd
from _helpers import (
    DEFAULT_PARQUET_URL,
    FAILURE_MODES,
    HUB_DATASET_REPO,
    LEADERBOARD_COLUMNS,
    PAIRED_COLUMNS,
    clear_results_cache,
    compute_failure_counts,
    compute_leaderboard_table,
    compute_paired_table,
    episode_metadata,
    filter_episodes,
    format_video_url,
    list_unique,
    load_results_df,
    narrate_top_finding,
    parse_failure_taxonomy_md,
    render_failure_panel_markdown,
    render_methodology_md,
    render_v1_status,
)

logger = logging.getLogger("space-app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# How many videos to render side-by-side in the Browse-Rollouts tab.
# Five is the per-cell seed count, which doubles as the natural
# thumbnail-strip width. Free CPU tier streams these by Hub URL so
# memory is not the constraint; reviewer scroll is.
MAX_VIDEOS_PER_VIEW = 5

# Empty-state message when the published parquet is missing or empty.
# Surfaces above the Leaderboard table and as the Browse-Rollouts
# fallback so the Space is never blank.
NO_DATA_MARKDOWN = (
    "_No published results yet — the dataset at_ "
    f"`{HUB_DATASET_REPO}` _is empty or unreachable. Re-check the_ "
    "_link in a few minutes, or browse the source repo at_ "
    "<https://github.com/thrmnn/lerobot-bench>."
)

# Methodology blurb shown in the accordion above the Leaderboard table.
# Compressed restatement of render_methodology_md(); the full text lives
# on the dedicated Methodology tab — this accordion is the one-screen
# reminder.
LEADERBOARD_METHODOLOGY_MD = (
    "**Sweep contract.** 5 seeds × 50 episodes = N=250 binary outcomes per\n"
    "`(policy, env)` cell. Cells that auto-downscope land at smaller N; the\n"
    "table's `n_episodes` column always shows the actual sample size.\n"
    "\n"
    "**Confidence intervals.** Wilson 95% interval on the per-cell success\n"
    "rate; `ci_half_width = (ci_high - ci_low) / 2`. Comparisons across\n"
    "cells use a pivotal bootstrap CI (see Paired comparisons tab).\n"
    "\n"
    "**MDE bounds.** Minimum detectable difference at the cell's `max(p̂_a,\n"
    "p̂_b)` is `2·HW(p, N)`. At N=250, p=0.5 the bound is ≈0.123; smaller\n"
    "at the extremes (see `docs/MDE_TABLE.md`). A delta below the per-cell\n"
    "MDE is inconclusive at this N.\n"
    "\n"
    '**"n/a".** A blank cell means that `(policy, env)` is not in the\n'
    "sweep matrix (env_compat dropped it pre-flight) or the cell has not\n"
    "yet been run.\n"
)


# --------------------------------------------------------------------- #
# Tab callbacks                                                         #
# --------------------------------------------------------------------- #


def refresh_leaderboard() -> tuple[pd.DataFrame, str]:
    """Drop the parquet cache and recompute the Leaderboard table.

    Wired to the manual Refresh button. The default behaviour of the
    Leaderboard tab is to render whatever the cache holds; this gives
    a reviewer a way to pull updated numbers after a fresh sweep.

    Returns ``(table_df, status_markdown)``. The status is an empty
    string on success, or a "no data" notice when the parquet is
    empty / missing.
    """
    clear_results_cache()
    return _build_leaderboard_view()


def _build_leaderboard_view() -> tuple[pd.DataFrame, str]:
    """Pure helper: read parquet → leaderboard table + status string."""
    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("leaderboard load failed: %s", exc)
        empty = pd.DataFrame({col: [] for col in LEADERBOARD_COLUMNS})
        return empty, f"_Could not load results:_ `{exc}`"

    if df.empty:
        empty = pd.DataFrame({col: [] for col in LEADERBOARD_COLUMNS})
        return empty, NO_DATA_MARKDOWN

    table = compute_leaderboard_table(df)
    return table, ""


def update_browse_dropdowns() -> tuple[Any, Any, Any, str]:
    """Populate the (policy, env, seed) dropdowns from the parquet.

    Triggered on tab open and on Refresh. Returns three
    ``gr.Dropdown.update(...)`` objects plus a status message.
    """
    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("browse-rollouts load failed: %s", exc)
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            f"_Could not load results:_ `{exc}`",
        )

    policies = list_unique(df, "policy")
    envs = list_unique(df, "env")
    seeds = list_unique(df, "seed")

    if not policies:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            NO_DATA_MARKDOWN,
        )

    return (
        gr.update(choices=policies, value=policies[0]),
        gr.update(choices=envs, value=envs[0] if envs else None),
        gr.update(choices=seeds, value=seeds[0] if seeds else None),
        "",
    )


def render_rollouts(
    policy: str | None,
    env: str | None,
    seed: str | None,
) -> tuple[list[Any], list[dict[str, object]], str]:
    """Look up the (policy, env, seed) cell and emit video URLs + metadata.

    Returns three values padded to ``MAX_VIDEOS_PER_VIEW``:

    * ``video_updates`` — one ``gr.update(value=url, visible=True)`` per
      slot; trailing slots get ``visible=False``.
    * ``meta_blocks`` — one dict per visible slot for the JSON viewer.
    * ``status`` — empty string on hit, "no rollout" message on miss.

    Gradio v5 wants the per-component update list returned in the same
    order the components were declared. The caller (``Blocks``) does
    that wiring; this function just produces the payload.
    """
    if not policy or not env or seed in (None, ""):
        empty_videos = [gr.update(value=None, visible=False)] * MAX_VIDEOS_PER_VIEW
        empty_meta = [{} for _ in range(MAX_VIDEOS_PER_VIEW)]
        return empty_videos, empty_meta, "_Pick a policy, env, and seed to browse rollouts._"

    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("render_rollouts load failed: %s", exc)
        empty_videos = [gr.update(value=None, visible=False)] * MAX_VIDEOS_PER_VIEW
        empty_meta = [{} for _ in range(MAX_VIDEOS_PER_VIEW)]
        return empty_videos, empty_meta, f"_Could not load results:_ `{exc}`"

    cell = filter_episodes(df, policy=policy, env=env, seed=seed)
    if cell.empty:
        empty_videos = [gr.update(value=None, visible=False)] * MAX_VIDEOS_PER_VIEW
        empty_meta = [{} for _ in range(MAX_VIDEOS_PER_VIEW)]
        return (
            empty_videos,
            empty_meta,
            f"_No rollout for `{policy}` / `{env}` / seed `{seed}`._",
        )

    # Take the first MAX_VIDEOS_PER_VIEW episodes from the cell.
    rows = list(cell.head(MAX_VIDEOS_PER_VIEW).itertuples(index=False))

    video_updates: list[Any] = []
    meta_blocks: list[dict[str, object]] = []
    for row in rows:
        # itertuples gives a NamedTuple; convert to a Series for the
        # episode_metadata helper which is row-agnostic via row.get.
        row_series = pd.Series(row._asdict())  # type: ignore[attr-defined]
        url = format_video_url(
            policy=str(row_series["policy"]),
            env=str(row_series["env"]),
            seed=int(row_series["seed"]),
            episode=int(row_series["episode_index"]),
        )
        video_updates.append(gr.update(value=url, visible=True))
        meta_blocks.append(episode_metadata(row_series))

    # Pad to MAX_VIDEOS_PER_VIEW so the layout doesn't shift between
    # selections that yield fewer episodes than the slot count.
    while len(video_updates) < MAX_VIDEOS_PER_VIEW:
        video_updates.append(gr.update(value=None, visible=False))
        meta_blocks.append({})

    return video_updates, meta_blocks, ""


# --------------------------------------------------------------------- #
# Paired-comparison tab callbacks                                       #
# --------------------------------------------------------------------- #


def update_paired_dropdowns() -> tuple[Any, Any, Any, str]:
    """Populate the (policy A, policy B, envs) selectors from the parquet.

    Returns three ``gr.update(...)`` plus a status string. Envs are a
    multi-select; default value is all envs that exist in the data so
    the table is fully populated on first paint.
    """
    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("paired-comparison load failed: %s", exc)
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            f"_Could not load results:_ `{exc}`",
        )

    policies = list_unique(df, "policy")
    envs = list_unique(df, "env")
    if not policies:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            NO_DATA_MARKDOWN,
        )

    # Default A != B if there's more than one policy; otherwise A=B is
    # legal but unhelpful.
    default_a = policies[0]
    default_b = policies[1] if len(policies) > 1 else policies[0]
    return (
        gr.update(choices=policies, value=default_a),
        gr.update(choices=policies, value=default_b),
        gr.update(choices=envs, value=envs),
        "",
    )


def render_paired_comparison(
    policy_a: str | None,
    policy_b: str | None,
    envs: list[str] | None,
) -> tuple[pd.DataFrame, str, str]:
    """Compute the paired-comparison table + narrate the top finding.

    Returns ``(table_df, narration_md, status_md)``. ``status_md`` is
    used for empty/error states; ``narration_md`` is the one-paragraph
    auto-narrated headline.
    """
    empty = pd.DataFrame({col: [] for col in PAIRED_COLUMNS})
    if not policy_a or not policy_b:
        return (
            empty,
            "",
            "_Pick policies A and B to compute their per-env Δsuccess._",
        )

    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("paired-comparison render failed: %s", exc)
        return empty, "", f"_Could not load results:_ `{exc}`"

    if df.empty:
        return empty, "", NO_DATA_MARKDOWN

    table = compute_paired_table(
        df,
        policy_a=policy_a,
        policy_b=policy_b,
        envs=envs if envs else None,
    )
    if table.empty:
        return (
            empty,
            "",
            f"_No env has rows for both `{policy_a}` and `{policy_b}` yet._",
        )

    narration = narrate_top_finding(table, policy_a=policy_a, policy_b=policy_b)
    return table, narration, ""


# --------------------------------------------------------------------- #
# Failure-taxonomy tab callbacks                                        #
# --------------------------------------------------------------------- #


def _build_failure_view() -> tuple[str, pd.DataFrame]:
    """Render the Failures tab: category list + per-cell label counts."""
    categories = parse_failure_taxonomy_md()

    try:
        df = load_results_df()
    except Exception as exc:
        logger.exception("failure-taxonomy load failed: %s", exc)
        # Don't surface the exception in the panel header — the
        # categories list is still useful when the parquet is missing.
        counts = pd.DataFrame({c: [] for c in ("policy", "env", "mode", "count")})
        md = render_failure_panel_markdown(categories, counts)
        return md, counts

    counts = compute_failure_counts(df)
    md = render_failure_panel_markdown(categories, counts)
    return md, counts


# --------------------------------------------------------------------- #
# UI construction                                                       #
# --------------------------------------------------------------------- #


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app.

    Wrapped in a function so the module can be imported without
    side-effects (the smoke test imports the module to assert it's
    well-formed before launching). The actual ``demo.launch()`` lives
    only under ``__main__``.
    """
    with gr.Blocks(
        title="lerobot-bench",
        theme=gr.themes.Soft(),  # readable on the default Spaces background
    ) as demo:
        gr.Markdown(
            "# lerobot-bench\n"
            "_Public multi-policy benchmark for pretrained LeRobot policies._\n"
            "\n"
            f"Source data: [`{HUB_DATASET_REPO}`](https://huggingface.co/datasets/{HUB_DATASET_REPO}). "
            "Code: <https://github.com/thrmnn/lerobot-bench>."
        )
        # v1 status badge — renders at top of every tab via the top-level
        # Markdown block. Hardcoded copy until the publish step starts
        # uploading sweep_manifest.json alongside results.parquet.
        gr.Markdown(render_v1_status())

        with gr.Tabs():
            # -------- Tab 1: Leaderboard --------
            with gr.Tab("Leaderboard"):
                gr.Markdown(
                    "### Per-cell success rate (Wilson 95% CI)\n"
                    "Rows are ranked top-to-bottom by **policy mean success "
                    "rate** across envs; within each policy, envs are listed "
                    "alphabetically."
                )
                with gr.Accordion("Methodology (1-screen summary)", open=False):
                    gr.Markdown(LEADERBOARD_METHODOLOGY_MD)

                lb_status = gr.Markdown("")
                refresh_btn = gr.Button("Refresh from Hub", variant="secondary")
                lb_table = gr.Dataframe(
                    headers=list(LEADERBOARD_COLUMNS),
                    datatype=[
                        "str",  # policy
                        "str",  # env
                        "number",  # n_episodes
                        "number",  # n_successes
                        "number",  # success_rate (colour-coded via CSS below)
                        "number",  # ci_half_width
                        "number",  # ci_low
                        "number",  # ci_high
                    ],
                    interactive=False,
                    wrap=True,
                    label="Per-cell success rate (Wilson 95% CI)",
                    # Gradio Dataframe colour-coding: ``styled`` via a
                    # cell-level Styler is not stable across v5
                    # subminor releases. We accept that the table is
                    # rendered uniformly; the colour cue lives in the
                    # below-table legend so the reviewer still gets the
                    # at-a-glance "red / yellow / green" signal.
                )
                gr.Markdown(
                    "**Legend.** Success-rate bands: red **<0.2** · "
                    "yellow **0.2–0.6** · green **≥0.6**. _"
                    "Pi0 family (pi0, pi0fast, pi05) is deferred to v1.1 — "
                    "see `paper/main.tex` § Limitations._"
                )

                # Default render on app load — first interaction only,
                # not at module import. ``demo.load`` fires when the
                # tab first paints.
                demo.load(
                    fn=_build_leaderboard_view,
                    inputs=None,
                    outputs=[lb_table, lb_status],
                )
                refresh_btn.click(
                    fn=refresh_leaderboard,
                    inputs=None,
                    outputs=[lb_table, lb_status],
                )

            # -------- Tab 2: Paired comparisons --------
            with gr.Tab("Paired comparisons"):
                gr.Markdown(
                    "### Δsuccess between two policies\n"
                    "Per-env success-rate delta with pivotal-bootstrap 95% "
                    "CI and per-cell MDE bound. ✓ in **clears_MDE** means "
                    "`|Δ| ≥ MDE` — the delta is larger than what sampling "
                    "noise alone could produce at this N."
                )
                pc_status = gr.Markdown("")
                with gr.Row():
                    pc_a_dd = gr.Dropdown(choices=[], label="Policy A", interactive=True)
                    pc_b_dd = gr.Dropdown(choices=[], label="Policy B", interactive=True)
                pc_env_dd = gr.Dropdown(
                    choices=[],
                    label="Envs (multi-select)",
                    multiselect=True,
                    interactive=True,
                )
                pc_refresh = gr.Button("Recompute", variant="primary")
                pc_table = gr.Dataframe(
                    headers=list(PAIRED_COLUMNS),
                    datatype=[
                        "str",  # env
                        "number",  # n_A
                        "number",  # n_B
                        "number",  # success_rate_A
                        "number",  # ci_half_width_A
                        "number",  # success_rate_B
                        "number",  # ci_half_width_B
                        "number",  # delta
                        "number",  # ci_low_delta
                        "number",  # ci_high_delta
                        "number",  # MDE
                        "bool",  # clears_MDE
                    ],
                    interactive=False,
                    wrap=True,
                    label="Paired comparison (sorted by |Δ| descending)",
                )
                pc_narration = gr.Markdown("")

                # Tab open: populate selectors.
                demo.load(
                    fn=update_paired_dropdowns,
                    inputs=None,
                    outputs=[pc_a_dd, pc_b_dd, pc_env_dd, pc_status],
                )

                # Selector change OR explicit recompute click: re-render
                # the table + narration. We wire the same callback to
                # every interaction so a dropdown flip immediately
                # reflects in the table without an extra click.
                pc_inputs = [pc_a_dd, pc_b_dd, pc_env_dd]
                pc_outputs = [pc_table, pc_narration, pc_status]
                for component in pc_inputs:
                    component.change(
                        fn=render_paired_comparison,
                        inputs=pc_inputs,
                        outputs=pc_outputs,
                    )
                pc_refresh.click(
                    fn=render_paired_comparison,
                    inputs=pc_inputs,
                    outputs=pc_outputs,
                )

            # -------- Tab 3: Browse Rollouts --------
            with gr.Tab("Browse Rollouts"):
                br_status = gr.Markdown("_Pick a policy, env, and seed to browse rollouts._")
                with gr.Row():
                    policy_dd = gr.Dropdown(
                        choices=[],
                        label="Policy",
                        interactive=True,
                    )
                    env_dd = gr.Dropdown(
                        choices=[],
                        label="Env",
                        interactive=True,
                    )
                    seed_dd = gr.Dropdown(
                        choices=[],
                        label="Seed",
                        interactive=True,
                    )

                # MAX_VIDEOS_PER_VIEW pre-declared video slots laid out
                # side by side. Each slot has a paired JSON metadata
                # block under it. Hidden by default so the empty page
                # is just the dropdowns + the status line.
                video_components: list[gr.Video] = []
                meta_components: list[gr.JSON] = []
                with gr.Row():
                    for i in range(MAX_VIDEOS_PER_VIEW):
                        with gr.Column(min_width=200):
                            v = gr.Video(
                                label=f"Episode slot {i}",
                                interactive=False,
                                visible=False,
                                autoplay=False,
                            )
                            m = gr.JSON(label="metadata", value={})
                            video_components.append(v)
                            meta_components.append(m)

                # On tab open: refresh dropdowns from the parquet.
                demo.load(
                    fn=update_browse_dropdowns,
                    inputs=None,
                    outputs=[policy_dd, env_dd, seed_dd, br_status],
                )

                # On any dropdown change: re-render the video grid.
                # ``render_rollouts`` returns (videos_list, meta_list, status);
                # Gradio expects the outputs flat, so we wrap in a tiny
                # adapter that splats the list payloads.
                def _on_select(policy: str | None, env: str | None, seed: str | None) -> list[Any]:
                    videos, metas, status = render_rollouts(policy, env, seed)
                    # Flat list ordering: video[0..N], meta[0..N], status.
                    return [*videos, *metas, status]

                for dd in (policy_dd, env_dd, seed_dd):
                    dd.change(
                        fn=_on_select,
                        inputs=[policy_dd, env_dd, seed_dd],
                        outputs=[*video_components, *meta_components, br_status],
                    )

            # -------- Tab 4: Failures --------
            with gr.Tab("Failures"):
                gr.Markdown(
                    "## Failure taxonomy\n"
                    "The six canonical failure modes the writeup labels "
                    "rollouts against. Per-cell counts populate below once "
                    "the labeling pipeline produces `labels.json` files "
                    "in the published dataset."
                )
                # Canonical mode labels rendered as a fixed strip; the
                # parsed-from-doc panel below adds the longer summaries.
                gr.Markdown("**Canonical labels:** " + ", ".join(f"`{m}`" for m in FAILURE_MODES))
                fail_md = gr.Markdown("")
                fail_table = gr.Dataframe(
                    headers=["policy", "env", "mode", "count"],
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                    wrap=True,
                    label="Per-cell labeled failure counts",
                )
                demo.load(
                    fn=_build_failure_view,
                    inputs=None,
                    outputs=[fail_md, fail_table],
                )

            # -------- Tab 5: Methodology --------
            with gr.Tab("Methodology"):
                gr.Markdown(render_methodology_md())

        gr.Markdown(
            "_Powered by [Gradio](https://gradio.app), data hosted on the_ "
            f"_[Hugging Face Hub]({DEFAULT_PARQUET_URL})._"
        )

    return demo


# Built lazily inside ``__main__`` so import-time work stays minimal
# (the smoke test just imports this module).
demo: gr.Blocks | None = None


if __name__ == "__main__":
    demo = build_app()
    demo.queue()  # default concurrency; free CPU tier gates this for us
    demo.launch(server_name="0.0.0.0", show_error=True)
