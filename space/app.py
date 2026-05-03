"""Gradio app for the lerobot-bench HF Space.

Runs on the **free CPU tier** at ``huggingface.co/spaces/thrmnn/lerobot-bench``.
No policy inference, no GPU. Three tabs:

1. **Leaderboard** — pre-aggregated success-rate table with Wilson CIs,
   read from ``thrmnn/lerobot-bench-results-v1/results.parquet`` on Hub.
2. **Browse Rollouts** — three dropdowns ``(policy, env, seed)`` →
   side-by-side ``gr.Video`` players. Each video URL is a direct Hub
   ``resolve/main`` link; the Space never proxies bytes.
3. **Methodology** — markdown rendered from
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
    HUB_DATASET_REPO,
    LEADERBOARD_COLUMNS,
    clear_results_cache,
    compute_leaderboard_table,
    episode_metadata,
    filter_episodes,
    format_video_url,
    list_unique,
    load_results_df,
    render_methodology_md,
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

        with gr.Tabs():
            # -------- Tab 1: Leaderboard --------
            with gr.Tab("Leaderboard"):
                lb_status = gr.Markdown("")
                refresh_btn = gr.Button("Refresh from Hub", variant="secondary")
                lb_table = gr.Dataframe(
                    headers=list(LEADERBOARD_COLUMNS),
                    datatype=[
                        "str",  # policy
                        "str",  # env
                        "number",  # n_episodes
                        "number",  # n_successes
                        "number",  # success_rate
                        "number",  # ci_half_width
                        "number",  # ci_low
                        "number",  # ci_high
                    ],
                    interactive=False,
                    wrap=True,
                    label="Per-cell success rate (Wilson 95% CI)",
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

            # -------- Tab 2: Browse Rollouts --------
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

            # -------- Tab 3: Methodology --------
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
