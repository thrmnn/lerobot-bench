"""Local-first sweep dashboard for the operator running an overnight sweep.

This is **not** the public-facing HF Space (that lives in ``space/``).
This dashboard runs on the operator's laptop, reads sweep state + videos
**from disk only** (no Hub fetches), and is meant to answer one question:

    "I just launched ``make sweep`` -- what's it actually doing, and
    is the calibration matrix shape I shipped sane?"

Three tabs:

1. **Sweep progress** -- live table of ``(policy, env)`` cells with
   status / seeds-done / ETA, auto-refreshing every 5 s. Reads
   ``results/<run>/sweep_manifest.json``.
2. **Calibration inspector** -- table of the latest
   ``results/calibration-*.json`` cells with timing, VRAM, and the
   downscope reason.
3. **Rollout preview** -- (policy, env, seed, episode) dropdown
   cascade -> HTML5 ``gr.Video`` player. Scans the local results dir
   and the Windows-mounted Robotics-Data drive for MP4s.

All non-trivial plumbing lives in :mod:`_helpers`. The companion
``tests/test_dashboard.py`` imports that module directly so the
dashboard's data layer is exercised without Gradio in the test
environment (Gradio is in the ``[space]`` extras but not ``[dev]``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from _helpers import (
    CALIBRATION_COLUMNS,
    DEFAULT_LOG_TAIL_LINES,
    DEFAULT_VIDEO_ROOTS,
    PROGRESS_COLUMNS,
    build_calibration_table,
    build_progress_table,
    clear_video_cache,
    discover_sweep_logs,
    discover_sweep_runs,
    env_dashboard_logs_dir,
    env_dashboard_results_dir,
    find_latest_calibration,
    find_video_path,
    format_log_lines_html,
    load_calibration_report,
    load_manifest,
    scan_video_index,
    summarize_log,
    tail_log_lines,
    video_index_options,
)

logger = logging.getLogger("dashboard-app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# How often the progress tab repaints. 5 s is plenty for an overnight
# sweep; the manifest writer flushes every cell boundary, so anything
# tighter would just burn battery without showing fresher data.
PROGRESS_REFRESH_SECONDS = 5.0


# --------------------------------------------------------------------- #
# Tab 1: Sweep progress                                                 #
# --------------------------------------------------------------------- #


def _runs_dropdown_choices() -> tuple[list[tuple[str, str]], str | None]:
    """Discover sweep runs and return ``(choices, default)``.

    Choices are ``[(label, manifest_path_str), ...]`` so Gradio shows
    a human label while the callback receives the absolute path. The
    default is the most recently started run -- the one the operator
    most likely cares about.
    """
    runs = discover_sweep_runs(env_dashboard_results_dir())
    if not runs:
        return [], None
    choices = [(r.label, str(r.manifest_path)) for r in runs]
    return choices, str(runs[0].manifest_path)


def refresh_progress(
    manifest_path_str: str | None,
) -> tuple[pd.DataFrame, str]:
    """Re-read the manifest and recompute the progress table.

    Called by the 5 s timer and the manual Refresh button. Returns
    ``(table_df, status_markdown)``. Status is empty on success or a
    helpful hint when no manifest is selected.
    """
    if not manifest_path_str:
        empty = pd.DataFrame({c: [] for c in PROGRESS_COLUMNS})
        return empty, _no_sweep_hint()

    manifest = load_manifest(Path(manifest_path_str))
    if not manifest:
        empty = pd.DataFrame({c: [] for c in PROGRESS_COLUMNS})
        return empty, (
            f"_Manifest at `{manifest_path_str}` is missing or unreadable. "
            "Has the sweep started yet?_"
        )

    table = build_progress_table(manifest)
    summary = _summarise_progress(table)
    return table, summary


def _no_sweep_hint() -> str:
    """Helpful empty-state for the progress tab."""
    return (
        "_No sweep manifest found under_ "
        f"`{env_dashboard_results_dir()}`. _Start one with_ `make sweep ARGS=...` "
        "_or set_ `DASHBOARD_RESULTS_DIR` _to point at a different results root._"
    )


def _summarise_progress(table: pd.DataFrame) -> str:
    """One-line summary above the progress table.

    Headline counts are by cell-status; the per-policy breakdown lives
    inside the table. Empty table -> a hint, not a misleading "0/0".
    """
    if table.empty:
        return "_No cells in this manifest._"
    counts = table["status"].value_counts().to_dict()
    parts = [f"**{int(v)} {k}**" for k, v in counts.items()]
    return " · ".join(parts)


def _on_run_selected(manifest_path_str: str | None) -> tuple[pd.DataFrame, str]:
    """Dropdown change -> re-render the progress table for that run."""
    return refresh_progress(manifest_path_str)


def _on_runs_refresh() -> tuple[Any, pd.DataFrame, str]:
    """Re-scan disk for new runs and re-render the table for the newest.

    Wired to the "Re-scan runs" button. Returns
    ``(dropdown_update, table_df, status_markdown)``.
    """
    choices, default = _runs_dropdown_choices()
    table, summary = refresh_progress(default)
    return (
        gr.update(choices=choices, value=default),
        table,
        summary,
    )


# --------------------------------------------------------------------- #
# Tab 2: Calibration                                                    #
# --------------------------------------------------------------------- #


def refresh_calibration() -> tuple[pd.DataFrame, str]:
    """Find the latest calibration JSON and build its table.

    Returns ``(table_df, status_markdown)``. Status describes which
    file was read (path + cell count) so the operator can confirm
    they're inspecting the right run.
    """
    path = find_latest_calibration(env_dashboard_results_dir())
    if path is None:
        empty = pd.DataFrame({c: [] for c in CALIBRATION_COLUMNS})
        return empty, (
            "_No calibration JSON found under_ "
            f"`{env_dashboard_results_dir()}`. _Run_ `make calibrate` _first._"
        )

    report = load_calibration_report(path)
    if not report:
        empty = pd.DataFrame({c: [] for c in CALIBRATION_COLUMNS})
        return empty, f"_Could not read calibration JSON at_ `{path}`."

    table = build_calibration_table(report)
    n_cells = len(table)
    ts = report.get("timestamp_utc", "unknown")
    sha = report.get("git_sha", "unknown")[:8]
    return (
        table,
        (f"Reading **{path.name}** (timestamp `{ts}`, git `{sha}`); {n_cells} cell(s) in matrix."),
    )


# --------------------------------------------------------------------- #
# Tab 3: Rollout preview                                                #
# --------------------------------------------------------------------- #


def _refresh_video_dropdowns() -> tuple[Any, Any, Any, Any, str]:
    """Scan disk for videos and populate the four dropdowns.

    Returns ``(policy_update, env_update, seed_update, episode_update,
    status_markdown)``. Status reports the count + roots scanned so
    the operator can confirm the Windows-mounted drive was found.
    """
    index = scan_video_index(DEFAULT_VIDEO_ROOTS)
    opts = video_index_options(index)

    def _first_or_none(values: list[str]) -> str | None:
        return values[0] if values else None

    status = _video_index_status(index)
    return (
        gr.update(choices=opts["policy"], value=_first_or_none(opts["policy"])),
        gr.update(choices=opts["env"], value=_first_or_none(opts["env"])),
        gr.update(choices=opts["seed"], value=_first_or_none(opts["seed"])),
        gr.update(choices=opts["episode"], value=_first_or_none(opts["episode"])),
        status,
    )


def _video_index_status(index: Any) -> str:
    roots = ", ".join(f"`{r}`" for r in index.roots)
    return f"Indexed **{index.n_videos}** MP4(s) across {roots}."


def _on_video_rescan() -> tuple[Any, Any, Any, Any, str]:
    """Manual re-scan: clear the cache then repopulate dropdowns."""
    clear_video_cache()
    return _refresh_video_dropdowns()


def _on_video_select(
    policy: str | None,
    env: str | None,
    seed: str | None,
    episode: str | None,
) -> tuple[Any, str]:
    """Look up the MP4 for the current dropdown combination.

    Returns ``(video_update, status_markdown)``. On miss returns a
    "no rollout for this combination" message so the page never
    crashes on a dropdown shuffle.
    """
    index = scan_video_index(DEFAULT_VIDEO_ROOTS)
    path = find_video_path(index, policy=policy, env=env, seed=seed, episode=episode)
    if path is None:
        return gr.update(value=None, visible=False), (
            f"_No rollout for `{policy}` / `{env}` / seed `{seed}` / ep `{episode}`._"
        )
    return gr.update(value=str(path), visible=True), f"Playing `{path.name}`."


# --------------------------------------------------------------------- #
# Tab 4: Live event log                                                 #
# --------------------------------------------------------------------- #
#
# Tails ``logs/sweep-*.log`` and renders the last N lines as colour-coded
# HTML, with a 2 s polling timer. Pure helpers (line classification,
# tail, log discovery) live in ``_helpers`` so we can test them without
# Gradio installed.

LOG_REFRESH_SECONDS = 2.0
ALL_LOG_CATEGORIES: tuple[str, ...] = ("dispatch", "success", "error", "breach", "other")
LOG_FILTER_CHOICES: tuple[str, ...] = ("all", "dispatch", "success", "error", "breach")


def _log_dropdown_choices() -> tuple[list[tuple[str, str]], str | None]:
    """Discover ``sweep-*.log`` files; newest first.

    Returns ``(choices, default_value)`` shaped like the sweep-run
    dropdown so the dropdown shows the basename but the callback gets
    the absolute path. Default is the newest file (the in-flight sweep,
    if there is one) -- empty if no logs exist yet.
    """
    logs = discover_sweep_logs()
    if not logs:
        return [], None
    choices = [(p.name, str(p)) for p in logs]
    return choices, str(logs[0])


def _filter_categories(filter_value: str) -> tuple[str, ...] | None:
    """Map the radio choice to the set passed to :func:`format_log_lines_html`.

    ``"all"`` -> ``None`` (no filtering). Any other value is treated as
    a single-category filter. The unused buckets (``"other"`` when the
    operator picks a specific filter) are simply omitted.
    """
    if filter_value == "all" or not filter_value:
        return None
    if filter_value not in ALL_LOG_CATEGORIES:
        return None
    return (filter_value,)


def refresh_event_log(
    log_path_str: str | None,
    filter_value: str,
    tail_n: int,
) -> tuple[str, str]:
    """Re-read the selected log and return ``(html_block, header_md)``.

    Empty / missing log -> empty HTML + a hint above. The HTML is wrapped
    in a ``<pre>`` so Gradio renders it monospace; the parent component
    is ``gr.HTML`` so the colour spans take effect (a ``gr.Code`` would
    HTML-escape them).
    """
    if not log_path_str:
        return "", _no_log_hint()
    path = Path(log_path_str)
    lines = tail_log_lines(path, n=tail_n)
    if not lines:
        return "", (
            f"_Log at_ `{log_path_str}` _is empty or missing. Tail will populate "
            "as the sweep dispatches its first cell._"
        )
    categories = _filter_categories(filter_value)
    body = format_log_lines_html(lines, categories=categories)
    counts = summarize_log(lines)
    header = (
        f"Showing last **{len(lines)}** line(s) from `{path.name}` - "
        f"**{counts['dispatch']}** dispatched - "
        f"**{counts['success']}** completed - "
        f"**{counts['error']}** errors - "
        f"**{counts['breach']}** breaches"
    )
    html_block = (
        '<pre style="white-space:pre-wrap;font-family:ui-monospace,monospace;'
        "font-size:12px;line-height:1.35;margin:0;padding:8px;"
        "background:rgba(0,0,0,0.03);border-radius:6px;max-height:560px;"
        f'overflow:auto">{body}</pre>'
    )
    return html_block, header


def _no_log_hint() -> str:
    """Empty-state for the event-log tab."""
    return (
        "_No sweep logs found under_ "
        f"`{env_dashboard_logs_dir()}`. _Start one with_ `make sweep ARGS=...` "
        "_or set_ `DASHBOARD_LOGS_DIR` _to point at a different log root._"
    )


def _on_logs_rescan() -> tuple[Any, str, str]:
    """Re-discover sweep logs on disk; repaint the dropdown + view.

    Returns ``(dropdown_update, html_block, header_md)``. The current
    filter / tail values are not known here (the rescan button has no
    inputs); the timer that fires every 2 s will repaint with the
    current selection a moment later, so we just paint the new
    default's content with default filter/tail.
    """
    choices, default = _log_dropdown_choices()
    html_block, header = refresh_event_log(default, "all", DEFAULT_LOG_TAIL_LINES)
    return gr.update(choices=choices, value=default), html_block, header


# --------------------------------------------------------------------- #
# UI construction                                                       #
# --------------------------------------------------------------------- #


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app.

    Wrapped in a function so the module can be imported without
    side-effects. The smoke check in ``tests/test_dashboard.py`` only
    imports ``_helpers`` -- ``app.py`` is exercised end-to-end when
    the operator runs ``make dashboard``.
    """
    with gr.Blocks(
        title="lerobot-bench dashboard (local)",
        theme=gr.themes.Default(),
    ) as demo:
        gr.Markdown(
            "# lerobot-bench - local sweep dashboard\n"
            f"_Reading from disk at_ `{env_dashboard_results_dir()}`. "
            "_This is the operator tool; the public HF Space lives separately under_ "
            "`space/`."
        )

        with gr.Tabs():
            # -------- Tab 1: Sweep progress --------
            with gr.Tab("Sweep progress"):
                _build_progress_tab(demo)

            # -------- Tab 2: Calibration inspector --------
            with gr.Tab("Calibration inspector"):
                _build_calibration_tab(demo)

            # -------- Tab 3: Rollout preview --------
            with gr.Tab("Rollout preview"):
                _build_rollout_tab(demo)

            # -------- Tab 4: Live event log --------
            with gr.Tab("Live event log"):
                _build_event_log_tab(demo)

    return demo


def _build_progress_tab(demo: gr.Blocks) -> None:
    """Render the sweep-progress tab. Side-effects on ``demo``."""
    initial_choices, initial_default = _runs_dropdown_choices()

    with gr.Row():
        run_dd = gr.Dropdown(
            choices=initial_choices,
            value=initial_default,
            label="Sweep run",
            interactive=True,
            scale=4,
        )
        rescan_btn = gr.Button("Re-scan runs", variant="secondary", scale=1)
    summary_md = gr.Markdown("")
    table = gr.Dataframe(
        headers=list(PROGRESS_COLUMNS),
        datatype=[
            "str",  # policy
            "str",  # env
            "str",  # status
            "number",  # seeds_done
            "number",  # seeds_total
            "number",  # episodes_done
            "number",  # episodes_total
            "str",  # last_update_utc
            "number",  # eta_minutes
        ],
        interactive=False,
        wrap=True,
        label="Per-(policy, env) cell progress",
    )

    # Initial paint -- on first tab open.
    demo.load(
        fn=lambda: refresh_progress(initial_default),
        inputs=None,
        outputs=[table, summary_md],
    )

    # Run dropdown change -> repaint table for the selected manifest.
    run_dd.change(
        fn=_on_run_selected,
        inputs=[run_dd],
        outputs=[table, summary_md],
    )

    # Re-scan: re-discover runs from disk and repaint.
    rescan_btn.click(
        fn=_on_runs_refresh,
        inputs=None,
        outputs=[run_dd, table, summary_md],
    )

    # 5 s polling. gr.Timer is the Gradio 5 way; the click stays for
    # operators who want immediate refresh.
    timer = gr.Timer(value=PROGRESS_REFRESH_SECONDS)
    timer.tick(
        fn=_on_run_selected,
        inputs=[run_dd],
        outputs=[table, summary_md],
    )


def _build_calibration_tab(demo: gr.Blocks) -> None:
    """Render the calibration-inspector tab."""
    status_md = gr.Markdown("")
    refresh_btn = gr.Button("Reload latest calibration", variant="secondary")
    cal_table = gr.Dataframe(
        headers=list(CALIBRATION_COLUMNS),
        datatype=[
            "str",  # policy
            "str",  # env
            "str",  # status
            "number",  # mean_step_ms
            "number",  # p95_step_ms
            "number",  # vram_peak_mb
            "number",  # recommended_seeds
            "number",  # recommended_episodes
            "str",  # reason
        ],
        interactive=False,
        wrap=True,
        label="Calibration cells (auto-downscope recommendations)",
    )

    demo.load(
        fn=refresh_calibration,
        inputs=None,
        outputs=[cal_table, status_md],
    )
    refresh_btn.click(
        fn=refresh_calibration,
        inputs=None,
        outputs=[cal_table, status_md],
    )


def _build_rollout_tab(demo: gr.Blocks) -> None:
    """Render the rollout-preview tab."""
    status_md = gr.Markdown("")
    with gr.Row():
        policy_dd = gr.Dropdown(choices=[], label="Policy", interactive=True)
        env_dd = gr.Dropdown(choices=[], label="Env", interactive=True)
        seed_dd = gr.Dropdown(choices=[], label="Seed", interactive=True)
        ep_dd = gr.Dropdown(choices=[], label="Episode", interactive=True)
    rescan_btn = gr.Button("Re-scan video disks", variant="secondary")
    video = gr.Video(
        label="Rollout",
        interactive=False,
        autoplay=False,
        visible=False,
    )

    # Initial paint -- on first tab open.
    demo.load(
        fn=_refresh_video_dropdowns,
        inputs=None,
        outputs=[policy_dd, env_dd, seed_dd, ep_dd, status_md],
    )

    rescan_btn.click(
        fn=_on_video_rescan,
        inputs=None,
        outputs=[policy_dd, env_dd, seed_dd, ep_dd, status_md],
    )

    # Any dropdown change -> resolve and (un)render the video.
    for dd in (policy_dd, env_dd, seed_dd, ep_dd):
        dd.change(
            fn=_on_video_select,
            inputs=[policy_dd, env_dd, seed_dd, ep_dd],
            outputs=[video, status_md],
        )


def _build_event_log_tab(demo: gr.Blocks) -> None:
    """Render the live-event-log tab.

    Polls the selected ``sweep-*.log`` file every 2 s, tails the last
    N lines, classifies + colour-codes them, and renders the result in
    a ``gr.HTML`` block. Auto-scroll is implemented client-side via a
    ``gr.Checkbox`` that controls the wrapper element's overflow
    behaviour through a small JavaScript snippet on the tail block.

    Side-effects on ``demo``: wires the timer and click handlers.
    """
    initial_choices, initial_default = _log_dropdown_choices()

    with gr.Row():
        log_dd = gr.Dropdown(
            choices=initial_choices,
            value=initial_default,
            label="Sweep log file",
            interactive=True,
            scale=4,
        )
        rescan_btn = gr.Button("Re-scan logs", variant="secondary", scale=1)

    with gr.Row():
        filter_radio = gr.Radio(
            choices=list(LOG_FILTER_CHOICES),
            value="all",
            label="Filter",
            interactive=True,
            scale=3,
        )
        tail_slider = gr.Slider(
            minimum=50,
            maximum=500,
            value=DEFAULT_LOG_TAIL_LINES,
            step=10,
            label="Tail lines",
            interactive=True,
            scale=2,
        )
        autoscroll = gr.Checkbox(
            value=True,
            label="Auto-scroll",
            interactive=True,
            scale=1,
        )

    header_md = gr.Markdown("")
    log_view = gr.HTML(
        value="",
        label="Live log",
    )

    # Initial paint on first tab open.
    demo.load(
        fn=refresh_event_log,
        inputs=[log_dd, filter_radio, tail_slider],
        outputs=[log_view, header_md],
    )

    # Dropdown / filter / slider changes -> immediate repaint.
    log_dd.change(
        fn=refresh_event_log,
        inputs=[log_dd, filter_radio, tail_slider],
        outputs=[log_view, header_md],
    )
    filter_radio.change(
        fn=refresh_event_log,
        inputs=[log_dd, filter_radio, tail_slider],
        outputs=[log_view, header_md],
    )
    tail_slider.change(
        fn=refresh_event_log,
        inputs=[log_dd, filter_radio, tail_slider],
        outputs=[log_view, header_md],
    )

    # Re-scan: re-discover logs from disk; repaint dropdown + view.
    rescan_btn.click(
        fn=_on_logs_rescan,
        inputs=None,
        outputs=[log_dd, log_view, header_md],
    )

    # 2 s polling. The timer reads whatever dropdown/filter/tail are
    # currently selected, so the operator can toggle filters without
    # the timer overwriting them.
    timer = gr.Timer(value=LOG_REFRESH_SECONDS)
    timer.tick(
        fn=_event_log_tick,
        inputs=[log_dd, filter_radio, tail_slider, autoscroll],
        outputs=[log_view, header_md],
    )


def _event_log_tick(
    log_path_str: str | None,
    filter_value: str,
    tail_n: int,
    autoscroll_on: bool,
) -> tuple[str, str]:
    """Timer-driven repaint. Adds an autoscroll script if enabled.

    Auto-scroll is implemented by appending a ``<script>`` tag that
    scrolls the inner ``<pre>`` to its bottom on render. The tag is
    re-evaluated by Gradio on every tick (the HTML block is replaced
    wholesale), so the page stays pinned to the tail while autoscroll
    is on, and stops being yanked back as soon as the operator
    unchecks the box.
    """
    body, header = refresh_event_log(log_path_str, filter_value, int(tail_n))
    if not body:
        return body, header
    if autoscroll_on:
        body = (
            body + "<script>(()=>{const p=document.querySelectorAll('pre');"
            "if(p.length){const e=p[p.length-1];e.scrollTop=e.scrollHeight;}})();</script>"
        )
    return body, header


# Built lazily inside ``__main__`` so import-time work stays minimal.
demo: gr.Blocks | None = None


if __name__ == "__main__":
    demo = build_app()
    demo.queue()
    demo.launch(server_name="127.0.0.1", show_error=True)
