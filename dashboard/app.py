"""Local-first sweep dashboard for the operator running an overnight sweep.

This is **not** the public-facing HF Space (that lives in ``space/``).
This dashboard runs on the operator's laptop, reads sweep state + videos
**from disk only** (no Hub fetches), and is meant to answer one question:

    "I just launched ``make sweep`` -- what's it actually doing, and
    is the calibration matrix shape I shipped sane?"

Tabs:

1. **Sweep progress** -- live table of ``(policy, env)`` cells with
   status / seeds-done / ETA, auto-refreshing every 5 s. Reads
   ``results/<run>/sweep_manifest.json``.
2. **Calibration inspector** -- table of the latest
   ``results/calibration-*.json`` cells with timing, VRAM, and the
   downscope reason.
3. **Rollout preview** -- (policy, env, seed, episode) dropdown
   cascade -> HTML5 ``gr.Video`` player. Scans the local results dir
   and the Windows-mounted Robotics-Data drive for MP4s. Episode
   selection defaults to the cell's representative rollout.
4. **Live event log** -- colour-coded tail of ``logs/sweep-*.log``.
5. **Policies** / **Envs** -- scientific-context cards for every
   policy / sim env in the v1 matrix (repo + revision, license,
   paper-reported-vs-ours, task description, observation modality).

All non-trivial plumbing lives in :mod:`_helpers`. The companion
``tests/test_dashboard.py`` imports that module directly so the
dashboard's data layer is exercised without Gradio in the test
environment (Gradio is in the ``[space]`` extras but not ``[dev]``).
"""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from _helpers import (
    CALIBRATION_COLUMNS,
    DEFAULT_LOG_TAIL_LINES,
    DEFAULT_VIDEO_ROOTS,
    EPISODE_SELECT_BEST,
    EPISODE_SELECT_FIRST,
    EPISODE_SELECT_REPRESENTATIVE,
    HEALTH_AMBER,
    HEALTH_GREEN,
    HEALTH_RED,
    LEADERBOARD_COLUMNS,
    PROGRESS_COLUMNS,
    V1_INCONCLUSIVE_BAND,
    AnomalyReport,
    HalfWidthCurve,
    MissionKPIs,
    StaleDataCache,
    ThrottleState,
    build_calibration_table,
    build_env_card_markdown,
    build_halfwidth_curve,
    build_live_leaderboard,
    build_policy_card_markdown,
    build_progress_table,
    clear_video_cache,
    column_glossary_markdown,
    compute_mission_kpis,
    discover_sweep_logs,
    discover_sweep_runs,
    env_dashboard_logs_dir,
    env_dashboard_results_dir,
    env_dropdown_choices,
    extract_row_click_target,
    find_latest_calibration,
    find_video_path,
    format_bytes_gb,
    format_log_lines_html,
    leaderboard_dataframe,
    load_calibration_report,
    load_manifest,
    load_results_parquet,
    load_with_stale_fallback,
    methodology_markdown,
    per_tab_intro_markdown,
    persistent_header_markdown,
    policy_dropdown_choices,
    read_system_memory,
    read_throttle_state,
    resolved_paths_banner_markdown,
    run_anomaly_review,
    scan_video_index,
    select_representative_episode,
    summarize_log,
    tail_log_lines,
    video_index_options,
)
from matplotlib.figure import Figure

logger = logging.getLogger("dashboard-app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# How often the progress tab repaints. 5 s is plenty for an overnight
# sweep; the manifest writer flushes every cell boundary, so anything
# tighter would just burn battery without showing fresher data.
PROGRESS_REFRESH_SECONDS = 5.0

# --------------------------------------------------------------------- #
# Stale-data resilience caches (audit item 9)                           #
# --------------------------------------------------------------------- #
#
# One :class:`StaleDataCache` per data-loading panel that needs to keep
# showing last-known-good values when the underlying file is mid-write.
# Module-level state is fine here: the dashboard is single-tenant by
# design (one operator, one browser, one sweep) and the cache contents
# are derived purely from disk -- a process restart re-warms them on
# the first successful refresh.
_progress_cache = StaleDataCache()
_calibration_cache = StaleDataCache()


def reset_stale_caches() -> None:
    """Reset the module-level caches.

    Test seam: ``tests/test_dashboard.py`` swaps in its own
    :class:`StaleDataCache` instances by calling the underlying helper
    directly; this function exists so an operator running the
    dashboard in a long-lived session can clear the cache manually if
    they ever need to (currently unwired but kept for parity with the
    other ``clear_*`` helpers in ``_helpers``).
    """
    global _progress_cache, _calibration_cache
    _progress_cache = StaleDataCache()
    _calibration_cache = StaleDataCache()


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
) -> tuple[pd.DataFrame, str, str, Figure]:
    """Re-read the manifest + parquet and recompute the progress view.

    Called by the 5 s timer and the manual Refresh button. Returns the
    4-tuple ``(table_df, status_markdown, stale_warning_markdown,
    halfwidth_figure)``:

    * ``status_markdown`` is empty on success or a helpful hint when no
      manifest is selected.
    * ``stale_warning_markdown`` is populated by
      :func:`load_with_stale_fallback` when the manifest file is
      mid-write (audit item 9); the table then falls back to the last
      known-good snapshot.
    * ``halfwidth_figure`` is the Wilson CI half-width vs N plot for the
      running (or most-recently-completed) cell.

    The per-episode ``results.parquet`` sits next to the manifest and is
    read through :func:`load_results_parquet`, which tolerates a
    mid-write file by serving the last-good frame.
    """
    if not manifest_path_str:
        empty = pd.DataFrame({c: [] for c in PROGRESS_COLUMNS})
        # No manifest selected isn't a "stale" failure -- it's just the
        # empty initial state. Don't bump the failure counter.
        return empty, _no_sweep_hint(), "", _halfwidth_figure(None)

    manifest_path = Path(manifest_path_str)
    # The parquet read has its own last-good cache (``_RESULTS_CACHE`` in
    # ``_helpers``); the manifest read goes through ``_progress_cache``
    # below. Two caches because the two files fail independently.
    results_df = load_results_parquet(manifest_path.parent / "results.parquet")

    def _loader() -> pd.DataFrame:
        # Wrapped in a callable so ``load_with_stale_fallback`` can
        # catch a partial-read OSError / JSONDecodeError mid-write
        # without us having to duplicate the try/except in every panel.
        manifest = load_manifest(manifest_path)
        if not manifest:
            # Treat an empty/unreadable manifest like a failure so the
            # cache falls back to the last-good table. ``load_manifest``
            # itself swallows the OSError; raising here re-surfaces it
            # to the stale-data layer.
            raise FileNotFoundError(f"manifest at {manifest_path_str} unreadable or empty")
        return build_progress_table(manifest, results_df=results_df)

    def _empty() -> pd.DataFrame:
        return pd.DataFrame({c: [] for c in PROGRESS_COLUMNS})

    table, warning = load_with_stale_fallback(
        _progress_cache,
        _loader,
        empty_factory=_empty,
    )
    # On a stale-data event the friendly summary line still sits above
    # the table (so the operator's eye doesn't lose context); the
    # warning sits below in its own component.
    summary = _summarise_progress(table)
    curve = build_halfwidth_curve(table, results_df)
    return table, summary, warning, _halfwidth_figure(curve)


def _halfwidth_figure(curve: HalfWidthCurve | None) -> Figure:
    """Render the Wilson CI half-width vs N plot as a matplotlib Figure.

    X is N (1..current); Y is the closed-form Wilson 95% half-width at
    the cell's running p-hat. A horizontal reference line marks the v1
    Wilson inconclusive band (2*HW at p=0.5, N=250 = 0.123 from
    ``docs/MDE_TABLE.md``) -- a cell whose single-cell half-width is
    still above that is nowhere near resolving a real effect.

    ``curve=None`` (cold start, no cell with episodes on disk) renders
    a centred "no data yet" placeholder so the ``gr.Plot`` slot is
    never blank.
    """
    fig = Figure(figsize=(6.4, 3.4), dpi=100)
    ax = fig.add_subplot(111)

    if curve is None:
        ax.text(
            0.5,
            0.5,
            "No running or completed cell with episodes on disk yet.",
            ha="center",
            va="center",
            fontsize=10,
            color="#666",
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    ax.plot(curve.n_values, curve.halfwidths, color="#2563eb", lw=1.8)
    ax.axhline(
        V1_INCONCLUSIVE_BAND,
        color="#dc2626",
        ls="--",
        lw=1.2,
        label=f"v1 inconclusive band 2·HW = {V1_INCONCLUSIVE_BAND:.3f}",
    )
    ax.scatter(
        [curve.n_current],
        [curve.halfwidths[-1]],
        color="#2563eb",
        zorder=5,
        s=28,
    )
    state = "currently running" if curve.is_running else "most recently completed"
    ax.set_title(
        f"Wilson CI half-width vs N — {curve.policy} / {curve.env} ({state})",
        fontsize=10,
    )
    ax.set_xlabel("N (episodes)")
    ax.set_ylabel("Wilson 95% half-width")
    ax.text(
        0.98,
        0.92,
        f"p̂ = {curve.p_hat:.2f}, N = {curve.n_current}, HW = {curve.halfwidths[-1]:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444",
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.82))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


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


def _on_run_selected(manifest_path_str: str | None) -> tuple[pd.DataFrame, str, str, Figure]:
    """Dropdown change -> re-render the progress view for that run.

    Returns the 4-tuple shape (table, summary, stale_warning, halfwidth
    figure) so the timer + dropdown handlers share one wiring with both
    the stale-data fallback (audit item 9) and the half-width plot.
    """
    return refresh_progress(manifest_path_str)


def _on_runs_refresh() -> tuple[Any, pd.DataFrame, str, str, Figure]:
    """Re-scan disk for new runs and re-render the view for the newest.

    Wired to the "Re-scan runs" button. Returns ``(dropdown_update,
    table_df, status_markdown, stale_warning_markdown, halfwidth_figure)``.
    """
    choices, default = _runs_dropdown_choices()
    table, summary, warning, fig = refresh_progress(default)
    return (
        gr.update(choices=choices, value=default),
        table,
        summary,
        warning,
        fig,
    )


def _on_progress_row_click(
    table: pd.DataFrame,
    evt: gr.SelectData,
) -> tuple[Any, str, str, Any, Any, Any]:
    """Drill-down handler for clicks on the Sweep-progress dataframe.

    Wired to ``gr.Dataframe.select`` on the progress table (audit item
    8). Returns a 6-tuple:

    1. ``tabs_update``: ``gr.update(selected=...)`` to flip the active
       tab to "Rollout preview" on success, or a no-op on failure.
    2. ``status_md``: unchanged (no overwrite) on success, a one-line
       warning on a non-actionable click. We piggy-back on the
       progress-tab summary widget so the warning shows up where the
       operator's mouse already is.
    3. ``warning_md``: same as ``status_md`` -- exposed separately so
       the test suite can introspect the helper return without a
       Gradio event loop.
    4-6. ``policy_update`` / ``env_update`` / ``seed_update``:
       ``gr.update(value=...)`` for the rollout-tab dropdowns. No-op
       on non-actionable rows.
    """
    row_index = evt.index[0] if evt.index else None
    target = extract_row_click_target(
        table,
        row_index,
        columns=list(table.columns) if hasattr(table, "columns") else None,
    )
    if not target.actionable:
        return (
            gr.update(),
            target.warning,
            target.warning,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        gr.update(selected=ROLLOUT_TAB_ID),
        "",
        "",
        gr.update(value=target.policy),
        gr.update(value=target.env),
        gr.update(value=target.seed),
    )


# Stable tab IDs for the ``gr.Tabs`` container. Strings rather than
# ints so the row-click handler can pass a meaningful selector;
# Gradio matches on either an ``id`` attribute or the label.
MISSION_TAB_ID = "mission"
PROGRESS_TAB_ID = "progress"
CALIBRATION_TAB_ID = "calibration"
ROLLOUT_TAB_ID = "rollouts"
EVENTS_TAB_ID = "events"
ABOUT_TAB_ID = "about"


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


# The episode-selection radio choices. Labels carry the cherry-pick
# caveat for "Best" so an operator skimming the tab is not misled into
# treating the best episode as typical.
EPISODE_SELECT_RADIO_CHOICES: list[tuple[str, str]] = [
    ("Representative (default)", EPISODE_SELECT_REPRESENTATIVE),
    ("Best (highest return / shortest success)", EPISODE_SELECT_BEST),
    ("First", EPISODE_SELECT_FIRST),
]


def _episode_options_for(
    index: Any,
    *,
    policy: str | None,
    env: str | None,
    seed: str | None,
) -> list[str]:
    """Episode values available on disk for one (policy, env, seed)."""
    if not policy or not env or seed in (None, ""):
        return []
    try:
        seed_int = int(seed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    eps = sorted(ep for (p, e, s, ep) in index.by_key if p == policy and e == env and s == seed_int)
    return [str(e) for e in eps]


def _default_episode_value(
    episodes: list[str],
    *,
    policy: str | None,
    env: str | None,
    seed: str | None,
    mode: str,
) -> str | None:
    """Resolve the default episode for the dropdown given the selection mode.

    Computes the representative / best episode from the per-episode
    results parquet via :func:`select_representative_episode`. Falls
    back to first-on-disk when the cell has no parquet rows yet.
    """
    if not episodes:
        return None
    if not policy or not env or seed in (None, ""):
        return episodes[0]
    chosen = select_representative_episode(
        _latest_results_df(),
        policy=policy,
        env=env,
        seed=seed,
        mode=mode,
        available_episodes=[int(e) for e in episodes],
    )
    if chosen is None:
        return episodes[0]
    chosen_str = str(chosen)
    return chosen_str if chosen_str in episodes else episodes[0]


def _refresh_video_dropdowns(
    mode: str = EPISODE_SELECT_REPRESENTATIVE,
) -> tuple[Any, Any, Any, Any, str]:
    """Scan disk for videos and populate the four dropdowns.

    Returns ``(policy_update, env_update, seed_update, episode_update,
    status_markdown)``. The episode dropdown defaults to the
    *representative* episode of the first (policy, env, seed) on disk
    (per ``mode``) rather than first-alphabetical. Status reports the
    count + roots scanned so the operator can confirm the
    Windows-mounted drive was found.
    """
    index = scan_video_index(DEFAULT_VIDEO_ROOTS)
    opts = video_index_options(index)

    def _first_or_none(values: list[str]) -> str | None:
        return values[0] if values else None

    policy = _first_or_none(opts["policy"])
    env = _first_or_none(opts["env"])
    seed = _first_or_none(opts["seed"])
    episodes = _episode_options_for(index, policy=policy, env=env, seed=seed)
    ep_value = _default_episode_value(episodes, policy=policy, env=env, seed=seed, mode=mode)

    status = _video_index_status(index)
    return (
        gr.update(choices=opts["policy"], value=policy),
        gr.update(choices=opts["env"], value=env),
        gr.update(choices=opts["seed"], value=seed),
        gr.update(choices=episodes or opts["episode"], value=ep_value),
        status,
    )


def _video_index_status(index: Any) -> str:
    roots = ", ".join(f"`{r}`" for r in index.roots)
    return f"Indexed **{index.n_videos}** MP4(s) across {roots}."


def _on_video_rescan(mode: str) -> tuple[Any, Any, Any, Any, str]:
    """Manual re-scan: clear the cache then repopulate dropdowns."""
    clear_video_cache()
    return _refresh_video_dropdowns(mode)


def _on_episode_mode_change(
    policy: str | None,
    env: str | None,
    seed: str | None,
    mode: str,
) -> Any:
    """Re-pick the episode value when the selection radio (or a cell) changes.

    Recomputes the episode dropdown's value for the current
    (policy, env, seed) under the new ``mode``, leaving the available
    choices intact. Wired to the episode-selection radio and to the
    policy / env / seed dropdowns.
    """
    index = scan_video_index(DEFAULT_VIDEO_ROOTS)
    episodes = _episode_options_for(index, policy=policy, env=env, seed=seed)
    ep_value = _default_episode_value(episodes, policy=policy, env=env, seed=seed, mode=mode)
    return gr.update(choices=episodes, value=ep_value)


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


def _refresh_persistent_header() -> str:
    """Re-render the persistent project-context header.

    Wrapped in a function so the header's live "cell N/total" badge
    repaints on the 5 s timer alongside the progress tab. The header
    is plain markdown -- the badge is recomputed from the newest
    manifest under :func:`env_dashboard_results_dir`.
    """
    return persistent_header_markdown()


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app.

    Wrapped in a function so the module can be imported without
    side-effects. The smoke check in ``tests/test_dashboard.py`` only
    imports ``_helpers`` -- ``app.py`` is exercised end-to-end when
    the operator runs ``make dashboard``.

    Layout: a persistent project-context header (markdown) renders
    above the tab strip on every tab so the reviewer always sees what
    lerobot-bench is, what the dashboard is doing, and how much of the
    sweep is done. A one-line resolved-paths banner sits underneath --
    the previous footgun was a dashboard pointing at an empty worktree
    ``results/`` and looking broken when it was actually fine.
    """
    with gr.Blocks(
        title="lerobot-bench dashboard (local)",
        theme=gr.themes.Default(),
    ) as demo:
        # Persistent project-context header. Live progress badge inside.
        header_md = gr.Markdown(persistent_header_markdown())
        gr.Markdown(resolved_paths_banner_markdown())

        # Repaint the header on the same 5 s cadence as the progress
        # tab so the in-flight "M/N seeds done" badge stays fresh
        # regardless of which tab the operator is looking at.
        header_timer = gr.Timer(value=PROGRESS_REFRESH_SECONDS)
        header_timer.tick(
            fn=_refresh_persistent_header,
            inputs=None,
            outputs=[header_md],
        )

        with gr.Tabs():
            # -------- Tab 0: Mission Control (default landing) --------
            with gr.Tab("Mission Control", id=MISSION_TAB_ID):
                _build_mission_control_tab(demo)

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

            # -------- Tab 5: Policies --------
            with gr.Tab("Policies"):
                _build_policies_tab(demo)

            # -------- Tab 6: Envs --------
            with gr.Tab("Envs"):
                _build_envs_tab(demo)

            # -------- Tab 7: About --------
            with gr.Tab("About"):
                _build_about_tab()

    return demo


# --------------------------------------------------------------------- #
# Tab 5: Policies (scientific-context panel)                            #
# --------------------------------------------------------------------- #


def _latest_results_df() -> pd.DataFrame | None:
    """Load the per-episode parquet of the newest sweep run, if any.

    Reuses :func:`discover_sweep_runs` + :func:`load_results_parquet`
    (the shared stale-tolerant parquet reader) so the Policies tab and
    the Rollout tab share one definition of "our re-run numbers".
    Returns ``None`` when no run / parquet is on disk yet -- the policy
    card then shows ``(pending)`` for every cell.
    """
    runs = discover_sweep_runs(env_dashboard_results_dir())
    if not runs:
        return None
    return load_results_parquet(runs[0].manifest_path.parent / "results.parquet")


def _render_policy_card(policy_name: str | None) -> str:
    """Render the science card for the selected policy."""
    if not policy_name:
        return "_Select a policy above._"
    return build_policy_card_markdown(policy_name, results_df=_latest_results_df())


def _build_policies_tab(demo: gr.Blocks) -> None:
    """Render the Policies tab -- a per-policy scientific-context card.

    Side-effects on ``demo``: wires the dropdown change + initial load.
    """
    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(
            "**What this tab shows.** A scientific-context card for every "
            "policy in the v1 matrix: HF Hub repo + pinned revision, "
            "license, architecture / training data, and a "
            "paper-reported-vs-our-re-run table with a colour-chipped "
            "delta per supported env. `(pending)` in the Ours column "
            "means that cell is not in the results parquet yet."
        )

    choices = policy_dropdown_choices()
    default = choices[0] if choices else None
    policy_dd = gr.Dropdown(
        choices=choices,
        value=default,
        label="Policy",
        interactive=True,
    )
    card_md = gr.Markdown(_render_policy_card(default))

    policy_dd.change(fn=_render_policy_card, inputs=[policy_dd], outputs=[card_md])
    demo.load(fn=lambda: _render_policy_card(default), inputs=None, outputs=[card_md])


# --------------------------------------------------------------------- #
# Tab 6: Envs (scientific-context panel)                                #
# --------------------------------------------------------------------- #


def _render_env_card(env_name: str | None) -> str:
    """Render the science card for the selected env."""
    if not env_name:
        return "_Select an env above._"
    return build_env_card_markdown(env_name)


def _build_envs_tab(demo: gr.Blocks) -> None:
    """Render the Envs tab -- a per-env scientific-context card.

    Side-effects on ``demo``: wires the dropdown change + initial load.
    """
    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(
            "**What this tab shows.** A scientific-context card for every "
            "sim env in the v1 matrix: the task, observation modality, "
            "what the env discriminates between policies, plus the "
            "runtime `max_steps` / `success_threshold` and the source "
            "paper."
        )

    choices = env_dropdown_choices()
    default = choices[0] if choices else None
    env_dd = gr.Dropdown(
        choices=choices,
        value=default,
        label="Env",
        interactive=True,
    )
    card_md = gr.Markdown(_render_env_card(default))

    env_dd.change(fn=_render_env_card, inputs=[env_dd], outputs=[card_md])
    demo.load(fn=lambda: _render_env_card(default), inputs=None, outputs=[card_md])


def _build_about_tab() -> None:
    """Render the About tab -- methodology + scope + reading guide.

    Pure markdown, no interactive widgets; the prose lives in
    :func:`methodology_markdown` so the test suite covers the H2
    structure without importing Gradio.
    """
    gr.Markdown(methodology_markdown())


# --------------------------------------------------------------------- #
# Tab 0: Mission Control (default landing tab)                          #
# --------------------------------------------------------------------- #
#
# One glanceable screen the operator leaves open on a tablet to
# supervise the ~20 h sweep. Four stacked sections, large text, colour,
# auto-refreshing every 5 s. All the data shaping lives in
# ``_helpers.py``; the functions below only format the helper output
# into HTML / Dataframe / Markdown components.

# Health-banner colours. Keyed on the _helpers HEALTH_* severities.
_HEALTH_BANNER_STYLE: dict[str, tuple[str, str, str]] = {
    # severity -> (background, text colour, leading glyph)
    HEALTH_GREEN: ("#16a34a", "#ffffff", "✓"),
    HEALTH_AMBER: ("#d97706", "#ffffff", "⚠"),
    HEALTH_RED: ("#dc2626", "#ffffff", "⚠"),
}


def _kpi_tile_html(label: str, value: str, *, danger: bool = False) -> str:
    """Render one big KPI tile. ``danger`` paints the value red."""
    value_colour = "#dc2626" if danger else "inherit"
    return (
        '<div style="flex:1;min-width:130px;padding:12px 16px;'
        'background:rgba(0,0,0,0.04);border-radius:10px;text-align:center">'
        f'<div style="font-size:12px;text-transform:uppercase;letter-spacing:0.5px;'
        f'color:#666;margin-bottom:4px">{html.escape(label)}</div>'
        f'<div style="font-size:26px;font-weight:700;line-height:1.1;'
        f'color:{value_colour}">{html.escape(value)}</div></div>'
    )


def _render_health_banner(kpis: MissionKPIs) -> str:
    """Render the prominent green/amber/red health banner as HTML."""
    bg, fg, glyph = _HEALTH_BANNER_STYLE.get(kpis.health, _HEALTH_BANNER_STYLE[HEALTH_AMBER])
    return (
        f'<div style="padding:16px 20px;border-radius:12px;background:{bg};'
        f"color:{fg};font-size:22px;font-weight:700;text-align:center;"
        f'margin-bottom:4px">{glyph}&nbsp;&nbsp;{html.escape(kpis.health_message)}</div>'
    )


def _render_kpi_strip(kpis: MissionKPIs) -> str:
    """Render the KPI tile strip as a flexbox HTML row."""
    tiles = [
        _kpi_tile_html(
            "Cells done",
            f"{kpis.cells_done}/{kpis.denom}  ({kpis.percent_done:.0f}%)",
        ),
        _kpi_tile_html("Cells failed", str(kpis.cells_failed), danger=kpis.cells_failed > 0),
        _kpi_tile_html("Running now", kpis.running_label),
        _kpi_tile_html("Elapsed", kpis.elapsed_label),
        _kpi_tile_html("ETA", kpis.eta_label),
        _kpi_tile_html(
            "Sweep state",
            kpis.state,
            danger=kpis.state == "THROTTLED-frozen",
        ),
    ]
    return '<div style="display:flex;flex-wrap:wrap;gap:10px">' + "".join(tiles) + "</div>"


def _render_anomaly_panel(report: AnomalyReport) -> str:
    """Render the anomaly-alerts section: green clean / red flagged list."""
    if report.error:
        return (
            '<div style="padding:12px 16px;border-radius:10px;'
            'background:rgba(0,0,0,0.04);color:#666;font-size:14px">'
            f"Anomaly review: {html.escape(report.error)}.</div>"
        )
    if report.ok:
        return (
            '<div style="padding:14px 18px;border-radius:10px;background:#16a34a;'
            'color:#fff;font-size:16px;font-weight:600">✓ No anomalies — '
            f"all {report.n_cells_reviewed} reviewed cell(s) look healthy.</div>"
        )
    items = "".join(f'<li style="margin:2px 0">{html.escape(line)}</li>' for line in report.lines)
    return (
        '<div style="padding:14px 18px;border-radius:10px;background:#dc2626;'
        'color:#fff;font-size:14px">'
        f'<div style="font-size:16px;font-weight:700;margin-bottom:6px">'
        f"⚠ {report.n_cells_flagged} cell(s) flagged "
        f"({report.n_cells_reviewed} reviewed)</div>"
        f'<ul style="margin:0;padding-left:20px">{items}</ul></div>'
    )


def _memory_bar(label: str, used: int | None, total: int | None) -> str:
    """Render a labelled used/total memory bar; ``—`` when undiscoverable."""
    if used is None or total is None or total <= 0:
        return (
            f'<div style="margin:6px 0"><b>{html.escape(label)}:</b> '
            "&mdash; <i>(not discoverable)</i></div>"
        )
    pct = min(100.0, used / total * 100.0)
    bar_colour = "#dc2626" if pct >= 90 else ("#d97706" if pct >= 75 else "#16a34a")
    return (
        f'<div style="margin:6px 0"><b>{html.escape(label)}:</b> '
        f"{format_bytes_gb(used)} / {format_bytes_gb(total)} ({pct:.0f}%)"
        '<div style="height:10px;border-radius:5px;background:rgba(0,0,0,0.08);'
        'margin-top:3px">'
        f'<div style="height:100%;width:{pct:.0f}%;border-radius:5px;'
        f'background:{bar_colour}"></div></div></div>'
    )


def _render_resource_panel(throttle: ThrottleState) -> str:
    """Render the resource + throttle section as HTML."""
    mem = read_system_memory()
    if mem is None:
        host_block = (
            '<div style="margin:6px 0"><b>Host RAM:</b> &mdash; '
            "<i>(/proc/meminfo unreadable)</i></div>"
        )
    else:
        host_block = _memory_bar("Host RAM", mem.used_bytes, mem.total_bytes)

    cgroup_block = _memory_bar(
        "Sweep cgroup memory",
        throttle.memory_current,
        throttle.memory_max,
    )

    if not throttle.running:
        throttle_line = "<b>Throttle state:</b> sweep process not running"
    elif throttle.frozen is None:
        throttle_line = (
            f"<b>Throttle state:</b> sweep running (PID {throttle.pid}); "
            "cgroup freeze state not discoverable"
        )
    elif throttle.frozen:
        throttle_line = (
            f'<b>Throttle state:</b> <span style="color:#dc2626;font-weight:700">'
            f"FROZEN</span> (PID {throttle.pid}) — watchdog throttled the sweep"
        )
    else:
        throttle_line = (
            f'<b>Throttle state:</b> <span style="color:#16a34a;font-weight:700">'
            f"RUNNING</span> (PID {throttle.pid})"
        )

    return (
        '<div style="padding:12px 16px;border-radius:10px;'
        'background:rgba(0,0,0,0.04);font-size:14px">'
        + host_block
        + cgroup_block
        + f'<div style="margin:6px 0">{throttle_line}</div></div>'
    )


def refresh_mission_control() -> tuple[str, str, pd.DataFrame, str, str]:
    """Recompute every Mission Control section. Driven by the 5 s timer.

    Returns ``(banner_html, kpi_html, leaderboard_df, anomaly_html,
    resource_html)``. Reads the newest sweep run's manifest + parquet;
    every section degrades to an empty / neutral state when nothing is
    on disk yet rather than crashing the tick.
    """
    runs = discover_sweep_runs(env_dashboard_results_dir())
    manifest: dict[str, Any] = {}
    results_path: Path | None = None
    if runs:
        manifest = load_manifest(runs[0].manifest_path)
        results_path = runs[0].manifest_path.parent / "results.parquet"

    throttle = read_throttle_state()
    kpis = compute_mission_kpis(manifest, throttled=bool(throttle.frozen))

    results_df = load_results_parquet(results_path)
    leaderboard = leaderboard_dataframe(build_live_leaderboard(results_df))
    anomalies = run_anomaly_review(results_path)

    return (
        _render_health_banner(kpis),
        _render_kpi_strip(kpis),
        leaderboard,
        _render_anomaly_panel(anomalies),
        _render_resource_panel(throttle),
    )


def _build_mission_control_tab(demo: gr.Blocks) -> None:
    """Render the Mission Control tab -- the default landing screen.

    Four stacked sections (KPI strip + health banner, live leaderboard,
    anomaly alerts, resource/throttle), auto-refreshing every 5 s.
    """
    gr.Markdown(
        "## Mission Control\n"
        "_One-screen sweep supervision -- leave this open. "
        "Auto-refreshes every 5 s._"
    )

    banner = gr.HTML()
    kpi_strip = gr.HTML()

    gr.Markdown("### Live results forming")
    leaderboard = gr.Dataframe(
        headers=list(LEADERBOARD_COLUMNS),
        datatype=["str", "str", "str", "number", "number"],
        interactive=False,
        wrap=True,
        label="Per-policy mean success rate (Wilson 95% CI) across completed cells",
    )

    gr.Markdown("### Anomaly alerts")
    anomaly_panel = gr.HTML()

    gr.Markdown("### Resource + throttle")
    resource_panel = gr.HTML()

    refresh_btn = gr.Button("Refresh now", variant="secondary")

    mission_outputs = [banner, kpi_strip, leaderboard, anomaly_panel, resource_panel]

    demo.load(fn=refresh_mission_control, inputs=None, outputs=mission_outputs)
    refresh_btn.click(fn=refresh_mission_control, inputs=None, outputs=mission_outputs)

    timer = gr.Timer(value=PROGRESS_REFRESH_SECONDS)
    timer.tick(fn=refresh_mission_control, inputs=None, outputs=mission_outputs)


def _build_progress_tab(demo: gr.Blocks) -> None:
    """Render the sweep-progress tab. Side-effects on ``demo``."""
    initial_choices, initial_default = _runs_dropdown_choices()

    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(per_tab_intro_markdown("progress"))

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
            "str",  # success_rate_so_far (formatted; "—" below n=25)
            "str",  # wilson_ci_so_far ("[lo, hi]" or "—")
            "str",  # seed_spread ("⚠ 0.NN" / "0.NN" / "—")
            "str",  # last_update_utc
            "number",  # eta_minutes
        ],
        interactive=False,
        wrap=True,
        label="Per-(policy, env) cell progress",
    )
    # Stale-data warning slot (audit item 9). Sits directly below the
    # table so a mid-write manifest surfaces "showing last-good data"
    # right where the operator is reading. Empty string on a clean read.
    stale_warning_md = gr.Markdown("")
    gr.Markdown(column_glossary_markdown("progress"))

    # Wilson CI half-width vs N plot for the running (or last-done) cell.
    halfwidth_plot = gr.Plot(
        label="Wilson CI half-width vs N — currently-running cell",
    )

    # All four handlers feed the same 4 outputs in ``refresh_progress``
    # order: table, summary, stale-warning, half-width plot.
    progress_outputs = [table, summary_md, stale_warning_md, halfwidth_plot]

    # Initial paint -- on first tab open.
    demo.load(
        fn=lambda: refresh_progress(initial_default),
        inputs=None,
        outputs=progress_outputs,
    )

    # Run dropdown change -> repaint table for the selected manifest.
    run_dd.change(
        fn=_on_run_selected,
        inputs=[run_dd],
        outputs=progress_outputs,
    )

    # Re-scan: re-discover runs from disk and repaint.
    rescan_btn.click(
        fn=_on_runs_refresh,
        inputs=None,
        outputs=[run_dd, *progress_outputs],
    )

    # 5 s polling. gr.Timer is the Gradio 5 way; the click stays for
    # operators who want immediate refresh.
    timer = gr.Timer(value=PROGRESS_REFRESH_SECONDS)
    timer.tick(
        fn=_on_run_selected,
        inputs=[run_dd],
        outputs=progress_outputs,
    )


def _build_calibration_tab(demo: gr.Blocks) -> None:
    """Render the calibration-inspector tab."""
    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(per_tab_intro_markdown("calibration"))

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
            "str",  # std_step_ms (formatted; "—" when raw step times absent)
            "number",  # n_steps
            "str",  # latency_skew ("⚠ skewed" or "")
            "number",  # vram_peak_mb
            "number",  # recommended_seeds
            "number",  # recommended_episodes
            "str",  # reason
        ],
        interactive=False,
        wrap=True,
        label="Calibration cells (auto-downscope recommendations)",
    )
    gr.Markdown(column_glossary_markdown("calibration"))

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
    """Render the rollout-preview tab.

    The episode dropdown defaults to the *representative* episode of
    the selected cell (modal outcome, step count closest to the cell
    median) rather than first-alphabetical; the "Episode selection"
    radio lets the operator switch to the best (cherry-pickable) or
    first episode instead.
    """
    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(per_tab_intro_markdown("rollouts"))

    status_md = gr.Markdown("")
    with gr.Row():
        policy_dd = gr.Dropdown(choices=[], label="Policy", interactive=True)
        env_dd = gr.Dropdown(choices=[], label="Env", interactive=True)
        seed_dd = gr.Dropdown(choices=[], label="Seed", interactive=True)
        ep_dd = gr.Dropdown(choices=[], label="Episode", interactive=True)
    episode_mode = gr.Radio(
        choices=EPISODE_SELECT_RADIO_CHOICES,
        value=EPISODE_SELECT_REPRESENTATIVE,
        label="Episode selection",
        info=(
            "Representative = modal outcome, step count closest to the "
            "cell median. Best may be cherry-picked. First = lowest "
            "episode index."
        ),
        interactive=True,
    )
    rescan_btn = gr.Button("Re-scan video disks", variant="secondary")
    video = gr.Video(
        label="Rollout",
        interactive=False,
        autoplay=False,
        visible=False,
    )

    # Initial paint -- on first tab open. Episode defaults to the
    # representative episode of the first cell on disk.
    demo.load(
        fn=lambda: _refresh_video_dropdowns(EPISODE_SELECT_REPRESENTATIVE),
        inputs=None,
        outputs=[policy_dd, env_dd, seed_dd, ep_dd, status_md],
    )

    rescan_btn.click(
        fn=_on_video_rescan,
        inputs=[episode_mode],
        outputs=[policy_dd, env_dd, seed_dd, ep_dd, status_md],
    )

    # policy / env / seed change -> re-pick the representative episode
    # for the new cell, then resolve the video.
    for dd in (policy_dd, env_dd, seed_dd):
        dd.change(
            fn=_on_episode_mode_change,
            inputs=[policy_dd, env_dd, seed_dd, episode_mode],
            outputs=[ep_dd],
        )

    # Episode-selection radio change -> re-pick the episode value.
    episode_mode.change(
        fn=_on_episode_mode_change,
        inputs=[policy_dd, env_dd, seed_dd, episode_mode],
        outputs=[ep_dd],
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

    with gr.Accordion("What this tab shows", open=False):
        gr.Markdown(per_tab_intro_markdown("events"))

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
    # GRADIO_SERVER_NAME lets an operator bind to 0.0.0.0 for tablet
    # access over the tailnet; defaults to loopback.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        show_error=True,
    )
