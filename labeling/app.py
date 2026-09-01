#!/usr/bin/env python3
"""Rank별 사업자 anomaly 구간 라벨링 앱.

실행:
    python labeling/app.py
    # 또는
    cd labeling && python app.py

브라우저가 열리면 사업자를 선택하세요 (선택 즉시 로드).

새 날짜의 data/*.csv를 추가한 경우:
    python labeling/preprocess.py
    python labeling/app.py

전처리는 data/labels/*_labels.json을 수정하지 않습니다.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from threading import Timer

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tool import (  # noqa: E402
    LABEL_HIGHLIGHT_END,
    LABEL_HIGHLIGHT_START,
    add_label,
    build_figure,
    clamp_time_range,
    data_time_bounds,
    display_metric,
    display_plmn,
    format_kst,
    freeze_shape_editing,
    label_highlight_overlays,
    label_line,
    load_labels,
    load_or_build_ranking,
    load_plmn,
    metric_columns,
    m971_daily_mean_series,
    is_rate_metric,
    is_synthetic_overlay,
    overlay_metrics_available,
    rate_metrics_available,
    M971_COL,
    M971_DAILY_AVG_KEY,
    parse_time,
    pending_anchor_shape,
    pending_range_fill_shape,
    ranked_hover_html,
    ranked_metric_pairs,
    remove_label,
    save_labels,
    to_plot_time,
    to_plot_times,
    update_label,
    value_cursor_overlays,
)

MAX_POINTS = 1500
HOST = "127.0.0.1"
PORT = 8050

rank_df = load_or_build_ranking(top_n=None)
plmn_options = [
    {
        "label": f"#{int(r.rank):03d}  {display_plmn(r.PLMN)}  ({int(r.M971_sum):,})",
        "value": r.PLMN,
    }
    for r in rank_df.itertuples(index=False)
]
plmn_ids = [o["value"] for o in plmn_options]

state: dict = {
    "df": None,
    "doc": None,
    "plmn": None,
    "rank": None,
    "metrics": [],
    # Currently drawn series (subset of metrics). Empty = draw nothing.
    "visible_metrics": [],
    "metric_rev": 0,
    "zoom_start": None,
    "zoom_end": None,
    # Explicit full-vs-zoomed flag (avoids stale view-range sync races).
    "x_full_view": True,
    "y_min": None,
    "y_max": None,
    "y_auto": True,
    "y_gen": 0,
    "y_rendered": None,
    "zoom_guard_until": 0.0,
    "select_guard_until": 0.0,
    "zoom_rendered": None,
    "zoom_rendered_prev": None,
    "label_range_anchor": None,
    "highlight_id": None,
    "highlight_shape_idxs": None,
    "label_rev": 0,
    "place_rev": 0,
    "confirm_action": None,
    "value_cursor_pos": None,
    "hover_pos": None,
    "_last_click": (None, 0.0),
    "show_anomalies": True,
}


def _sample_plot_ms(df: pd.DataFrame) -> list[int]:
    """KST wall-clock times as UTC-ms for Plotly naive date-axis snapping."""
    vals = to_plot_times(df["time"])
    return vals.astype("datetime64[ms]").astype("int64").tolist()


def _empty_figure():
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        height=420,
        autosize=True,
        margin=dict(l=40, r=20, t=40, b=40),
        annotations=[
            dict(
                text="사업자를 선택하세요",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#888"),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _label_options():
    items = (state["doc"] or {}).get("labels", [])
    return [{"label": label_line(x), "value": x["id"]} for x in items]


def _plot_metrics() -> list[str]:
    """Metrics currently drawn on the graph (stable column order + overlays)."""
    all_m = list(state.get("metrics") or [])
    overlays = overlay_metrics_available(state.get("df"), all_m)
    visible = state.get("visible_metrics")
    if visible is None:
        return all_m + [o for o in overlays if o not in all_m]
    selected = set(visible)
    out = [m for m in all_m if m in selected]
    for key in overlays:
        if key in selected and key not in out:
            out.append(key)
    return out


def _metrics_by_view_sum() -> list[str]:
    """Metrics with a non-zero value in the on-screen window, ranked by sum (desc)."""
    df = state.get("df")
    metrics = list(state.get("metrics") or [])
    if df is None or not len(df) or not metrics:
        return metrics
    start_ts = state.get("zoom_start")
    end_ts = state.get("zoom_end")
    view = df
    if start_ts is not None and end_ts is not None:
        mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
        if bool(mask.any()):
            view = df.loc[mask]
    try:
        sums = view[metrics].sum(numeric_only=True)
    except Exception:
        return metrics

    def sort_key(m: str) -> float:
        try:
            v = float(sums.get(m, 0) or 0)
        except (TypeError, ValueError):
            return 0.0
        if v != v:  # NaN
            return 0.0
        return v

    def has_nonzero(m: str) -> bool:
        try:
            col = view[m]
        except KeyError:
            return False
        return bool((col.notna() & (col != 0)).any())

    active = [m for m in metrics if has_nonzero(m)]
    ranked = sorted(active, key=sort_key, reverse=True)
    pinned = overlay_metrics_available(df, metrics)
    rest = [m for m in ranked if m not in pinned]
    return pinned + rest


def _metric_filter_ui() -> tuple[list[dict], list[str]]:
    """Checklist options (by on-screen sum desc) + checked values."""
    ranked = _metrics_by_view_sum()
    opts = [{"label": display_metric(m), "value": m} for m in ranked]
    selected = set(state.get("visible_metrics") or [])
    vals = [m for m in ranked if m in selected]
    return opts, vals


def _visible_from_filter_selection(selected) -> list[str]:
    """Map checklist values to drawn metrics (raw columns + overlay toggles)."""
    all_m = list(state.get("metrics") or [])
    selected_set = set(selected or [])
    overlays = overlay_metrics_available(state.get("df"), all_m)
    out = [m for m in all_m if m in selected_set]
    for key in overlays:
        if key in selected_set and key not in out:
            out.append(key)
    return out


def _label_by_id(label_id):
    for item in (state["doc"] or {}).get("labels", []):
        if item["id"] == label_id:
            return item
    return None


def _label_at_time(ts, *, point_tol_seconds: float | None = None):
    """Return the best label at `ts` (prefer divider edges, then tightest range)."""
    doc = state.get("doc") or {}
    zoom_start = state.get("zoom_start")
    zoom_end = state.get("zoom_end")
    if zoom_start is not None and zoom_end is not None:
        win_sec = max((zoom_end - zoom_start).total_seconds(), 1.0)
        edge_tol = max(15 * 60.0, win_sec * 0.012)
    else:
        edge_tol = float(point_tol_seconds if point_tol_seconds is not None else 150.0)

    edge_hits: list[tuple[float, dict]] = []
    range_hits: list[tuple[float, dict]] = []
    for item in doc.get("labels", []):
        try:
            start_ts = parse_time(item["start"])
            end_ts = parse_time(item["end"])
        except (ValueError, TypeError, KeyError):
            continue
        kind = (item.get("kind") or "point").lower()
        is_point = kind == "point" or start_ts == end_ts
        if is_point:
            dist = abs((ts - start_ts).total_seconds())
            if dist <= edge_tol:
                edge_hits.append((dist, item))
            continue
        for edge in (start_ts, end_ts):
            dist = abs((ts - edge).total_seconds())
            if dist <= edge_tol:
                edge_hits.append((dist, item))
        if start_ts <= ts <= end_ts:
            range_hits.append(((end_ts - start_ts).total_seconds(), item))
    if edge_hits:
        edge_hits.sort(key=lambda x: x[0])
        return edge_hits[0][1]
    if not range_hits:
        return None
    range_hits.sort(key=lambda x: x[0])
    return range_hits[0][1]


def _select_or_toggle_label_hit(hit: dict, click_mode: str):
    """Select a graph-clicked label, or clear highlight if already selected.

    Zoom is unchanged (unlike 「라벨 선택해제」 which resets to 전체).
    """
    state["label_range_anchor"] = None
    kind = (hit.get("kind") or "point").lower()
    tag = "점" if kind == "point" else "구간"
    already = (
        state.get("highlight_id") is not None
        and str(state["highlight_id"]) == str(hit["id"])
    )
    if already:
        state["highlight_id"] = None
        state.pop("_offscreen_cleared", None)
        return (
            _build_graph(click_mode),
            no_update,
            None,
            "라벨 선택이 해제되었습니다.",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )
    state["highlight_id"] = hit["id"]
    state.pop("_offscreen_cleared", None)
    # Block stale plotly_relayout echoes from undoing this highlight.
    _arm_select_guard()
    if click_mode == "edit_range":
        status = f"선택 ({tag}): {label_line(hit)} — 빨간 경계를 드래그하세요."
    else:
        status = f"선택 ({tag}): {label_line(hit)} · 「선택 구간으로 줌」으로 확대"
    return (
        _build_graph(click_mode),
        _label_options(),
        hit["id"],
        status,
        _cancel_style(click_mode),
        no_update,
        no_update,
    )


def _zoom_to_label_item(item: dict) -> None:
    """Zoom X around a label (same window math as the web viewer) and highlight it."""
    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    kind = (item.get("kind") or "point").lower()
    is_point = kind == "point" or start_ts == end_ts
    # Point: fixed window — same base as before, plus two more 「줌아웃」
    # steps → × (1/0.7)^9 instead of ^7.
    # Range: tight fit to the label (min 30 min), then widen × (1/0.7)^7.
    fixed_point = pd.Timedelta(minutes=30) * ((1.0 / 0.7) ** 9)
    if is_point:
        mid = start_ts
        wide = fixed_point
    elif end_ts > start_ts:
        mid = start_ts + (end_ts - start_ts) / 2
        tight = max(end_ts - start_ts, pd.Timedelta(minutes=30))
        wide = tight * ((1.0 / 0.7) ** 7)
    else:
        mid = start_ts
        wide = fixed_point
    half = wide / 2
    state["zoom_start"], state["zoom_end"] = _clamp_zoom(mid - half, mid + half)
    state["highlight_id"] = item["id"]
    state["x_full_view"] = False
    state.pop("_offscreen_cleared", None)
    _reset_y()
    _arm_select_guard()


def _cancel_style(click_mode: str):
    if click_mode == "label_range" and state["label_range_anchor"] is not None:
        return {"display": "inline-block"}
    return {"display": "none"}


def _bump_label_rev() -> None:
    state["label_rev"] = int(state.get("label_rev") or 0) + 1


def _bump_place_rev() -> None:
    """Bump when range-placement overlays change (Plotly uirevision)."""
    state["place_rev"] = int(state.get("place_rev") or 0) + 1


def _shape_name(shape) -> str | None:
    if shape is None:
        return None
    if isinstance(shape, dict):
        raw = shape.get("name")
        return str(raw) if raw else None
    if getattr(shape, "name", None):
        return str(shape.name)
    if hasattr(shape, "to_plotly_json"):
        raw = (shape.to_plotly_json() or {}).get("name")
        return str(raw) if raw else None
    return None


def _shape_json(shape) -> dict:
    if isinstance(shape, dict):
        return shape
    if hasattr(shape, "to_plotly_json"):
        return shape.to_plotly_json() or {}
    return {}


def _edit_label_shape_meta(
    fig, highlight_id: str, doc: dict | None = None
) -> dict[str, int]:
    """Shape indices for live edge drag (fill + edges + highlight lines)."""
    hid = str(highlight_id)
    out: dict[str, int] = {}
    shapes = fig.layout.shapes or ()
    for i, shape in enumerate(shapes):
        name = _shape_name(shape)
        if name == f"label_fill:{hid}":
            out["fill"] = i
        elif name == f"label_edge_start:{hid}":
            out["edge_start"] = i
        elif name == f"label_edge_end:{hid}":
            out["edge_end"] = i
        elif name == LABEL_HIGHLIGHT_START:
            out["hi_start"] = i
        elif name == LABEL_HIGHLIGHT_END:
            out["hi_end"] = i

    if out.get("fill") is not None and out.get("edge_start") is not None:
        return out

    item = None
    if doc:
        item = next(
            (x for x in doc.get("labels", []) if str(x.get("id")) == hid),
            None,
        )
    if item is None:
        return out

    s = str(to_plot_time(pd.to_datetime(item["start"], utc=True)))
    e = str(to_plot_time(pd.to_datetime(item["end"], utc=True)))
    is_range = item.get("kind") == "range" or s != e
    for i, shape in enumerate(shapes):
        j = _shape_json(shape)
        stype = j.get("type")
        if is_range and stype == "rect" and out.get("fill") is None:
            if str(j.get("x0")) == s and str(j.get("x1")) == e:
                out["fill"] = i
        elif stype == "line":
            x = str(j.get("x0"))
            if x == s and i != out.get("hi_start") and out.get("edge_start") is None:
                out["edge_start"] = i
            elif (
                is_range
                and x == e
                and i != out.get("hi_end")
                and out.get("edge_end") is None
            ):
                out["edge_end"] = i
    return out


def _build_graph(click_mode: str):
    if state["df"] is None or state["doc"] is None:
        return _empty_figure()

    title = (
        f"#{state['rank']:03d} {display_plmn(state['plmn'])} | "
        f"labels={len(state['doc']['labels'])}"
    )
    fig = build_figure(
        state["df"],
        state["doc"],
        metrics=_plot_metrics(),
        color_metrics=state.get("metrics") or None,
        start=state["zoom_start"],
        end=state["zoom_end"],
        title=title,
        hover_values=False,
        max_metrics=-1,
        max_points=MAX_POINTS,
        show_labels=bool(state.get("show_anomalies", True)),
        highlight_id=state.get("highlight_id"),
    )
    # Extra revision bump so Dash/Plotly never keep a previous WebGL buffer /
    # axis range after 줌인·이동 buttons.
    state["fig_gen"] = int(state.get("fig_gen") or 0) + 1
    tmin, tmax = data_time_bounds(state["df"])
    x0 = state["zoom_start"] if state["zoom_start"] is not None else tmin
    x1 = state["zoom_end"] if state["zoom_end"] is not None else tmax
    x0, x1 = _clamp_zoom(x0, x1)
    state["zoom_start"], state["zoom_end"] = x0, x1
    x0_plot, x1_plot = to_plot_time(x0), to_plot_time(x1)
    rendered = (x0, x1)
    prev = state.get("zoom_rendered")
    if prev is not None and not _same_time_window(rendered, prev):
        state["zoom_rendered_prev"] = prev
    state["zoom_rendered"] = rendered
    # Keep layout.uirevision stable across zoom (PLMN + metric_rev). Bumping
    # metric_rev on filter changes forces Scattergl to drop removed series —
    # otherwise Plotly.react can leave ghost WebGL traces after 전체 해제.
    metric_rev = int(state.get("metric_rev") or 0)
    label_rev = int(state.get("label_rev") or 0)
    place_rev = int(state.get("place_rev") or 0)
    fig.update_layout(
        datarevision=(
            f"{fig.layout.datarevision}:{state['fig_gen']}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}"
        ),
        uirevision=(
            f"{state.get('plmn') or 'labeling'}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}"
        ),
        # Match dcc.Graph style height so select/redraw never shrinks the plot.
        # autosize stays True so width keeps filling the host.
        height=420,
        autosize=True,
    )

    fig.update_layout(
        dragmode=(
            False
            if click_mode in ("edit_range", "label_range", "label_point", "inspect")
            else (
                "pan"
                if click_mode in ("pan", "pan_keep_y")
                else "zoom"
            )
        )
    )
    # Always lock y for mouse drag: zoom/pan are time-axis only. Vertical scale
    # is controlled with Y+ / Y- / Y 자동.
    y_locked = True
    # Traces carry the full series with only the x range narrowed, so plotly's own
    # autorange would size y against data outside the window.
    y_bounds = _effective_y_bounds()
    state["y_rendered"] = y_bounds
    # Per-axis uirevision makes Plotly accept server-driven y changes (e.g. Y 자동).
    # y_gen only — baking float bounds into uirevision remounts axes on every zoom.
    if y_bounds is None:
        fig.update_yaxes(
            fixedrange=y_locked,
            autorange=True,
            uirevision=f"y:{state.get('y_gen', 0)}",
        )
    else:
        fig.update_yaxes(
            fixedrange=y_locked,
            autorange=False,
            range=list(y_bounds),
            uirevision=f"y:{state.get('y_gen', 0)}",
        )

    if any(is_rate_metric(m) for m in _plot_metrics()):
        fig.update_layout(
            yaxis2=dict(
                title="rate",
                overlaying="y",
                side="right",
                range=[0, 1],
                fixedrange=True,
                showgrid=False,
                uirevision=f"y2:{state.get('y_gen', 0)}",
            ),
            margin=dict(
                l=fig.layout.margin.l or 40,
                r=55,
                t=fig.layout.margin.t or 60,
                b=fig.layout.margin.b or 40,
            ),
        )

    if state["label_range_anchor"] is not None:
        before_shapes = len(fig.layout.shapes or ())
        fig.add_shape(**pending_range_fill_shape(state["label_range_anchor"]))
        state["_pending_fill_index"] = before_shapes
        fig.add_shape(**pending_anchor_shape(state["label_range_anchor"]))
        # No on-plot tooltip annotation — it covers the series while placing.
    else:
        state["_pending_fill_index"] = None

    if state.get("highlight_id") and state.get("show_anomalies", True):
        item = _label_by_id(state["highlight_id"])
        if item is not None:
            before = len(fig.layout.shapes or ())
            # Custom JS hit-tests these edges; Plotly native shape edit stays off.
            hs, hn = label_highlight_overlays(
                item,
                editable=False,
                line_width=4 if click_mode == "edit_range" else 1,
            )
            for s in hs:
                fig.add_shape(**s)
            for a in hn:
                fig.add_annotation(**a)
            if click_mode == "edit_range" and len(hs) >= 2:
                state["highlight_shape_idxs"] = {
                    "start": before,
                    "end": before + 1,
                }
            elif click_mode == "edit_range" and len(hs) == 1:
                state["highlight_shape_idxs"] = {
                    "start": before,
                    "end": before,
                }
            else:
                state["highlight_shape_idxs"] = None
        else:
            state["highlight_shape_idxs"] = None
    else:
        state["highlight_shape_idxs"] = None

    # Force Plotly to accept the server x window (and refreshed high-res traces)
    # after pan/zoom. A constant layout uirevision alone can keep a stale range.
    fig.update_xaxes(
        range=[x0_plot, x1_plot],
        autorange=False,
        # Custom place-time spike draws one dotted line; keep Plotly spikes off
        # in tip modes so zoom/pan/inspect don't show a double vertical.
        showspikes=(
            click_mode
            not in (
                "edit_range",
                "label_range",
                "label_point",
                "inspect",
                "pan",
                "pan_keep_y",
                "zoom",
            )
        ),
    )
    if click_mode == "edit_range":
        fig.update_layout(hovermode=False)

    # Expose edge indices + data time bounds for browser pan clamping.
    idxs = state.get("highlight_shape_idxs") if click_mode == "edit_range" else None
    meta = {
        "edit_edges": [],
        "data_x": [str(to_plot_time(tmin)), str(to_plot_time(tmax))],
        "zoom_x": [str(x0_plot), str(x1_plot)],
    }
    samples = state.get("sample_ms")
    if samples is None and state["df"] is not None and len(state["df"]):
        samples = _sample_plot_ms(state["df"])
        state["sample_ms"] = samples
    if click_mode in (
        "edit_range",
        "label_range",
        "label_point",
        "inspect",
        "pan",
        "pan_keep_y",
        "zoom",
    ) and samples:
        meta["sample_ms"] = samples
    if click_mode == "label_range" and state.get("label_range_anchor") is not None:
        meta["pending_range_start"] = str(
            to_plot_time(state["label_range_anchor"])
        )
        if state.get("_pending_fill_index") is not None:
            meta["pending_fill_index"] = int(state["_pending_fill_index"])
    if idxs:
        edges_meta = []
        if idxs.get("start") is not None:
            edges_meta.append(
                {"name": LABEL_HIGHLIGHT_START, "index": int(idxs["start"])}
            )
        if idxs.get("end") is not None and idxs.get("end") != idxs.get("start"):
            edges_meta.append(
                {"name": LABEL_HIGHLIGHT_END, "index": int(idxs["end"])}
            )
        meta["edit_edges"] = edges_meta
    if click_mode == "edit_range" and state.get("highlight_id"):
        hid = str(state["highlight_id"])
        meta["edit_label_id"] = hid
        shape_idxs = _edit_label_shape_meta(fig, hid, state.get("doc"))
        if shape_idxs:
            meta["edit_label_shapes"] = shape_idxs
    fig.update_layout(meta=meta)

    if state.get("value_cursor_pos") is not None:
        row = state["df"].iloc[state["value_cursor_pos"]]
        shape, note = value_cursor_overlays(row["time"])
        fig.add_shape(**shape)
        fig.add_annotation(**note)
        meta["value_cursor_pos"] = int(state["value_cursor_pos"])
        fig.update_layout(meta=meta)

    freeze_shape_editing(fig)
    return fig


def _snap_to_data_time(ts):
    """Snap to the nearest sample (data is 5-minute). Works for any Y click."""
    df = state["df"]
    if df is None or not len(df):
        return ts
    ts = pd.to_datetime(ts, utc=True)
    pos = int((df["time"] - ts).abs().to_numpy().argmin())
    return df.iloc[pos]["time"]


def _cancel_pending_range(click_mode: str):
    """Clear in-progress range placement (Esc / 시작점 취소)."""
    state["label_range_anchor"] = None
    _bump_place_rev()
    return (
        _build_graph(click_mode),
        no_update,
        no_update,
        "구간 설정 취소",
        _cancel_style(click_mode),
        no_update,
        no_update,
    )


def _place_label_click(ts, click_mode: str):
    """Handle point/range label placement for a plotted time click."""
    if click_mode == "label_point":
        state["label_range_anchor"] = None
        ts = _snap_to_data_time(ts)
        before = {x["id"] for x in state["doc"].get("labels", [])}
        add_label(
            state["doc"],
            kind="point",
            tag="anomaly",
            start=ts,
            metrics=["ALL"],
        )
        after = [x for x in state["doc"]["labels"] if x["id"] not in before]
        lid = after[0]["id"] if after else None
        state["highlight_id"] = lid
        opts = _label_options()
        _bump_label_rev()
        _arm_place_click_guard()
        return (
            _build_graph(click_mode),
            opts,
            lid,
            html.Span(
                f"✔ [점] anomaly 추가됨 ({lid}) {format_kst(ts)} — Save Labels로 저장",
                style={"color": "green"},
            ),
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if click_mode != "label_range":
        return None

    anchor = state["label_range_anchor"]
    if anchor is None:
        state["label_range_anchor"] = _snap_to_data_time(ts)
        _bump_place_rev()
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            (
                f"① 시작 {format_kst(state['label_range_anchor'])} "
                "· 오른쪽 끝 클릭 · Esc 취소"
            ),
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    ts = _snap_to_data_time(ts)
    if ts <= anchor:
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            (
                f"끝점은 시작보다 오른쪽이어야 합니다 "
                f"(시작 {format_kst(anchor)}). 오른쪽에서 다시 클릭하세요."
            ),
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    a, b = anchor, ts
    state["label_range_anchor"] = None
    _bump_place_rev()
    before = {x["id"] for x in state["doc"].get("labels", [])}
    add_label(
        state["doc"],
        kind="range",
        tag="anomaly",
        start=a,
        end=b,
        metrics=["ALL"],
    )
    after = [x for x in state["doc"]["labels"] if x["id"] not in before]
    lid = after[0]["id"] if after else None
    state["highlight_id"] = lid
    opts = _label_options()
    when = f"{format_kst(a)} → {format_kst(b)}"
    _bump_label_rev()
    state["_range_label_done"] = int(time.time() * 1000)
    state["_keep_highlight_id"] = True
    _arm_place_click_guard()
    return (
        _build_graph(click_mode),
        opts,
        lid,
        html.Span(
            f"✔ [구간] anomaly 추가됨 ({lid}) {when} — Save Labels로 저장 · 이동(Y자동)",
            style={"color": "green"},
        ),
        {"display": "none"},
        no_update,
        no_update,
    )


def _shape_x_from_relayout(relayout, idx: int):
    """Read the x position of shapes[idx] after an edit, if present."""
    if relayout is None:
        return None
    for key in (f"shapes[{idx}].x0", f"shapes[{idx}].x1"):
        if key in relayout:
            try:
                return parse_time(relayout[key])
            except (ValueError, TypeError):
                return None
    shapes = relayout.get("shapes")
    if isinstance(shapes, (list, tuple)) and 0 <= idx < len(shapes):
        shape = shapes[idx] or {}
        raw = shape.get("x0", shape.get("x1"))
        if raw is None:
            return None
        try:
            return parse_time(raw)
        except (ValueError, TypeError):
            return None
    return None


def _apply_label_edge_from_relayout(relayout) -> str | None:
    """Update the selected label when a highlight edge was dragged."""
    label_id = state.get("highlight_id")
    idxs = state.get("highlight_shape_idxs") or {}
    if not label_id or state["doc"] is None or not idxs:
        return None
    item = _label_by_id(label_id)
    if item is None:
        return None

    start_idx = idxs.get("start")
    end_idx = idxs.get("end")
    if start_idx is None:
        return None

    new_start = _shape_x_from_relayout(relayout, start_idx)
    new_end = (
        _shape_x_from_relayout(relayout, end_idx)
        if end_idx is not None and end_idx != start_idx
        else None
    )
    if new_start is None and new_end is None:
        return None

    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    if new_start is not None:
        start_ts = _snap_to_data_time(new_start)
    if new_end is not None:
        end_ts = _snap_to_data_time(new_end)
    elif item.get("kind") == "point" or start_idx == end_idx:
        end_ts = start_ts

    update_label(state["doc"], label_id, start=start_ts, end=end_ts)
    _bump_label_rev()
    item = _label_by_id(label_id)
    when = f"{format_kst(item['start'])} → {format_kst(item['end'])}"
    return f"구간 조절: {when} — Save Labels로 저장"


def _apply_label_edge_from_drag(payload) -> str | None:
    """Update the selected label from a horizontal-only edge drag event."""
    if not payload or state["doc"] is None:
        return None
    label_id = state.get("highlight_id")
    if not label_id:
        return None
    item = _label_by_id(label_id)
    if item is None:
        return None
    name = payload.get("name") or ""
    try:
        ts = _snap_to_data_time(parse_time(payload.get("x")))
    except (ValueError, TypeError):
        return None

    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    if name == LABEL_HIGHLIGHT_START:
        start_ts = ts
        if item.get("kind") == "point":
            end_ts = ts
    elif name == LABEL_HIGHLIGHT_END:
        end_ts = ts
    else:
        return None

    update_label(state["doc"], label_id, start=start_ts, end=end_ts)
    _bump_label_rev()
    _bump_place_rev()
    item = _label_by_id(label_id)
    when = f"{format_kst(item['start'])} → {format_kst(item['end'])}"
    return f"구간 조절: {when} — Save Labels로 저장"


def _clamp_zoom(start_ts, end_ts):
    tmin, tmax = data_time_bounds(state["df"])
    return clamp_time_range(start_ts, end_ts, tmin, tmax)


def _visible_y_bounds() -> tuple[float, float] | None:
    df = state["df"]
    metrics = [
        m
        for m in _plot_metrics()
        if not is_rate_metric(m) and not is_synthetic_overlay(m)
    ]
    if df is None or not len(df) or not metrics:
        return None
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    view = df
    if start_ts is not None and end_ts is not None:
        mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
        if mask.any():
            view = df.loc[mask]
    series = view[metrics]
    ymin = float(series.min(numeric_only=True).min())
    ymax = float(series.max(numeric_only=True).max())
    if M971_DAILY_AVG_KEY in _plot_metrics():
        dm = state.get("m971_daily_mean")
        if dm is not None:
            dmv = dm.loc[view.index]
            if len(dmv):
                ymin = min(ymin, float(dmv.min()))
                ymax = max(ymax, float(dmv.max()))
    if not (ymin == ymin and ymax == ymax):  # NaN check
        return None
    # Metrics are counts, so the baseline stays at zero unless data goes below it.
    base = min(0.0, ymin)
    if ymax <= base:
        return base, base + 1.0
    return base, ymax + (ymax - base) * 0.05


def _effective_y_bounds() -> tuple[float, float] | None:
    """Range to draw: the visible-window fit while auto, else the pinned range."""
    if state.get("y_auto") or state.get("y_min") is None or state.get("y_max") is None:
        return _visible_y_bounds()
    return float(state["y_min"]), float(state["y_max"])


def _scale_y(factor: float) -> bool:
    """Keep the Y floor fixed; only the top moves (Y+ = shrink, Y- = expand).

    X-axis stays at the bottom of the plot. Do not move ``y_min`` — moving the
    floor makes the series look like it jumps up and the plot gets shorter.
    """
    if state["df"] is None:
        return False
    bounds = _effective_y_bounds()
    if bounds is None:
        return False
    lo, hi = float(bounds[0]), float(bounds[1])
    # Keep an already-pinned floor; on first leave from auto, snap non-neg → 0.
    if state.get("y_min") is not None and not state.get("y_auto"):
        lo = float(state["y_min"])
    elif lo >= 0:
        lo = 0.0
    span = hi - lo
    if not (span > 0):
        return False
    top = lo + span * factor
    if top <= lo:
        return False
    state["y_min"], state["y_max"] = lo, float(top)
    state["y_auto"] = False
    state["y_gen"] = int(state.get("y_gen") or 0) + 1
    return True


def _reset_y() -> None:
    state["y_min"] = None
    state["y_max"] = None
    state["y_auto"] = True
    state["y_gen"] = int(state.get("y_gen") or 0) + 1


def _select_value_pos(pos: int):
    df = state["df"]
    if df is None or not len(df):
        return None
    pos = max(0, min(int(pos), len(df) - 1))
    state["value_cursor_pos"] = pos
    row = df.iloc[pos]
    panel = _hover_panel_at_row(row)
    return row, panel


def _hover_panel_at_row(row: pd.Series):
    """Bottom '이 시점 특성값' panel — visible metrics with non-zero values only."""
    visible = _plot_metrics()
    cols = [c for c in visible if not is_synthetic_overlay(c)]
    if not cols and M971_DAILY_AVG_KEY not in visible:
        return html.I("표시 중인 metric이 없습니다. 위에서 metric을 선택하세요.")
    extra: list[tuple[str, float]] = []
    if M971_DAILY_AVG_KEY in visible:
        dm = state.get("m971_daily_mean")
        if dm is not None:
            try:
                v = float(dm.loc[row.name])
            except (KeyError, TypeError, ValueError):
                v = float("nan")
            if v == v and v != 0:
                extra.append((M971_DAILY_AVG_KEY, v))
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True) + extra
    if not pairs:
        return html.I("이 시점에 0이 아닌 특성값이 없습니다.")
    return dcc.Markdown(
        ranked_hover_html(row["time"], row, cols, extra_pairs=extra or None),
        dangerously_allow_html=True,
    )


def _refresh_hover_panel():
    """Rebuild hover panel for the last hovered / inspected sample, if any."""
    df = state.get("df")
    if df is None or not len(df):
        return no_update
    pos = state.get("value_cursor_pos")
    if pos is None:
        pos = state.get("hover_pos")
    if pos is None:
        return no_update
    pos = max(0, min(int(pos), len(df) - 1))
    return _hover_panel_at_row(df.iloc[pos])


def _pin_y_before_x_change() -> None:
    """Freeze the currently drawn y range so the next rebuild won't refit."""
    pinned = state.get("y_rendered")
    if pinned is None:
        pinned = _effective_y_bounds()
    if pinned is None:
        return
    state["y_min"], state["y_max"] = float(pinned[0]), float(pinned[1])
    state["y_auto"] = False


def _shift_zoom(
    direction: int, fraction: float = 0.5, *, y_auto: bool = True
) -> bool:
    """Pan the time window; never past the dataset start/end. True if it moved."""
    if state["df"] is None:
        return False
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    if start_ts is None or end_ts is None:
        start_ts, end_ts = data_time_bounds(state["df"])
    width = end_ts - start_ts
    if width <= pd.Timedelta(0):
        return False
    before = (start_ts, end_ts)
    delta = width * fraction * direction
    new_start, new_end = _clamp_zoom(start_ts + delta, end_ts + delta)
    if (new_start, new_end) == before:
        return False
    if not y_auto:
        _pin_y_before_x_change()
    state["zoom_start"], state["zoom_end"] = new_start, new_end
    if y_auto:
        _reset_y()
    _note_x_window_changed()
    _arm_zoom_guard()
    return True


def _scale_x(factor: float, *, y_auto: bool = True) -> bool:
    """Shrink (factor < 1) or expand (factor > 1) the time window around its center."""
    if state["df"] is None:
        return False
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    if start_ts is None or end_ts is None:
        start_ts, end_ts = data_time_bounds(state["df"])
    width = end_ts - start_ts
    if width <= pd.Timedelta(0):
        return False
    mid = start_ts + width / 2
    half = width * factor / 2
    if half <= pd.Timedelta(0):
        return False
    # Keep at least a few samples visible when zooming in.
    tmin, tmax = data_time_bounds(state["df"])
    min_half = pd.Timedelta(minutes=15)
    if half < min_half and factor < 1:
        half = min_half
    new_start, new_end = _clamp_zoom(mid - half, mid + half)
    # Zoom-out already at full data bounds: nothing to do.
    if factor > 1 and _same_time_window((start_ts, end_ts), (tmin, tmax)):
        state["x_full_view"] = True
        return False
    if _same_time_window((new_start, new_end), (start_ts, end_ts)):
        return False
    if not y_auto:
        _pin_y_before_x_change()
    state["zoom_start"], state["zoom_end"] = new_start, new_end
    if y_auto:
        _reset_y()
    _note_x_window_changed()
    if factor < 1:
        _clear_highlight_if_offscreen()
    _arm_zoom_guard()
    return True


def _scale_x_from_left(factor: float, *, y_auto: bool = True) -> bool:
    """Shrink the time window while keeping the left edge fixed."""
    if state["df"] is None:
        return False
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    if start_ts is None or end_ts is None:
        start_ts, end_ts = data_time_bounds(state["df"])
    width = end_ts - start_ts
    if width <= pd.Timedelta(0) or factor >= 1:
        return False
    new_width = width * factor
    min_width = pd.Timedelta(minutes=30)
    if new_width < min_width:
        new_width = min_width
    if new_width >= width:
        return False
    _, tmax = data_time_bounds(state["df"])
    new_end = start_ts + new_width
    if new_end > tmax:
        new_end = tmax
    if new_end <= start_ts or new_end >= end_ts:
        return False
    if not y_auto:
        _pin_y_before_x_change()
    state["zoom_start"] = start_ts
    state["zoom_end"] = new_end
    if y_auto:
        _reset_y()
    _note_x_window_changed()
    _clear_highlight_if_offscreen()
    _arm_zoom_guard()
    return True


def _relayout_has_shape_edit(relayout) -> bool:
    if not relayout:
        return False
    if "shapes" in relayout:
        return True
    return any(str(k).startswith("shapes[") for k in relayout)


def _relayout_moved_non_edge(relayout) -> bool:
    """True when a shape other than the highlight edges was repositioned."""
    if not _relayout_has_shape_edit(relayout):
        return False
    idxs = state.get("highlight_shape_idxs") or {}
    edge_idxs = {i for i in (idxs.get("start"), idxs.get("end")) if i is not None}
    if "shapes" in relayout:
        # Full replace: treat as non-edge unless we already applied an edge update.
        return True
    touched = set()
    for key in relayout:
        text = str(key)
        if not text.startswith("shapes["):
            continue
        try:
            touched.add(int(text.split("[", 1)[1].split("]", 1)[0]))
        except (IndexError, ValueError):
            return True
    return bool(touched - edge_idxs)


def _same_time_window(
    a: tuple[pd.Timestamp, pd.Timestamp] | None,
    b: tuple[pd.Timestamp, pd.Timestamp] | None,
    *,
    tol_ratio: float = 0.02,
) -> bool:
    if not a or not b:
        return False
    a0, a1 = a
    b0, b1 = b
    try:
        span = max(abs((a1 - a0).total_seconds()), abs((b1 - b0).total_seconds()), 1e-6)
        return (
            abs((a0 - b0).total_seconds()) / span < tol_ratio
            and abs((a1 - b1).total_seconds()) / span < tol_ratio
        )
    except Exception:
        return False


def _label_overlaps_x_window(item: dict, z0, z1) -> bool:
    """True if the label's time span intersects [z0, z1]."""
    try:
        a = parse_time(item["start"])
        b = parse_time(item.get("end") or item["start"])
    except (ValueError, TypeError, KeyError):
        return False
    lo, hi = (a, b) if a <= b else (b, a)
    return lo <= z1 and hi >= z0


def _clear_highlight_if_offscreen() -> bool:
    """Deselect the highlighted label when it lies outside the current X window.

    Returns True if a selection was cleared.
    """
    hid = state.get("highlight_id")
    if hid is None or state.get("df") is None:
        return False
    item = _label_by_id(hid)
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if item is None or z0 is None or z1 is None:
        state["highlight_id"] = None
        state["_offscreen_cleared"] = True
        return True
    if _label_overlaps_x_window(item, z0, z1):
        return False
    state["highlight_id"] = None
    state["_offscreen_cleared"] = True
    return True


def _is_full_x_view() -> bool:
    """True when the time window is the full dataset (「전체」).

    Prefer the explicit flag, but also treat a near-full zoom window as full so
    a stale ``x_full_view=False`` (e.g. after Plotly double-click autorange left
    the browser at full while the server stayed zoomed) does not auto-zoom on
    list select.
    """
    if bool(state.get("x_full_view", True)):
        return True
    if state.get("df") is None:
        return True
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if z0 is None or z1 is None:
        return True
    return _same_time_window((z0, z1), data_time_bounds(state["df"]))


def _note_x_window_changed() -> None:
    """Refresh x_full_view after zoom_start/end change."""
    if state.get("df") is None:
        state["x_full_view"] = True
        return
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if z0 is None or z1 is None:
        state["x_full_view"] = True
        return
    state["x_full_view"] = _same_time_window((z0, z1), data_time_bounds(state["df"]))


def _arm_zoom_guard(seconds: float = 2.0) -> None:
    """Ignore browser axis echoes after a server-driven zoom/pan (prevents freeze loops)."""
    state["zoom_guard_until"] = time.monotonic() + seconds


def _arm_select_guard(seconds: float = 2.0) -> None:
    """After select/zoom-to-label: block stale relayout from clearing highlight."""
    until = time.monotonic() + seconds
    state["select_guard_until"] = until
    state["zoom_guard_until"] = until


def _arm_place_click_guard(seconds: float = 0.3) -> None:
    """Drop the trailing clickData echo after placement completes."""
    state["place_click_guard_until"] = time.monotonic() + seconds


def _place_click_guarded() -> bool:
    return time.monotonic() < float(state.get("place_click_guard_until") or 0)


def _make_axis_cmd(
    status: str = "",
    *,
    apply_x: bool = True,
    touch_y: bool = True,
    guard_seconds: float = 4.0,
    rebuild_ms: int = 300,
) -> dict:
    """Payload for instant client relayout + deferred high-res rebuild."""
    if state["df"] is None:
        return {}
    tmin, tmax = data_time_bounds(state["df"])
    x0 = state["zoom_start"] if state["zoom_start"] is not None else tmin
    x1 = state["zoom_end"] if state["zoom_end"] is not None else tmax
    x0, x1 = _clamp_zoom(x0, x1)
    state["zoom_start"], state["zoom_end"] = x0, x1
    rendered = (x0, x1)
    prev = state.get("zoom_rendered")
    if prev is not None and not _same_time_window(rendered, prev):
        state["zoom_rendered_prev"] = prev
    state["zoom_rendered"] = rendered
    _arm_zoom_guard(guard_seconds)
    if touch_y:
        yb = _effective_y_bounds()
        state["y_rendered"] = yb
        y_payload = [float(yb[0]), float(yb[1])] if yb else None
    else:
        # Keep the pinned / currently drawn y; do not push a new y range.
        yb = None
        if state.get("y_min") is not None and state.get("y_max") is not None:
            yb = (float(state["y_min"]), float(state["y_max"]))
        elif state.get("y_rendered") is not None:
            yb = state["y_rendered"]
        state["y_rendered"] = yb
        y_payload = None
    state["axis_cmd_seq"] = int(state.get("axis_cmd_seq") or 0) + 1
    return {
        "seq": state["axis_cmd_seq"],
        # Drag zoom/pan already set x in the browser; re-applying x fights the drag.
        "x": [str(to_plot_time(x0)), str(to_plot_time(x1))] if apply_x else None,
        "y": y_payload,
        "touch_y": bool(touch_y),
        "status": status,
        "rebuild_ms": int(rebuild_ms),
    }


def _apply_axes_from_relayout(
    relayout, *, ignore_guard: bool = False, y_auto: bool = True
) -> bool:
    if not relayout or state["df"] is None:
        return False
    if (not ignore_guard) and time.monotonic() < float(
        state.get("zoom_guard_until") or 0
    ):
        return False

    start_ts = end_ts = None
    try:
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            start_ts = parse_time(relayout["xaxis.range[0]"])
            end_ts = parse_time(relayout["xaxis.range[1]"])
        elif "xaxis.range" in relayout:
            start_ts = parse_time(relayout["xaxis.range"][0])
            end_ts = parse_time(relayout["xaxis.range"][1])
        elif relayout.get("xaxis.autorange"):
            # Double-click zoom-out: keep server in sync with the browser full view.
            state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
            state["x_full_view"] = True
            if y_auto:
                _reset_y()
            return True
    except (ValueError, TypeError):
        return False

    if start_ts is not None and end_ts is not None and end_ts > start_ts:
        incoming = _clamp_zoom(start_ts, end_ts)
        # Ignore echoes of the window we just drew, and of the previous window
        # (button zoom is often undone by a delayed replay of the old range).
        if _same_time_window(incoming, state.get("zoom_rendered")):
            return False
        if _same_time_window(incoming, state.get("zoom_rendered_prev")):
            return False
        prev0, prev1 = state.get("zoom_start"), state.get("zoom_end")
        prev_w = None
        if prev0 is not None and prev1 is not None and prev1 > prev0:
            prev_w = (prev1 - prev0).total_seconds()
        if y_auto:
            state["zoom_start"], state["zoom_end"] = incoming
            _reset_y()
        else:
            pinned = state.get("y_rendered")
            if pinned is None:
                pinned = _effective_y_bounds()
            state["zoom_start"], state["zoom_end"] = incoming
            if pinned is not None:
                state["y_min"], state["y_max"] = float(pinned[0]), float(pinned[1])
                state["y_auto"] = False
        _note_x_window_changed()
        # Drag / box zoom-in that leaves the selection off-screen → clear it.
        new_w = (incoming[1] - incoming[0]).total_seconds()
        if prev_w is not None and new_w < prev_w * 0.98:
            _clear_highlight_if_offscreen()
        return True

    if relayout.get("yaxis.autorange"):
        _reset_y()
        return True
    return False


def _pin_y(y_range) -> bool:
    """Store an explicit y window from Y+/Y- (not from graph relayout echoes)."""
    try:
        ymin = float(y_range[0])
        ymax = float(y_range[1])
    except (ValueError, TypeError):
        return False
    if ymax <= ymin:
        return False
    drawn = state.get("y_rendered")
    if state.get("y_auto") and drawn is not None and _same_range(drawn, (ymin, ymax)):
        return False
    state["y_min"], state["y_max"] = ymin, ymax
    state["y_auto"] = False
    state["y_gen"] = int(state.get("y_gen") or 0) + 1
    return True


def _sync_zoom_from_view(view_range) -> None:
    """Adopt the x range the browser is actually showing.

    Y is not synced from the browser: zoom/pan echo the previous y and would
    undo Y 자동. Vertical scale stays server-driven (Y+/Y-/Y 자동).

    Disabled from `_main_body` (stale Store races). Kept for emergency/debug.
    Never overwrite an intentional zoom with a full-window echo.
    """
    if state["df"] is None or not view_range:
        return
    if time.monotonic() < float(state.get("zoom_guard_until") or 0):
        return
    if isinstance(view_range, dict):
        x_range = view_range.get("x")
    elif isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        x_range = view_range
    else:
        return

    if x_range and len(x_range) == 2:
        try:
            start_ts = parse_time(x_range[0])
            end_ts = parse_time(x_range[1])
        except (ValueError, TypeError):
            start_ts = end_ts = None
        if start_ts is not None and end_ts is not None and end_ts > start_ts:
            incoming = _clamp_zoom(start_ts, end_ts)
            if _same_time_window(incoming, state.get("zoom_rendered")):
                return
            if _same_time_window(incoming, state.get("zoom_rendered_prev")):
                return
            # Refuse stale full-window echoes while we believe we are zoomed.
            if not state.get("x_full_view", True) and _same_time_window(
                incoming, data_time_bounds(state["df"])
            ):
                return
            state["zoom_start"], state["zoom_end"] = incoming
            _note_x_window_changed()


def _same_range(a: tuple[float, float], b: tuple[float, float]) -> bool:
    span = max(abs(a[1] - a[0]), 1e-9)
    return abs(a[0] - b[0]) / span < 1e-3 and abs(a[1] - b[1]) / span < 1e-3


def _do_load(plmn: str, click_mode: str):
    rank = int(rank_df.loc[rank_df["PLMN"] == plmn, "rank"].iloc[0])
    df = load_plmn(plmn)
    metrics = metric_columns(df)
    overlays = overlay_metrics_available(df, metrics)
    doc = load_labels(plmn, rank=rank)
    tmin, tmax = data_time_bounds(df)
    dm = m971_daily_mean_series(df)
    state.update(
        df=df,
        doc=doc,
        plmn=plmn,
        rank=rank,
        metrics=metrics,
        visible_metrics=list(metrics) + [o for o in overlays if o not in metrics],
        m971_daily_mean=dm,
        metric_rev=int(state.get("metric_rev") or 0) + 1,
        zoom_start=tmin,
        zoom_end=tmax,
        x_full_view=True,
        y_min=None,
        y_max=None,
        y_auto=True,
        y_gen=0,
        y_rendered=None,
        label_range_anchor=None,
        highlight_id=None,
        highlight_shape_idxs=None,
        value_cursor_pos=None,
        hover_pos=None,
        # Epoch-ms of KST-naive plot times (matches Plotly axis / to_plot_time).
        sample_ms=_sample_plot_ms(df),
    )
    opts = _label_options()
    m_opts, m_vals = _metric_filter_ui()
    return (
        _build_graph(click_mode),
        opts,
        None,
        "",
        _cancel_style(click_mode),
        html.I("그래프에 커서를 올리면 이 시점의 특성값이 내림차순으로 표시됩니다."),
        plmn,
        m_opts,
        m_vals,
    )


# Prefetch rank #1 so the first paint already shows data (no empty "선택하세요" flash).
_START_PLMN = plmn_ids[0] if plmn_ids else None
_START_LABEL_OPTS: list = []
_START_LABEL_VALUE = None
_START_FIGURE = None
_START_METRIC_OPTS: list = []
_START_METRIC_VALS: list = []
if _START_PLMN:
    _do_load(_START_PLMN, "zoom")
    _START_FIGURE = _build_graph("zoom")
    _START_LABEL_OPTS = _label_options()
    _START_LABEL_VALUE = None
    _START_METRIC_OPTS, _START_METRIC_VALS = _metric_filter_ui()
else:
    _START_FIGURE = _empty_figure()


app = Dash(__name__, title="Anomaly Labeling")
app.layout = html.Div(
    [
        html.H3("1) 사업자 선택", style={"margin": "4px 0"}),
        html.Div(
            [
                html.Button("◀ Prev", id="btn-prev", n_clicks=0),
                dcc.Dropdown(
                    id="dd-plmn",
                    options=plmn_options,
                    value=plmn_options[0]["value"] if plmn_options else None,
                    clearable=False,
                    style={"width": "560px", "display": "inline-block"},
                ),
                html.Button("Next ▶", id="btn-next", n_clicks=0),
            ],
            style={
                "display": "flex",
                "gap": "8px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
        ),
        html.H3("2) Anomaly", style={"margin": "10px 0 4px"}),
        html.Div(
            "anomaly 라벨: 구간은 왼쪽 시작 → 오른쪽 끝 2클릭, 점은 시점 1클릭으로 추가하세요. "
            "목록에서 선택 후 구간 편집으로 경계를 조절할 수 있습니다.",
            style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="label-list",
                    options=_START_LABEL_OPTS,
                    value=_START_LABEL_VALUE,
                    clearable=True,
                    searchable=False,
                    placeholder="anomaly 구간 선택",
                    style={"width": "680px", "flex": "0 0 680px"},
                ),
                html.Button("라벨 선택해제", id="btn-clear-selection", n_clicks=0),
                html.Button("선택 구간으로 줌", id="btn-zoom-selected", n_clicks=0),
                html.Button("선택 라벨 삭제", id="btn-delete", n_clicks=0),
                html.Button("Save Labels", id="btn-save", n_clicks=0),
                html.Button("Reload Saved", id="btn-reload", n_clicks=0),
            ],
            style={
                "display": "flex",
                "gap": "8px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "marginBottom": "8px",
            },
        ),
        html.Div(
            id="confirm-modal",
            style={
                "display": "none",
                "position": "fixed",
                "inset": "0",
                "zIndex": 2000,
                "alignItems": "center",
                "justifyContent": "center",
            },
            children=[
                html.Div(
                    id="confirm-modal-backdrop",
                    n_clicks=0,
                    style={
                        "position": "absolute",
                        "inset": "0",
                        "background": "rgba(15, 23, 42, 0.45)",
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            id="confirm-modal-title",
                            children="확인",
                            style={"margin": "0 0 10px", "fontSize": "18px"},
                        ),
                        html.P(
                            id="confirm-modal-message",
                            children="",
                            style={
                                "margin": "0 0 18px",
                                "lineHeight": "1.5",
                                "whiteSpace": "pre-wrap",
                                "color": "#334155",
                            },
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "취소",
                                    id="btn-confirm-cancel",
                                    n_clicks=0,
                                    style={
                                        "padding": "8px 16px",
                                        "border": "1px solid #cbd5e1",
                                        "borderRadius": "6px",
                                        "background": "#fff",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Button(
                                    "확인",
                                    id="btn-confirm-ok",
                                    n_clicks=0,
                                    style={
                                        "padding": "8px 16px",
                                        "border": "none",
                                        "borderRadius": "6px",
                                        "background": "#2563eb",
                                        "color": "#fff",
                                        "cursor": "pointer",
                                        "fontWeight": "600",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "flex-end",
                                "gap": "8px",
                            },
                        ),
                    ],
                    style={
                        "position": "relative",
                        "zIndex": 1,
                        "width": "min(420px, calc(100vw - 32px))",
                        "background": "#fff",
                        "borderRadius": "10px",
                        "padding": "20px 22px",
                        "boxShadow": "0 18px 40px rgba(15, 23, 42, 0.25)",
                    },
                ),
            ],
        ),
        html.Div(
            [
                html.H3("3) 그래프", style={"margin": "0"}),
                dcc.RadioItems(
                    id="anomaly-overlay-mode",
                    options=[
                        {"label": "anomaly 표시", "value": "show"},
                        {"label": "anomaly 숨김", "value": "hide"},
                    ],
                    value="show",
                    inline=True,
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
                "margin": "10px 0 4px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                dcc.RadioItems(
                    id="click-mode",
                    options=[
                        {"label": "줌", "value": "zoom"},
                        {"label": "이동(Y자동)", "value": "pan"},
                        {"label": "이동(Y고정)", "value": "pan_keep_y"},
                        {"label": "값 탐색(클릭+←→)", "value": "inspect"},
                        {"label": "구간 라벨(2클릭)", "value": "label_range"},
                        {"label": "점 라벨(1클릭)", "value": "label_point"},
                        {"label": "구간 편집", "value": "edit_range"},
                    ],
                    value="zoom",
                    inline=True,
                    style={"marginRight": "12px"},
                ),
                html.Button("◀", id="btn-pan-left", n_clicks=0, title="왼쪽(과거)으로 이동"),
                html.Button("▶", id="btn-pan-right", n_clicks=0, title="오른쪽(미래)으로 이동"),
                html.Button("Y+", id="btn-y-in", n_clicks=0, title="세로축 확대"),
                html.Button("Y-", id="btn-y-out", n_clicks=0, title="세로축 축소"),
                html.Button("Y 자동", id="btn-y-auto", n_clicks=0, title="세로축 자동 맞춤"),
                html.Button("줌인", id="btn-zoom-in", n_clicks=0, title="시간축 한 단계 확대 (중앙 기준)"),
                html.Button(
                    "줌인(왼쪽)",
                    id="btn-zoom-in-left",
                    n_clicks=0,
                    title="왼쪽 시작점 고정 후 시간축 확대",
                ),
                html.Button("줌아웃", id="btn-zoom-out", n_clicks=0, title="시간축 한 단계 축소"),
                html.Button("전체", id="btn-reset-zoom", n_clicks=0, title="가로·세로 전체 보기"),
                html.Button(
                    "시작점 취소",
                    id="btn-cancel-range",
                    n_clicks=0,
                    style={"display": "none"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "nowrap",
                "alignItems": "center",
                "gap": "8px",
                "border": "1px solid #e0e0e0",
                "padding": "6px",
                "marginBottom": "4px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.B("표시 metric", style={"marginRight": "8px"}),
                        html.Button(
                            "전체 선택",
                            id="btn-metrics-all",
                            n_clicks=0,
                            title="모든 metric을 다시 표시",
                        ),
                        html.Button(
                            "전체 해제",
                            id="btn-metrics-none",
                            n_clicks=0,
                            title="모든 metric 숨김 (Y축 스케일 유지)",
                        ),
                        html.Span(
                            "체크 해제하면 그래프에서 숨깁니다",
                            style={
                                "fontSize": "12px",
                                "color": "#666",
                                "marginLeft": "8px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "flexWrap": "wrap",
                        "gap": "4px",
                        "marginBottom": "4px",
                    },
                ),
                dcc.Checklist(
                    id="metric-filter",
                    options=_START_METRIC_OPTS,
                    value=_START_METRIC_VALS,
                    inline=True,
                    labelStyle={
                        "display": "inline-block",
                        "marginRight": "12px",
                        "marginBottom": "2px",
                        "fontSize": "12px",
                        "whiteSpace": "nowrap",
                    },
                    style={"lineHeight": "1.6"},
                ),
            ],
            style={
                "border": "1px solid #e0e0e0",
                "padding": "6px 8px",
                "marginBottom": "4px",
                "maxHeight": "110px",
                "overflowY": "auto",
                "background": "#fafafa",
            },
        ),
        dcc.Graph(
            id="graph",
            figure=_START_FIGURE,
            config={
                "responsive": True,
                "displayModeBar": True,
                "showTips": False,
            },
            style={"width": "100%", "height": "420px"},
        ),
        html.Div(id="status", style={"display": "none"}),
        html.B("이 시점 특성값 (값 내림차순)"),
        html.Div(
            id="hover-panel",
            children=html.I(
                "그래프에 커서를 올리면 이 시점의 특성값이 내림차순으로 표시됩니다."
            ),
            style={
                "height": "240px",
                "overflow": "auto",
                "border": "1px solid #ddd",
                "padding": "8px",
                "width": "100%",
                "boxSizing": "border-box",
            },
        ),
        dcc.Store(id="key-event"),
        dcc.Store(id="key-listener-state"),
        dcc.Store(id="view-range"),
        dcc.Store(id="edge-drag-event"),
        dcc.Store(id="shape-click-event"),
        dcc.Store(id="axis-cmd"),
        dcc.Store(id="axis-cmd-ack"),
        dcc.Store(id="rebuild-trigger"),
        dcc.Store(id="range-label-done"),
    ],
    style={"fontFamily": "sans-serif", "padding": "12px", "maxWidth": "100%"},
)


@app.callback(
    Output("graph", "figure"),
    Output("label-list", "options"),
    Output("label-list", "value"),
    Output("status", "children"),
    Output("btn-cancel-range", "style"),
    Output("hover-panel", "children"),
    Output("dd-plmn", "value"),
    Output("metric-filter", "options"),
    Output("metric-filter", "value"),
    Output("range-label-done", "data"),
    Input("dd-plmn", "value"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("click-mode", "value"),
    Input("anomaly-overlay-mode", "value"),
    Input("btn-cancel-range", "n_clicks"),
    Input("btn-confirm-ok", "n_clicks"),
    Input("btn-clear-selection", "n_clicks"),
    Input("btn-zoom-selected", "n_clicks"),
    Input("label-list", "value"),
    Input("key-event", "data"),
    Input("edge-drag-event", "data"),
    Input("shape-click-event", "data"),
    Input("graph", "clickData"),
    Input("graph", "hoverData"),
    Input("graph", "relayoutData"),
    State("click-mode", "value"),
    State("view-range", "data"),
    prevent_initial_call=False,
)
def _main(
    plmn,
    n_prev,
    n_next,
    _mode_change,
    anomaly_overlay_mode,
    n_cancel,
    n_confirm_ok,
    n_clear_selection,
    n_zoom_sel,
    selected_label,
    key_event,
    edge_drag,
    shape_click,
    click_data,
    hover_data,
    relayout,
    click_mode,
    view_range,
):
    result = _main_body(
        plmn,
        n_prev,
        n_next,
        _mode_change,
        anomaly_overlay_mode,
        n_cancel,
        n_confirm_ok,
        n_clear_selection,
        n_zoom_sel,
        selected_label,
        key_event,
        edge_drag,
        shape_click,
        click_data,
        hover_data,
        relayout,
        click_mode,
        view_range,
    )
    if result is None:
        return (no_update,) * 10
    out = list(result) if len(result) == 9 else list(result) + [no_update, no_update]
    # Keep filter order in sync with the current on-screen window whenever the
    # figure is rebuilt (zoom / pan / PLMN load already set options explicitly).
    if out[0] is not no_update and state.get("df") is not None and out[7] is no_update:
        m_opts, _ = _metric_filter_ui()
        out[7] = m_opts
    out.append(state.pop("_range_label_done", no_update))
    return tuple(out)


@app.callback(
    Output("click-mode", "value", allow_duplicate=True),
    Output("rebuild-trigger", "data", allow_duplicate=True),
    Input("range-label-done", "data"),
    prevent_initial_call=True,
)
def _finish_range_label(seq):
    if not seq:
        return no_update, no_update
    return "pan", {"seq": seq, "t": time.time()}


@app.callback(
    Output("click-mode", "value", allow_duplicate=True),
    Input("btn-zoom-selected", "n_clicks"),
    Input("label-list", "value"),
    Input("btn-reset-zoom", "n_clicks"),
    State("label-list", "value"),
    prevent_initial_call=True,
)
def _sync_click_mode_on_nav(_n_zoom, _list_value, _n_reset, selected_label):
    """선택 구간 줌 → 이동(Y자동); 「전체」 → 줌."""
    tid = getattr(callback_context, "triggered_id", None)
    if tid == "btn-reset-zoom":
        return "zoom"
    if tid == "btn-zoom-selected":
        if not selected_label or _label_by_id(selected_label) is None:
            return no_update
        return "pan"
    if tid == "label-list":
        if not selected_label:
            return no_update
        # Same rule as main: only when already zoomed (not full view).
        if _is_full_x_view():
            return no_update
        if _label_by_id(selected_label) is None:
            return no_update
        return "pan"
    return no_update


@app.callback(
    Output("confirm-modal", "style"),
    Output("confirm-modal-title", "children"),
    Output("confirm-modal-message", "children"),
    Output("btn-confirm-ok", "children"),
    Output("btn-confirm-ok", "style"),
    Output("status", "children", allow_duplicate=True),
    Input("btn-delete", "n_clicks"),
    Input("btn-save", "n_clicks"),
    Input("btn-reload", "n_clicks"),
    Input("btn-confirm-cancel", "n_clicks"),
    Input("btn-confirm-ok", "n_clicks"),
    Input("confirm-modal-backdrop", "n_clicks"),
    State("label-list", "value"),
    prevent_initial_call=True,
)
def _confirm_modal_ui(
    _n_del, _n_save, _n_reload, _n_cancel, _n_ok, _n_backdrop, selected_label
):
    tid = getattr(callback_context, "triggered_id", None)
    hidden = {
        "display": "none",
        "position": "fixed",
        "inset": "0",
        "zIndex": 2000,
        "alignItems": "center",
        "justifyContent": "center",
    }
    shown = {**hidden, "display": "flex"}
    ok_base = {
        "padding": "8px 16px",
        "border": "none",
        "borderRadius": "6px",
        "color": "#fff",
        "cursor": "pointer",
        "fontWeight": "600",
    }
    if tid == "btn-delete":
        if not selected_label:
            state["confirm_action"] = None
            return (
                hidden,
                no_update,
                no_update,
                no_update,
                no_update,
                "삭제할 라벨을 선택하세요.",
            )
        item = _label_by_id(selected_label)
        detail = label_line(item) if item else str(selected_label)
        state["confirm_action"] = "delete"
        return (
            shown,
            "라벨 삭제",
            f"선택한 라벨을 삭제할까요?\n\n{detail}",
            "삭제",
            {**ok_base, "background": "#dc2626"},
            no_update,
        )
    if tid == "btn-save":
        if state.get("doc") is None:
            state["confirm_action"] = None
            return (
                hidden,
                no_update,
                no_update,
                no_update,
                no_update,
                "저장할 라벨이 없습니다.",
            )
        plmn = state.get("plmn") or ""
        n = len((state.get("doc") or {}).get("labels") or [])
        state["confirm_action"] = "save"
        return (
            shown,
            "라벨 저장",
            f"현재 라벨을 파일에 저장할까요?\n\n{display_plmn(plmn)} · {n}개",
            "저장",
            {**ok_base, "background": "#2563eb"},
            no_update,
        )
    if tid == "btn-reload":
        if state.get("plmn") is None:
            state["confirm_action"] = None
            return (
                hidden,
                no_update,
                no_update,
                no_update,
                no_update,
                "불러올 사업자가 없습니다.",
            )
        plmn = state.get("plmn") or ""
        state["confirm_action"] = "reload"
        return (
            shown,
            "라벨 다시 불러오기",
            f"저장되지 않은 변경이 있으면 사라집니다.\n파일에서 라벨을 다시 불러올까요?\n\n{display_plmn(plmn)}",
            "불러오기",
            {**ok_base, "background": "#2563eb"},
            no_update,
        )
    if tid in ("btn-confirm-cancel", "btn-confirm-ok", "confirm-modal-backdrop"):
        if tid != "btn-confirm-ok":
            state["confirm_action"] = None
        return hidden, no_update, no_update, no_update, no_update, no_update
    return hidden, no_update, no_update, no_update, no_update, no_update


def _main_body(
    plmn,
    n_prev,
    n_next,
    _mode_change,
    anomaly_overlay_mode,
    n_cancel,
    n_confirm_ok,
    n_clear_selection,
    n_zoom_sel,
    selected_label,
    key_event,
    edge_drag,
    shape_click,
    click_data,
    hover_data,
    relayout,
    click_mode,
    view_range,
):
    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    click_mode = click_mode or "zoom"
    state["show_anomalies"] = (anomaly_overlay_mode or "show") != "hide"

    # ----- load on startup / dropdown change / prev / next -----
    triggered_id = getattr(callback_context, "triggered_id", None)
    boot = triggered_id is None or (not prop) or prop == "."
    target = plmn
    if prop == "btn-prev.n_clicks" and plmn in plmn_ids:
        i = plmn_ids.index(plmn)
        if i > 0:
            target = plmn_ids[i - 1]
        else:
            return (no_update,) * 7
    elif prop == "btn-next.n_clicks" and plmn in plmn_ids:
        i = plmn_ids.index(plmn)
        if i < len(plmn_ids) - 1:
            target = plmn_ids[i + 1]
        else:
            return (no_update,) * 7

    want_load = (
        boot
        or prop in ("dd-plmn.value", "btn-prev.n_clicks", "btn-next.n_clicks")
        or (state.get("df") is None and bool(target))
    )
    if want_load:
        if not target:
            return (
                no_update,
                no_update,
                no_update,
                "사업자를 선택하세요.",
                no_update,
                no_update,
                no_update,
            )
        already = target == state.get("plmn") and state.get("df") is not None
        if already and prop == "dd-plmn.value":
            # Echo after Prev/Next wrote the same value back.
            return (no_update,) * 7
        if already:
            # Startup prefetch already loaded rank #1 — keep it, clear status.
            opts = _label_options()
            m_opts, m_vals = _metric_filter_ui()
            return (
                _build_graph(click_mode),
                opts,
                no_update,
                "",
                _cancel_style(click_mode),
                no_update,
                no_update,
                m_opts,
                m_vals,
            )
        return _do_load(target, click_mode)

    if state["df"] is None or state["doc"] is None:
        return (
            no_update,
            no_update,
            no_update,
            "사업자를 선택하세요.",
            no_update,
            no_update,
            no_update,
        )

    # Drag zoom/pan update state in `_drag_axis_nav`. Do not replay the
    # browser view-range Store on every main event — it often lags behind
    # list/button zooms and resets server zoom to full while the plot still
    # shows the zoomed window (zoom-out no-ops; zoom-in then jumps wide).

    # ----- click mode -----
    if prop == "click-mode.value":
        state["label_range_anchor"] = None
        state.pop("place_click_guard_until", None)
        if click_mode != "inspect":
            state["value_cursor_pos"] = None
        keep_highlight = state.pop("_keep_highlight_id", False)
        if selected_label and not keep_highlight:
            state["highlight_id"] = selected_label
        if click_mode == "zoom":
            status = "그래프에서 좌우로 드래그해 시간축을 확대하세요."
        elif click_mode == "pan":
            status = "좌우로 드래그해 이동하세요. 놓을 때 Y 자동이 적용됩니다."
        elif click_mode == "pan_keep_y":
            status = (
                "좌우로 드래그해 이동하세요. Y축은 유지되고, "
                "해상도만 바로 갱신됩니다."
            )
        elif click_mode == "inspect":
            status = "그래프에서 시점을 클릭하거나 마우스를 올린 뒤 ←/→ 키로 값을 탐색하세요."
        elif click_mode == "label_range":
            status = (
                "① 왼쪽 시작 클릭 → ② 오른쪽 끝 클릭 · Esc 취소"
            )
        elif click_mode == "label_point":
            status = "그래프에서 시점을 한 번 클릭하면 점(세로선) 라벨이 추가됩니다."
        elif click_mode == "edit_range":
            status = (
                "빨간 경계선 위에서만 ↔ 커서가 됩니다. 선을 좌우로 드래그하세요. "
                "화면 이동은 ◀ ▶ 를 사용하세요."
                if state.get("highlight_id")
                else (
                    "분홍 구간(또는 목록)을 클릭해 선택한 뒤 "
                    "좌·우 경계선을 드래그하세요."
                )
            )
        else:
            status = "① 왼쪽 시작 클릭 → ② 오른쪽 끝 클릭 · Esc 취소"
        return (
            _build_graph(click_mode),
            _label_options(),
            state.get("highlight_id"),
            status,
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "anomaly-overlay-mode.value":
        shown = "표시" if state.get("show_anomalies", True) else "숨김"
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            f"anomaly 오버레이: {shown}",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "btn-cancel-range.n_clicks":
        return _cancel_pending_range(click_mode)

    if prop == "btn-confirm-ok.n_clicks":
        action = state.pop("confirm_action", None)
        if action == "save":
            path = save_labels(state["doc"])
            return (
                no_update,
                _label_options(),
                state.get("highlight_id"),
                html.Span(f"Saved: {path}", style={"color": "green"}),
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        if action == "reload":
            state["doc"] = load_labels(state["plmn"], rank=state["rank"])
            state["highlight_id"] = None
            state["label_range_anchor"] = None
            if state.get("df") is not None:
                state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
                state["x_full_view"] = True
                _reset_y()
            opts = _label_options()
            return (
                _build_graph(click_mode),
                opts,
                None,
                "라벨 파일을 다시 불러왔습니다.",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        if action == "delete":
            if not selected_label:
                return (
                    no_update,
                    no_update,
                    no_update,
                    "삭제할 라벨을 선택하세요.",
                    no_update,
                    no_update,
                    no_update,
                )
            remove_label(state["doc"], selected_label)
            if state.get("highlight_id") == selected_label:
                state["highlight_id"] = None
            _bump_label_rev()
            opts = _label_options()
            return (
                _build_graph(click_mode),
                opts,
                opts[0]["value"] if opts else None,
                f"삭제됨: {selected_label} — Save Labels로 저장",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        return (no_update,) * 7

    if prop == "btn-clear-selection.n_clicks":
        state["highlight_id"] = None
        state["label_range_anchor"] = None
        if state.get("df") is not None:
            state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
            state["x_full_view"] = True
            _reset_y()
        return (
            _build_graph(click_mode),
            no_update,
            None,
            "라벨 선택이 해제되었습니다.",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "btn-zoom-selected.n_clicks":
        item = _label_by_id(selected_label)
        if item is None:
            return (no_update,) * 7
        _zoom_to_label_item(item)
        # Switch UI to 이동(Y자동); click-mode Output is set by sibling callback.
        return (
            _build_graph("pan"),
            _label_options(),
            item["id"],
            f"선택 구간으로 줌: {label_line(item)}",
            _cancel_style("pan"),
            no_update,
            no_update,
        )

    if prop == "label-list.value":
        if not selected_label:
            # Clear highlight only. Do NOT reset zoom — graph deselect also
            # writes value=None and must leave a zoomed window intact.
            # 「라벨 선택해제」 / 「전체」 are what reset to full view.
            state["highlight_id"] = None
            state["label_range_anchor"] = None
            state.pop("_offscreen_cleared", None)
            return (
                _build_graph(click_mode),
                no_update,
                None,
                "라벨 선택이 해제되었습니다.",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        item = _label_by_id(selected_label)
        if item is None:
            state["highlight_id"] = selected_label
            state.pop("_offscreen_cleared", None)
            _arm_select_guard()
            return (
                _build_graph(click_mode),
                no_update,
                selected_label,
                "",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        # Criterion is view state only (not whether a label is selected):
        # full → select only; already zoomed → select + zoom to the new item.
        do_zoom = not _is_full_x_view()
        if do_zoom:
            _zoom_to_label_item(item)
            mode = "pan"
            status = f"선택 구간으로 줌: {label_line(item)}"
        else:
            state["highlight_id"] = item["id"]
            state.pop("_offscreen_cleared", None)
            _arm_select_guard()
            mode = click_mode
            if click_mode == "edit_range":
                status = "빨간 좌·우 경계선만 좌우로 드래그하세요. 화면 이동은 ◀ ▶ 버튼을 사용하세요."
            else:
                status = f"선택: {label_line(item)}"
        return (
            _build_graph(mode),
            no_update,
            selected_label,
            status,
            _cancel_style(mode),
            no_update,
            no_update,
        )

    if prop == "key-event.data":
        if key_event and key_event.get("key") == "Escape":
            if state.get("label_range_anchor") is not None:
                return _cancel_pending_range(click_mode)
            return (no_update,) * 7
        if click_mode != "inspect" or not key_event:
            return (no_update,) * 7
        if key_event.get("key") not in ("ArrowLeft", "ArrowRight"):
            return (no_update,) * 7
        current = state.get("value_cursor_pos")
        if current is None:
            current = state.get("hover_pos")
        if current is None:
            return (no_update,) * 7
        step = -1 if key_event.get("key") == "ArrowLeft" else 1
        steps = max(1, int(key_event.get("steps") or 1))
        selected = _select_value_pos(current + step * steps)
        if selected is None:
            return (no_update,) * 7
        row, panel = selected
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            f"값 탐색: {format_kst(row['time'])} · ←/→ 키로 이동",
            _cancel_style(click_mode),
            panel,
            no_update,
        )

    if prop == "graph.hoverData" and hover_data and hover_data.get("points"):
        x = hover_data["points"][0].get("x")
        if x is None:
            return (no_update,) * 7
        ts = parse_time(x)
        nearest = (state["df"]["time"] - ts).abs().idxmin()
        state["hover_pos"] = int(state["df"].index.get_loc(nearest))
        row = state["df"].loc[nearest]
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            _hover_panel_at_row(row),
            no_update,
        )

    if prop == "edge-drag-event.data":
        if click_mode != "edit_range":
            return (no_update,) * 7
        edge_status = _apply_label_edge_from_drag(edge_drag)
        if edge_status is None:
            return (no_update,) * 7
        return (
            _build_graph(click_mode),
            _label_options(),
            state.get("highlight_id"),
            edge_status,
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "shape-click-event.data" and shape_click and shape_click.get("x") is not None:
        # Click on plot area (shapes themselves are not clickable).
        if state.get("df") is None or state.get("doc") is None:
            return (no_update,) * 7
        try:
            ts = parse_time(shape_click["x"])
        except (ValueError, TypeError):
            return (no_update,) * 7
        last_ts, last_at = state.get("_last_click", (None, 0.0))
        now = time.monotonic()
        range_end_pending = (
            click_mode == "label_range"
            and state.get("label_range_anchor") is not None
        )
        # Drop echo of the same sample (shape-click + clickData).
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        if (
            not range_end_pending
            and click_mode in ("edit_range", "inspect", "label_point")
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        state["_last_click"] = (ts, now)

        # Range/point placement.
        if click_mode in ("label_range", "label_point"):
            if _place_click_guarded():
                return (no_update,) * 7
            placed = _place_label_click(ts, click_mode)
            return placed if placed is not None else (no_update,) * 7

        # 값 탐색: empty-area clicks snap to the nearest sample (same as line clicks).
        if click_mode == "inspect":
            pos = int((state["df"]["time"] - ts).abs().to_numpy().argmin())
            selected = _select_value_pos(pos)
            if selected is None:
                return (no_update,) * 7
            row, panel = selected
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                f"값 탐색: {format_kst(row['time'])} · ←/→ 키로 이동",
                _cancel_style(click_mode),
                panel,
                no_update,
            )

        hit = _label_at_time(ts)
        if hit is None:
            return (no_update,) * 7
        return _select_or_toggle_label_hit(hit, click_mode)

    if prop == "graph.relayoutData":
        # Drag zoom/pan: Y 자동 + high-res rebuild go through `_drag_axis_nav`
        # (axis-cmd), same path as the toolbar buttons — avoids a second full
        # figure rebuild fighting the live drag.
        return (no_update,) * 7

    if prop == "graph.clickData" and click_data and click_data.get("points"):
        x = click_data["points"][0].get("x")
        if x is None:
            return (no_update,) * 7
        ts = parse_time(x)
        last_ts, last_at = state.get("_last_click", (None, 0.0))
        now = time.monotonic()
        range_end_pending = (
            click_mode == "label_range"
            and state.get("label_range_anchor") is not None
        )
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        if (
            not range_end_pending
            and click_mode in ("edit_range", "inspect", "label_point")
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        state["_last_click"] = (ts, now)

        if click_mode == "inspect":
            pos = int((state["df"]["time"] - ts).abs().to_numpy().argmin())
            selected = _select_value_pos(pos)
            if selected is None:
                return (no_update,) * 7
            row, panel = selected
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                f"값 탐색: {format_kst(row['time'])} · ←/→ 키로 이동",
                _cancel_style(click_mode),
                panel,
                no_update,
            )

        # label_range / label_point: clickData on traces (exact Plotly x).
        if click_mode in ("label_range", "label_point"):
            if _place_click_guarded():
                return (no_update,) * 7
            placed = _place_label_click(ts, click_mode)
            return placed if placed is not None else (no_update,) * 7

        hit = _label_at_time(ts)
        if hit is not None:
            return _select_or_toggle_label_hit(hit, click_mode)

        return (no_update,) * 7

    return (no_update,) * 7


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("metric-filter", "value", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("hover-panel", "children", allow_duplicate=True),
    Input("metric-filter", "value"),
    Input("btn-metrics-all", "n_clicks"),
    Input("btn-metrics-none", "n_clicks"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _metric_filter_changed(selected, _n_all, _n_none, click_mode):
    """Toggle drawn metrics. 전체 선택 → Y 자동; 전체 해제 → Y 유지."""
    if state.get("df") is None:
        return no_update, no_update, no_update, no_update
    all_m = list(state.get("metrics") or [])
    if not all_m:
        return no_update, no_update, no_update, no_update
    tid = getattr(callback_context, "triggered_id", None)
    click_mode = click_mode or "zoom"

    def _bump_metric_rev() -> None:
        state["metric_rev"] = int(state.get("metric_rev") or 0) + 1

    if tid == "btn-metrics-all":
        ranked = _metrics_by_view_sum()
        state["visible_metrics"] = list(ranked)
        _bump_metric_rev()
        _reset_y()
        return (
            _build_graph(click_mode),
            list(ranked),
            f"metric 전체 선택 ({len(ranked)})",
            _refresh_hover_panel(),
        )
    if tid == "btn-metrics-none":
        # Keep the pre-clear Y scale so the empty plot doesn't jump.
        _pin_y_before_x_change()
        state["y_gen"] = int(state.get("y_gen") or 0) + 1
        state["visible_metrics"] = []
        _bump_metric_rev()
        return (
            _build_graph(click_mode),
            [],
            "metric 전체 해제 (Y축 유지)",
            _refresh_hover_panel(),
        )
    new_vis = _visible_from_filter_selection(selected)
    if new_vis == list(state.get("visible_metrics") or []):
        return no_update, no_update, no_update, no_update
    state["visible_metrics"] = new_vis
    _bump_metric_rev()
    _reset_y()
    ranked = _metrics_by_view_sum()
    return (
        _build_graph(click_mode),
        no_update,
        f"표시 metric {len(new_vis)}/{len(ranked)}",
        _refresh_hover_panel(),
    )


@app.callback(
    Output("axis-cmd", "data"),
    Input("btn-pan-left", "n_clicks"),
    Input("btn-pan-right", "n_clicks"),
    Input("btn-y-in", "n_clicks"),
    Input("btn-y-out", "n_clicks"),
    Input("btn-y-auto", "n_clicks"),
    Input("btn-zoom-in", "n_clicks"),
    Input("btn-zoom-in-left", "n_clicks"),
    Input("btn-zoom-out", "n_clicks"),
    Input("btn-reset-zoom", "n_clicks"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _axis_nav(
    n_left,
    n_right,
    n_y_in,
    n_y_out,
    n_y_auto,
    n_zoom_in,
    n_zoom_in_left,
    n_zoom_out,
    n_reset,
    click_mode,
):
    """Instant axis moves: update state only; browser relayouts, then rebuilds."""
    if state["df"] is None:
        return no_update
    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    click_mode = click_mode or "zoom"
    pan_keep_y = click_mode == "pan_keep_y"

    if prop == "btn-pan-left.n_clicks":
        if not _shift_zoom(-1, y_auto=not pan_keep_y):
            return no_update
        return _make_axis_cmd("", touch_y=not pan_keep_y, rebuild_ms=80)
    if prop == "btn-pan-right.n_clicks":
        if not _shift_zoom(1, y_auto=not pan_keep_y):
            return no_update
        return _make_axis_cmd("", touch_y=not pan_keep_y, rebuild_ms=80)
    if prop == "btn-y-in.n_clicks":
        if not _scale_y(0.7):
            return no_update
        # Y-only: don't touch X, don't full-rebuild (rebuild remounts WebGL and
        # can shrink the plot height / shift the series upward).
        return _make_axis_cmd("세로축 확대", apply_x=False, rebuild_ms=-1)
    if prop == "btn-y-out.n_clicks":
        if not _scale_y(1.4):
            return no_update
        return _make_axis_cmd("세로축 축소", apply_x=False, rebuild_ms=-1)
    if prop == "btn-y-auto.n_clicks":
        _reset_y()
        return _make_axis_cmd("세로축 자동 맞춤", apply_x=False, rebuild_ms=-1)
    if prop == "btn-zoom-in.n_clicks":
        if not _scale_x(0.7, y_auto=not pan_keep_y):
            return no_update
        return _make_axis_cmd(
            "시간축 줌인",
            touch_y=not pan_keep_y,
            rebuild_ms=80,
        )
    if prop == "btn-zoom-in-left.n_clicks":
        if not _scale_x_from_left(0.7, y_auto=not pan_keep_y):
            return no_update
        return _make_axis_cmd(
            "시간축 줌인(왼쪽 고정)",
            touch_y=not pan_keep_y,
            rebuild_ms=80,
        )
    if prop == "btn-zoom-out.n_clicks":
        if not _scale_x(1.0 / 0.7, y_auto=not pan_keep_y):
            return no_update
        return _make_axis_cmd(
            "시간축 줌아웃",
            touch_y=not pan_keep_y,
            rebuild_ms=80,
        )
    if prop == "btn-reset-zoom.n_clicks":
        state["label_range_anchor"] = None
        state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
        state["x_full_view"] = True
        _reset_y()
        return _make_axis_cmd("")
    return no_update


@app.callback(
    Output("axis-cmd", "data", allow_duplicate=True),
    Input("graph", "relayoutData"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _drag_axis_nav(relayout, click_mode):
    """On drag-zoom / drag-pan: optional Y 자동, then deferred high-res rebuild."""
    if not relayout or state["df"] is None:
        return no_update
    has_x = (
        ("xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout)
        or "xaxis.range" in relayout
        or relayout.get("xaxis.autorange") is True
    )
    if not has_x:
        return no_update
    click_mode = click_mode or "zoom"
    # 이동(Y고정) only: keep y. Zoom / 이동(Y자동) / others: Y 자동.
    y_auto = click_mode != "pan_keep_y"
    # After select/zoom-to-label, respect zoom_guard so delayed plotly_relayout
    # echoes cannot overwrite the window / clear the red highlight. Outside that
    # window keep ignore_guard so toolbar-zoom guards do not block real drags.
    protect = time.monotonic() < float(state.get("select_guard_until") or 0)
    if not _apply_axes_from_relayout(
        relayout, ignore_guard=not protect, y_auto=y_auto
    ):
        return no_update
    return _make_axis_cmd(
        "",
        apply_x=False,
        touch_y=y_auto,
        guard_seconds=0.5,
        rebuild_ms=80 if click_mode in ("pan", "pan_keep_y") else 300,
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("metric-filter", "options", allow_duplicate=True),
    Output("label-list", "value", allow_duplicate=True),
    Input("rebuild-trigger", "data"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _rebuild_after_axis(trigger, click_mode):
    if not trigger or state["df"] is None:
        return no_update, no_update, no_update
    _arm_zoom_guard(3.0)
    m_opts, _ = _metric_filter_ui()
    # Sync dropdown when zoom-in cleared an off-screen selection — but only if
    # the user has not already re-selected something before this deferred rebuild.
    cleared = state.pop("_offscreen_cleared", False)
    list_val = (
        None if (cleared and state.get("highlight_id") is None) else no_update
    )
    return _build_graph(click_mode or "zoom"), m_opts, list_val


app.clientside_callback(
    """
    function(cmd) {
        if (!cmd) {
            return window.dash_clientside.no_update;
        }
        var hasX = cmd.x && cmd.x.length === 2;
        var hasY = cmd.y && cmd.y.length === 2;
        var touchY = (cmd.touch_y !== false);
        if (!hasX && !touchY) {
            // Still allow rebuild-only commands (pan with Y fixed).
        } else if (!hasX && !hasY && cmd.y !== null && touchY) {
            return window.dash_clientside.no_update;
        }
        var host = document.getElementById('graph');
        var gd = host && host.querySelector('.js-plotly-plot');
        if (!gd || !window.Plotly) {
            return window.dash_clientside.no_update;
        }
        if (hasX) {
            window.__ignoreDataXClampUntil = Date.now() + 4000;
        }
        var patch = {};
        if (hasX) {
            patch['xaxis.autorange'] = false;
            patch['xaxis.range'] = [cmd.x[0], cmd.x[1]];
        }
        if (touchY) {
            if (hasY) {
                patch['yaxis.autorange'] = false;
                patch['yaxis.range'] = [cmd.y[0], cmd.y[1]];
            } else {
                patch['yaxis.autorange'] = true;
            }
        }
        var apply = function() {
            if (window.__rebuildTimer) {
                clearTimeout(window.__rebuildTimer);
                window.__rebuildTimer = null;
            }
            var delay = (cmd.rebuild_ms != null) ? cmd.rebuild_ms : 300;
            // rebuild_ms < 0 → Y-only / no remount (keeps plot height & x-axis put).
            if (delay >= 0) {
                window.__rebuildTimer = setTimeout(function() {
                    window.dash_clientside.set_props('rebuild-trigger', {
                        data: {seq: cmd.seq, t: Date.now()}
                    });
                }, delay);
            }
            if (cmd.status) {
                window.dash_clientside.set_props('status', {children: cmd.status});
            }
        };
        if (Object.keys(patch).length) {
            window.__clampingDataX = true;
            // Keep layout height/domain stable when only Y changes.
            if (!hasX && hasY) {
                patch['yaxis.fixedrange'] = true;
            }
            window.Plotly.relayout(gd, patch).finally(function() {
                window.__clampingDataX = false;
                apply();
            });
        } else {
            apply();
        }
        return cmd.seq;
    }
    """,
    Output("axis-cmd-ack", "data"),
    Input("axis-cmd", "data"),
)


app.clientside_callback(
    """
    function(relayout) {
        var view = window.__readGraphView && window.__readGraphView();
        return view || window.dash_clientside.no_update;
    }
    """,
    Output("view-range", "data"),
    Input("graph", "relayoutData"),
)


# The mode radio keeps focus after being clicked, so the browser would move the
# radio selection on arrow keys. Claiming the event during the capture phase and
# dropping focus keeps the arrows on the value cursor.
app.clientside_callback(
    """
    function(mode) {
        window.__currentClickMode = mode;
        if (mode !== 'label_range') {
            var host0 = document.getElementById('graph');
            var gd0 = host0 && host0.querySelector('.js-plotly-plot');
            if (gd0 && window.__clearPendingRangePreview) {
                window.__clearPendingRangePreview(gd0);
            }
            if (gd0 && window.__scrubStalePendingRangeFill) {
                window.__scrubStalePendingRangeFill(gd0, true);
            }
        }
        window.__valueInspectMode = mode === 'inspect';
        window.__editRangeMode = mode === 'edit_range';
        window.__labelPlaceMode = (mode === 'label_range' || mode === 'label_point');
        // Empty-area time tip (+ spike): place / edit / inspect / pan / zoom.
        window.__showPlaceTimeTip = (
            mode === 'label_range' || mode === 'label_point'
            || mode === 'edit_range' || mode === 'inspect'
            || mode === 'pan' || mode === 'pan_keep_y'
            || mode === 'zoom'
        );
        if (!window.__showPlaceTimeTip) {
            var placeTip = document.getElementById('place-time-tip');
            if (placeTip) placeTip.style.display = 'none';
            var placeSpike = document.getElementById('place-time-spike');
            if (placeSpike) placeSpike.style.display = 'none';
        }
        window.__desiredDragmode = (
            (mode === 'edit_range' || mode === 'label_range'
                || mode === 'label_point' || mode === 'inspect')
                ? false
                : (mode === 'pan' || mode === 'pan_keep_y')
                    ? 'pan' : 'zoom'
        );

        function isTextEntry(node) {
            var tag = ((node && node.tagName) || '').toLowerCase();
            if (tag === 'textarea') return true;
            if (tag === 'input') {
                var type = (node.getAttribute('type') || 'text').toLowerCase();
                return ['radio', 'checkbox', 'button', 'submit'].indexOf(type) === -1;
            }
            return !!(node && node.isContentEditable);
        }

        if (!window.__valueInspectKeyHandler) {
            window.__inspectPendingStep = 0;
            window.__inspectInFlight = false;
            window.__inspectFlushTimer = null;

            window.__flushInspectKeys = function() {
                window.__inspectFlushTimer = null;
                if (!window.__valueInspectMode) {
                    window.__inspectPendingStep = 0;
                    return;
                }
                if (window.__inspectInFlight) {
                    // Wait for the current figure update, then send accumulated steps.
                    if (!window.__inspectFlushTimer) {
                        window.__inspectFlushTimer = setTimeout(window.__flushInspectKeys, 40);
                    }
                    return;
                }
                var delta = window.__inspectPendingStep || 0;
                if (!delta) return;
                window.__inspectPendingStep = 0;
                window.__inspectInFlight = true;
                clearTimeout(window.__inspectInFlightWatchdog);
                // If the server returns no_update (no figure event), unlock after a beat.
                window.__inspectInFlightWatchdog = setTimeout(function() {
                    window.__inspectInFlight = false;
                    if ((window.__inspectPendingStep || 0) !== 0 && window.__flushInspectKeys) {
                        window.__flushInspectKeys();
                    }
                }, 1500);
                window.dash_clientside.set_props('key-event', {
                    data: {
                        key: delta < 0 ? 'ArrowLeft' : 'ArrowRight',
                        steps: Math.abs(delta),
                        sequence: Date.now()
                    }
                });
            };

            window.__valueInspectKeyHandler = function(event) {
                if (isTextEntry(event.target)) return;
                // Esc cancels in-progress range placement (any mode).
                if (event.key === 'Escape') {
                    event.preventDefault();
                    event.stopPropagation();
                    window.dash_clientside.set_props('key-event', {
                        data: {key: 'Escape', sequence: Date.now()}
                    });
                    return;
                }
                if (!window.__valueInspectMode) return;
                if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
                event.preventDefault();
                event.stopPropagation();
                var step = event.key === 'ArrowLeft' ? -1 : 1;
                window.__inspectPendingStep = (window.__inspectPendingStep || 0) + step;
                if (window.__inspectFlushTimer) return;
                // Coalesce key-repeat into one server step burst (no clientside cursor fight).
                window.__inspectFlushTimer = setTimeout(window.__flushInspectKeys, 30);
            };
            window.addEventListener('keydown', window.__valueInspectKeyHandler, true);
        }

        if (!window.__valueInspectMode) {
            window.__inspectPendingStep = 0;
            window.__inspectInFlight = false;
        }

        var active = document.activeElement;
        if (window.__valueInspectMode && active && !isTextEntry(active) && active.blur) {
            active.blur();
        }
        window.dash_clientside.set_props('graph', {
            config: {
                responsive: true,
                displayModeBar: true,
                showTips: false,
                edits: {shapePosition: false}
            }
        });

        if (!document.getElementById('plotly-notifier-hide')) {
            var hideTip = document.createElement('style');
            hideTip.id = 'plotly-notifier-hide';
            hideTip.textContent = '.plotly-notifier{display:none!important;}';
            document.head.appendChild(hideTip);
        }

        if (!document.getElementById('edit-range-pointer-style')) {
            var style = document.createElement('style');
            style.id = 'edit-range-pointer-style';
            document.head.appendChild(style);
        }
        if (!window.__syncEditRangePointerStyle) {
            window.__syncEditRangePointerStyle = function() {
                var style = document.getElementById('edit-range-pointer-style');
                if (!style) return;
                var base = (
                    '#graph .shapelayer path,'
                    + '#graph .shapelayer rect{'
                    + 'pointer-events:none !important;}'
                );
                if (window.__editRangeMode) {
                    style.textContent = base
                        + '#graph .nsewdrag{cursor:default !important;}'
                        + '#graph .nsewdrag.edge-hit{cursor:ew-resize !important;}'
                        + '#graph .outline-controllers{display:none !important;}';
                } else {
                    style.textContent = base;
                }
            };
        }
        window.__syncEditRangePointerStyle();

        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            if (gd && window.Plotly && window.__desiredDragmode !== undefined) {
                // Stable layout.uirevision keeps the old dragmode; force the mode radio.
                window.Plotly.relayout(gd, {dragmode: window.__desiredDragmode});
            }
            if (gd && window.__installCustomEdgeEdit) {
                window.__installCustomEdgeEdit();
            }
            if (gd && mode !== 'edit_range') {
                var drag = gd.querySelector('.nsewdrag');
                if (drag) drag.classList.remove('edge-hit');
                var tip = document.getElementById('edge-drag-tip');
                if (tip) tip.style.display = 'none';
            }
        }, 80);
        return mode;
    }
    """,
    Output("key-listener-state", "data"),
    Input("click-mode", "value"),
)


# A figure drawn before its container settles renders the WebGL traces at the
# stale width while the SVG label overlays use the final geometry, so ranges look
# shifted until the first zoom. Re-measuring right after each update avoids that.
# Server-driven redraws emit no relayout event, so the stored view is refreshed
# here as well; otherwise the next action would replay a stale range.
app.clientside_callback(
    """
    function(figure) {
        if (!document.getElementById('edit-range-pointer-style')) {
            var style = document.createElement('style');
            style.id = 'edit-range-pointer-style';
            document.head.appendChild(style);
        }
        if (!window.__syncEditRangePointerStyle) {
            window.__syncEditRangePointerStyle = function() {
                var style = document.getElementById('edit-range-pointer-style');
                if (!style) return;
                var base = (
                    '#graph .shapelayer path,'
                    + '#graph .shapelayer rect{'
                    + 'pointer-events:none !important;}'
                );
                if (window.__editRangeMode) {
                    style.textContent = base
                        + '#graph .nsewdrag{cursor:default !important;}'
                        + '#graph .nsewdrag.edge-hit{cursor:ew-resize !important;}'
                        + '#graph .outline-controllers{display:none !important;}';
                } else {
                    style.textContent = base;
                }
            };
        }
        window.__syncEditRangePointerStyle();

        if (!window.__plotPlaceUtils) {
            window.__plotPlaceUtils = function(gd) {
                function plotLayer() {
                    return gd.querySelector('.nsewdrag')
                        || gd.querySelector('.xy')
                        || gd;
                }

                function plotAreaRect() {
                    var layer = plotLayer();
                    if (!layer) return null;
                    var bb = layer.getBoundingClientRect();
                    return {
                        left: bb.left, top: bb.top,
                        right: bb.right, bottom: bb.bottom,
                        width: bb.width, height: bb.height
                    };
                }

                function clientXToPlotX(clientX) {
                    var xa = gd._fullLayout && gd._fullLayout.xaxis;
                    if (!xa) return null;
                    var layer = plotLayer();
                    if (!layer) return null;
                    var bb = layer.getBoundingClientRect();
                    var px = Math.max(0, Math.min(bb.width, clientX - bb.left));
                    try {
                        if (typeof xa.p2d === 'function') return xa.p2d(px);
                        if (typeof xa.p2c === 'function' && typeof xa.c2d === 'function') {
                            return xa.c2d(xa.p2c(px));
                        }
                    } catch (err) {}
                    return null;
                }

                function plotXToMs(xVal) {
                    if (xVal == null) return NaN;
                    if (typeof xVal === 'number' && isFinite(xVal)) return xVal;
                    if (xVal instanceof Date) {
                        return Date.UTC(
                            xVal.getUTCFullYear(), xVal.getUTCMonth(), xVal.getUTCDate(),
                            xVal.getUTCHours(), xVal.getUTCMinutes(), xVal.getUTCSeconds()
                        );
                    }
                    var s = String(xVal).replace('T', ' ');
                    var y = +s.slice(0, 4), mo = +s.slice(5, 7), d = +s.slice(8, 10);
                    var h = +s.slice(11, 13), mi = +s.slice(14, 16), sec = +s.slice(17, 19) || 0;
                    if (!(y > 0) || !(mo > 0)) return NaN;
                    return Date.UTC(y, mo - 1, d, h, mi, sec);
                }

                function formatPlotNaive(ms) {
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-'
                        + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
                }

                function nearestSampleMs(ms, samples) {
                    if (!samples || !samples.length || !isFinite(ms)) return null;
                    var lo = 0;
                    var hi = samples.length - 1;
                    if (ms <= samples[0]) return samples[0];
                    if (ms >= samples[hi]) return samples[hi];
                    while (lo < hi) {
                        var mid = (lo + hi) >> 1;
                        if (samples[mid] < ms) lo = mid + 1;
                        else hi = mid;
                    }
                    var a = samples[Math.max(0, lo - 1)];
                    var b = samples[Math.min(samples.length - 1, lo)];
                    return (Math.abs(ms - a) <= Math.abs(ms - b)) ? a : b;
                }

                function snapClientX(clientX) {
                    var xVal = clientXToPlotX(clientX);
                    if (xVal == null) return null;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var samples = meta.sample_ms;
                    var ms = plotXToMs(xVal);
                    if (!isFinite(ms)) return String(xVal);
                    var snapped = (samples && samples.length)
                        ? nearestSampleMs(ms, samples) : null;
                    if (!isFinite(snapped)) {
                        var step = 5 * 60 * 1000;
                        snapped = Math.round(ms / step) * step;
                    }
                    return isFinite(snapped) ? formatPlotNaive(snapped) : String(xVal);
                }

                function inPlotArea(clientX, clientY) {
                    var r = plotAreaRect();
                    if (!r) return false;
                    return clientX >= r.left && clientX <= r.right
                        && clientY >= r.top && clientY <= r.bottom;
                }

                return {
                    clientXToPlotX: clientXToPlotX,
                    snapClientX: snapClientX,
                    plotXToMs: plotXToMs,
                    formatPlotNaive: formatPlotNaive,
                    inPlotArea: inPlotArea
                };
            };
        }

        if (!window.__readGraphView) {
            window.__readGraphView = function() {
                var host = document.getElementById('graph');
                var gd = host && host.querySelector('.js-plotly-plot');
                var layout = gd && gd._fullLayout;
                // The placeholder figure has its own default ranges; recording
                // them would later be replayed onto the real data.
                if (!gd || !gd._fullData || !gd._fullData.length) return null;
                if (!layout || !layout.xaxis || !layout.xaxis.range) return null;
                var view = {
                    x: [String(layout.xaxis.range[0]), String(layout.xaxis.range[1])]
                };
                if (layout.yaxis && layout.yaxis.range) {
                    view.y = [layout.yaxis.range[0], layout.yaxis.range[1]];
                }
                return view;
            };
        }

        // Custom edge edit: ↔ cursor + horizontal drag only on crimson edges.
        if (!window.__installCustomEdgeEdit) {
            window.__installCustomEdgeEdit = function() {
                var host = document.getElementById('graph');
                if (!host || host.__customEdgeEditBound) return;
                host.__customEdgeEditBound = true;
                var HIT_PX = 14;
                var dragState = null;

                function activeGd() {
                    return host.querySelector('.js-plotly-plot');
                }

                function dragEl(gd) {
                    return gd && gd.querySelector('.nsewdrag');
                }

                function setEdgeHit(gd, on) {
                    var el = dragEl(gd);
                    if (!el) return;
                    el.classList.toggle('edge-hit', !!on);
                }

                function ensureTip() {
                    var tip = document.getElementById('edge-drag-tip');
                    if (!tip) {
                        tip = document.createElement('div');
                        tip.id = 'edge-drag-tip';
                        tip.style.cssText = [
                            'position:fixed',
                            'z-index:99999',
                            'pointer-events:none',
                            'display:none',
                            'background:#1a1a1a',
                            'color:#fff',
                            'padding:5px 9px',
                            'border-radius:4px',
                            'font:12px/1.35 sans-serif',
                            'white-space:nowrap',
                            'box-shadow:0 2px 8px rgba(0,0,0,.28)'
                        ].join(';');
                        document.body.appendChild(tip);
                    }
                    return tip;
                }

                function formatPlotNaive(ms) {
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-'
                        + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
                }

                function nearestSampleMs(ms, samples) {
                    if (!samples || !samples.length || !isFinite(ms)) return null;
                    var lo = 0;
                    var hi = samples.length - 1;
                    if (ms <= samples[0]) return samples[0];
                    if (ms >= samples[hi]) return samples[hi];
                    while (lo <= hi) {
                        var mid = (lo + hi) >> 1;
                        if (samples[mid] < ms) lo = mid + 1;
                        else hi = mid - 1;
                    }
                    var a = samples[Math.max(0, lo - 1)];
                    var b = samples[Math.min(samples.length - 1, lo)];
                    return (Math.abs(ms - a) <= Math.abs(ms - b)) ? a : b;
                }

                function toMs(gd, xVal) {
                    if (xVal == null) return NaN;
                    if (typeof xVal === 'number' && isFinite(xVal)) return xVal;
                    var samples = ((gd.layout && gd.layout.meta) || {}).sample_ms;
                    if (xVal instanceof Date) {
                        var utcSlot = Date.UTC(
                            xVal.getUTCFullYear(), xVal.getUTCMonth(), xVal.getUTCDate(),
                            xVal.getUTCHours(), xVal.getUTCMinutes(), xVal.getUTCSeconds()
                        );
                        var localSlot = Date.UTC(
                            xVal.getFullYear(), xVal.getMonth(), xVal.getDate(),
                            xVal.getHours(), xVal.getMinutes(), xVal.getSeconds()
                        );
                        if (samples && samples.length) {
                            var nUtc = nearestSampleMs(utcSlot, samples);
                            var nLoc = nearestSampleMs(localSlot, samples);
                            if (nUtc != null && Math.abs(nUtc - utcSlot) <= Math.abs(nLoc - localSlot)) {
                                return utcSlot;
                            }
                            return localSlot;
                        }
                        return utcSlot;
                    }
                    var s = String(xVal).replace('T', ' ');
                    var y = +s.slice(0, 4);
                    var mo = +s.slice(5, 7);
                    var d = +s.slice(8, 10);
                    var h = +s.slice(11, 13);
                    var mi = +s.slice(14, 16);
                    var sec = +s.slice(17, 19) || 0;
                    if (!(y > 0) || !(mo > 0)) {
                        var t = Date.parse(s);
                        return isNaN(t) ? NaN : t;
                    }
                    return Date.UTC(y, mo - 1, d, h || 0, mi || 0, sec);
                }

                function formatTipTime(gd, xVal) {
                    var ms = toMs(gd, xVal);
                    if (!isFinite(ms)) return '';
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '년 ' + pad(d.getUTCMonth() + 1) + '월 '
                        + pad(d.getUTCDate()) + '일 ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes());
                }

                function showTip(ev, gd, xVal) {
                    var tip = ensureTip();
                    var text = formatTipTime(gd, xVal);
                    if (!text) {
                        tip.style.display = 'none';
                        return;
                    }
                    tip.textContent = text;
                    tip.style.display = 'block';
                    var left = ev.clientX + 14;
                    var top = ev.clientY - 32;
                    var w = tip.offsetWidth || 160;
                    if (left + w > window.innerWidth - 8) {
                        left = ev.clientX - w - 14;
                    }
                    if (top < 8) top = ev.clientY + 18;
                    tip.style.left = left + 'px';
                    tip.style.top = top + 'px';
                }

                function hideTip() {
                    var tip = document.getElementById('edge-drag-tip');
                    if (tip) tip.style.display = 'none';
                }

                function toAxisPx(xa, xVal) {
                    try {
                        var px = xa.d2p(xVal);
                        if (px == null || isNaN(px)) px = xa.d2p(new Date(xVal));
                        return px;
                    } catch (err) {
                        try { return xa.d2p(new Date(xVal)); } catch (err2) { return NaN; }
                    }
                }

                function collectEditEdges(gd) {
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var edges = meta.edit_edges || [];
                    if (edges.length) return edges;
                    var shapes = gd.layout.shapes || [];
                    var out = [];
                    for (var i = 0; i < shapes.length; i++) {
                        var s = shapes[i];
                        if (!s || !s.name) continue;
                        if (s.name === 'label_highlight_start'
                            || s.name === 'label_highlight_end') {
                            out.push({name: s.name, index: i});
                        }
                    }
                    return out;
                }

                function edgeHit(gd, clientX, clientY) {
                    if (!window.__editRangeMode) return null;
                    var fl = gd._fullLayout;
                    var xa = fl && fl.xaxis;
                    var edges = collectEditEdges(gd);
                    if (!xa || !edges.length) return null;
                    var el = dragEl(gd);
                    if (!el) return null;
                    var r = el.getBoundingClientRect();
                    if (clientX < r.left || clientX > r.right ||
                        clientY < r.top || clientY > r.bottom) {
                        return null;
                    }
                    var xPx = clientX - r.left;
                    var shapes = gd.layout.shapes || [];
                    var best = null;
                    var bestDist = HIT_PX + 1;
                    for (var i = 0; i < edges.length; i++) {
                        var info = edges[i];
                        var shape = shapes[info.index];
                        if (!shape) continue;
                        var xVal = shape.x0 != null ? shape.x0 : shape.x1;
                        var edgePx = toAxisPx(xa, xVal);
                        if (edgePx == null || isNaN(edgePx)) continue;
                        var dist = Math.abs(edgePx - xPx);
                        if (dist < bestDist) {
                            bestDist = dist;
                            best = info;
                        }
                    }
                    return bestDist <= HIT_PX ? best : null;
                }

                function snapToSample(gd, xVal) {
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var samples = meta.sample_ms;
                    if (!samples || !samples.length) return xVal;
                    var ms = toMs(gd, xVal);
                    if (!isFinite(ms)) return xVal;
                    var snapped = nearestSampleMs(ms, samples);
                    return snapped == null ? xVal : formatPlotNaive(snapped);
                }

                function xFromClientX(gd, clientX) {
                    var xa = gd._fullLayout.xaxis;
                    var el = dragEl(gd);
                    if (!el) return null;
                    var r = el.getBoundingClientRect();
                    var xPx = Math.max(0, Math.min(r.width, clientX - r.left));
                    return xa.p2d(xPx);
                }

                function sameX(a, b) {
                    return a != null && b != null && String(a) === String(b);
                }

                function isLabelFillRect(s) {
                    if (!s || s.type !== 'rect') return false;
                    var fc = String(s.fillcolor || '');
                    return fc.indexOf('220') >= 0 && fc.indexOf('60') >= 0;
                }

                function resolveLabelShapeIndices(gd) {
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var idxs = meta.edit_label_shapes;
                    if (idxs && idxs.fill != null) return idxs;

                    var shapes = gd.layout.shapes || [];
                    var hiStart = null;
                    var hiEnd = null;
                    var fillIdx = null;
                    var edgeStart = null;
                    var edgeEnd = null;
                    var i;
                    for (i = 0; i < shapes.length; i++) {
                        var nm = shapes[i].name;
                        if (nm === 'label_highlight_start') hiStart = i;
                        if (nm === 'label_highlight_end') hiEnd = i;
                    }
                    for (i = 0; i < shapes.length; i++) {
                        if (isLabelFillRect(shapes[i])) {
                            fillIdx = i;
                            break;
                        }
                    }
                    if (fillIdx != null) {
                        var fill = shapes[fillIdx];
                        var fx0 = fill.x0;
                        var fx1 = fill.x1;
                        for (i = 0; i < shapes.length; i++) {
                            var ln = shapes[i];
                            if (!ln || ln.type !== 'line') continue;
                            var lx = ln.x0;
                            if (sameX(lx, fx0) && i !== hiStart && edgeStart == null) {
                                edgeStart = i;
                            }
                            if (sameX(lx, fx1) && i !== hiEnd && edgeEnd == null) {
                                edgeEnd = i;
                            }
                        }
                    }
                    return {
                        fill: fillIdx,
                        edge_start: edgeStart,
                        edge_end: edgeEnd,
                        hi_start: hiStart,
                        hi_end: hiEnd
                    };
                }

                function formatXForServer(gd, xVal) {
                    if (xVal == null) return null;
                    if (typeof xVal === 'string') return xVal;
                    var ms = toMs(gd, xVal);
                    if (isFinite(ms)) return formatPlotNaive(ms);
                    return String(xVal);
                }

                function buildEdgeDragPatch(gd, dragName, hiIdx, xVal, idxs) {
                    idxs = idxs || resolveLabelShapeIndices(gd);
                    var patch = {};
                    var hi = hiIdx != null ? hiIdx : (
                        dragName === 'label_highlight_start' ? idxs.hi_start : idxs.hi_end
                    );
                    if (hi != null) {
                        patch['shapes[' + hi + '].x0'] = xVal;
                        patch['shapes[' + hi + '].x1'] = xVal;
                        patch['shapes[' + hi + '].y0'] = 0;
                        patch['shapes[' + hi + '].y1'] = 1;
                    }
                    if (dragName === 'label_highlight_start') {
                        if (idxs.fill != null) {
                            patch['shapes[' + idxs.fill + '].x0'] = xVal;
                        }
                        if (idxs.edge_start != null) {
                            patch['shapes[' + idxs.edge_start + '].x0'] = xVal;
                            patch['shapes[' + idxs.edge_start + '].x1'] = xVal;
                        }
                    } else if (dragName === 'label_highlight_end') {
                        if (idxs.fill != null) {
                            patch['shapes[' + idxs.fill + '].x1'] = xVal;
                        }
                        if (idxs.edge_end != null) {
                            patch['shapes[' + idxs.edge_end + '].x0'] = xVal;
                            patch['shapes[' + idxs.edge_end + '].x1'] = xVal;
                        }
                    }
                    return patch;
                }

                function onMove(ev) {
                    var gd = activeGd();
                    if (!gd) return;
                    if (dragState) {
                        ev.preventDefault();
                        ev.stopPropagation();
                        var xVal = snapToSample(gd, xFromClientX(gd, ev.clientX));
                        if (dragState.lastX === xVal) {
                            setEdgeHit(gd, true);
                            showTip(ev, gd, xVal);
                            return;
                        }
                        dragState.lastX = xVal;
                        var patch = buildEdgeDragPatch(
                            gd, dragState.name, dragState.index, xVal, dragState.shapeIdxs
                        );
                        if (window.Plotly) window.Plotly.relayout(gd, patch);
                        setEdgeHit(gd, true);
                        showTip(ev, gd, xVal);
                        return;
                    }
                    if (!window.__editRangeMode) {
                        setEdgeHit(gd, false);
                        hideTip();
                        return;
                    }
                    setEdgeHit(gd, !!edgeHit(gd, ev.clientX, ev.clientY));
                }

                function onDown(ev) {
                    if (!window.__editRangeMode || ev.button !== 0) return;
                    var gd = activeGd();
                    if (!gd) return;
                    var hit = edgeHit(gd, ev.clientX, ev.clientY);
                    if (!hit) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    if (host.setPointerCapture && ev.pointerId != null) {
                        try { host.setPointerCapture(ev.pointerId); } catch (err) {}
                    }
                    window.__edgeDragging = true;
                    var x0 = snapToSample(gd, xFromClientX(gd, ev.clientX));
                    dragState = {
                        index: hit.index,
                        name: hit.name,
                        lastX: x0,
                        gd: gd,
                        shapeIdxs: resolveLabelShapeIndices(gd)
                    };
                    setEdgeHit(gd, true);
                    showTip(ev, gd, x0);
                }

                function onUp(ev) {
                    if (!dragState) return;
                    var gd = dragState.gd || activeGd();
                    var name = dragState.name;
                    var lastX = dragState.lastX;
                    dragState = null;
                    window.__edgeDragging = false;
                    window.__skipNextShapeClick = true;
                    if (host.releasePointerCapture && ev.pointerId != null) {
                        try { host.releasePointerCapture(ev.pointerId); } catch (err) {}
                    }
                    hideTip();
                    if (!gd) return;
                    setEdgeHit(gd, !!edgeHit(gd, ev.clientX, ev.clientY));
                    var xVal = formatXForServer(gd, lastX);
                    if (xVal == null) return;
                    if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                    window.dash_clientside.set_props('edge-drag-event', {
                        data: {
                            name: name,
                            x: xVal,
                            sequence: Date.now()
                        }
                    });
                }

                function onWindowMove(ev) {
                    if (dragState) onMove(ev);
                }

                host.addEventListener('pointermove', onMove, true);
                host.addEventListener('pointerdown', onDown, true);
                host.addEventListener('pointerup', onUp, true);
                host.addEventListener('pointercancel', onUp, true);
                host.addEventListener('mousemove', onMove, true);
                host.addEventListener('mousedown', onDown, true);
                window.addEventListener('pointermove', onWindowMove, true);
                window.addEventListener('mousemove', onWindowMove, true);
                window.addEventListener('mouseup', onUp, true);
                window.addEventListener('pointerup', onUp, true);
            };
        }

        // Clamp pan/zoom so the window never leaves the dataset [tmin, tmax].
        // Only on final relayout (not every drag tick) to avoid fighting the gesture.
        if (!window.__clampPanToData) {
            window.__ms = function(v) {
                if (v == null) return NaN;
                if (typeof v === 'number') return v;
                var t = Date.parse(v);
                return isNaN(t) ? NaN : t;
            };
            window.__clampPanToData = function(gd) {
                if (!gd || gd.__dataXClamp) return;
                gd.__dataXClamp = true;
                gd.on('plotly_relayout', function(eventData) {
                    if (window.__clampingDataX || !eventData) return;
                    // Skip while a server figure is settling (avoids fighting zoom-in).
                    if (window.__ignoreDataXClampUntil &&
                        Date.now() < window.__ignoreDataXClampUntil) {
                        return;
                    }
                    // Never expand a deliberate zoom back via autorange echoes.
                    if (eventData['xaxis.autorange']) return;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var bounds = meta.data_x;
                    if (!bounds || bounds.length < 2) return;
                    var tmin = window.__ms(bounds[0]);
                    var tmax = window.__ms(bounds[1]);
                    if (!(tmax > tmin)) return;

                    var x0 = eventData['xaxis.range[0]'];
                    var x1 = eventData['xaxis.range[1]'];
                    if (x0 === undefined || x1 === undefined) {
                        if (Array.isArray(eventData['xaxis.range'])) {
                            x0 = eventData['xaxis.range'][0];
                            x1 = eventData['xaxis.range'][1];
                        }
                    }
                    if (x0 === undefined || x1 === undefined) return;

                    var a = window.__ms(x0);
                    var b = window.__ms(x1);
                    if (!(b > a)) return;
                    var width = b - a;
                    var full = tmax - tmin;
                    var na = a;
                    var nb = b;
                    if (width >= full) {
                        na = tmin;
                        nb = tmax;
                    } else {
                        if (na < tmin) {
                            na = tmin;
                            nb = tmin + width;
                        }
                        if (nb > tmax) {
                            nb = tmax;
                            na = tmax - width;
                        }
                        if (na < tmin) na = tmin;
                    }
                    if (Math.abs(na - a) < 0.5 && Math.abs(nb - b) < 0.5) return;
                    window.__clampingDataX = true;
                    window.Plotly.relayout(gd, {
                        'xaxis.range[0]': new Date(na),
                        'xaxis.range[1]': new Date(nb)
                    }).finally(function() {
                        window.__clampingDataX = false;
                    });
                });
            };
        }

        // Empty-area clicks → snapped time. Bound on #graph host so figure
        // redraws never drop listeners.
        if (!window.__installPlotClickHost) {
            window.__installPlotClickHost = function() {
                var host = document.getElementById('graph');
                if (!host || host.__plotClickHostBound) return;
                host.__plotClickHostBound = true;
                var lastEmit = {x: null, t: 0};

                host.addEventListener('click', function(ev) {
                    if (ev.button !== 0) return;
                    if (window.__skipNextShapeClick) {
                        window.__skipNextShapeClick = false;
                        return;
                    }
                    if (window.__edgeDragging) return;
                    var mode = window.__currentClickMode;
                    // Trace hits: Plotly clickData has the exact x (label modes).
                    if (mode === 'label_range' || mode === 'label_point') {
                        var t = ev.target;
                        if (t && t.closest && t.closest(
                            '.scatterlayer, .point, .points, .lines'
                        )) {
                            return;
                        }
                    }
                    var gd = host.querySelector('.js-plotly-plot');
                    if (!gd || !gd._fullLayout || !window.__plotPlaceUtils) return;
                    var plot = window.__plotPlaceUtils(gd);
                    if (!plot.inPlotArea(ev.clientX, ev.clientY)) return;
                    var xVal = plot.snapClientX(ev.clientX);
                    if (xVal == null) return;
                    var now = Date.now();
                    if (lastEmit.x === xVal && now - lastEmit.t < 80) return;
                    lastEmit = {x: xVal, t: now};
                    if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                    window.dash_clientside.set_props('shape-click-event', {
                        data: {x: xVal, sequence: now}
                    });
                }, true);
            };
        }

        if (!window.__installPendingRangePreview) {
            window.__installPendingRangePreview = function() {
                var host = document.getElementById('graph');
                if (!host || host.__pendingRangePreviewBound) return;
                host.__pendingRangePreviewBound = true;
                var raf = null;
                var lastX1 = null;

                function pendingFillIndex(gd) {
                    var shapes = (gd.layout && gd.layout.shapes) || [];
                    for (var i = 0; i < shapes.length; i++) {
                        if (shapes[i] && shapes[i].name === 'pending_range_fill') {
                            return i;
                        }
                    }
                    var meta = (gd.layout && gd.layout.meta) || {};
                    return meta.pending_fill_index;
                }

                host.addEventListener('pointermove', function(ev) {
                    if (window.__currentClickMode !== 'label_range') return;
                    var gd = host.querySelector('.js-plotly-plot');
                    if (!gd || !window.Plotly || !window.__plotPlaceUtils) return;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var start = meta.pending_range_start;
                    var idx = pendingFillIndex(gd);
                    if (idx == null || start == null) return;
                    var plot = window.__plotPlaceUtils(gd);
                    if (!plot.inPlotArea(ev.clientX, ev.clientY)) return;
                    var xVal = plot.snapClientX(ev.clientX);
                    if (xVal == null) return;
                    var startMs = plot.plotXToMs(start);
                    var curMs = plot.plotXToMs(xVal);
                    if (!isFinite(startMs) || !isFinite(curMs)) return;

                    var patch = {};
                    if (curMs > startMs) {
                        var x1 = plot.formatPlotNaive(curMs);
                        if (x1 === lastX1) return;
                        lastX1 = x1;
                        patch['shapes[' + idx + '].x0'] = start;
                        patch['shapes[' + idx + '].x1'] = x1;
                        patch['shapes[' + idx + '].fillcolor'] = 'rgba(220, 20, 60, 0.42)';
                    } else {
                        if (lastX1 == null) return;
                        lastX1 = null;
                        patch['shapes[' + idx + '].x1'] = start;
                        patch['shapes[' + idx + '].fillcolor'] = 'rgba(0,0,0,0)';
                    }
                    if (raf) cancelAnimationFrame(raf);
                    raf = requestAnimationFrame(function() {
                        window.Plotly.relayout(gd, patch);
                    });
                }, true);
            };

            if (!window.__clearPendingRangePreview) {
                window.__clearPendingRangePreview = function(gd) {
                    if (!gd || !window.Plotly) return;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var start = meta.pending_range_start;
                    var idx = null;
                    var shapes = gd.layout.shapes || [];
                    for (var i = 0; i < shapes.length; i++) {
                        if (shapes[i] && shapes[i].name === 'pending_range_fill') {
                            idx = i;
                            break;
                        }
                    }
                    if (idx == null) idx = meta.pending_fill_index;
                    if (idx == null || start == null) return;
                    window.Plotly.relayout(gd, {
                        ['shapes[' + idx + '].x1']: start,
                        ['shapes[' + idx + '].fillcolor']: 'rgba(0,0,0,0)'
                    });
                };
            }

            if (!window.__scrubStalePendingRangeFill) {
                window.__scrubStalePendingRangeFill = function(gd, force) {
                    if (!gd || !gd.layout || !window.Plotly) return;
                    // Never scrub while the user is placing a 2-click range.
                    if (!force && window.__currentClickMode === 'label_range') {
                        return;
                    }
                    var meta = (gd.layout && gd.layout.meta) || {};
                    if (!force && (meta.pending_fill_index != null
                        || meta.pending_range_start != null)) {
                        return;
                    }
                    var shapes = gd.layout.shapes;
                    if (!shapes || !shapes.length) return;
                    var patch = {};
                    for (var i = 0; i < shapes.length; i++) {
                        var s = shapes[i];
                        if (!s) continue;
                        if (s.name === 'pending_range_fill') {
                            patch['shapes[' + i + '].fillcolor'] = 'rgba(0,0,0,0)';
                            if (s.x0 != null) patch['shapes[' + i + '].x1'] = s.x0;
                        } else if (s.name === 'pending_range_anchor') {
                            patch['shapes[' + i + '].line.color'] = 'rgba(0,0,0,0)';
                            patch['shapes[' + i + '].line.width'] = 0;
                        }
                    }
                    if (Object.keys(patch).length) {
                        window.Plotly.relayout(gd, patch);
                    }
                };
            }

            // Empty-area time tooltip + royalblue dotted spike (match Plotly spikes).
            // Match Plotly hoverlabel look + %{x|%Y년 %m월 %d일 %H:%M} + 5-min snap.
            window.__installPlaceTimeTip = function(gd) {
                if (!gd || gd.__placeTimeTipBound) return;
                gd.__placeTimeTipBound = true;
                var onTrace = false;
                var raf = null;
                var lastText = '';
                var lastSpikeX = null;

                function ensureTip() {
                    var tip = document.getElementById('place-time-tip');
                    if (!tip) {
                        tip = document.createElement('div');
                        tip.id = 'place-time-tip';
                        // Mirror fig.layout.hoverlabel (white / monospace / 11px).
                        tip.style.cssText = [
                            'position:fixed',
                            'z-index:99999',
                            'pointer-events:none',
                            'display:none',
                            'background:#fff',
                            'color:#444',
                            'padding:6px 8px',
                            'border:1px solid #bbb',
                            'border-radius:2px',
                            'font:11px/1.4 monospace',
                            'white-space:nowrap',
                            'box-shadow:0 1px 3px rgba(0,0,0,.18)'
                        ].join(';');
                        document.body.appendChild(tip);
                    }
                    return tip;
                }

                function ensureSpike() {
                    var spike = document.getElementById('place-time-spike');
                    if (!spike) {
                        spike = document.createElement('div');
                        spike.id = 'place-time-spike';
                        // Match layout.xaxis spikedash/spikecolor/spikethickness.
                        spike.style.cssText = [
                            'position:fixed',
                            'z-index:99998',
                            'pointer-events:none',
                            'display:none',
                            'width:0',
                            'border-left:1px dotted royalblue',
                            'box-sizing:border-box'
                        ].join(';');
                        document.body.appendChild(spike);
                    }
                    return spike;
                }

                function hideTip() {
                    var tip = document.getElementById('place-time-tip');
                    if (tip) tip.style.display = 'none';
                    var spike = document.getElementById('place-time-spike');
                    if (spike) spike.style.display = 'none';
                    lastText = '';
                    lastSpikeX = null;
                }

                function formatPlotNaive(ms) {
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-'
                        + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
                }

                function toAxisPx(xa, xVal) {
                    try {
                        var px = xa.d2p(xVal);
                        if (px == null || isNaN(px)) px = xa.d2p(new Date(xVal));
                        return px;
                    } catch (err) {
                        try { return xa.d2p(new Date(xVal)); } catch (err2) { return NaN; }
                    }
                }

                function showSpikeAtMs(ms, plotRect) {
                    var xa = gd._fullLayout && gd._fullLayout.xaxis;
                    if (!xa || !plotRect) return;
                    var px = toAxisPx(xa, formatPlotNaive(ms));
                    if (px == null || isNaN(px)) return;
                    var left = plotRect.left + px;
                    if (left < plotRect.left || left > plotRect.right) return;
                    if (lastSpikeX === left) {
                        var spike0 = ensureSpike();
                        if (spike0.style.display === 'block') return;
                    }
                    lastSpikeX = left;
                    var spike = ensureSpike();
                    spike.style.left = left + 'px';
                    spike.style.top = plotRect.top + 'px';
                    spike.style.height = Math.max(0, plotRect.height) + 'px';
                    spike.style.display = 'block';
                }

                function nearestSampleMs(ms, samples) {
                    if (!samples || !samples.length || !isFinite(ms)) return NaN;
                    var lo = 0, hi = samples.length - 1;
                    if (ms <= samples[0]) return samples[0];
                    if (ms >= samples[hi]) return samples[hi];
                    while (lo <= hi) {
                        var mid = (lo + hi) >> 1;
                        if (samples[mid] < ms) lo = mid + 1;
                        else hi = mid - 1;
                    }
                    var a = samples[Math.max(0, lo - 1)];
                    var b = samples[Math.min(samples.length - 1, lo)];
                    return (Math.abs(ms - a) <= Math.abs(ms - b)) ? a : b;
                }

                // Fallback when sample_ms missing: round to 5-minute wall clock.
                function snapFiveMinMs(ms) {
                    if (!isFinite(ms)) return NaN;
                    var step = 5 * 60 * 1000;
                    return Math.round(ms / step) * step;
                }

                function plotXToMs(xVal) {
                    if (xVal == null) return NaN;
                    if (typeof xVal === 'number' && isFinite(xVal)) return xVal;
                    if (xVal instanceof Date) {
                        return Date.UTC(
                            xVal.getUTCFullYear(), xVal.getUTCMonth(), xVal.getUTCDate(),
                            xVal.getUTCHours(), xVal.getUTCMinutes(), xVal.getUTCSeconds()
                        );
                    }
                    var s = String(xVal).replace('T', ' ');
                    var y = +s.slice(0, 4), mo = +s.slice(5, 7), d = +s.slice(8, 10);
                    var h = +s.slice(11, 13), mi = +s.slice(14, 16), sec = +s.slice(17, 19) || 0;
                    if (!(y > 0) || !(mo > 0)) return NaN;
                    return Date.UTC(y, mo - 1, d, h, mi, sec);
                }

                // Same as Plotly hoverformat / hovertemplate %{x|%Y년 %m월 %d일 %H:%M}
                function formatHoverTime(ms) {
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '년 ' + pad(d.getUTCMonth() + 1) + '월 '
                        + pad(d.getUTCDate()) + '일 ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes());
                }

                function clientXToPlotX(clientX) {
                    var full = gd._fullLayout;
                    var xa = full && full.xaxis;
                    if (!xa) return null;
                    var layer = gd.querySelector('.nsewdrag') || gd;
                    var bb = layer.getBoundingClientRect();
                    var px = clientX - bb.left;
                    if (px < 0 || px > bb.width) return null;
                    try {
                        if (typeof xa.p2d === 'function') return xa.p2d(px);
                        if (typeof xa.p2c === 'function' && typeof xa.c2d === 'function') {
                            return xa.c2d(xa.p2c(px));
                        }
                    } catch (err) {}
                    return null;
                }

                gd.on('plotly_hover', function() {
                    onTrace = true;
                    // Hide tip text only — keep the single custom spike.
                    var tip = document.getElementById('place-time-tip');
                    if (tip) tip.style.display = 'none';
                    lastText = '';
                });
                gd.on('plotly_unhover', function() {
                    onTrace = false;
                });

                gd.addEventListener('pointermove', function(ev) {
                    if (!window.__showPlaceTimeTip) {
                        hideTip();
                        return;
                    }
                    // Hide while dragging (box-zoom / pan / edge edit).
                    if (ev.buttons || window.__edgeDragging) {
                        hideTip();
                        return;
                    }
                    var layer = gd.querySelector('.nsewdrag') || gd;
                    var bb = layer.getBoundingClientRect();
                    if (ev.clientX < bb.left || ev.clientX > bb.right ||
                        ev.clientY < bb.top || ev.clientY > bb.bottom) {
                        hideTip();
                        return;
                    }
                    var xVal = clientXToPlotX(ev.clientX);
                    if (xVal == null) {
                        hideTip();
                        return;
                    }
                    var ms = plotXToMs(xVal);
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var snapped = nearestSampleMs(ms, meta.sample_ms);
                    if (!isFinite(snapped)) snapped = snapFiveMinMs(ms);
                    if (!isFinite(snapped)) {
                        hideTip();
                        return;
                    }
                    // Always one custom spike at the snapped time (Plotly spikes off).
                    showSpikeAtMs(snapped, bb);
                    // On a series, Plotly hoverlabel already shows the time.
                    if (onTrace && !window.__editRangeMode) {
                        return;
                    }
                    var text = formatHoverTime(snapped);
                    if (text === lastText && document.getElementById('place-time-tip')
                        && document.getElementById('place-time-tip').style.display === 'block') {
                        var tip0 = document.getElementById('place-time-tip');
                        tip0.style.left = (ev.clientX + 14) + 'px';
                        tip0.style.top = Math.max(8, ev.clientY - 32) + 'px';
                        return;
                    }
                    lastText = text;
                    if (raf) cancelAnimationFrame(raf);
                    raf = requestAnimationFrame(function() {
                        var tip = ensureTip();
                        tip.textContent = text;
                        tip.style.display = 'block';
                        var left = ev.clientX + 14;
                        var top = ev.clientY - 32;
                        var w = tip.offsetWidth || 160;
                        if (left + w > window.innerWidth - 8) left = ev.clientX - w - 14;
                        if (top < 8) top = ev.clientY + 18;
                        tip.style.left = left + 'px';
                        tip.style.top = top + 'px';
                    });
                }, true);

                gd.addEventListener('pointerleave', hideTip, true);
            };
        }

        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            window.__ignoreDataXClampUntil = Date.now() + 1500;
            if (!gd || !window.Plotly) return;

            function afterGraphReady() {
                window.__clampPanToData(gd);
                window.__installPlotClickHost();
                window.__installCustomEdgeEdit();
                if (window.__installPendingRangePreview) {
                    window.__installPendingRangePreview();
                }
                if (window.__installPlaceTimeTip) {
                    window.__installPlaceTimeTip(gd);
                }
                if (window.__scrubStalePendingRangeFill) {
                    var forceScrub = window.__currentClickMode !== 'label_range';
                    window.__scrubStalePendingRangeFill(gd, forceScrub);
                }
                var drag = gd.querySelector('.nsewdrag');
                if (drag && !window.__editRangeMode) {
                    drag.classList.remove('edge-hit');
                }
                if (window.__desiredDragmode !== undefined) {
                    var cur = gd._fullLayout && gd._fullLayout.dragmode;
                    if (cur !== window.__desiredDragmode) {
                        window.Plotly.relayout(gd, {dragmode: window.__desiredDragmode});
                    }
                }
                window.__inspectInFlight = false;
                clearTimeout(window.__inspectInFlightWatchdog);
                if (window.__valueInspectMode && (window.__inspectPendingStep || 0) !== 0) {
                    if (!window.__inspectFlushTimer && window.__flushInspectKeys) {
                        window.__inspectFlushTimer = setTimeout(window.__flushInspectKeys, 0);
                    }
                }
            }

            // Force full layout sync so edge-drag relayout patches never stick
            // after the server sends the updated label shapes.
            if (figure && figure.data && figure.layout && !window.__edgeDragging) {
                var cfg = gd.config || {responsive: true, displayModeBar: true};
                var hostH = host.clientHeight || 420;
                var wantH = Math.max(hostH > 40 ? hostH : 420, 260);
                var layout = Object.assign({}, figure.layout, {
                    autosize: true,
                    height: wantH
                });
                window.Plotly.react(gd, figure.data, layout, cfg).then(function() {
                    try { window.Plotly.Plots.resize(gd); } catch (err) {}
                    afterGraphReady();
                });
                return;
            }

            var hostH = host.clientHeight || 420;
            var wantH = Math.max(hostH > 40 ? hostH : 420, 260);
            var curH = (gd.layout && gd.layout.height) || 0;
            var patch = {autosize: true};
            if (Math.abs(curH - wantH) > 1) {
                patch.height = wantH;
            }
            window.Plotly.relayout(gd, patch).then(function() {
                try { window.Plotly.Plots.resize(gd); } catch (err) {}
                afterGraphReady();
            });
        }, 80);
        return window.dash_clientside.no_update;
    }
    """,
    Output("graph", "className"),
    Input("graph", "figure"),
)


def main():
    url = f"http://{HOST}:{PORT}/"
    print(f"순위 로드: {len(rank_df)}개 PLMN")
    print(f"라벨링 앱 실행 중 → {url}")
    print("종료: Ctrl+C")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
