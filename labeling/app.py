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

전처리는 labeling/labels/*_labels.json을 수정하지 않습니다.
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
    display_plmn,
    format_kst,
    freeze_shape_editing,
    label_highlight_overlays,
    label_line,
    load_labels,
    load_or_build_ranking,
    load_plmn,
    metric_columns,
    parse_time,
    pending_anchor_annotation,
    pending_anchor_shape,
    ranked_hover_html,
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
    "zoom_start": None,
    "zoom_end": None,
    "y_min": None,
    "y_max": None,
    "y_auto": True,
    "y_gen": 0,
    "y_rendered": None,
    "zoom_guard_until": 0.0,
    "zoom_rendered": None,
    "zoom_rendered_prev": None,
    "label_range_anchor": None,
    "highlight_id": None,
    "highlight_shape_idxs": None,
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
        height=336,
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


def _zoom_to_label_item(item: dict) -> None:
    """Zoom X around a label (tight fit, then zoom-out ×7) and highlight it."""
    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    span = end_ts - start_ts
    pad = span / 2 if span > pd.Timedelta(0) else pd.Timedelta(hours=6)
    tight_start, tight_end = start_ts - pad, end_ts + pad
    tight_width = tight_end - tight_start
    mid = tight_start + tight_width / 2
    wide = tight_width * ((1.0 / 0.7) ** 7)
    half = wide / 2
    state["zoom_start"], state["zoom_end"] = _clamp_zoom(mid - half, mid + half)
    state["highlight_id"] = item["id"]
    _reset_y()
    _arm_zoom_guard()


def _cancel_style(click_mode: str):
    if click_mode == "label_range" and state["label_range_anchor"] is not None:
        return {"display": "inline-block"}
    return {"display": "none"}


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
        metrics=state["metrics"],
        start=state["zoom_start"],
        end=state["zoom_end"],
        title=title,
        hover_values=False,
        max_points=MAX_POINTS,
        show_labels=bool(state.get("show_anomalies", True)),
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
    # Keep layout.uirevision stable (PLMN). Bumping it every zoom remounts every
    # Scattergl trace and freezes the UI; datarevision handles data refresh.
    fig.update_layout(
        datarevision=f"{fig.layout.datarevision}:{state['fig_gen']}",
        uirevision=str(state.get("plmn") or "labeling"),
    )
    # Label fills must never be draggable as a whole — only edge lines in edit mode.
    freeze_shape_editing(fig)

    fig.update_layout(
        dragmode=(
            False
            if click_mode == "edit_range"
            else (
                "pan"
                if click_mode in ("pan", "pan_keep_y", "inspect")
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

    if state["label_range_anchor"] is not None:
        fig.add_shape(**pending_anchor_shape(state["label_range_anchor"]))
        fig.add_annotation(**pending_anchor_annotation(state["label_range_anchor"]))

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
        # Spikes look like editable guides; keep them off while adjusting edges.
        showspikes=(click_mode != "edit_range"),
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
    if click_mode == "edit_range":
        # Nearest-sample snap while dragging edges (irregular gaps included).
        samples = state.get("sample_ms")
        if samples is None and state["df"] is not None and len(state["df"]):
            samples = _sample_plot_ms(state["df"])
            state["sample_ms"] = samples
        if samples:
            meta["sample_ms"] = samples
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
    fig.update_layout(meta=meta)

    if state.get("value_cursor_pos") is not None:
        row = state["df"].iloc[state["value_cursor_pos"]]
        shape, note = value_cursor_overlays(row["time"])
        fig.add_shape(**shape)
        fig.add_annotation(**note)

    return fig


def _snap_to_data_time(ts):
    df = state["df"]
    if df is None or not len(df):
        return ts
    pos = int((df["time"] - ts).abs().to_numpy().argmin())
    return df.iloc[pos]["time"]


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
    item = _label_by_id(label_id)
    when = f"{format_kst(item['start'])} → {format_kst(item['end'])}"
    return f"구간 조절: {when} — Save Labels로 저장"


def _clamp_zoom(start_ts, end_ts):
    tmin, tmax = data_time_bounds(state["df"])
    return clamp_time_range(start_ts, end_ts, tmin, tmax)


def _visible_y_bounds() -> tuple[float, float] | None:
    df = state["df"]
    metrics = state["metrics"]
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
    """Scale the y window against the zero baseline, keeping it at the bottom."""
    if state["df"] is None:
        return False
    bounds = _effective_y_bounds()
    if bounds is None:
        return False
    lo, hi = bounds
    top = lo + (hi - lo) * factor
    if top <= lo:
        return False
    state["y_min"], state["y_max"] = lo, top
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
    panel = dcc.Markdown(
        ranked_hover_html(row["time"], row, state["metrics"]),
        dangerously_allow_html=True,
    )
    return row, panel


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
    # If already at the data bounds and zooming out, nothing changed.
    if factor > 1 and new_start == tmin and new_end == tmax:
        if start_ts == tmin and end_ts == tmax:
            return False
    if new_start == start_ts and new_end == end_ts:
        return False
    if not y_auto:
        _pin_y_before_x_change()
    state["zoom_start"], state["zoom_end"] = new_start, new_end
    if y_auto:
        _reset_y()
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


def _arm_zoom_guard(seconds: float = 2.0) -> None:
    """Ignore browser axis echoes after a server-driven zoom/pan (prevents freeze loops)."""
    state["zoom_guard_until"] = time.monotonic() + seconds


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
            # Full reset is only via the 전체 button — ignore Plotly echoes.
            return False
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
            state["zoom_start"], state["zoom_end"] = incoming


def _same_range(a: tuple[float, float], b: tuple[float, float]) -> bool:
    span = max(abs(a[1] - a[0]), 1e-9)
    return abs(a[0] - b[0]) / span < 1e-3 and abs(a[1] - b[1]) / span < 1e-3


def _do_load(plmn: str, click_mode: str):
    rank = int(rank_df.loc[rank_df["PLMN"] == plmn, "rank"].iloc[0])
    df = load_plmn(plmn)
    metrics = metric_columns(df)
    doc = load_labels(plmn, rank=rank)
    tmin, tmax = data_time_bounds(df)
    state.update(
        df=df,
        doc=doc,
        plmn=plmn,
        rank=rank,
        metrics=metrics,
        zoom_start=tmin,
        zoom_end=tmax,
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
    return (
        _build_graph(click_mode),
        opts,
        opts[0]["value"] if opts else None,
        "",
        _cancel_style(click_mode),
        html.I("그래프에 커서를 올리면 이 시점의 특성값이 내림차순으로 표시됩니다."),
        plmn,
    )


# Prefetch rank #1 so the first paint already shows data (no empty "선택하세요" flash).
_START_PLMN = plmn_ids[0] if plmn_ids else None
_START_LABEL_OPTS: list = []
_START_LABEL_VALUE = None
_START_FIGURE = None
if _START_PLMN:
    _do_load(_START_PLMN, "zoom")
    _START_FIGURE = _build_graph("zoom")
    _START_LABEL_OPTS = _label_options()
    _START_LABEL_VALUE = (
        _START_LABEL_OPTS[0]["value"] if _START_LABEL_OPTS else None
    )
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
        html.Div(
            [
                html.H3("2) 그래프", style={"margin": "0"}),
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
        dcc.Graph(
            id="graph",
            figure=_START_FIGURE,
            config={
                "responsive": True,
                "displayModeBar": True,
            },
            style={"width": "100%", "height": "360px"},
        ),
        html.Div(id="status", style={"minHeight": "1.4em", "margin": "4px 0"}),
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
        html.H3("3) 라벨 추가 / 수정", style={"margin": "10px 0 4px"}),
        html.Div(
            "anomaly 라벨: 구간은 시작·끝 2클릭, 점은 시점 1클릭으로 추가하세요. "
            "목록에서 선택 후 구간 편집으로 경계를 조절할 수 있습니다.",
            style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"},
        ),
        html.Div(
            [
                dcc.Input(
                    id="note",
                    type="text",
                    placeholder="Note",
                    style={"width": "420px"},
                ),
                html.Button("Save Labels", id="btn-save", n_clicks=0),
                html.Button("Reload Saved", id="btn-reload", n_clicks=0),
            ],
            style={
                "display": "flex",
                "gap": "8px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Button("라벨 선택해제", id="btn-clear-selection", n_clicks=0),
                html.Button("선택 라벨로 줌", id="btn-zoom-selected", n_clicks=0),
                html.Button("선택 라벨 삭제", id="btn-delete", n_clicks=0),
            ],
            style={"display": "flex", "gap": "8px", "marginTop": "8px"},
        ),
        html.Div(
            [
                html.B("저장된 라벨 (한 줄 = 라벨 1개)"),
                html.Span(
                    " — 한 줄을 클릭하면 그래프에서 노란색으로 강조됩니다.",
                    style={"fontSize": "12px", "color": "#666"},
                ),
            ],
            style={"marginTop": "8px"},
        ),
        dcc.RadioItems(
            id="label-list",
            options=_START_LABEL_OPTS,
            value=_START_LABEL_VALUE,
        ),
        dcc.Store(id="key-event"),
        dcc.Store(id="key-listener-state"),
        dcc.Store(id="view-range"),
        dcc.Store(id="edge-drag-event"),
        dcc.Store(id="shape-click-event"),
        dcc.Store(id="axis-cmd"),
        dcc.Store(id="axis-cmd-ack"),
        dcc.Store(id="rebuild-trigger"),
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
    Input("dd-plmn", "value"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("click-mode", "value"),
    Input("anomaly-overlay-mode", "value"),
    Input("btn-cancel-range", "n_clicks"),
    Input("btn-save", "n_clicks"),
    Input("btn-reload", "n_clicks"),
    Input("btn-delete", "n_clicks"),
    Input("btn-clear-selection", "n_clicks"),
    Input("btn-zoom-selected", "n_clicks"),
    Input("label-list", "value"),
    Input("key-event", "data"),
    Input("edge-drag-event", "data"),
    Input("shape-click-event", "data"),
    Input("graph", "clickData"),
    Input("graph", "hoverData"),
    Input("graph", "relayoutData"),
    State("note", "value"),
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
    n_save,
    n_reload,
    n_delete,
    n_clear_selection,
    n_zoom_sel,
    selected_label,
    key_event,
    edge_drag,
    shape_click,
    click_data,
    hover_data,
    relayout,
    note,
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
            return (
                _build_graph(click_mode),
                opts,
                no_update,
                "",
                _cancel_style(click_mode),
                no_update,
                no_update,
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

    # Axis buttons live in `_axis_nav`. Avoid replaying a stale browser view.
    _skip_view_sync = {
        "btn-zoom-selected.n_clicks",
        "shape-click-event.data",
        "graph.clickData",
    }
    if prop != "graph.relayoutData" and prop not in _skip_view_sync:
        _sync_zoom_from_view(view_range)

    # ----- click mode -----
    if prop == "click-mode.value":
        state["label_range_anchor"] = None
        if click_mode != "inspect":
            state["value_cursor_pos"] = None
        if selected_label:
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
            status = "① 시작점 클릭 → ② 끝점 클릭으로 구간 라벨 추가"
        elif click_mode == "label_point":
            status = "그래프에서 시점을 한 번 클릭하면 점(세로선) 라벨이 추가됩니다."
        elif click_mode == "edit_range":
            status = (
                "빨간 경계선 위에서만 ↔ 커서가 됩니다. 선을 좌우로 드래그하세요. "
                "화면 이동은 ◀ ▶ 를 사용하세요."
                if state.get("highlight_id")
                else "편집할 라벨을 목록에서 선택한 뒤 좌·우 경계선을 드래그하세요."
            )
        else:
            status = "① 시작점 클릭 → ② 끝점 클릭으로 구간 라벨 추가"
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
        state["label_range_anchor"] = None
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            "구간 시작점 취소됨",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "btn-save.n_clicks":
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
    if prop == "btn-reload.n_clicks":
        state["doc"] = load_labels(state["plmn"], rank=state["rank"])
        state["highlight_id"] = None
        state["label_range_anchor"] = None
        opts = _label_options()
        return (
            _build_graph(click_mode),
            opts,
            opts[0]["value"] if opts else None,
            "라벨 파일을 다시 불러왔습니다.",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )
    if prop == "btn-delete.n_clicks":
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

    if prop == "btn-clear-selection.n_clicks":
        state["highlight_id"] = None
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
        return (
            _build_graph(click_mode),
            _label_options(),
            item["id"],
            "",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "label-list.value":
        state["highlight_id"] = selected_label
        if selected_label and click_mode == "edit_range":
            status = "빨간 좌·우 경계선만 좌우로 드래그하세요. 화면 이동은 ◀ ▶ 버튼을 사용하세요."
        else:
            status = ""
        return (
            _build_graph(click_mode),
            no_update,
            selected_label,
            status,
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "key-event.data":
        if click_mode != "inspect" or not key_event:
            return (no_update,) * 7
        current = state.get("value_cursor_pos")
        if current is None:
            current = state.get("hover_pos")
        if current is None:
            return (no_update,) * 7
        step = -1 if key_event.get("key") == "ArrowLeft" else 1
        selected = _select_value_pos(current + step)
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
        html_str = ranked_hover_html(row["time"], row, state["metrics"])
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            dcc.Markdown(html_str, dangerously_allow_html=True),
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
        # Click on plot area near an anomaly divider (shapes are not clickable).
        if state.get("df") is None or state.get("doc") is None:
            return (no_update,) * 7
        if click_mode == "edit_range":
            return (no_update,) * 7
        selecting_range_end = (
            click_mode == "label_range" and state.get("label_range_anchor") is not None
        )
        if selecting_range_end:
            return (no_update,) * 7
        try:
            ts = parse_time(shape_click["x"])
        except (ValueError, TypeError):
            return (no_update,) * 7
        last_ts, last_at = state.get("_last_click", (None, 0.0))
        now = time.monotonic()
        if last_ts is not None and abs((ts - last_ts).total_seconds()) < 1 and now - last_at < 0.6:
            return (no_update,) * 7
        hit = _label_at_time(ts)
        if hit is None:
            return (no_update,) * 7
        state["_last_click"] = (ts, now)
        state["label_range_anchor"] = None
        _zoom_to_label_item(hit)
        kind = (hit.get("kind") or "point").lower()
        tag = "점" if kind == "point" else "구간"
        return (
            _build_graph(click_mode),
            _label_options(),
            hit["id"],
            f"선택·줌 ({tag}): {label_line(hit)}",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

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
        if last_ts == ts and now - last_at < 0.6:
            return (no_update,) * 7
        state["_last_click"] = (ts, now)

        if click_mode == "inspect":
            pos = int((state["df"]["time"] - ts).abs().to_numpy().argmin())
            row, panel = _select_value_pos(pos)
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                f"값 탐색: {format_kst(row['time'])} · ←/→ 키로 이동",
                _cancel_style(click_mode),
                panel,
                no_update,
            )

        # Prefer selecting an existing red label when the click lands on it
        # (except while placing the 2nd click of a new range).
        selecting_range_end = (
            click_mode == "label_range" and state.get("label_range_anchor") is not None
        )
        if not selecting_range_end:
            hit = _label_at_time(ts)
            if hit is not None:
                state["label_range_anchor"] = None
                _zoom_to_label_item(hit)
                kind = (hit.get("kind") or "point").lower()
                tag = "점" if kind == "point" else "구간"
                return (
                    _build_graph(click_mode),
                    _label_options(),
                    hit["id"],
                    f"선택·줌 ({tag}): {label_line(hit)}",
                    _cancel_style(click_mode),
                    no_update,
                    no_update,
                )

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
                note=(note or "").strip(),
            )
            after = [x for x in state["doc"]["labels"] if x["id"] not in before]
            lid = after[0]["id"] if after else None
            state["highlight_id"] = lid
            opts = _label_options()
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
            return (no_update,) * 7

        anchor = state["label_range_anchor"]
        if anchor is None:
            state["label_range_anchor"] = _snap_to_data_time(ts)
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                f"① 구간 시작: {format_kst(state['label_range_anchor'])} → ② 끝점을 클릭하세요",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )

        ts = _snap_to_data_time(ts)
        a, b = (anchor, ts) if anchor <= ts else (ts, anchor)
        state["label_range_anchor"] = None
        before = {x["id"] for x in state["doc"].get("labels", [])}
        add_label(
            state["doc"],
            kind="range",
            tag="anomaly",
            start=a,
            end=b,
            metrics=["ALL"],
            note=(note or "").strip(),
        )
        after = [x for x in state["doc"]["labels"] if x["id"] not in before]
        lid = after[0]["id"] if after else None
        state["highlight_id"] = lid
        opts = _label_options()
        when = f"{format_kst(a)} → {format_kst(b)}"
        return (
            _build_graph(click_mode),
            opts,
            lid,
            html.Span(
                f"✔ [구간] anomaly 추가됨 ({lid}) {when} — Save Labels로 저장",
                style={"color": "green"},
            ),
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    return (no_update,) * 7


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
        return _make_axis_cmd("세로축 확대")
    if prop == "btn-y-out.n_clicks":
        if not _scale_y(1.4):
            return no_update
        return _make_axis_cmd("세로축 축소")
    if prop == "btn-y-auto.n_clicks":
        _reset_y()
        return _make_axis_cmd("세로축 자동 맞춤")
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
    )
    if not has_x:
        return no_update
    click_mode = click_mode or "zoom"
    # 이동(Y고정) only: keep y. Zoom / 이동(Y자동) / others: Y 자동.
    y_auto = click_mode != "pan_keep_y"
    if not _apply_axes_from_relayout(relayout, ignore_guard=True, y_auto=y_auto):
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
    Input("rebuild-trigger", "data"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _rebuild_after_axis(trigger, click_mode):
    if not trigger or state["df"] is None:
        return no_update
    _arm_zoom_guard(3.0)
    return _build_graph(click_mode or "zoom")


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
            }
            var delay = (cmd.rebuild_ms != null) ? cmd.rebuild_ms : 300;
            window.__rebuildTimer = setTimeout(function() {
                window.dash_clientside.set_props('rebuild-trigger', {
                    data: {seq: cmd.seq, t: Date.now()}
                });
            }, delay);
            if (cmd.status) {
                window.dash_clientside.set_props('status', {children: cmd.status});
            }
        };
        if (Object.keys(patch).length) {
            window.__clampingDataX = true;
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
        window.__valueInspectMode = mode === 'inspect';
        window.__editRangeMode = mode === 'edit_range';
        window.__desiredDragmode = (
            mode === 'edit_range' ? false
            : (mode === 'pan' || mode === 'pan_keep_y' || mode === 'inspect')
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
            window.__valueInspectKeyHandler = function(event) {
                if (!window.__valueInspectMode) return;
                if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
                if (isTextEntry(event.target)) return;
                event.preventDefault();
                event.stopPropagation();
                window.dash_clientside.set_props('key-event', {
                    data: {key: event.key, sequence: Date.now()}
                });
            };
            window.addEventListener('keydown', window.__valueInspectKeyHandler, true);
        }

        var active = document.activeElement;
        if (window.__valueInspectMode && active && !isTextEntry(active) && active.blur) {
            active.blur();
        }
        window.dash_clientside.set_props('graph', {
            config: {
                responsive: true,
                displayModeBar: true,
                edits: {shapePosition: false}
            }
        });

        if (!document.getElementById('edit-range-pointer-style')) {
            var style = document.createElement('style');
            style.id = 'edit-range-pointer-style';
            document.head.appendChild(style);
        }
        document.getElementById('edit-range-pointer-style').textContent =
            // Label fills/lines must not steal clicks — selection uses clickData x.
            '#graph .shapelayer path,'
            + '#graph .shapelayer rect{'
            + 'pointer-events:none !important;}'
            + (mode === 'edit_range'
                ? (
                    '#graph .nsewdrag{cursor:default !important;}'
                    + '#graph .nsewdrag.edge-hit{cursor:ew-resize !important;}'
                    + '#graph .outline-controllers{display:none !important;}'
                  )
                : '');

        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            if (gd && window.Plotly && window.__desiredDragmode !== undefined) {
                // Stable layout.uirevision keeps the old dragmode; force the mode radio.
                window.Plotly.relayout(gd, {dragmode: window.__desiredDragmode});
            }
            if (gd && window.__installCustomEdgeEdit) {
                window.__installCustomEdgeEdit(gd);
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
            style.textContent =
                '#graph .shapelayer path,#graph .shapelayer rect{'
                + 'pointer-events:none !important;}';
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
            window.__installCustomEdgeEdit = function(gd) {
                if (!gd || gd.__customEdgeEdit) return;
                gd.__customEdgeEdit = true;
                var HIT_PX = 12;
                var dragState = null;

                function dragEl() {
                    return gd.querySelector('.nsewdrag');
                }

                function setEdgeHit(on) {
                    var el = dragEl();
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

                function formatTipTime(xVal) {
                    var ms = toMs(xVal);
                    if (!isFinite(ms)) return '';
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    // sample_ms / plot axis use KST wall encoded as UTC components.
                    return d.getUTCFullYear() + '년 ' + pad(d.getUTCMonth() + 1) + '월 '
                        + pad(d.getUTCDate()) + '일 ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes());
                }

                function showTip(ev, xVal) {
                    var tip = ensureTip();
                    var text = formatTipTime(xVal);
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

                function edgeHit(clientX, clientY) {
                    if (!window.__editRangeMode) return null;
                    var fl = gd._fullLayout;
                    var xa = fl && fl.xaxis;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var edges = meta.edit_edges || [];
                    if (!xa || !edges.length) return null;
                    var el = dragEl();
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

                function toMs(xVal) {
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

                function snapToSample(xVal) {
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var samples = meta.sample_ms;
                    if (!samples || !samples.length) return xVal;
                    var ms = toMs(xVal);
                    if (!isFinite(ms)) return xVal;
                    var snapped = nearestSampleMs(ms, samples);
                    return snapped == null ? xVal : formatPlotNaive(snapped);
                }

                function xFromClientX(clientX) {
                    var xa = gd._fullLayout.xaxis;
                    var el = dragEl();
                    var r = el.getBoundingClientRect();
                    var xPx = Math.max(0, Math.min(r.width, clientX - r.left));
                    return xa.p2d(xPx);
                }

                function onMove(ev) {
                    if (dragState) {
                        ev.preventDefault();
                        ev.stopPropagation();
                        var xVal = snapToSample(xFromClientX(ev.clientX));
                        if (dragState.lastX === xVal) {
                            setEdgeHit(true);
                            showTip(ev, xVal);
                            return;
                        }
                        dragState.lastX = xVal;
                        var idx = dragState.index;
                        var patch = {};
                        patch['shapes[' + idx + '].x0'] = xVal;
                        patch['shapes[' + idx + '].x1'] = xVal;
                        patch['shapes[' + idx + '].y0'] = 0;
                        patch['shapes[' + idx + '].y1'] = 1;
                        if (window.Plotly) window.Plotly.relayout(gd, patch);
                        setEdgeHit(true);
                        showTip(ev, xVal);
                        return;
                    }
                    if (!window.__editRangeMode) {
                        setEdgeHit(false);
                        hideTip();
                        return;
                    }
                    setEdgeHit(!!edgeHit(ev.clientX, ev.clientY));
                }

                function onDown(ev) {
                    if (!window.__editRangeMode || ev.button !== 0) return;
                    var hit = edgeHit(ev.clientX, ev.clientY);
                    if (!hit) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    var x0 = snapToSample(xFromClientX(ev.clientX));
                    dragState = {index: hit.index, name: hit.name, lastX: x0};
                    setEdgeHit(true);
                    showTip(ev, x0);
                }

                function onUp(ev) {
                    if (!dragState) return;
                    var name = dragState.name;
                    var idx = dragState.index;
                    dragState = null;
                    hideTip();
                    var shapes = (gd.layout && gd.layout.shapes) || [];
                    var shape = shapes[idx];
                    var xVal = shape ? (shape.x0 != null ? shape.x0 : shape.x1) : null;
                    setEdgeHit(!!edgeHit(ev.clientX, ev.clientY));
                    if (xVal == null) return;
                    xVal = snapToSample(xVal);
                    window.dash_clientside.set_props('edge-drag-event', {
                        data: {
                            name: name,
                            x: String(xVal),
                            sequence: Date.now()
                        }
                    });
                }

                gd.addEventListener('mousemove', onMove, true);
                gd.addEventListener('mousedown', onDown, true);
                window.addEventListener('mouseup', onUp, true);
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

        // Click near anomaly divider lines → zoom (shapes themselves are not clickable).
        if (!window.__installAnomalyShapeClick) {
            window.__installAnomalyShapeClick = function(gd) {
                if (!gd) return;
                var drag = gd.querySelector('.nsewdrag');
                if (!drag || drag.__anomalyZoomBound) return;
                drag.__anomalyZoomBound = true;
                var ptr = null;

                function formatPlotX(val) {
                    if (val == null || !isFinite(+new Date(val)) && typeof val !== 'string') {
                        if (typeof val === 'number' && isFinite(val)) {
                            var d = new Date(val);
                            var pad = function(n) { return (n < 10 ? '0' : '') + n; };
                            return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-'
                                + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':'
                                + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
                        }
                        return null;
                    }
                    if (typeof val === 'string') return val;
                    var dt = (val instanceof Date) ? val : new Date(val);
                    if (isNaN(dt.getTime())) return null;
                    var p = function(n) { return (n < 10 ? '0' : '') + n; };
                    // Plot uses KST-naive wall clock encoded as UTC components.
                    return dt.getUTCFullYear() + '-' + p(dt.getUTCMonth() + 1) + '-'
                        + p(dt.getUTCDate()) + ' ' + p(dt.getUTCHours()) + ':'
                        + p(dt.getUTCMinutes()) + ':' + p(dt.getUTCSeconds());
                }

                function clientXToPlotX(clientX) {
                    var full = gd._fullLayout;
                    var xa = full && full.xaxis;
                    if (!xa) return null;
                    var layer = gd.querySelector('.nsewdrag');
                    if (!layer) return null;
                    var bb = layer.getBoundingClientRect();
                    var px = clientX - bb.left;
                    if (px < 0 || px > bb.width) return null;
                    try {
                        if (typeof xa.p2d === 'function') return formatPlotX(xa.p2d(px));
                        if (typeof xa.p2c === 'function' && typeof xa.c2d === 'function') {
                            return formatPlotX(xa.c2d(xa.p2c(px)));
                        }
                    } catch (err) {}
                    return null;
                }

                drag.addEventListener('pointerdown', function(ev) {
                    ptr = {x: ev.clientX, y: ev.clientY};
                });
                drag.addEventListener('click', function(ev) {
                    if (window.__editRangeMode) return;
                    if (ptr && Math.hypot(ev.clientX - ptr.x, ev.clientY - ptr.y) > 8) return;
                    var xVal = clientXToPlotX(ev.clientX);
                    if (xVal == null) return;
                    if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                    window.dash_clientside.set_props('shape-click-event', {
                        data: {x: xVal, sequence: Date.now()}
                    });
                });
            };
        }

        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            window.__ignoreDataXClampUntil = Date.now() + 1500;
            if (!gd || !window.Plotly) return;
            window.__clampPanToData(gd);
            window.__installCustomEdgeEdit(gd);
            window.__installAnomalyShapeClick(gd);
            var drag = gd.querySelector('.nsewdrag');
            if (drag && !window.__editRangeMode) {
                drag.classList.remove('edge-hit');
            }
            // Re-apply mode dragmode after figure updates (uirevision keeps stale zoom/pan).
            if (window.__desiredDragmode !== undefined) {
                var cur = gd._fullLayout && gd._fullLayout.dragmode;
                if (cur !== window.__desiredDragmode) {
                    window.Plotly.relayout(gd, {dragmode: window.__desiredDragmode});
                }
            }
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
