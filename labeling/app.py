#!/usr/bin/env python3
"""Rank별 사업자 anomaly 구간 라벨링 앱 (노트북과 동일 기능).

실행:
    python labeling/app.py
    # 또는
    cd labeling && python app.py

브라우저가 열리면 PLMN을 고른 뒤 Load 하세요.
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
    add_label,
    build_figure,
    clamp_time_range,
    data_time_bounds,
    display_plmn,
    format_kst,
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
    "label_range_anchor": None,
    "highlight_id": None,
    "value_cursor_pos": None,
    "_last_click": (None, 0.0),
}


def _empty_figure():
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        height=336,
        margin=dict(l=40, r=20, t=40, b=40),
        annotations=[
            dict(
                text="PLMN을 선택하고 Load를 누르세요",
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
    )
    fig.update_layout(dragmode="pan" if click_mode in ("pan", "inspect") else "zoom")
    fig.update_yaxes(fixedrange=True)

    if state["label_range_anchor"] is not None:
        fig.add_shape(**pending_anchor_shape(state["label_range_anchor"]))
        fig.add_annotation(**pending_anchor_annotation(state["label_range_anchor"]))

    if state.get("highlight_id"):
        item = _label_by_id(state["highlight_id"])
        if item is not None:
            hs, hn = label_highlight_overlays(item)
            for s in hs:
                fig.add_shape(**s)
            for a in hn:
                fig.add_annotation(**a)

    if state.get("value_cursor_pos") is not None:
        row = state["df"].iloc[state["value_cursor_pos"]]
        shape, note = value_cursor_overlays(row["time"])
        fig.add_shape(**shape)
        fig.add_annotation(**note)

    return fig


def _clamp_zoom(start_ts, end_ts):
    tmin, tmax = data_time_bounds(state["df"])
    return clamp_time_range(start_ts, end_ts, tmin, tmax)


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


def _shift_zoom(direction: int, fraction: float = 0.5):
    if state["df"] is None:
        return
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    if start_ts is None or end_ts is None:
        start_ts, end_ts = data_time_bounds(state["df"])
    width = end_ts - start_ts
    if width <= pd.Timedelta(0):
        return
    delta = width * fraction * direction
    state["zoom_start"], state["zoom_end"] = _clamp_zoom(
        start_ts + delta, end_ts + delta
    )


def _apply_xaxis_from_relayout(relayout) -> bool:
    if not relayout or state["df"] is None:
        return False
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        start_ts = parse_time(relayout["xaxis.range[0]"])
        end_ts = parse_time(relayout["xaxis.range[1]"])
    elif "xaxis.range" in relayout:
        start_ts = parse_time(relayout["xaxis.range"][0])
        end_ts = parse_time(relayout["xaxis.range"][1])
    elif relayout.get("xaxis.autorange"):
        state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
        return True
    else:
        return False
    state["zoom_start"], state["zoom_end"] = _clamp_zoom(start_ts, end_ts)
    return True


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
        label_range_anchor=None,
        highlight_id=None,
        value_cursor_pos=None,
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
                html.Button("Load", id="btn-load", n_clicks=0, style={"marginLeft": "8px"}),
            ],
            style={
                "display": "flex",
                "gap": "8px",
                "alignItems": "center",
                "flexWrap": "wrap",
            },
        ),
        html.H3("2) 그래프", style={"margin": "10px 0 4px"}),
        html.Div(
            [
                dcc.RadioItems(
                    id="click-mode",
                    options=[
                        {"label": "줌", "value": "zoom"},
                        {"label": "이동(팬)", "value": "pan"},
                        {"label": "값 탐색(클릭+←→)", "value": "inspect"},
                        {"label": "구간 라벨(2클릭)", "value": "label_range"},
                    ],
                    value="zoom",
                    inline=True,
                    style={"marginRight": "12px"},
                ),
                html.Button("◀", id="btn-pan-left", n_clicks=0, title="왼쪽(과거)으로 이동"),
                html.Button("▶", id="btn-pan-right", n_clicks=0, title="오른쪽(미래)으로 이동"),
                html.Button("전체", id="btn-reset-zoom", n_clicks=0),
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
            figure=_empty_figure(),
            config={"responsive": True, "displayModeBar": True},
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
            "라벨은 항상 anomaly 구간입니다. 그래프에서 시작·끝을 클릭해 추가하세요.",
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
                html.B("저장된 라벨 (한 줄 = 라벨 1개)"),
                html.Span(
                    " — 한 줄을 클릭하면 그래프에서 노란색으로 강조됩니다.",
                    style={"fontSize": "12px", "color": "#666"},
                ),
            ],
            style={"marginTop": "8px"},
        ),
        dcc.RadioItems(id="label-list", options=[], value=None),
        html.Div(
            [
                html.Button("라벨 선택해제", id="btn-clear-selection", n_clicks=0),
                html.Button("선택 라벨로 줌", id="btn-zoom-selected", n_clicks=0),
                html.Button("선택 라벨 삭제", id="btn-delete", n_clicks=0),
            ],
            style={"display": "flex", "gap": "8px", "marginTop": "6px"},
        ),
        dcc.Store(id="key-event"),
        dcc.Store(id="key-listener-state"),
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
    Input("btn-load", "n_clicks"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("click-mode", "value"),
    Input("btn-pan-left", "n_clicks"),
    Input("btn-pan-right", "n_clicks"),
    Input("btn-reset-zoom", "n_clicks"),
    Input("btn-cancel-range", "n_clicks"),
    Input("btn-save", "n_clicks"),
    Input("btn-reload", "n_clicks"),
    Input("btn-delete", "n_clicks"),
    Input("btn-clear-selection", "n_clicks"),
    Input("btn-zoom-selected", "n_clicks"),
    Input("label-list", "value"),
    Input("key-event", "data"),
    Input("graph", "clickData"),
    Input("graph", "hoverData"),
    Input("graph", "relayoutData"),
    State("dd-plmn", "value"),
    State("note", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _main(
    n_load,
    n_prev,
    n_next,
    _mode_change,
    n_left,
    n_right,
    n_reset,
    n_cancel,
    n_save,
    n_reload,
    n_delete,
    n_clear_selection,
    n_zoom_sel,
    selected_label,
    key_event,
    click_data,
    hover_data,
    relayout,
    plmn,
    note,
    click_mode,
):
    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    click_mode = click_mode or "zoom"

    # ----- load / prev / next -----
    if prop in ("btn-load.n_clicks", "btn-prev.n_clicks", "btn-next.n_clicks"):
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
        if not target:
            return (
                no_update,
                no_update,
                no_update,
                "PLMN을 선택하세요.",
                no_update,
                no_update,
                no_update,
            )
        return _do_load(target, click_mode)

    if state["df"] is None or state["doc"] is None:
        return (
            no_update,
            no_update,
            no_update,
            "먼저 Load를 누르세요.",
            no_update,
            no_update,
            no_update,
        )

    # ----- click mode -----
    if prop == "click-mode.value":
        state["label_range_anchor"] = None
        if click_mode != "inspect":
            state["value_cursor_pos"] = None
        if click_mode == "zoom":
            status = "그래프에서 좌우로 드래그해 시간축을 확대하세요."
        elif click_mode == "pan":
            status = "좌우로 드래그해 시간축을 이동하세요."
        elif click_mode == "inspect":
            status = "그래프에서 시점을 클릭한 뒤 ←/→ 키로 값을 탐색하세요."
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

    if prop == "btn-pan-left.n_clicks":
        _shift_zoom(-1)
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            "",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )
    if prop == "btn-pan-right.n_clicks":
        _shift_zoom(1)
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            "",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )
    if prop == "btn-reset-zoom.n_clicks":
        state["label_range_anchor"] = None
        state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            "",
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
        start_ts = parse_time(item["start"])
        end_ts = parse_time(item["end"])
        span = end_ts - start_ts
        pad = span / 2 if span > pd.Timedelta(0) else pd.Timedelta(hours=6)
        state["zoom_start"], state["zoom_end"] = _clamp_zoom(
            start_ts - pad, end_ts + pad
        )
        state["highlight_id"] = item["id"]
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
        return (
            _build_graph(click_mode),
            no_update,
            selected_label,
            "",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "key-event.data":
        if (
            click_mode != "inspect"
            or not key_event
            or state.get("value_cursor_pos") is None
        ):
            return (no_update,) * 7
        step = -1 if key_event.get("key") == "ArrowLeft" else 1
        selected = _select_value_pos(state["value_cursor_pos"] + step)
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

    if prop == "graph.relayoutData" and relayout:
        if _apply_xaxis_from_relayout(relayout):
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                "",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
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
        if click_mode != "label_range":
            return (no_update,) * 7

        anchor = state["label_range_anchor"]
        if anchor is None:
            state["label_range_anchor"] = ts
            return (
                _build_graph(click_mode),
                no_update,
                no_update,
                f"① 구간 시작: {format_kst(ts)} → ② 끝점을 클릭하세요",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )

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


# The mode radio keeps focus after being clicked, so the browser would move the
# radio selection on arrow keys. Claiming the event during the capture phase and
# dropping focus keeps the arrows on the value cursor.
app.clientside_callback(
    """
    function(mode) {
        window.__valueInspectMode = mode === 'inspect';

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
        return mode;
    }
    """,
    Output("key-listener-state", "data"),
    Input("click-mode", "value"),
)


# A figure drawn before its container settles renders the WebGL traces at the
# stale width while the SVG label overlays use the final geometry, so ranges look
# shifted until the first zoom. Re-measuring right after each update avoids that.
app.clientside_callback(
    """
    function(figure) {
        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            if (gd && window.Plotly) {
                window.Plotly.Plots.resize(gd);
            }
        }, 60);
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
