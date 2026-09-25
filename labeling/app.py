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

전처리는 data/ib_data/labels/*_labels.json을 수정하지 않습니다.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from threading import Timer
from typing import Any

import pandas as pd
import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tool import (  # noqa: E402
    LABEL_HIGHLIGHT_END,
    LABEL_HIGHLIGHT_START,
    LABEL_REASON_DEFS,
    ANOMALY_SCORE_KEY,
    add_label,
    apply_threshold_mode,
    build_figure,
    chrono_split_bounds,
    clamp_time_range,
    data_time_bounds,
    display_metric,
    display_plmn,
    enrich_predictions_for_ui,
    is_anomalytransformer_prediction,
    is_argos_prediction,
    primary_threshold_label,
    format_kst,
    freeze_shape_editing,
    label_highlight_overlays,
    label_line,
    labels_missing_reasons,
    load_labels,
    load_or_build_ranking,
    load_plmn,
    load_predictions,
    list_prediction_sources,
    anomaly_score_at_time,
    ensure_score_column,
    score_at_row,
    score_hover_arrays,
    metric_columns,
    is_rate_metric,
    overlay_metrics_available,
    pred_line,
    rate_metrics_available,
    A_RATE_KEY,
    M971_COL,
    S_RATE_KEY,
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
    # Currently drawn series (subset of metric_universe). Empty = draw nothing.
    "visible_metrics": [],
    # Metrics eligible in the filter under the active model run (None = all).
    "metric_universe": None,
    "metric_rev": 0,
    "zoom_start": None,
    "zoom_end": None,
    # Explicit full-vs-zoomed flag (avoids stale view-range sync races).
    "x_full_view": True,
    "x_gen": 0,
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
    "show_model_preds": True,
    "predictions": None,
    "pred_source": "omnianomaly",
    "selected_pred_id": None,
    "threshold_mode": "pot",  # "pot" | "best_f1"
    # Draft label reasons (applied to new placements; mirrored from selected label).
    "draft_reasons": [],
    "draft_fail_metrics": [],
    "draft_note": "",
    # Label id currently shown in the reason panel (flush target when switching).
    "reason_bound_id": None,
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


def _label_split_tag(item: dict) -> str:
    """Classify a human label into Train / Valid / Test by its start time."""
    df = state.get("df")
    if df is None or not len(df):
        return ""
    bounds = chrono_split_bounds(df, state.get("predictions"))
    if not bounds:
        return ""
    try:
        t = parse_time(item.get("start"))
    except (TypeError, ValueError):
        return ""
    if t < bounds["valid_start"]:
        return "Train"
    if t < bounds["test_start"]:
        return "Valid"
    return "Test"


def _label_options():
    items = (state["doc"] or {}).get("labels", [])
    opts = []
    for x in items:
        tag = _label_split_tag(x)
        prefix = f"[{tag}] " if tag else ""
        opts.append({"label": f"{prefix}{label_line(x)}", "value": x["id"]})
    return opts


def _fail_metric_options() -> list[dict[str, str]]:
    """Multi-select options for fail_surge (counter metrics, rates excluded)."""
    cols = list(state.get("metrics") or [])
    if not cols and state.get("df") is not None:
        cols = metric_columns(state["df"])
    opts = []
    for c in cols:
        if is_rate_metric(c):
            continue
        opts.append({"label": display_metric(c), "value": c})
    return opts


def _reason_checklist_options() -> list[dict[str, str]]:
    return [{"label": label, "value": code} for code, label in LABEL_REASON_DEFS]


def _store_draft_reasons(
    reasons: list[str] | None,
    fail_metrics: list[str] | None,
    note: str | None,
) -> None:
    state["draft_reasons"] = list(reasons or [])
    state["draft_fail_metrics"] = list(fail_metrics or [])
    state["draft_note"] = (note or "").strip()


def _draft_reason_kwargs() -> dict[str, Any]:
    return {
        "reasons": list(state.get("draft_reasons") or []),
        "fail_metrics": list(state.get("draft_fail_metrics") or []),
        "note": state.get("draft_note") or "",
    }


def _flush_reasons_to_label(
    label_id: str | None,
    *,
    reasons: list[str] | None,
    fail_metrics: list[str] | None,
    note: str | None,
) -> bool:
    """Persist current reason UI fields onto a label. Returns True if updated."""
    if not label_id or state.get("doc") is None:
        return False
    if _label_by_id(label_id) is None:
        return False
    codes = list(reasons or [])
    mets = list(fail_metrics or [])
    if "fail_surge" not in codes:
        mets = []
    update_label(
        state["doc"],
        label_id,
        reasons=codes,
        fail_metrics=mets,
        note=note or "",
    )
    _store_draft_reasons(codes, mets, note)
    _bump_label_rev()
    return True


def _ui_feature_mode_for_display(preds: dict | None = None) -> str | None:
    """Attribution/list labels: comb_share uses the same names as comb (no ÷M971)."""
    preds = preds if preds is not None else (state.get("predictions") or {})
    mode = preds.get("feature_mode")
    if mode == "comb_share":
        return "comb"
    return mode


def _model_pred_options():
    pred = state.get("predictions") or {}
    items = pred.get("labels") or []
    mode = state.get("threshold_mode") or pred.get("threshold_mode") or "pot"
    mode_tag = "POT" if mode == "pot" else "best-F1"
    # Short labels — full attribution is shown in the panel below (keeps
    # dropdown usable when Anomaly Transformer emits hundreds of segments).
    return [
        {
            "label": f"[{mode_tag}] {pred_line(x, with_contrib=False)}",
            "value": x["id"],
        }
        for x in items
    ]


def _pred_by_id(pred_id: str | None) -> dict | None:
    if not pred_id:
        return None
    for item in (state.get("predictions") or {}).get("labels") or []:
        if item.get("id") == pred_id:
            return item
    return None


def _model_pred_source_options():
    plmn = state.get("plmn")
    if not plmn:
        return [{"label": "전체 메트릭", "value": "omnianomaly"}]
    return list_prediction_sources(plmn)


def _default_pred_source(options: list[dict] | None = None) -> str:
    opts = options if options is not None else _model_pred_source_options()
    values = [o["value"] for o in opts]
    cur = state.get("pred_source") or "omnianomaly"
    if cur in values:
        return cur
    return values[0] if values else "omnianomaly"


def _fmt_prf(
    precision: float | None,
    recall: float | None,
    f1: float | None,
    *,
    tp=None,
    fp=None,
    fn=None,
) -> str:
    def _f(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    counts = ""
    if tp is not None or fp is not None or fn is not None:
        counts = f"  TP={tp} FP={fp} FN={fn}"
    return f"P={_f(precision)}  R={_f(recall)}  F1={_f(f1)}{counts}"


def _fmt_metric_line(m: dict | None, title: str) -> str:
    if not m:
        return f"{title}: —"
    return (
        f"{title}: "
        + _fmt_prf(
            m.get("precision"),
            m.get("recall"),
            m.get("f1"),
            tp=m.get("TP"),
            fp=m.get("FP"),
            fn=m.get("FN"),
        )
    )


def _mode_block(
    title: str,
    *,
    active: bool,
    thr: float | None,
    paper: dict | None,
    live: dict | None,
    note: str,
) -> html.Div:
    border = "#7c3aed" if active else "#e2e8f0"
    bg = "#f5f3ff" if active else "#f8fafc"
    badge = " ← 그래프 오버레이" if active else ""
    children: list[Any] = [
        html.Div(
            f"{title}{badge}",
            style={
                "fontWeight": "700",
                "color": "#5b21b6" if active else "#334155",
                "marginBottom": "4px",
            },
        ),
        html.Div(
            note,
            style={"fontSize": "11px", "color": "#64748b", "marginBottom": "4px"},
        ),
    ]
    thr_s = f"{thr:.4f}" if isinstance(thr, (int, float)) else "—"
    children.append(
        html.Div(
            f"threshold={thr_s}",
            style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "2px"},
        )
    )
    if paper:
        children.append(
            html.Div(
                "저장 metrics (PA): "
                + _fmt_prf(
                    paper.get("precision"),
                    paper.get("recall"),
                    paper.get("f1"),
                    tp=paper.get("TP"),
                    fp=paper.get("FP"),
                    fn=paper.get("FN"),
                ),
                style={"fontFamily": "monospace", "fontSize": "12px"},
            )
        )
    if live:
        children.append(
            html.Div(
                _fmt_metric_line(live.get("point"), "Point-wise (live)"),
                style={"fontFamily": "monospace", "fontSize": "12px"},
            )
        )
        children.append(
            html.Div(
                _fmt_metric_line(live.get("point_adjust"), "Point-adjust PA (live)"),
                style={"fontFamily": "monospace", "fontSize": "12px"},
            )
        )
    if not paper and not live:
        children.append(html.I("이 모드의 메트릭이 없습니다."))
    return html.Div(
        children,
        style={
            "border": f"1px solid {border}",
            "borderRadius": "6px",
            "padding": "8px 10px",
            "marginBottom": "6px",
            "background": bg,
        },
    )


def _argos_metrics_children(pred: dict) -> list[Any]:
    paper = pred.get("paper_metrics") or {}
    point = paper.get("point") or {}
    event = paper.get("event") or {}
    mode = paper.get("mode") or pred.get("source") or "argos"
    metric = paper.get("metric") or pred.get("metric") or "?"
    split = paper.get("eval_split") or pred.get("eval_split") or "valid"
    ens = bool(paper.get("ensemble") or "ensemble" in str(pred.get("source") or ""))
    title = (
        f"Detection metrics — Argos ensemble ({mode} · fuse={paper.get('fuse') or '?'} · {split})"
        if ens
        else f"Detection metrics — Argos rule ({mode} · {metric} · {split})"
    )
    rows: list[Any] = [
        html.Div(
            title,
            style={
                "fontWeight": "700",
                "color": "#5b21b6",
                "marginBottom": "6px",
            },
        ),
        html.Div(
            (
                "multi-metric rules → fused score (−#axes); binary thr ≈ OR"
                if ens
                else "이진 규칙 출력 · continuous score / POT·best-F1 없음"
            ),
            style={"fontSize": "11px", "color": "#64748b", "marginBottom": "6px"},
        ),
    ]
    if ens and paper.get("ensemble_metrics"):
        rows.append(
            html.Div(
                "axes: " + ", ".join(str(x) for x in paper.get("ensemble_metrics") or []),
                style={"fontSize": "11px", "color": "#475569", "marginBottom": "4px"},
            )
        )
    if point:
        rows.append(
            html.Div(
                "Point-wise: "
                + _fmt_prf(
                    point.get("precision"),
                    point.get("recall"),
                    point.get("f1"),
                    tp=point.get("TP"),
                    fp=point.get("FP"),
                    fn=point.get("FN"),
                ),
                style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "2px"},
            )
        )
    else:
        rows.append(html.I("저장된 point metrics가 없습니다."))
    if event:
        er = event.get("event_recall")
        er_s = f"{er:.4f}" if isinstance(er, (int, float)) else "—"
        rows.append(
            html.Div(
                f"Event: recall={er_s}  "
                f"hit={event.get('n_hit_events')} / gt={event.get('n_gt_events')}  "
                f"pred_events={event.get('n_pred_events')}",
                style={"fontFamily": "monospace", "fontSize": "12px"},
            )
        )
    rows.append(
        html.Div(
            f"예측 구간 {len(pred.get('labels') or [])}개"
            + (
                f" · rule={os.path.basename(str(pred.get('rule_path')))}"
                if pred.get("rule_path")
                else ""
            ),
            style={"fontSize": "12px", "marginTop": "6px", "color": "#334155"},
        )
    )
    return rows


def _threshold_mode_options(pred: dict | None = None, source: str | None = None) -> list[dict]:
    """Radio labels for primary thr — POT (OA) vs percentile (AT)."""
    pred = pred if pred is not None else (state.get("predictions") or {})
    src = source if source is not None else state.get("pred_source")
    primary = primary_threshold_label(pred, source=src)
    return [
        {"label": primary["radio"], "value": "pot"},
        {"label": "best-F1 (oracle)", "value": "best_f1"},
    ]


def _detection_metrics_children() -> list[Any]:
    pred = state.get("predictions") or {}
    if not pred.get("labels") and not pred.get("thresholds") and not state.get("pred_source"):
        return [
            html.I(
                "모델 예측 파일이 없습니다. "
                "`run_plmn.py`로 학습·export 후 다시 로드하세요."
            )
        ]
    if is_argos_prediction(pred) or str(state.get("pred_source") or "").startswith(
        "argos"
    ):
        return _argos_metrics_children(pred)
    mode = state.get("threshold_mode") or pred.get("threshold_mode") or "pot"
    thresholds = pred.get("thresholds") or {}
    paper = pred.get("paper_metrics") or {}
    live_by = pred.get("live_by_mode") or {}
    primary = primary_threshold_label(pred, source=state.get("pred_source"))
    # Prefer note from saved paper block when present.
    pot_paper = paper.get("pot") or None
    pot_note = (pot_paper or {}).get("note") or primary["note"]
    rows: list[Any] = [
        html.Div(
            primary["heading"],
            style={
                "fontWeight": "700",
                "color": "#5b21b6",
                "marginBottom": "6px",
            },
        )
    ]
    rows.append(
        _mode_block(
            primary["title"],
            active=(mode == "pot"),
            thr=thresholds.get("pot"),
            paper=pot_paper,
            live=live_by.get("pot"),
            note=pot_note,
        )
    )
    rows.append(
        _mode_block(
            "[best-F1]",
            active=(mode == "best_f1"),
            thr=thresholds.get("best_f1"),
            paper=paper.get("best_f1") or None,
            live=live_by.get("best_f1"),
            note="oracle · valid 사람 라벨로 threshold 탐색",
        )
    )
    overlay_name = primary["overlay"] if mode == "pot" else "best-F1"
    rows.append(
        html.Div(
            f"현재 그래프 오버레이 = {overlay_name} "
            f"· 예측 구간 {len(pred.get('labels') or [])}개",
            style={"fontSize": "12px", "marginTop": "4px", "color": "#334155"},
        )
    )
    return rows


def _enrich_current_predictions(*, threshold_mode: str | None = None) -> None:
    """Recompute dual-threshold view for the loaded PLMN + prediction source."""
    df = state.get("df")
    preds = state.get("predictions")
    if df is None or not preds:
        return
    mode = threshold_mode or state.get("threshold_mode") or "pot"
    human = ((state.get("doc") or {}).get("labels")) or []
    state["predictions"] = enrich_predictions_for_ui(
        df, preds, human, threshold_mode=mode
    )
    state["threshold_mode"] = state["predictions"].get("threshold_mode") or mode


def _model_attr_children(item: dict | None):
    if item is None:
        n = len((state.get("predictions") or {}).get("labels") or [])
        if n == 0:
            return html.I(
                "모델 예측 파일이 없거나 구간이 없습니다. "
                "`run_plmn.py`로 학습·export 후 다시 로드하세요."
            )
        return html.I(
            f"모델 예측 {n}개 — 보라색 구간/점을 클릭하거나 목록에서 고르면 "
            "이상으로 본 기여 metric(상위)이 표시됩니다."
        )

    pred_doc = state.get("predictions") or {}
    feat_mode = _ui_feature_mode_for_display(pred_doc)
    tops = item.get("top_metrics") or []
    if not tops:
        return html.Span(
            [
                html.B(pred_line(item, feature_mode=feat_mode)),
                html.Br(),
                html.I(
                    "기여 metric 정보가 없습니다. "
                    "(POT export 구간만 attribution이 포함됩니다. "
                    "best-F1은 점수·threshold로 재구성한 구간입니다.)"
                ),
            ]
        )
    rows = [
        html.Li(
            f"#{t.get('rank')} "
            f"{display_metric(str(t.get('metric')))}  "
            f"기여={float(t.get('contribution', 0)):.2f}  "
            f"(log_prob={float(t.get('log_prob', 0)):.2f})"
        )
        for t in tops
        if t.get("metric")
    ]
    note = (
        pred_doc.get("attribution") or {}
    ).get("note") or (
        "기여 = train 중앙값(log_prob) − 구간 평균(log_prob); "
        "클수록 해당 metric이 평소보다 잘 복원되지 않음"
    )
    if pred_doc.get("feature_mode") == "comb_share":
        note = (
            f"{note} · 학습은 comb_share(÷M971)지만 "
            "그래프·metric 필터는 comb와 동일(raw + S_RATE/A_RATE)"
        )
    return html.Div(
        [
            html.Div(
                "이상 사유 (모델 attribution)",
                style={
                    "fontWeight": "700",
                    "color": "#5b21b6",
                    "marginBottom": "4px",
                },
            ),
            html.Div(pred_line(item, feature_mode=feat_mode), style={"marginBottom": "4px"}),
            html.Ul(rows, style={"margin": "0 0 4px 18px", "padding": 0}),
            html.Div(note, style={"fontSize": "11px", "color": "#64748b"}),
        ]
    )


def _plot_metrics() -> list[str]:
    """Metrics currently drawn on the graph (within the active universe)."""
    universe = _metric_universe()
    visible = state.get("visible_metrics")
    if visible is None:
        return list(universe)
    selected = set(visible)
    return [m for m in universe if m in selected]


def _metrics_by_view_sum() -> list[str]:
    """Filter checklist candidates: active universe, ranked by on-screen sum."""
    df = state.get("df")
    universe = _metric_universe()
    if df is None or not len(df) or not universe:
        return list(universe)
    start_ts = state.get("zoom_start")
    end_ts = state.get("zoom_end")
    view = df
    if start_ts is not None and end_ts is not None:
        mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
        if bool(mask.any()):
            view = df.loc[mask]

    # Rates / overlays first (stable), then raw metrics by sum; zeros kept at end
    # so comb-universe items remain toggleable even when inactive in-view.
    pinned = [
        m
        for m in overlay_metrics_available(df, list(state.get("metrics") or []))
        if m in universe
    ]
    rest = [m for m in universe if m not in pinned]

    def sort_key(m: str) -> float:
        if m not in view.columns:
            return 0.0
        try:
            v = float(view[m].sum())
        except (TypeError, ValueError):
            return 0.0
        if v != v:  # NaN
            return 0.0
        return v

    def has_nonzero(m: str) -> bool:
        if m not in view.columns:
            return False
        col = view[m]
        return bool((col.notna() & (col != 0)).any())

    active = sorted([m for m in rest if has_nonzero(m)], key=sort_key, reverse=True)
    inactive = [m for m in rest if m not in active]
    # Subset runs (comb): keep every universe item toggleable.
    # Full (paperib): hide in-view zeros from the checklist (previous behavior).
    if _pred_uses_feature_subset():
        return pinned + active + inactive
    return pinned + active


def _metric_filter_ui() -> tuple[list[dict], list[str]]:
    """Checklist options (= metric universe) + checked values."""
    ranked = _metrics_by_view_sum()
    opts = [{"label": display_metric(m), "value": m} for m in ranked]
    selected = set(state.get("visible_metrics") or [])
    vals = [m for m in ranked if m in selected]
    return opts, vals


def _full_visible_metrics() -> list[str]:
    metrics = list(state.get("metrics") or [])
    overlays = overlay_metrics_available(state.get("df"), metrics)
    return list(metrics) + [o for o in overlays if o not in metrics]


def _pred_uses_feature_subset(preds: dict | None = None) -> bool:
    preds = preds if preds is not None else (state.get("predictions") or {})
    cols = list(preds.get("feature_columns") or [])
    mode = preds.get("feature_mode")
    source = str(preds.get("source") or state.get("pred_source") or "")
    return mode in ("comb", "comb_share", "selected") or (
        mode not in ("all",) and source not in ("", "omnianomaly") and bool(cols)
    )


def _comb_ui_metric_columns(preds: dict) -> list[str]:
    """Same checklist/plot set for comb and comb_share: raw counters + rates."""
    raw = list(preds.get("comb_raw_counters") or [])
    if not raw:
        # Fallback: drop rates from feature_columns, then re-append UI rates.
        raw = [
            c
            for c in (preds.get("feature_columns") or [])
            if c and not is_rate_metric(str(c))
        ]
    cols: list[str] = []
    seen: set[str] = set()
    for c in [*raw, S_RATE_KEY, A_RATE_KEY]:
        if c in seen:
            continue
        seen.add(c)
        cols.append(c)
    return cols


def _metric_universe_for_predictions(preds: dict | None = None) -> list[str]:
    """Eligible metrics for filter/plot under the active model run."""
    preds = preds if preds is not None else (state.get("predictions") or {})
    full = _full_visible_metrics()
    if not _pred_uses_feature_subset(preds):
        return full
    mode = preds.get("feature_mode")
    source = str(preds.get("source") or state.get("pred_source") or "")
    # comb_share trains on ÷M971, but UI shows the same series as comb.
    if mode in ("comb", "comb_share") or source.endswith("_comb") or source.endswith(
        "_comb_share"
    ):
        cols = _comb_ui_metric_columns(preds)
    else:
        cols = list(preds.get("feature_columns") or [])
    available = set(full)
    out = [c for c in cols if c in available]
    return out or full


def _metric_universe() -> list[str]:
    stored = state.get("metric_universe")
    if stored is not None:
        return list(stored)
    return _full_visible_metrics()


def _apply_pred_metric_visibility(preds: dict | None = None) -> None:
    """Limit filter universe to the run's features; all of them start visible."""
    universe = _metric_universe_for_predictions(preds)
    state["metric_universe"] = universe
    state["visible_metrics"] = list(universe)
    state["metric_rev"] = int(state.get("metric_rev") or 0) + 1


def _visible_from_filter_selection(selected) -> list[str]:
    """Map checklist values to drawn metrics (only within the active universe)."""
    universe = _metric_universe()
    selected_set = set(selected or [])
    return [m for m in universe if m in selected_set]


def _label_by_id(label_id):
    for item in (state["doc"] or {}).get("labels", []):
        if item["id"] == label_id:
            return item
    return None


def _label_at_time(ts, *, point_tol_seconds: float | None = None):
    """Return the best label at `ts` (prefer divider edges, then tightest range)."""
    doc = state.get("doc") or {}
    return _interval_hit_at_time(
        doc.get("labels") or [],
        ts,
        point_tol_seconds=point_tol_seconds,
    )


def _pred_at_time(ts, *, point_tol_seconds: float | None = None):
    """Return the best model-prediction segment at `ts` (same hit rules as labels)."""
    if not state.get("show_model_preds", True):
        return None
    preds = state.get("predictions") or {}
    return _interval_hit_at_time(
        preds.get("labels") or [],
        ts,
        point_tol_seconds=point_tol_seconds,
    )


def _interval_hit_at_time(
    items: list,
    ts,
    *,
    point_tol_seconds: float | None = None,
):
    """Best point/range item at `ts` (prefer divider edges, then tightest range)."""
    zoom_start = state.get("zoom_start")
    zoom_end = state.get("zoom_end")
    if zoom_start is not None and zoom_end is not None:
        win_sec = max((zoom_end - zoom_start).total_seconds(), 1.0)
        edge_tol = max(15 * 60.0, win_sec * 0.012)
    else:
        edge_tol = float(point_tol_seconds if point_tol_seconds is not None else 150.0)

    edge_hits: list[tuple[float, dict]] = []
    range_hits: list[tuple[float, dict]] = []
    for item in items:
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

    Zoom is unchanged on deselection. In 구간 편집, re-clicking the same
    label keeps the selection so edge drags are not cancelled by a toggle.
    """
    state["label_range_anchor"] = None
    kind = (hit.get("kind") or "point").lower()
    tag = "점" if kind == "point" else "구간"
    already = (
        state.get("highlight_id") is not None
        and str(state["highlight_id"]) == str(hit["id"])
    )
    if already:
        if click_mode == "edit_range":
            return (
                _build_graph(click_mode),
                _label_options(),
                hit["id"],
                f"선택 유지 ({tag}): 빨간 경계를 드래그하세요.",
                _cancel_style(click_mode),
                no_update,
                no_update,
            )
        state["highlight_id"] = None
        state.pop("_offscreen_cleared", None)
        _store_draft_reasons([], [], "")
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
    """Zoom X around a label and highlight it."""
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
    # Same as 줌인/전체: bump layout uirevision so Plotly.react accepts the
    # new x window and the Y-auto ranges from `_reset_y` (stable uirevision
    # otherwise keeps the previous metrics/rate/score y ranges).
    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["_force_ui_rev"] = time.time()
    _arm_select_guard()


def _zoom_to_model_pred_item(item: dict) -> None:
    """Zoom X around a model prediction (does not touch human label highlight)."""
    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    kind = (item.get("kind") or "point").lower()
    is_point = kind == "point" or start_ts == end_ts
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
    state["selected_pred_id"] = item.get("id")
    state["x_full_view"] = False
    state.pop("_offscreen_cleared", None)
    _reset_y()
    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["_force_ui_rev"] = time.time()
    _arm_select_guard()


def _zoom_to_chrono_split(which: str) -> bool:
    """Zoom to train / valid / test / full chronological split. Returns False if N/A."""
    df = state.get("df")
    if df is None or not len(df):
        return False
    if which == "full":
        state["zoom_start"], state["zoom_end"] = data_time_bounds(df)
        state["x_full_view"] = True
        state["label_range_anchor"] = None
        _reset_y()
        return True
    bounds = chrono_split_bounds(df, state.get("predictions"))
    if not bounds:
        return False
    if which == "train":
        s, e = bounds["t_min"], bounds["train_end"]
    elif which == "valid":
        s, e = bounds["valid_start"], bounds["valid_end"]
    elif which == "test":
        s, e = bounds["test_start"], bounds["t_max"]
    else:
        return False
    state["zoom_start"], state["zoom_end"] = _clamp_zoom(s, e)
    state["x_full_view"] = False
    state["label_range_anchor"] = None
    state.pop("_offscreen_cleared", None)
    _reset_y()
    return True


def _model_pred_ids() -> list[str]:
    return [
        str(x["id"])
        for x in (state.get("predictions") or {}).get("labels") or []
        if x.get("id")
    ]


def _neighbor_model_pred_id(current: str | None, delta: int) -> str | None:
    """Prev/next model segment id (clamped at ends)."""
    ids = _model_pred_ids()
    if not ids:
        return None
    if current is None or str(current) not in ids:
        return ids[0] if delta >= 0 else ids[-1]
    i = ids.index(str(current))
    j = max(0, min(len(ids) - 1, i + int(delta)))
    return ids[j]


def _snap_inspect_to_item(item: dict):
    """Place 값 탐색 cursor at the anomaly start time; return (row, panel) or None."""
    df = state.get("df")
    if df is None or not len(df):
        return None
    start_ts = parse_time(item["start"])
    nearest = (df["time"] - start_ts).abs().idxmin()
    return _select_value_pos(int(df.index.get_loc(nearest)))


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


def _has_score_panel() -> bool:
    """Predictions include an anomaly-score series (may still be empty in-window)."""
    preds = state.get("predictions") or {}
    if not state.get("show_model_preds", True):
        return False
    ss = preds.get("score_series") or {}
    return bool(ss.get("times") and ss.get("scores"))


def _score_subplot_active() -> bool:
    """Match ``build_figure``: score row only when this x-window has finite scores.

    If we assume a 3-row layout while the figure is 2-row (train / no-score
    zoom), metrics Y-auto is written to a missing yaxis3 and the metrics panel
    (yaxis2) gets the rate scale instead — Y 자동 looks broken after anomaly zoom.
    """
    if not _has_score_panel():
        return False
    return _visible_score_y_bounds() is not None


def _metric_axis_row() -> int:
    return 3 if _score_subplot_active() else 2


def _rate_axis_row() -> int:
    return 2 if _score_subplot_active() else 1


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
        hover_values=True,
        max_metrics=-1,
        max_points=MAX_POINTS,
        show_labels=bool(state.get("show_anomalies", True)),
        highlight_id=state.get("highlight_id"),
        predictions=state.get("predictions"),
        show_predictions=bool(state.get("show_model_preds", True)),
        highlight_pred_id=state.get("selected_pred_id"),
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
        # Accept new x (and Y-auto) on figure replace — same need as axis-cmd.
        state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["zoom_rendered"] = rendered
    # Keep layout.uirevision stable across zoom (PLMN + metric_rev). Bumping
    # metric_rev on filter changes forces Scattergl to drop removed series —
    # otherwise Plotly.react can leave ghost WebGL traces after 전체 해제.
    metric_rev = int(state.get("metric_rev") or 0)
    label_rev = int(state.get("label_rev") or 0)
    place_rev = int(state.get("place_rev") or 0)
    cursor_rev = int(state.get("cursor_rev") or 0)
    x_gen = int(state.get("x_gen") or 0)
    force_ui = state.pop("_force_ui_rev", None)
    force_bit = f":f{force_ui}" if force_ui is not None else ""
    fig.update_layout(
        datarevision=(
            f"{fig.layout.datarevision}:{state['fig_gen']}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}:c{cursor_rev}:x{x_gen}{force_bit}"
        ),
        # Include x_gen so Plotly accepts a new time window on rebuild.
        # 「전체」 also sets _force_ui_rev for a one-shot hard UI reset.
        uirevision=(
            f"{state.get('plmn') or 'labeling'}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}:c{cursor_rev}:x{x_gen}{force_bit}"
        ),
        # Match dcc.Graph style height so select/redraw never shrinks the plot.
        # autosize stays True so width keeps filling the host.
        height=480 if _score_subplot_active() else 420,
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
    # is controlled with Y+ / Y- / Y 자동 (metrics panel). Rate / score panels
    # auto-fit the visible x-window.
    y_locked = True
    y_bounds = _effective_y_bounds()
    rate_bounds = _visible_rate_y_bounds()
    score_active = _score_subplot_active()
    score_bounds = _visible_score_y_bounds() if score_active else None
    state["y_rendered"] = y_bounds
    state["rate_y_rendered"] = rate_bounds
    state["score_y_rendered"] = score_bounds
    rate_row = _rate_axis_row()
    metric_row = _metric_axis_row()
    # Include score in y_gen uirevision so Y 자동 / zoom refresh the score panel.
    score_rev = f"ys:{state.get('y_gen', 0)}"
    if score_active:
        if score_bounds is None:
            fig.update_yaxes(
                title_text="score",
                autorange=True,
                fixedrange=True,
                uirevision=score_rev,
                row=1,
                col=1,
            )
        else:
            fig.update_yaxes(
                title_text="score",
                range=list(score_bounds),
                autorange=False,
                fixedrange=True,
                uirevision=score_rev,
                row=1,
                col=1,
            )
    if rate_bounds is None:
        fig.update_yaxes(
            title_text="rate",
            autorange=True,
            fixedrange=True,
            uirevision=f"y1:{state.get('y_gen', 0)}",
            row=rate_row,
            col=1,
        )
    else:
        fig.update_yaxes(
            title_text="rate",
            range=list(rate_bounds),
            autorange=False,
            fixedrange=True,
            uirevision=f"y1:{state.get('y_gen', 0)}",
            row=rate_row,
            col=1,
        )
    if y_bounds is None:
        fig.update_yaxes(
            title_text="value",
            fixedrange=y_locked,
            autorange=True,
            uirevision=f"y:{state.get('y_gen', 0)}",
            row=metric_row,
            col=1,
        )
    else:
        fig.update_yaxes(
            title_text="value",
            fixedrange=y_locked,
            autorange=False,
            range=list(y_bounds),
            uirevision=f"y:{state.get('y_gen', 0)}",
            row=metric_row,
            col=1,
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
    # after pan/zoom. Only set range on axes; do NOT set per-axis uirevision here —
    # that breaks subplot `matches="x"` and can leave 「전체」 stuck on the old window.
    # layout.uirevision carries x_gen instead.
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
        # sample_ms is heavy (~full PLMN): push once per load; clientside cache
        # + react merge keep it across pan/zoom. score_* is smaller and must be
        # present whenever the score panel is on — otherwise a later Plotly.react
        # that omits them wipes layout.meta before the tip ever ingests, and the
        # tip shows "(없음)" even on valid.
        # edit_range always gets sample_ms so edge drag can snap to samples.
        meta["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0)
        # Clientside tip: only the real score subplot may show score text.
        meta["score_panel"] = bool(_score_subplot_active())
        if not state.get("_hover_arrays_pushed") or click_mode == "edit_range":
            meta["sample_ms"] = samples
            state["_hover_arrays_pushed"] = True
        if _has_score_panel():
            score_arr = state.get("_score_hover")
            if score_arr is None:
                score_arr = score_hover_arrays(
                    state.get("predictions")
                    if state.get("show_model_preds", True)
                    else None
                )
                state["_score_hover"] = score_arr
            if score_arr is not None:
                meta["score_ms"] = score_arr[0]
                meta["score_vals"] = score_arr[1]
            thr = None
            preds = state.get("predictions") or {}
            ss = preds.get("score_series") or {}
            if ss.get("threshold") is not None:
                thr = float(ss["threshold"])
            elif preds.get("threshold") is not None:
                thr = float(preds["threshold"])
            if thr is not None:
                meta["score_threshold"] = thr
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
        fig.add_shape(**value_cursor_overlays(row["time"]))
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
        # Keep prior selection's reasons on that label; new label starts blank.
        prev = state.get("reason_bound_id") or state.get("highlight_id")
        if prev:
            _flush_reasons_to_label(
                prev,
                reasons=state.get("draft_reasons"),
                fail_metrics=state.get("draft_fail_metrics"),
                note=state.get("draft_note"),
            )
        _store_draft_reasons([], [], "")
        before = {x["id"] for x in state["doc"].get("labels", [])}
        add_label(
            state["doc"],
            kind="point",
            start=ts,
            reasons=[],
            fail_metrics=[],
            note="",
        )
        after = [x for x in state["doc"]["labels"] if x["id"] not in before]
        lid = after[0]["id"] if after else None
        state["highlight_id"] = lid
        state["reason_bound_id"] = lid
        opts = _label_options()
        _bump_label_rev()
        _arm_place_click_guard()
        return (
            _build_graph(click_mode),
            opts,
            lid,
            html.Span(
                f"✔ [점] anomaly 추가됨 ({lid}) {format_kst(ts)} — "
                "Save Labels로 저장 · 판정 기준을 아래에서 선택하세요",
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
    prev = state.get("reason_bound_id") or state.get("highlight_id")
    if prev:
        _flush_reasons_to_label(
            prev,
            reasons=state.get("draft_reasons"),
            fail_metrics=state.get("draft_fail_metrics"),
            note=state.get("draft_note"),
        )
    _store_draft_reasons([], [], "")
    before = {x["id"] for x in state["doc"].get("labels", [])}
    add_label(
        state["doc"],
        kind="range",
        start=a,
        end=b,
        reasons=[],
        fail_metrics=[],
        note="",
    )
    after = [x for x in state["doc"]["labels"] if x["id"] not in before]
    lid = after[0]["id"] if after else None
    state["highlight_id"] = lid
    state["reason_bound_id"] = lid
    opts = _label_options()
    when = f"{format_kst(a)} → {format_kst(b)}"
    state["_range_label_done"] = int(time.time() * 1000)
    state["_keep_highlight_id"] = True
    _arm_place_click_guard()
    return (
        _build_graph(click_mode),
        opts,
        lid,
        html.Span(
            f"✔ [구간] anomaly 추가됨 ({lid}) {when} — "
            "Save Labels로 저장 · 이동(Y자동) · 판정 기준을 아래에서 선택하세요",
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
    label_id = state.get("highlight_id") or payload.get("label_id")
    if not label_id:
        return None
    if not state.get("highlight_id"):
        state["highlight_id"] = label_id
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
        if not is_rate_metric(m)
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
    if not (ymin == ymin and ymax == ymax):  # NaN check
        return None
    # Metrics are counts, so the baseline stays at zero unless data goes below it.
    base = min(0.0, ymin)
    if ymax <= base:
        return base, base + 1.0
    return base, ymax + (ymax - base) * 0.05


def _visible_score_y_bounds() -> tuple[float, float] | None:
    """Y range for the anomaly-score panel (visible x-window)."""
    df = state["df"]
    preds = state.get("predictions") if state.get("show_model_preds", True) else None
    if df is None or not len(df) or not preds:
        return None
    if ANOMALY_SCORE_KEY not in df.columns:
        return None
    start_ts = state["zoom_start"]
    end_ts = state["zoom_end"]
    view = df
    if start_ts is not None and end_ts is not None:
        mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
        if mask.any():
            view = df.loc[mask]
    score_vals = view[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64, copy=False)
    finite = score_vals[np.isfinite(score_vals)]
    if len(finite) == 0:
        return None
    ymin = float(finite.min())
    ymax = float(finite.max())
    thr = None
    ss = (preds or {}).get("score_series") or {}
    if ss.get("threshold") is not None:
        thr = float(ss["threshold"])
    elif (preds or {}).get("threshold") is not None:
        thr = float(preds["threshold"])
    if thr is not None:
        ymin = min(ymin, thr)
        ymax = max(ymax, thr)
    if ymax <= ymin:
        return ymin - 1.0, ymax + 1.0
    pad = (ymax - ymin) * 0.08
    return ymin - pad, ymax + pad


def _visible_rate_y_bounds() -> tuple[float, float] | None:
    """Y range for the top SUCCESS/COMB rate panel (visible x-window)."""
    df = state["df"]
    metrics = [m for m in _plot_metrics() if is_rate_metric(m)]
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
    base = min(0.0, ymin)
    if ymax <= base:
        return base, base + 0.01
    pad = (ymax - base) * 0.08
    return base, ymax + pad


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
    # Force Plotly to accept the new cursor shape (stable uirevision can skip it).
    state["cursor_rev"] = int(state.get("cursor_rev") or 0) + 1
    row = df.iloc[pos]
    panel = _hover_panel_at_row(row)
    return row, panel


def _hover_panel_at_row(row: pd.Series):
    """Bottom '이 시점 특성값' panel — visible metrics with non-zero values only."""
    visible = _plot_metrics()
    cols = list(visible)
    if not cols:
        return html.I("표시 중인 metric이 없습니다. 위에서 metric을 선택하세요.")
    score = None
    if state.get("show_model_preds", True):
        score = score_at_row(row)
        if score is None:
            score = anomaly_score_at_time(state.get("predictions"), row["time"])

    score_block = None
    if score is not None:
        thr = None
        preds = state.get("predictions") or {}
        ss = preds.get("score_series") or {}
        if ss.get("threshold") is not None:
            thr = float(ss["threshold"])
        elif preds.get("threshold") is not None:
            thr = float(preds["threshold"])
        thr_txt = f"  (threshold={thr:.2f})" if thr is not None else ""
        flag = " · anomaly" if thr is not None and score < thr else ""
        score_block = html.Div(
            f"Anomaly score = {score:.2f}{thr_txt}{flag}",
            style={
                "marginBottom": "6px",
                "padding": "4px 8px",
                "background": "#efe7ff",
                "border": "1px solid #c4b5fd",
                "borderRadius": "4px",
                "fontFamily": "monospace",
                "fontSize": "12px",
                "color": "#5b21b6",
                "fontWeight": "600",
            },
        )
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    if not pairs and score_block is None:
        return html.I("이 시점에 0이 아닌 특성값이 없습니다.")
    body = dcc.Markdown(
        ranked_hover_html(row["time"], row, cols),
        dangerously_allow_html=True,
    )
    if score_block is None:
        return body
    return html.Div([score_block, body])


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

    if factor > 1:
        # Already (near) full → snap to exact full once so the button always
        # does something visible when the plot looks zoomed-in-ish.
        if _same_time_window((start_ts, end_ts), (tmin, tmax)):
            exact = start_ts == tmin and end_ts == tmax and bool(
                state.get("x_full_view", True)
            )
            if exact:
                return False
            new_start, new_end = tmin, tmax
        elif _same_time_window((new_start, new_end), (start_ts, end_ts)):
            # Edge clamp left the span unchanged — jump to full.
            new_start, new_end = tmin, tmax
    elif _same_time_window((new_start, new_end), (start_ts, end_ts)):
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
    if prev is None or not _same_time_window(rendered, prev):
        if prev is not None and not _same_time_window(rendered, prev):
            state["zoom_rendered_prev"] = prev
        # Bump so Plotly accepts the new x range on rebuild (stable layout
        # uirevision otherwise restores the previous zoom — 「전체」 needed 2 clicks).
        state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["zoom_rendered"] = rendered
    _arm_zoom_guard(guard_seconds)
    score_payload = None
    rate_payload = None
    if touch_y:
        yb = _effective_y_bounds()
        state["y_rendered"] = yb
        y_payload = [float(yb[0]), float(yb[1])] if yb else None
        rb = _visible_rate_y_bounds()
        state["rate_y_rendered"] = rb
        rate_payload = [float(rb[0]), float(rb[1])] if rb else None
        if _score_subplot_active():
            sb = _visible_score_y_bounds()
            state["score_y_rendered"] = sb
            score_payload = [float(sb[0]), float(sb[1])] if sb else None
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
        "y_rate": rate_payload,
        "y_score": score_payload,
        "touch_y": bool(touch_y),
        "status": status,
        "rebuild_ms": int(rebuild_ms),
    }


def _relayout_x_window(relayout) -> tuple[Any, Any] | str | None:
    """Extract shared-x window from a Plotly relayout payload.

    Dual-panel figures emit ``xaxis`` and/or ``xaxis2`` keys depending on which
    subplot was dragged. Either is authoritative for the shared time window.
    Returns ``(start, end)``, the string ``\"autorange\"``, or ``None``.
    """
    if not relayout:
        return None
    if relayout.get("xaxis.autorange") or relayout.get("xaxis2.autorange") or relayout.get(
        "xaxis3.autorange"
    ):
        return "autorange"
    for prefix in ("xaxis", "xaxis2", "xaxis3"):
        k0, k1 = f"{prefix}.range[0]", f"{prefix}.range[1]"
        if k0 in relayout and k1 in relayout:
            return relayout[k0], relayout[k1]
        key = f"{prefix}.range"
        if key in relayout and isinstance(relayout[key], (list, tuple)) and len(relayout[key]) >= 2:
            return relayout[key][0], relayout[key][1]
    return None


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
        xwin = _relayout_x_window(relayout)
        if xwin == "autorange":
            # Double-click zoom-out: keep server in sync with the browser full view.
            state["zoom_start"], state["zoom_end"] = data_time_bounds(state["df"])
            state["x_full_view"] = True
            if y_auto:
                _reset_y()
            return True
        if isinstance(xwin, tuple):
            start_ts = parse_time(xwin[0])
            end_ts = parse_time(xwin[1])
    except (ValueError, TypeError):
        return False

    if start_ts is not None and end_ts is not None and end_ts > start_ts:
        incoming = _clamp_zoom(start_ts, end_ts)
        # Ignore echoes of the window we just drew. A delayed replay of the
        # *previous* window is only ignored while zoom_guard is active — after
        # that, treat it as an intentional revisit (drag back) so Y 자동 runs.
        if _same_time_window(incoming, state.get("zoom_rendered")):
            if y_auto:
                old_y = state.get("y_rendered")
                _reset_y()
                new_y = _effective_y_bounds()
                if (
                    old_y is not None
                    and new_y is not None
                    and _same_range(old_y, new_y)
                ):
                    return False
                return True
            return False
        if _same_time_window(incoming, state.get("zoom_rendered_prev")):
            # Stale echo of the previous window (common after button zoom).
            # After the guard expires, treat the same range as a real revisit so
            # Y 자동 is applied again.
            if time.monotonic() < float(state.get("zoom_guard_until") or 0):
                return False
            # Past guard: intentional revisit of a prior window.
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

    if relayout.get("yaxis.autorange") or relayout.get("yaxis2.autorange"):
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


def _force_adopt_browser_x(view_range) -> bool:
    """Always adopt browser X when it is a valid window (drag-zoom sync).

    Unlike ``_adopt_browser_x_if_tighter``, this does not require the browser to
    be tighter — after a drag-zoom the Store view-range is often stale, so the
    live snap from ``zoom-gesture`` must overwrite server state unconditionally.
    """
    if state["df"] is None or not view_range:
        return False
    if isinstance(view_range, dict):
        x_range = view_range.get("x")
    elif isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        x_range = view_range
    else:
        return False
    if not x_range or len(x_range) != 2:
        return False
    try:
        b0 = parse_time(x_range[0])
        b1 = parse_time(x_range[1])
    except (ValueError, TypeError):
        return False
    if not (b1 > b0):
        return False
    browser = _clamp_zoom(b0, b1)
    s0, s1 = state.get("zoom_start"), state.get("zoom_end")
    if s0 is not None and s1 is not None and _same_time_window(browser, (s0, s1)):
        return False
    state["zoom_start"], state["zoom_end"] = browser
    _note_x_window_changed()
    return True


def _adopt_browser_x_if_tighter(view_range) -> bool:
    """If the plot is zoomed in more than server state, adopt the browser X.

    Drag-zoom can update the Plotly view while a zoom_guard blocks
    ``_apply_axes_from_relayout``, leaving server state at full range. Zoom-out
    then no-ops. Call this before toolbar zoom-in/out.
    """
    if state["df"] is None or not view_range:
        return False
    if isinstance(view_range, dict):
        x_range = view_range.get("x")
    elif isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        x_range = view_range
    else:
        return False
    if not x_range or len(x_range) != 2:
        return False
    try:
        b0 = parse_time(x_range[0])
        b1 = parse_time(x_range[1])
    except (ValueError, TypeError):
        return False
    if not (b1 > b0):
        return False
    browser = _clamp_zoom(b0, b1)
    tmin, tmax = data_time_bounds(state["df"])
    s0 = state.get("zoom_start")
    s1 = state.get("zoom_end")
    if s0 is None or s1 is None:
        server = (tmin, tmax)
    else:
        server = (s0, s1)
    browser_w = (browser[1] - browser[0]).total_seconds()
    server_w = max((server[1] - server[0]).total_seconds(), 1e-6)
    # Browser shows a meaningfully tighter window → trust it.
    if browser_w >= server_w * 0.95:
        return False
    if _same_time_window(browser, server):
        return False
    state["zoom_start"], state["zoom_end"] = browser
    _note_x_window_changed()
    return True


def _same_range(a: tuple[float, float], b: tuple[float, float]) -> bool:
    span = max(abs(a[1] - a[0]), 1e-9)
    return abs(a[0] - b[0]) / span < 1e-3 and abs(a[1] - b[1]) / span < 1e-3


def _do_load(plmn: str, click_mode: str):
    rank = int(rank_df.loc[rank_df["PLMN"] == plmn, "rank"].iloc[0])
    df = load_plmn(plmn)
    doc = load_labels(plmn, rank=rank)
    # Resolve prediction source after plmn is known (options depend on files).
    state["plmn"] = plmn
    src_opts = list_prediction_sources(plmn)
    pred_source = _default_pred_source(src_opts)
    preds = load_predictions(plmn, source=pred_source)
    df = ensure_score_column(df, preds, source=pred_source, plmn=plmn)
    metrics = metric_columns(df)
    tmin, tmax = data_time_bounds(df)
    mode = state.get("threshold_mode") or "pot"
    preds = enrich_predictions_for_ui(
        df, preds, doc.get("labels") or [], threshold_mode=mode
    )
    state.update(
        df=df,
        doc=doc,
        predictions=preds,
        pred_source=pred_source,
        threshold_mode=preds.get("threshold_mode") or mode,
        plmn=plmn,
        rank=rank,
        metrics=metrics,
        zoom_start=tmin,
        zoom_end=tmax,
        x_full_view=True,
        x_gen=0,
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
        _score_hover=None,
        _hover_arrays_pushed=False,
        hover_cache_seq=int(state.get("hover_cache_seq") or 0) + 1,
        selected_pred_id=None,
    )
    _apply_pred_metric_visibility(preds)
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
_START_MODEL_PRED_OPTS: list = []
_START_MODEL_SRC_OPTS: list = []
_START_MODEL_SRC_VALUE = "omnianomaly"
_START_THR_OPTS: list = [
    {"label": "POT (논문 primary)", "value": "pot"},
    {"label": "best-F1 (oracle)", "value": "best_f1"},
]
if _START_PLMN:
    _do_load(_START_PLMN, "zoom")
    _START_FIGURE = _build_graph("zoom")
    _START_LABEL_OPTS = _label_options()
    _START_LABEL_VALUE = None
    _START_METRIC_OPTS, _START_METRIC_VALS = _metric_filter_ui()
    _START_MODEL_PRED_OPTS = _model_pred_options()
    _START_MODEL_SRC_OPTS = _model_pred_source_options()
    _START_MODEL_SRC_VALUE = _default_pred_source(_START_MODEL_SRC_OPTS)
    _START_THR_OPTS = _threshold_mode_options()
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
            [
                html.Div(
                    [
                        html.B("판정 기준", style={"marginRight": "8px"}),
                        html.Span(
                            "선택 라벨에 저장 · 새로 찍는 라벨에도 적용 (복수 가능)",
                            style={"fontSize": "12px", "color": "#666"},
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
                    id="label-reason-codes",
                    options=_reason_checklist_options(),
                    value=[],
                    inline=True,
                    labelStyle={
                        "display": "inline-block",
                        "marginRight": "14px",
                        "marginBottom": "2px",
                        "fontSize": "12px",
                        "whiteSpace": "nowrap",
                    },
                    style={"lineHeight": "1.7"},
                ),
                html.Div(
                    [
                        html.Span(
                            "fail_surge 지표",
                            style={
                                "fontSize": "12px",
                                "fontWeight": "600",
                                "marginRight": "8px",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        dcc.Dropdown(
                            id="label-fail-metrics",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="FAIL/TO 등 폭증 지표 (복수)",
                            style={"flex": "1 1 360px", "minWidth": "260px"},
                        ),
                    ],
                    id="label-fail-metrics-wrap",
                    style={
                        "display": "none",
                        "alignItems": "center",
                        "gap": "4px",
                        "marginTop": "6px",
                        "flexWrap": "wrap",
                    },
                ),
                dcc.Textarea(
                    id="label-reason-note",
                    value="",
                    placeholder="메모 (선택, other면 권장)",
                    style={
                        "width": "100%",
                        "height": "48px",
                        "marginTop": "6px",
                        "fontSize": "12px",
                        "resize": "vertical",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "메모·기준 반영",
                            id="btn-apply-reason",
                            n_clicks=0,
                            title="같은 라벨에 머문 채 메모를 즉시 저장 (다른 라벨로 옮기거나 Save 할 때도 자동 저장됨)",
                            style={"fontSize": "12px"},
                        ),
                        html.Span(
                            id="reason-status",
                            style={
                                "fontSize": "12px",
                                "color": "#64748b",
                                "marginLeft": "10px",
                            },
                        ),
                    ],
                    style={"marginTop": "6px", "display": "flex", "alignItems": "center"},
                ),
            ],
            style={
                "border": "1px solid #e0e0e0",
                "padding": "8px 10px",
                "marginBottom": "8px",
                "background": "#fafafa",
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
                html.Div(
                    [
                        html.Span(
                            "회색=TRAIN · 보라=VALID · 노랑=TEST  (점선=경계)",
                            style={
                                "fontSize": "12px",
                                "color": "#64748b",
                                "marginRight": "8px",
                            },
                        ),
                        html.Button(
                            "Train 구간",
                            id="btn-zoom-train",
                            n_clicks=0,
                            title="train 구간으로 줌",
                        ),
                        html.Button(
                            "Valid 구간",
                            id="btn-zoom-valid",
                            n_clicks=0,
                            title="valid 구간으로 줌 (모델 평가)",
                        ),
                        html.Button(
                            "Test 구간",
                            id="btn-zoom-test",
                            n_clicks=0,
                            title="test 구간으로 줌 (hold-out)",
                        ),
                        html.Button(
                            "전체 구간",
                            id="btn-zoom-full-split",
                            n_clicks=0,
                            title="train+valid+test 전체 보기",
                        ),
                        dcc.RadioItems(
                            id="anomaly-overlay-mode",
                            options=[
                                {"label": "사람+모델", "value": "both"},
                                {"label": "사람만", "value": "human"},
                                {"label": "모델만", "value": "model"},
                                {"label": "숨김", "value": "hide"},
                            ],
                            value="both",
                            inline=True,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                        "flexWrap": "wrap",
                    },
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
                html.B("모델 run", style={"marginRight": "8px"}),
                dcc.Dropdown(
                    id="model-pred-source",
                    options=_START_MODEL_SRC_OPTS,
                    value=_START_MODEL_SRC_VALUE,
                    clearable=False,
                    searchable=False,
                    placeholder="모델 · feature view 선택",
                    style={"width": "min(480px, 100%)", "flex": "1 1 280px"},
                ),
                html.B("Threshold", style={"marginLeft": "8px", "marginRight": "6px"}),
                dcc.RadioItems(
                    id="threshold-mode",
                    options=_START_THR_OPTS,
                    value="pot",
                    inline=True,
                    style={"fontSize": "13px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "flexWrap": "wrap",
                "margin": "0 0 4px",
            },
        ),
        html.Div(
            [
                html.B("모델 예측·기여 metric", style={"marginRight": "8px"}),
                html.Button(
                    "◀",
                    id="btn-model-pred-prev",
                    n_clicks=0,
                    title="이전 모델 예측 구간",
                ),
                dcc.Dropdown(
                    id="model-pred-list",
                    options=_START_MODEL_PRED_OPTS,
                    value=None,
                    clearable=True,
                    searchable=False,
                    placeholder="모델이 잡은 구간 선택",
                    style={"width": "min(640px, 100%)", "flex": "1 1 360px"},
                ),
                html.Button(
                    "▶",
                    id="btn-model-pred-next",
                    n_clicks=0,
                    title="다음 모델 예측 구간",
                ),
                html.Button(
                    "모델 구간으로 줌",
                    id="btn-zoom-model-pred",
                    n_clicks=0,
                    title="선택한 모델 예측 구간으로 다시 확대",
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "flexWrap": "wrap",
                "margin": "0 0 4px",
            },
        ),
        html.Div(
            id="model-pred-attr",
            children=_model_attr_children(None),
            style={
                "fontSize": "12px",
                "color": "#334155",
                "marginBottom": "8px",
                "padding": "6px 10px",
                "background": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "6px",
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
            style={"width": "100%", "height": "480px"},
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
        html.Div(
            id="detection-metrics",
            children=_detection_metrics_children(),
            style={
                "fontSize": "12px",
                "color": "#334155",
                "marginTop": "12px",
                "marginBottom": "8px",
                "padding": "8px 10px",
                "background": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "6px",
            },
        ),
        dcc.Store(id="key-event"),
        dcc.Store(id="key-listener-state"),
        dcc.Store(id="view-range"),
        dcc.Store(id="zoom-gesture"),
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
    Output("model-pred-source", "options"),
    Output("model-pred-source", "value"),
    Input("dd-plmn", "value"),
    prevent_initial_call=False,
)
def _sync_model_pred_source(plmn):
    opts = _model_pred_source_options()
    return opts, _default_pred_source(opts)


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("model-pred-list", "options", allow_duplicate=True),
    Output("model-pred-list", "value", allow_duplicate=True),
    Output("model-pred-attr", "children", allow_duplicate=True),
    Output("detection-metrics", "children", allow_duplicate=True),
    Output("metric-filter", "options", allow_duplicate=True),
    Output("metric-filter", "value", allow_duplicate=True),
    Output("threshold-mode", "style"),
    Output("threshold-mode", "options"),
    Input("model-pred-source", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _switch_model_pred_source(source, click_mode):
    plmn = state.get("plmn")
    if not plmn or not source:
        return (no_update,) * 9
    state["pred_source"] = source
    preds = load_predictions(plmn, source=source)
    state["selected_pred_id"] = None
    if state.get("df") is not None:
        state["df"] = ensure_score_column(
            state["df"],
            preds,
            source=source,
            plmn=plmn,
        )
        human = ((state.get("doc") or {}).get("labels")) or []
        mode = state.get("threshold_mode") or "pot"
        state["predictions"] = enrich_predictions_for_ui(
            state["df"], preds, human, threshold_mode=mode
        )
        state["threshold_mode"] = (
            state["predictions"].get("threshold_mode") or mode
        )
    else:
        state["predictions"] = preds
    state["_score_hover"] = None
    state["_hover_arrays_pushed"] = False
    state["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0) + 1
    _apply_pred_metric_visibility(state["predictions"])
    opts = _model_pred_options()
    m_opts, m_vals = _metric_filter_ui()
    is_argos = is_argos_prediction(state.get("predictions")) or str(source).startswith(
        "argos"
    )
    thr_style = (
        {"fontSize": "13px", "opacity": "0.35", "pointerEvents": "none"}
        if is_argos
        else {"fontSize": "13px"}
    )
    thr_opts = _threshold_mode_options(state.get("predictions"), source)
    return (
        _build_graph(click_mode or "zoom"),
        opts,
        None,
        _model_attr_children(None),
        _detection_metrics_children(),
        m_opts,
        m_vals,
        thr_style,
        thr_opts,
    )


@app.callback(
    Output("model-pred-list", "options"),
    Output("model-pred-list", "value"),
    Input("dd-plmn", "value"),
    prevent_initial_call=False,
)
def _sync_model_pred_list(plmn):
    opts = _model_pred_options()
    return opts, None


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("model-pred-list", "options", allow_duplicate=True),
    Output("model-pred-list", "value", allow_duplicate=True),
    Output("model-pred-attr", "children", allow_duplicate=True),
    Output("detection-metrics", "children", allow_duplicate=True),
    Input("threshold-mode", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _switch_threshold_mode(mode, click_mode):
    mode = mode or "pot"
    state["threshold_mode"] = mode
    df = state.get("df")
    preds = state.get("predictions")
    if df is None or not preds:
        return (no_update,) * 5
    if preds.get("labels_by_mode"):
        state["predictions"] = apply_threshold_mode(
            df, preds, threshold_mode=mode
        )
    else:
        human = ((state.get("doc") or {}).get("labels")) or []
        state["predictions"] = enrich_predictions_for_ui(
            df, preds, human, threshold_mode=mode
        )
    state["threshold_mode"] = (
        state["predictions"].get("threshold_mode") or mode
    )
    state["selected_pred_id"] = None
    state["_score_hover"] = None
    state["_hover_arrays_pushed"] = False
    state["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0) + 1
    return (
        _build_graph(click_mode or "zoom"),
        _model_pred_options(),
        None,
        _model_attr_children(None),
        _detection_metrics_children(),
    )


@app.callback(
    Output("model-pred-attr", "children"),
    Output("detection-metrics", "children"),
    Input("model-pred-list", "value"),
    Input("dd-plmn", "value"),
    Input("model-pred-source", "value"),
    prevent_initial_call=False,
)
def _show_model_pred_attr(pred_id, _plmn, _source):
    state["selected_pred_id"] = pred_id
    return (
        _model_attr_children(_pred_by_id(pred_id)),
        _detection_metrics_children(),
    )


@app.callback(
    Output("model-pred-list", "value", allow_duplicate=True),
    Input("btn-model-pred-prev", "n_clicks"),
    Input("btn-model-pred-next", "n_clicks"),
    State("model-pred-list", "value"),
    prevent_initial_call=True,
)
def _nav_model_pred(_n_prev, _n_next, current):
    tid = getattr(callback_context, "triggered_id", None)
    if tid == "btn-model-pred-prev":
        nxt = _neighbor_model_pred_id(current, -1)
    elif tid == "btn-model-pred-next":
        nxt = _neighbor_model_pred_id(current, 1)
    else:
        return no_update
    if nxt is None or nxt == current:
        return no_update
    return nxt


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("click-mode", "value", allow_duplicate=True),
    Output("hover-panel", "children", allow_duplicate=True),
    Input("model-pred-list", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _on_model_pred_selected(pred_id, click_mode):
    """Select model segment → zoom to it and show attribution (via list value)."""
    state["selected_pred_id"] = pred_id
    if state.get("df") is None:
        return (no_update,) * 4
    item = _pred_by_id(pred_id)
    if item is None:
        status = "모델 예측 선택이 해제되었습니다."
        if not state.get("show_model_preds", True):
            return no_update, status, no_update, no_update
        return (
            _build_graph(click_mode or "zoom"),
            status,
            no_update,
            no_update,
        )
    _zoom_to_model_pred_item(item)
    snapped = _snap_inspect_to_item(item)
    panel = (
        snapped[1]
        if snapped is not None
        else html.I("값 탐색 모드 — 클릭 또는 ←/→ 로 시점을 고르세요.")
    )
    status = (
        f"모델 예측 선택·줌: "
        f"{pred_line(item, feature_mode=_ui_feature_mode_for_display())}"
    )
    return _build_graph("inspect"), status, "inspect", panel


@app.callback(
    Output("model-pred-list", "value", allow_duplicate=True),
    Input("shape-click-event", "data"),
    Input("graph", "clickData"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _pick_model_pred_on_graph(shape_click, click_data, click_mode):
    """Click a purple model band/point → select it (shows attribution above)."""
    click_mode = click_mode or "zoom"
    if click_mode in ("label_range", "label_point", "edit_range"):
        return no_update
    if not state.get("show_model_preds", True):
        return no_update
    if state.get("df") is None:
        return no_update

    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    ts = None
    if prop == "shape-click-event.data" and shape_click and shape_click.get("x") is not None:
        try:
            ts = parse_time(shape_click["x"])
        except (ValueError, TypeError):
            return no_update
    elif prop == "graph.clickData" and click_data and click_data.get("points"):
        x = click_data["points"][0].get("x")
        if x is None:
            return no_update
        try:
            ts = parse_time(x)
        except (ValueError, TypeError):
            return no_update
    else:
        return no_update

    hit = _pred_at_time(ts)
    if hit is None:
        return no_update
    hit_id = hit.get("id")
    if not hit_id:
        return no_update
    # Toggle off if the same segment is clicked again.
    if state.get("selected_pred_id") is not None and str(state["selected_pred_id"]) == str(
        hit_id
    ):
        return None
    return hit_id


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("click-mode", "value", allow_duplicate=True),
    Output("hover-panel", "children", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("btn-zoom-model-pred", "n_clicks"),
    State("model-pred-list", "value"),
    prevent_initial_call=True,
)
def _zoom_model_pred(n_clicks, pred_id):
    if not n_clicks:
        return (no_update,) * 4
    item = _pred_by_id(pred_id)
    if item is None:
        return (no_update,) * 4
    _zoom_to_model_pred_item(item)
    snapped = _snap_inspect_to_item(item)
    panel = (
        snapped[1]
        if snapped is not None
        else html.I("값 탐색 모드 — 클릭 또는 ←/→ 로 시점을 고르세요.")
    )
    status = (
        f"모델 구간으로 줌: "
        f"{pred_line(item, feature_mode=_ui_feature_mode_for_display())}"
    )
    return _build_graph("inspect"), "inspect", panel, status


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
    State("label-reason-codes", "value"),
    State("label-fail-metrics", "value"),
    State("label-reason-note", "value"),
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
    reason_codes,
    fail_metrics,
    reason_note,
):
    tid = getattr(callback_context, "triggered_id", None)
    deselecting = (
        tid == "btn-clear-selection"
        or (tid == "label-list" and not selected_label)
    )
    if deselecting:
        _store_draft_reasons([], [], "")
    elif tid != "label-list":
        # label-list switches are owned by _sync_label_reasons (flush + load).
        # Keep draft in sync with UI so newly placed labels pick up current reasons.
        codes = list(reason_codes or [])
        mets = list(fail_metrics or []) if "fail_surge" in codes else []
        _store_draft_reasons(codes, mets, reason_note)
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
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _sync_click_mode_on_nav(_n_zoom, _list_value, _n_reset, selected_label, click_mode):
    """선택 구간 줌 → 값 탐색; 「전체」 → 줌. 구간 편집 모드는 유지."""
    tid = getattr(callback_context, "triggered_id", None)
    if tid == "btn-reset-zoom":
        # Avoid a redundant click-mode rewrite that can race with axis-cmd.
        if (click_mode or "zoom") == "zoom":
            return no_update
        return "zoom"
    if tid == "btn-zoom-selected":
        if not selected_label or _label_by_id(selected_label) is None:
            return no_update
        return "inspect"
    if tid == "label-list":
        if not selected_label:
            return no_update
        # Stay in 구간 편집 when picking another label from the list.
        if (click_mode or "") == "edit_range":
            return no_update
        # Same rule as main: only when already zoomed (not full view).
        if _is_full_x_view():
            return no_update
        if _label_by_id(selected_label) is None:
            return no_update
        return "inspect"
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
        missing = labels_missing_reasons(state.get("doc"))
        warn = ""
        if missing:
            sample = ", ".join(missing[:5])
            more = f" 외 {len(missing) - 5}개" if len(missing) > 5 else ""
            warn = (
                f"\n\n⚠ 판정 기준 미입력 {len(missing)}개: {sample}{more}\n"
                "가능하면 아래에서 기준을 고른 뒤 저장하세요."
            )
        state["confirm_action"] = "save"
        return (
            shown,
            "라벨 저장",
            f"현재 라벨을 파일에 저장할까요?\n\n{display_plmn(plmn)} · {n}개{warn}",
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


def _fail_metrics_wrap_style(reasons: list[str] | None) -> dict[str, str]:
    show = "fail_surge" in (reasons or [])
    return {
        "display": "flex" if show else "none",
        "alignItems": "center",
        "gap": "4px",
        "marginTop": "6px",
        "flexWrap": "wrap",
    }


@app.callback(
    Output("label-reason-codes", "value"),
    Output("label-fail-metrics", "value"),
    Output("label-fail-metrics", "options"),
    Output("label-fail-metrics-wrap", "style"),
    Output("label-reason-note", "value"),
    Output("reason-status", "children"),
    Output("label-list", "options", allow_duplicate=True),
    Input("label-list", "value"),
    Input("label-reason-codes", "value"),
    Input("label-fail-metrics", "value"),
    Input("btn-apply-reason", "n_clicks"),
    Input("dd-plmn", "value"),
    State("label-reason-note", "value"),
    prevent_initial_call="initial_duplicate",
)
def _sync_label_reasons(
    selected_label,
    reason_codes,
    fail_metrics,
    _n_apply,
    _plmn,
    note,
):
    """Keep reason UI ↔ selected/new label draft in sync."""
    tid = getattr(callback_context, "triggered_id", None)
    fail_opts = _fail_metric_options()
    list_opts = no_update

    if tid in ("label-list", "dd-plmn", None):
        # Before switching away, flush typed note/checklist to the previously
        # bound label (note only lives in the textarea until then).
        prev = state.get("reason_bound_id")
        if tid == "label-list" and prev and str(prev) != str(selected_label or ""):
            if _flush_reasons_to_label(
                prev,
                reasons=reason_codes,
                fail_metrics=fail_metrics,
                note=note,
            ):
                list_opts = _label_options()

        item = _label_by_id(selected_label) if selected_label else None
        if item is not None:
            codes = list(item.get("reasons") or [])
            mets = list(item.get("fail_metrics") or [])
            note_s = item.get("note") or ""
            state["reason_bound_id"] = selected_label
            _store_draft_reasons(codes, mets, note_s)
            return (
                codes,
                mets,
                fail_opts,
                _fail_metrics_wrap_style(codes),
                note_s,
                "",
                list_opts,
            )
        # Deselect / no label: clear reason UI so the next placement starts blank.
        state["reason_bound_id"] = None
        _store_draft_reasons([], [], "")
        return (
            [],
            [],
            fail_opts,
            _fail_metrics_wrap_style([]),
            "",
            "",
            list_opts,
        )

    codes = list(reason_codes or [])
    mets = list(fail_metrics or [])
    note_s = note if note is not None else ""
    if "fail_surge" not in codes:
        mets = []
    _store_draft_reasons(codes, mets, note_s)

    reason_msg = ""
    bound = selected_label or state.get("reason_bound_id")
    if bound and state.get("doc") is not None:
        if _flush_reasons_to_label(
            bound,
            reasons=codes,
            fail_metrics=mets,
            note=note_s,
        ):
            state["reason_bound_id"] = bound
            list_opts = _label_options()
            item = _label_by_id(bound)
            if item is not None:
                reason_msg = f"반영됨 · {label_line(item)}"

    return (
        codes,
        mets,
        fail_opts,
        _fail_metrics_wrap_style(codes),
        no_update if tid != "btn-apply-reason" else note_s,
        reason_msg,
        list_opts,
    )


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
    state["show_anomalies"] = (anomaly_overlay_mode or "both") in ("both", "human")
    state["show_model_preds"] = (anomaly_overlay_mode or "both") in ("both", "model")

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
            if state.get("value_cursor_pos") is not None:
                state["value_cursor_pos"] = None
                # Bump so Plotly drops the royalblue cursor shape (uirevision).
                state["cursor_rev"] = int(state.get("cursor_rev") or 0) + 1
        else:
            # Fresh inspect session: don't let a prior click debounce swallow the first pick.
            state["_last_click"] = (None, 0.0)
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
        mode = anomaly_overlay_mode or "both"
        state["_hover_arrays_pushed"] = False
        state["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0) + 1
        n_model = len((state.get("predictions") or {}).get("labels") or [])
        return (
            _build_graph(click_mode),
            no_update,
            no_update,
            f"오버레이: {mode} (모델 구간 {n_model}개)",
            _cancel_style(click_mode),
            no_update,
            no_update,
        )

    if prop == "btn-cancel-range.n_clicks":
        return _cancel_pending_range(click_mode)

    if prop == "btn-confirm-ok.n_clicks":
        action = state.pop("confirm_action", None)
        if action == "save":
            bound = selected_label or state.get("reason_bound_id")
            if bound:
                _flush_reasons_to_label(
                    bound,
                    reasons=state.get("draft_reasons"),
                    fail_metrics=state.get("draft_fail_metrics"),
                    note=state.get("draft_note"),
                )
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
        state.pop("_offscreen_cleared", None)
        _store_draft_reasons([], [], "")
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
        snapped = _snap_inspect_to_item(item)
        panel = snapped[1] if snapped is not None else no_update
        row = snapped[0] if snapped is not None else None
        status = f"선택 구간으로 줌: {label_line(item)}"
        if row is not None:
            status = f"{status} · 값 탐색 {format_kst(row['time'])}"
        return (
            _build_graph("inspect"),
            _label_options(),
            item["id"],
            status,
            _cancel_style("inspect"),
            panel,
            no_update,
        )

    if prop == "label-list.value":
        if not selected_label:
            # Clear highlight only. Do NOT reset zoom — keep the current window.
            state["highlight_id"] = None
            state["label_range_anchor"] = None
            state.pop("_offscreen_cleared", None)
            _store_draft_reasons([], [], "")
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
        # Exception: stay in 구간 편집 (do not jump to 값 탐색).
        do_zoom = not _is_full_x_view() and click_mode != "edit_range"
        if do_zoom:
            _zoom_to_label_item(item)
            snapped = _snap_inspect_to_item(item)
            mode = "inspect"
            status = f"선택 구간으로 줌: {label_line(item)}"
            if snapped is not None:
                status = f"{status} · 값 탐색 {format_kst(snapped[0]['time'])}"
            panel = snapped[1] if snapped is not None else no_update
        else:
            state["highlight_id"] = item["id"]
            state.pop("_offscreen_cleared", None)
            _arm_select_guard()
            mode = click_mode
            panel = no_update
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
            panel,
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
        # Drop echo of the same sample (shape-click + clickData) only.
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
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
        # Drop echo of the same sample (shape-click + clickData) only.
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
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
    """Toggle drawn metrics within the active universe. 전체 선택 → Y 자동; 전체 해제 → Y 유지."""
    if state.get("df") is None:
        return no_update, no_update, no_update, no_update
    universe = _metric_universe()
    if not universe:
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
    Input("btn-zoom-train", "n_clicks"),
    Input("btn-zoom-valid", "n_clicks"),
    Input("btn-zoom-test", "n_clicks"),
    Input("btn-zoom-full-split", "n_clicks"),
    State("click-mode", "value"),
    State("view-range", "data"),
    prevent_initial_call=True,
)
def _axis_nav(
    n_left,
    n_right,
    n_y_in,
    n_y_out,
    n_y_auto,
    n_train,
    n_valid,
    n_test,
    n_full_split,
    click_mode,
    view_range,
):
    """Instant axis moves: update state only; browser relayouts, then rebuilds.

    Zoom-in / zoom-out / 「전체」 replace the figure directly (see
    ``_zoom_x_hard`` / ``_reset_zoom_hard``) so Plotly uirevision cannot
    swallow the new time window.
    """
    if state["df"] is None:
        return no_update
    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    click_mode = click_mode or "zoom"
    pan_keep_y = click_mode == "pan_keep_y"

    if prop in (
        "btn-pan-left.n_clicks",
        "btn-pan-right.n_clicks",
    ):
        _adopt_browser_x_if_tighter(view_range)

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
        return _make_axis_cmd("세로축 확대", apply_x=False, rebuild_ms=-1)
    if prop == "btn-y-out.n_clicks":
        if not _scale_y(1.4):
            return no_update
        return _make_axis_cmd("세로축 축소", apply_x=False, rebuild_ms=-1)
    if prop == "btn-y-auto.n_clicks":
        _reset_y()
        return _make_axis_cmd("세로축 자동 맞춤", apply_x=False, rebuild_ms=-1)
    if prop == "btn-zoom-train.n_clicks":
        if not _zoom_to_chrono_split("train"):
            return no_update
        return _make_axis_cmd("Train 구간", rebuild_ms=80)
    if prop == "btn-zoom-valid.n_clicks":
        if not _zoom_to_chrono_split("valid"):
            return no_update
        return _make_axis_cmd("Valid 구간", rebuild_ms=80)
    if prop == "btn-zoom-test.n_clicks":
        if not _zoom_to_chrono_split("test"):
            return no_update
        return _make_axis_cmd("Test 구간", rebuild_ms=80)
    if prop == "btn-zoom-full-split.n_clicks":
        if not _zoom_to_chrono_split("full"):
            return no_update
        return _make_axis_cmd("전체 구간", rebuild_ms=80)
    return no_update


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("zoom-gesture", "data"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _zoom_x_hard(gesture, click_mode):
    """줌인 / 줌인(왼쪽) / 줌아웃: live browser X → scale → replace figure.

    ``zoom-gesture`` is filled clientside with ``__readGraphView()`` on the same
    click, so drag-zoom windows are visible even when server state / view-range
    Store are still at full.
    """
    if state.get("df") is None or not gesture:
        return no_update, no_update
    action = gesture.get("action")
    mode = click_mode or "zoom"
    pan_keep_y = mode == "pan_keep_y"
    # Always sync from the live snap first (drag-zoom after 「전체」).
    live = gesture.get("view")
    if live:
        _force_adopt_browser_x(live)

    if action == "in":
        if not _scale_x(0.7, y_auto=not pan_keep_y):
            return no_update, no_update
        status = "시간축 줌인"
    elif action == "in_left":
        if not _scale_x_from_left(0.7, y_auto=not pan_keep_y):
            return no_update, no_update
        status = "시간축 줌인(왼쪽 고정)"
    elif action == "out":
        if not _scale_x(1.0 / 0.7, y_auto=not pan_keep_y):
            return no_update, no_update
        status = "시간축 줌아웃"
    else:
        return no_update, no_update

    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["_force_ui_rev"] = time.time()
    _arm_zoom_guard(4.0)
    return _build_graph(mode), status


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("btn-reset-zoom", "n_clicks"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _reset_zoom_hard(n_clicks, click_mode):
    """「전체」: replace the figure directly (bypass axis-cmd / uirevision races)."""
    if not n_clicks or state.get("df") is None:
        return no_update, no_update
    state["label_range_anchor"] = None
    tmin, tmax = data_time_bounds(state["df"])
    state["zoom_start"], state["zoom_end"] = tmin, tmax
    state["x_full_view"] = True
    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["zoom_rendered"] = (tmin, tmax)
    state["zoom_rendered_prev"] = None
    state["_force_ui_rev"] = time.time()
    _reset_y()
    _arm_zoom_guard(4.0)
    mode = click_mode or "zoom"
    return _build_graph(mode), "전체 보기"


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
    has_x = _relayout_x_window(relayout) is not None
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
    function(nIn, nInLeft, nOut) {
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        if (!trig || !trig.prop_id) {
            return window.dash_clientside.no_update;
        }
        var action = null;
        if (trig.prop_id.indexOf('btn-zoom-in-left') === 0) action = 'in_left';
        else if (trig.prop_id.indexOf('btn-zoom-in') === 0) action = 'in';
        else if (trig.prop_id.indexOf('btn-zoom-out') === 0) action = 'out';
        if (!action) return window.dash_clientside.no_update;

        // Drop pending rebuilds that could race the hard figure replace.
        if (window.__rebuildTimer) {
            clearTimeout(window.__rebuildTimer);
            window.__rebuildTimer = null;
        }
        window.__ignoreDataXClampUntil = Date.now() + 5000;

        // Live plot range — required after drag-zoom when server/view-range lag.
        var view = window.__readGraphView && window.__readGraphView();
        return {
            action: action,
            view: view || null,
            t: Date.now()
        };
    }
    """,
    Output("zoom-gesture", "data"),
    Input("btn-zoom-in", "n_clicks"),
    Input("btn-zoom-in-left", "n_clicks"),
    Input("btn-zoom-out", "n_clicks"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(n) {
        if (!n) return window.dash_clientside.no_update;
        if (window.__rebuildTimer) {
            clearTimeout(window.__rebuildTimer);
            window.__rebuildTimer = null;
        }
        window.__ignoreDataXClampUntil = Date.now() + 5000;
        return {t: Date.now(), action: 'reset-zoom'};
    }
    """,
    Output("axis-cmd-ack", "data", allow_duplicate=True),
    Input("btn-reset-zoom", "n_clicks"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(cmd) {
        if (!cmd) {
            return window.dash_clientside.no_update;
        }
        var hasX = cmd.x && cmd.x.length === 2;
        var hasY = cmd.y && cmd.y.length === 2;
        var hasYRate = cmd.y_rate && cmd.y_rate.length === 2;
        var hasYScore = cmd.y_score && cmd.y_score.length === 2;
        var touchY = (cmd.touch_y !== false);
        if (!hasX && !touchY) {
            // Still allow rebuild-only commands (pan with Y fixed).
        } else if (!hasX && !hasY && !hasYRate && !hasYScore && cmd.y !== null && touchY) {
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
            // Shared time axis across 2–3 subplot rows.
            patch['xaxis2.autorange'] = false;
            patch['xaxis2.range'] = [cmd.x[0], cmd.x[1]];
            patch['xaxis3.autorange'] = false;
            patch['xaxis3.range'] = [cmd.x[0], cmd.x[1]];
        }
        if (touchY) {
            var fl = gd._fullLayout || {};
            var hasScoreRow = !!fl.yaxis3;
            // 3-row: yaxis=score, yaxis2=rate, yaxis3=metrics
            // 2-row: yaxis=rate, yaxis2=metrics
            var metricsKey = hasScoreRow ? 'yaxis3' : 'yaxis2';
            var rateKey = hasScoreRow ? 'yaxis2' : 'yaxis';
            var scoreKey = hasScoreRow ? 'yaxis' : null;
            if (hasY) {
                patch[metricsKey + '.autorange'] = false;
                patch[metricsKey + '.range'] = [cmd.y[0], cmd.y[1]];
                patch[metricsKey + '.fixedrange'] = true;
            } else {
                patch[metricsKey + '.autorange'] = true;
                patch[metricsKey + '.fixedrange'] = true;
            }
            if (hasYRate) {
                patch[rateKey + '.autorange'] = false;
                patch[rateKey + '.range'] = [cmd.y_rate[0], cmd.y_rate[1]];
                patch[rateKey + '.fixedrange'] = true;
            } else {
                patch[rateKey + '.autorange'] = true;
                patch[rateKey + '.fixedrange'] = true;
            }
            if (scoreKey) {
                if (hasYScore) {
                    patch[scoreKey + '.autorange'] = false;
                    patch[scoreKey + '.range'] = [cmd.y_score[0], cmd.y_score[1]];
                    patch[scoreKey + '.fixedrange'] = true;
                } else {
                    patch[scoreKey + '.autorange'] = true;
                    patch[scoreKey + '.fixedrange'] = true;
                }
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
            if (!hasX && (hasY || hasYRate || hasYScore)) {
                var yKey2 = (gd._fullLayout && gd._fullLayout.yaxis3) ? 'yaxis3' : 'yaxis2';
                patch[yKey2 + '.fixedrange'] = true;
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
        // Tip engine on for place/edit/inspect/pan/zoom. 값 탐색에서는
        // anomaly-score 패널에서만 tip을 띄움 (pointermove 필터).
        window.__showPlaceTimeTip = (
            mode === 'label_range' || mode === 'label_point'
            || mode === 'edit_range' || mode === 'inspect'
            || mode === 'pan' || mode === 'pan_keep_y'
            || mode === 'zoom'
        );
        // Leaving 값 탐색: drop the blue dotted spike immediately (server
        // figure rebuild clears the solid value_cursor shape).
        if (mode !== 'inspect') {
            var leaveSpike = document.getElementById('place-time-spike');
            if (leaveSpike) leaveSpike.style.display = 'none';
            var leaveTip = document.getElementById('place-time-tip');
            if (leaveTip) leaveTip.style.display = 'none';
        }
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

        // Always redefine so hot reloads pick up coord / snap fixes.
        window.__plotPlaceUtils = function(gd) {
                function plotAreaRect() {
                    // Prefer top x-axis subplot so p2d pixels match clientX.
                    try {
                        var fl = gd._fullLayout;
                        var plot0 = fl && fl._plots && (fl._plots.xy || fl._plots.x2y2);
                        if (plot0 && plot0.plot && plot0.plot.getBoundingClientRect) {
                            var r0 = plot0.plot.getBoundingClientRect();
                            if (r0.width > 2 && r0.height > 2) {
                                return {
                                    left: r0.left, top: r0.top,
                                    right: r0.right, bottom: r0.bottom,
                                    width: r0.width, height: r0.height
                                };
                            }
                        }
                    } catch (err) {}
                    if (window.__combinedPlotRect) {
                        var dual = window.__combinedPlotRect(gd);
                        if (dual) return dual;
                    }
                    var layer = gd.querySelector('.nsewdrag')
                        || gd.querySelector('.xy')
                        || gd;
                    if (!layer) return null;
                    var bb = layer.getBoundingClientRect();
                    return {
                        left: bb.left, top: bb.top,
                        right: bb.right, bottom: bb.bottom,
                        width: bb.width, height: bb.height
                    };
                }

                function yHitRect() {
                    // Allow clicks across stacked panels.
                    if (window.__combinedPlotRect) {
                        var dual = window.__combinedPlotRect(gd);
                        if (dual) return dual;
                    }
                    return plotAreaRect();
                }

                function clientXToPlotX(clientX) {
                    var xa = gd._fullLayout && gd._fullLayout.xaxis;
                    if (!xa) return null;
                    var bb = plotAreaRect();
                    if (!bb) return null;
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

                function sampleList() {
                    var meta = (gd.layout && gd.layout.meta) || {};
                    if (meta.sample_ms && meta.sample_ms.length) return meta.sample_ms;
                    if (window.__plotHoverCache && window.__plotHoverCache.sample_ms
                        && window.__plotHoverCache.sample_ms.length) {
                        return window.__plotHoverCache.sample_ms;
                    }
                    return null;
                }

                function snapClientX(clientX) {
                    var xVal = clientXToPlotX(clientX);
                    if (xVal == null) return null;
                    var samples = sampleList();
                    var ms = plotXToMs(xVal);
                    if (!(typeof ms === 'number' && isFinite(ms))) return String(xVal);
                    var snapped = (samples && samples.length)
                        ? nearestSampleMs(ms, samples) : null;
                    // NOTE: isFinite(null)===true in JS — must check null explicitly.
                    if (snapped == null || !(typeof snapped === 'number' && isFinite(snapped))) {
                        // Data is 5-minute; snap wall-clock when sample list missing.
                        var step = 5 * 60 * 1000;
                        snapped = Math.round(ms / step) * step;
                    }
                    return (typeof snapped === 'number' && isFinite(snapped))
                        ? formatPlotNaive(snapped) : String(xVal);
                }

                function inPlotArea(clientX, clientY) {
                    var xR = plotAreaRect();
                    var yR = yHitRect() || xR;
                    if (!xR || !yR) return false;
                    return clientX >= xR.left && clientX <= xR.right
                        && clientY >= yR.top && clientY <= yR.bottom;
                }

                return {
                    clientXToPlotX: clientXToPlotX,
                    snapClientX: snapClientX,
                    plotXToMs: plotXToMs,
                    formatPlotNaive: formatPlotNaive,
                    inPlotArea: inPlotArea
                };
            };

        if (!window.__readGraphView) {
            window.__readGraphView = function() {
                var host = document.getElementById('graph');
                var gd = host && host.querySelector('.js-plotly-plot');
                var layout = gd && gd._fullLayout;
                // The placeholder figure has its own default ranges; recording
                // them would later be replayed onto the real data.
                if (!gd || !gd._fullData || !gd._fullData.length) return null;
                if (!layout) return null;
                var xr = (layout.xaxis && layout.xaxis.range)
                    || (layout.xaxis2 && layout.xaxis2.range)
                    || (layout.xaxis3 && layout.xaxis3.range);
                if (!xr) return null;
                var view = {
                    x: [String(xr[0]), String(xr[1])]
                };
                // Metrics panel is the bottom subplot (yaxis3 with score, else yaxis2).
                var yAxis = layout.yaxis3 || layout.yaxis2 || layout.yaxis;
                if (yAxis && yAxis.range) {
                    view.y = [yAxis.range[0], yAxis.range[1]];
                }
                return view;
            };
        }

        // Union of both subplot plot areas (top xy + bottom x2y2).
        // Always redefine so hot reloads pick up the dual-panel fix.
        window.__combinedPlotRect = function(gd) {
            try {
                if (!gd) return null;
                var rects = [];
                var fl = gd._fullLayout;
                var plots = fl && fl._plots;
                if (plots) {
                    Object.keys(plots).forEach(function(key) {
                        var el = plots[key] && plots[key].plot;
                        if (!el || !el.getBoundingClientRect) return;
                        var r = el.getBoundingClientRect();
                        if (r.width > 2 && r.height > 2) rects.push(r);
                    });
                }
                if (!rects.length) {
                    var drags = gd.querySelectorAll('.nsewdrag');
                    for (var i = 0; i < drags.length; i++) {
                        var r2 = drags[i].getBoundingClientRect();
                        if (r2.width > 2 && r2.height > 2) rects.push(r2);
                    }
                }
                if (!rects.length) return null;
                var left = rects[0].left;
                var right = rects[0].right;
                var topY = rects[0].top;
                var bottomY = rects[0].bottom;
                for (var j = 1; j < rects.length; j++) {
                    left = Math.min(left, rects[j].left);
                    right = Math.max(right, rects[j].right);
                    topY = Math.min(topY, rects[j].top);
                    bottomY = Math.max(bottomY, rects[j].bottom);
                }
                return {
                    left: left,
                    top: topY,
                    right: right,
                    bottom: bottomY,
                    width: right - left,
                    height: bottomY - topY
                };
            } catch (err) {
                return null;
            }
        };

        window.__installSharedPanelSync = function(gd) {
                if (!gd || gd.__sharedPanelSyncV3) return;
                gd.__sharedPanelSyncV3 = true;

                function xPairFromEvent(ed) {
                    if (!ed) return null;
                    var keys = ['xaxis', 'xaxis2', 'xaxis3'];
                    for (var i = 0; i < keys.length; i++) {
                        var ax = keys[i];
                        var a = ed[ax + '.range[0]'];
                        var b = ed[ax + '.range[1]'];
                        if (a === undefined || b === undefined) {
                            if (Array.isArray(ed[ax + '.range'])) {
                                a = ed[ax + '.range'][0];
                                b = ed[ax + '.range'][1];
                            }
                        }
                        if (a !== undefined && b !== undefined) {
                            return {a: a, b: b, from: ax};
                        }
                    }
                    return null;
                }

                // Final sync only. Live plotly_relayouting + Plotly.relayout during
                // box-zoom cancels the gesture and jumps to a bad range (flat lines).
                gd.on('plotly_relayout', function(ed) {
                    if (window.__syncingSharedX || window.__clampingDataX || !ed) return;
                    if (ed['xaxis.autorange'] || ed['xaxis2.autorange'] || ed['xaxis3.autorange']) return;
                    var pair = xPairFromEvent(ed);
                    if (!pair) return;
                    var patch = {};
                    ['xaxis', 'xaxis2', 'xaxis3'].forEach(function(ax) {
                        if (ax === pair.from) return;
                        patch[ax + '.autorange'] = false;
                        patch[ax + '.range'] = [pair.a, pair.b];
                    });
                    if (!Object.keys(patch).length) return;
                    window.__syncingSharedX = true;
                    window.Plotly.relayout(gd, patch).finally(function() {
                        window.__syncingSharedX = false;
                    });
                });

                // Hide Plotly's per-subplot zoombox; we draw a dual-panel band.
                if (!document.getElementById('dual-zoom-style')) {
                    var st = document.createElement('style');
                    st.id = 'dual-zoom-style';
                    st.textContent = [
                        '.js-plotly-plot .zoombox,',
                        '.js-plotly-plot .zoombox-corners{',
                        'opacity:0!important;',
                        'pointer-events:none!important;',
                        '}'
                    ].join('');
                    document.head.appendChild(st);
                }

                // Visual-only zoom band across both panels (does not touch axes).
                var dragStart = null;
                function ensureBand() {
                    var band = document.getElementById('dual-zoom-band');
                    if (!band) {
                        band = document.createElement('div');
                        band.id = 'dual-zoom-band';
                        band.style.cssText = [
                            'position:fixed',
                            'z-index:99990',
                            'pointer-events:none',
                            'display:none',
                            'background:rgba(65,105,225,0.16)',
                            'border:1px solid rgba(65,105,225,0.6)',
                            'box-sizing:border-box'
                        ].join(';');
                        document.body.appendChild(band);
                    }
                    return band;
                }
                function hideBand() {
                    var band = document.getElementById('dual-zoom-band');
                    if (band) band.style.display = 'none';
                }
                function onDown(ev) {
                    if (ev.button !== 0) return;
                    if (window.__editRangeMode || window.__labelPlaceMode
                        || window.__valueInspectMode || window.__edgeDragging) {
                        return;
                    }
                    if (window.__desiredDragmode !== 'zoom') return;
                    var rect = window.__combinedPlotRect(gd);
                    if (!rect) return;
                    if (ev.clientX < rect.left || ev.clientX > rect.right
                        || ev.clientY < rect.top || ev.clientY > rect.bottom) {
                        return;
                    }
                    dragStart = {x: ev.clientX, rect: rect};
                }
                function onMove(ev) {
                    if (!dragStart || !(ev.buttons & 1)) {
                        if (!(ev.buttons & 1)) {
                            dragStart = null;
                            hideBand();
                        }
                        return;
                    }
                    if (window.__desiredDragmode !== 'zoom') {
                        hideBand();
                        return;
                    }
                    var rect = window.__combinedPlotRect(gd) || dragStart.rect;
                    var x0 = Math.max(rect.left, Math.min(dragStart.x, ev.clientX));
                    var x1 = Math.min(rect.right, Math.max(dragStart.x, ev.clientX));
                    var band = ensureBand();
                    band.style.left = x0 + 'px';
                    band.style.top = rect.top + 'px';
                    band.style.width = Math.max(0, x1 - x0) + 'px';
                    band.style.height = Math.max(0, rect.height) + 'px';
                    band.style.display = 'block';
                }
                function onUp() {
                    dragStart = null;
                    hideBand();
                }
                // Bubble phase so Plotly's own zoom gesture is not disturbed.
                gd.addEventListener('mousedown', onDown, false);
                window.addEventListener('mousemove', onMove, true);
                window.addEventListener('mouseup', onUp, true);
                window.addEventListener('blur', onUp, true);
            };

        // Custom edge edit: ↔ cursor + horizontal drag only on crimson edges.
        if (!window.__installCustomEdgeEdit) {
            window.__installCustomEdgeEdit = function() {
                var host = document.getElementById('graph');
                if (!host || host.__customEdgeEditBound) return;
                host.__customEdgeEditBound = true;
                var HIT_PX = 18;
                var dragState = null;

                function activeGd() {
                    return host.querySelector('.js-plotly-plot');
                }

                function dragEls(gd) {
                    if (!gd) return [];
                    return Array.prototype.slice.call(
                        gd.querySelectorAll('.nsewdrag')
                    );
                }

                function dragEl(gd) {
                    var els = dragEls(gd);
                    return els.length ? els[0] : null;
                }

                function setEdgeHit(gd, on) {
                    dragEls(gd).forEach(function(el) {
                        el.classList.toggle('edge-hit', !!on);
                    });
                }

                function plotHitRect(gd) {
                    return window.__combinedPlotRect
                        ? window.__combinedPlotRect(gd)
                        : null;
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
                    // Prefer the top x-axis plot rect so d2p pixels match clientX.
                    var r = null;
                    try {
                        var plot0 = fl._plots && (fl._plots.xy || fl._plots.x2y2);
                        if (plot0 && plot0.plot && plot0.plot.getBoundingClientRect) {
                            r = plot0.plot.getBoundingClientRect();
                        }
                    } catch (err) {}
                    if (!r || !(r.width > 2)) {
                        r = plotHitRect(gd);
                    }
                    if (!r) {
                        var el = dragEl(gd);
                        if (!el) return null;
                        r = el.getBoundingClientRect();
                    }
                    // Y: allow hits across stacked panels (combined rect).
                    var yR = plotHitRect(gd) || r;
                    if (clientX < r.left || clientX > r.right ||
                        clientY < yR.top || clientY > yR.bottom) {
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
                    if ((!samples || !samples.length) && window.__plotHoverCache) {
                        samples = window.__plotHoverCache.sample_ms;
                    }
                    var ms = toMs(gd, xVal);
                    if (!isFinite(ms)) {
                        return (typeof xVal === 'string') ? xVal : String(xVal);
                    }
                    var snapped = null;
                    if (samples && samples.length) {
                        snapped = nearestSampleMs(ms, samples);
                    }
                    if (snapped == null || !(typeof snapped === 'number' && isFinite(snapped))) {
                        // Always snap to 5-minute wall clock (data cadence).
                        var step = 5 * 60 * 1000;
                        snapped = Math.round(ms / step) * step;
                    }
                    return formatPlotNaive(snapped);
                }

                function xFromClientX(gd, clientX) {
                    var xa = gd._fullLayout.xaxis;
                    var r = null;
                    try {
                        var fl = gd._fullLayout;
                        var plot0 = fl._plots && (fl._plots.xy || fl._plots.x2y2);
                        if (plot0 && plot0.plot && plot0.plot.getBoundingClientRect) {
                            r = plot0.plot.getBoundingClientRect();
                        }
                    } catch (err) {}
                    if (!r || !(r.width > 2)) {
                        r = plotHitRect(gd);
                    }
                    if (!r) {
                        var el = dragEl(gd);
                        if (!el) return null;
                        r = el.getBoundingClientRect();
                    }
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
                    var meta = (gd.layout && gd.layout.meta) || {};
                    window.dash_clientside.set_props('edge-drag-event', {
                        data: {
                            name: name,
                            x: xVal,
                            label_id: meta.edit_label_id || null,
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
                    if (eventData['xaxis.autorange'] || eventData['xaxis2.autorange']
                        || eventData['xaxis3.autorange']) return;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var bounds = meta.data_x;
                    if (!bounds || bounds.length < 2) return;
                    var tmin = window.__ms(bounds[0]);
                    var tmax = window.__ms(bounds[1]);
                    if (!(tmax > tmin)) return;

                    var x0, x1;
                    var axKeys = ['xaxis', 'xaxis2', 'xaxis3'];
                    for (var ai = 0; ai < axKeys.length; ai++) {
                        var ax = axKeys[ai];
                        x0 = eventData[ax + '.range[0]'];
                        x1 = eventData[ax + '.range[1]'];
                        if (x0 === undefined || x1 === undefined) {
                            if (Array.isArray(eventData[ax + '.range'])) {
                                x0 = eventData[ax + '.range'][0];
                                x1 = eventData[ax + '.range'][1];
                            }
                        }
                        if (x0 !== undefined && x1 !== undefined) break;
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
                    if (Math.abs(na - a) < 0.5 && Math.abs(nb - b) < 0.5) {
                        // In-bounds: shared-panel sync mirrors the other axis.
                        return;
                    }
                    // Keep the same value type Plotly emitted (string/number).
                    function shiftRaw(orig, delta) {
                        var t = window.__ms(orig);
                        if (!isFinite(t)) return orig;
                        var next = t + delta;
                        if (typeof orig === 'number') return next;
                        var d = new Date(next);
                        function pad(n) { return (n < 10 ? '0' : '') + n; }
                        return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-'
                            + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':'
                            + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
                    }
                    var out0 = shiftRaw(x0, na - a);
                    var out1 = shiftRaw(x1, nb - b);
                    window.__clampingDataX = true;
                    window.Plotly.relayout(gd, {
                        'xaxis.autorange': false,
                        'xaxis.range': [out0, out1],
                        'xaxis2.autorange': false,
                        'xaxis2.range': [out0, out1],
                        'xaxis3.autorange': false,
                        'xaxis3.range': [out0, out1]
                    }).finally(function() {
                        window.__clampingDataX = false;
                    });
                });
            };
        }

        // Empty-area / band clicks → snapped time. Always redefine; rebind with V5.
        window.__installPlotClickHost = function() {
                var host = document.getElementById('graph');
                if (!host) return;
                if (host.__plotClickHostBoundV5) return;
                host.__plotClickHostBoundV5 = true;
                var lastEmit = {x: null, t: 0};
                var ptrDown = null;

                function emitSnap(clientX, clientY) {
                    var gd = host.querySelector('.js-plotly-plot');
                    if (!gd || !gd._fullLayout || !window.__plotPlaceUtils) return false;
                    var plot = window.__plotPlaceUtils(gd);
                    if (!plot.inPlotArea(clientX, clientY)) return false;
                    var xVal = plot.snapClientX(clientX);
                    if (xVal == null) return false;
                    var now = Date.now();
                    if (lastEmit.x === xVal && now - lastEmit.t < 120) return false;
                    lastEmit = {x: xVal, t: now};
                    if (!window.dash_clientside || !window.dash_clientside.set_props) return false;
                    window.dash_clientside.set_props('shape-click-event', {
                        data: {x: xVal, sequence: now}
                    });
                    return true;
                }

                host.addEventListener('pointerdown', function(ev) {
                    if (ev.button !== 0) return;
                    ptrDown = {x: ev.clientX, y: ev.clientY};
                }, true);

                host.addEventListener('click', function(ev) {
                    if (ev.button !== 0) return;
                    // 값 탐색: pointerup handles WebGL score-panel picks; avoid double emit.
                    if (window.__currentClickMode === 'inspect') return;
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
                    emitSnap(ev.clientX, ev.clientY);
                }, true);

                // Short clicks: select labels even when Plotly zoom/pan swallows click.
                // Also covers 값 탐색 WebGL score-panel picks.
                host.addEventListener('pointerup', function(ev) {
                    if (ev.button !== 0) return;
                    if (window.__edgeDragging) return;
                    var down = ptrDown;
                    ptrDown = null;
                    if (window.__skipNextShapeClick) {
                        window.__skipNextShapeClick = false;
                        return;
                    }
                    var mode = window.__currentClickMode;
                    // Placement modes use click (empty) + clickData (traces).
                    if (mode === 'label_range' || mode === 'label_point') return;
                    if (down) {
                        var dx = Math.abs(ev.clientX - down.x);
                        var dy = Math.abs(ev.clientY - down.y);
                        if (dx > 6 || dy > 6) return; // pan/zoom drag
                    }
                    emitSnap(ev.clientX, ev.clientY);
                }, true);
            };

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
        }

        // Always (re)define — not gated on pending-preview install.
        window.__installPlaceTimeTip = function(gd) {
                if (!gd) return;
                if (gd.__placeTimeTipBoundV8) return;
                gd.__placeTimeTipBoundV8 = true;
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
                            'white-space:pre-line',
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
                    var fl = gd._fullLayout;
                    var xa = fl && fl.xaxis;
                    if (!xa) return;
                    // Prefer the combined top+bottom subplot area so the cursor
                    // spans both rate and metric panels.
                    var rect = (window.__combinedPlotRect && window.__combinedPlotRect(gd))
                        || plotRect;
                    if (!rect) return;
                    var px = toAxisPx(xa, formatPlotNaive(ms));
                    if (px == null || isNaN(px)) return;
                    var left = rect.left + px;
                    if (left < rect.left || left > rect.right) return;
                    if (lastSpikeX === left) {
                        var spike0 = ensureSpike();
                        if (spike0.style.display === 'block') return;
                    }
                    lastSpikeX = left;
                    var spike = ensureSpike();
                    spike.style.left = left + 'px';
                    spike.style.top = rect.top + 'px';
                    spike.style.height = Math.max(0, rect.height) + 'px';
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

                function nearestScore(ms, scoreMs, scoreVals) {
                    if (!scoreMs || !scoreVals || !scoreMs.length || !isFinite(ms)) {
                        return null;
                    }
                    var lo = 0, hi = scoreMs.length - 1;
                    if (ms <= scoreMs[0]) {
                        return (Math.abs(ms - scoreMs[0]) <= 5 * 60 * 1000)
                            ? scoreVals[0] : null;
                    }
                    if (ms >= scoreMs[hi]) {
                        return (Math.abs(ms - scoreMs[hi]) <= 5 * 60 * 1000)
                            ? scoreVals[hi] : null;
                    }
                    while (lo <= hi) {
                        var mid = (lo + hi) >> 1;
                        if (scoreMs[mid] < ms) lo = mid + 1;
                        else hi = mid - 1;
                    }
                    var ia = Math.max(0, lo - 1);
                    var ib = Math.min(scoreMs.length - 1, lo);
                    var a = scoreMs[ia], b = scoreMs[ib];
                    var pick = (Math.abs(ms - a) <= Math.abs(ms - b)) ? ia : ib;
                    if (Math.abs(ms - scoreMs[pick]) > 5 * 60 * 1000) return null;
                    var v = scoreVals[pick];
                    return (typeof v === 'number' && isFinite(v)) ? v : null;
                }

                function ingestHoverMeta(rawMeta) {
                    var meta = rawMeta || {};
                    if ((!meta.sample_ms || !meta.sample_ms.length)
                        && gd._fullLayout && gd._fullLayout.meta) {
                        meta = Object.assign({}, gd._fullLayout.meta, meta);
                    }
                    var cache = window.__plotHoverCache || {};
                    if (meta.hover_cache_seq != null && cache.seq != null
                        && meta.hover_cache_seq !== cache.seq) {
                        cache = {seq: meta.hover_cache_seq};
                    }
                    if (meta.sample_ms && meta.sample_ms.length) {
                        cache.sample_ms = meta.sample_ms;
                    }
                    if (meta.score_ms && meta.score_ms.length) {
                        cache.score_ms = meta.score_ms;
                        cache.score_vals = meta.score_vals;
                    }
                    if (meta.score_threshold != null) {
                        cache.score_threshold = meta.score_threshold;
                    }
                    if (meta.hover_cache_seq != null) cache.seq = meta.hover_cache_seq;
                    window.__plotHoverCache = cache;
                    return cache;
                }
                // Figure updates call this before Plotly.react so arrays are cached
                // even if the user has not hovered yet.
                window.__ingestHoverMeta = ingestHoverMeta;
                window.__withCachedHoverMeta = function(layout) {
                    if (!layout) return layout;
                    var meta = Object.assign({}, layout.meta || {});
                    ingestHoverMeta(meta);
                    var cache = window.__plotHoverCache || {};
                    if (cache.sample_ms && cache.sample_ms.length
                        && !(meta.sample_ms && meta.sample_ms.length)) {
                        meta.sample_ms = cache.sample_ms;
                    }
                    if (cache.score_ms && cache.score_ms.length
                        && !(meta.score_ms && meta.score_ms.length)) {
                        meta.score_ms = cache.score_ms;
                        meta.score_vals = cache.score_vals;
                    }
                    if (cache.score_threshold != null && meta.score_threshold == null) {
                        meta.score_threshold = cache.score_threshold;
                    }
                    layout.meta = meta;
                    return layout;
                };

                function scorePanelVisible() {
                    // 3-row layout: yaxis=score, yaxis2=rate, yaxis3=metrics.
                    // 2-row (no score subplot): never treat the top drag as score —
                    // otherwise the tip covers S_RATE hover with "Anomaly score".
                    var fl = gd._fullLayout;
                    if (fl && fl.yaxis3) return true;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    return meta.score_panel === true;
                }

                function inScorePanel(clientY) {
                    try {
                        if (!scorePanelVisible()) return false;
                        var meta = (gd.layout && gd.layout.meta) || {};
                        var cache = ingestHoverMeta(meta);
                        var hasScore = (cache.score_ms && cache.score_ms.length)
                            || cache.score_threshold != null
                            || meta.score_threshold != null;
                        if (!hasScore) return false;
                        // Score is the topmost nsewdrag only when the score row exists.
                        var drags = gd.querySelectorAll('.nsewdrag');
                        var best = null;
                        var bestTop = Infinity;
                        for (var i = 0; i < drags.length; i++) {
                            var r = drags[i].getBoundingClientRect();
                            if (r.width <= 2 || r.height <= 2) continue;
                            if (r.top < bestTop) {
                                bestTop = r.top;
                                best = r;
                            }
                        }
                        if (!best) return false;
                        return clientY >= best.top && clientY <= best.bottom;
                    } catch (err) {
                        return false;
                    }
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

                function formatScoreTip(snapped, cache, wantScore) {
                    var text = formatHoverTime(snapped);
                    if (!wantScore) return text;
                    var score = nearestScore(snapped, cache.score_ms, cache.score_vals);
                    var thr = (typeof cache.score_threshold === 'number'
                        && isFinite(cache.score_threshold))
                        ? cache.score_threshold : null;
                    if (score == null) {
                        text += '\\nAnomaly score = (없음)';
                        if (thr != null) text += '\\nthreshold = ' + thr.toFixed(2);
                        return text;
                    }
                    text += '\\nAnomaly score = ' + score.toFixed(2);
                    if (thr != null) {
                        text += '\\nthreshold = ' + thr.toFixed(2);
                        if (score < thr) text += ' · anomaly';
                    }
                    return text;
                }

                function clientXToPlotX(clientX) {
                    var full = gd._fullLayout;
                    var xa = full && full.xaxis;
                    if (!xa) return null;
                    var bb = (window.__combinedPlotRect && window.__combinedPlotRect(gd));
                    if (!bb) {
                        var layer = gd.querySelector('.nsewdrag') || gd;
                        bb = layer.getBoundingClientRect();
                    }
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

                function placeTipAt(ev, text) {
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
                }

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
                    var rect = window.__combinedPlotRect
                        ? window.__combinedPlotRect(gd) : null;
                    if (!rect) {
                        var layer = gd.querySelector('.nsewdrag') || gd;
                        var bb = layer.getBoundingClientRect();
                        rect = {
                            left: bb.left, top: bb.top,
                            right: bb.right, bottom: bb.bottom,
                            width: bb.width, height: bb.height
                        };
                    }
                    if (ev.clientX < rect.left || ev.clientX > rect.right ||
                        ev.clientY < rect.top || ev.clientY > rect.bottom) {
                        hideTip();
                        return;
                    }
                    var overScore = inScorePanel(ev.clientY);
                    // 값 탐색: floating tip only on anomaly-score panel (bottom panel has the rest).
                    if (window.__valueInspectMode && !overScore) {
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
                    var cache = ingestHoverMeta(meta);
                    var snapped = nearestSampleMs(ms, cache.sample_ms);
                    if (snapped == null || !(typeof snapped === 'number' && isFinite(snapped))) {
                        snapped = snapFiveMinMs(ms);
                    }
                    if (snapped == null || !(typeof snapped === 'number' && isFinite(snapped))) {
                        hideTip();
                        return;
                    }
                    // Always one custom spike at the snapped time (Plotly spikes off).
                    showSpikeAtMs(snapped, rect);
                    // Anomaly-score panel: time + score tip.
                    // Other panels: empty-area time tip only (Plotly owns series hover).
                    var wantScore = overScore;
                    if (!overScore) {
                        var hoverdata = gd._hoverdata;
                        var hoveringOther = !!(hoverdata && hoverdata.length);
                        if (hoveringOther && !window.__editRangeMode) {
                            var tipHide = document.getElementById('place-time-tip');
                            if (tipHide) tipHide.style.display = 'none';
                            lastText = '';
                            return;
                        }
                    }
                    var text = formatScoreTip(snapped, cache, wantScore);
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
                        if (!overScore) {
                            var hd = gd._hoverdata;
                            if (hd && hd.length && !window.__editRangeMode) {
                                var tipHide2 = document.getElementById('place-time-tip');
                                if (tipHide2) tipHide2.style.display = 'none';
                                lastText = '';
                                return;
                            }
                        }
                        placeTipAt(ev, text);
                    });
                }, true);

                gd.addEventListener('pointerleave', hideTip, true);
            };

        setTimeout(function() {
            var host = document.getElementById('graph');
            var gd = host && host.querySelector('.js-plotly-plot');
            window.__ignoreDataXClampUntil = Date.now() + 1500;
            if (!gd || !window.Plotly) return;

            // Standalone hover-cache helpers (tip install may not have run yet).
            if (!window.__ingestHoverMeta) {
                window.__ingestHoverMeta = function(rawMeta) {
                    var meta = rawMeta || {};
                    var cache = window.__plotHoverCache || {};
                    if (meta.hover_cache_seq != null && cache.seq != null
                        && meta.hover_cache_seq !== cache.seq) {
                        cache = {seq: meta.hover_cache_seq};
                    }
                    if (meta.sample_ms && meta.sample_ms.length) {
                        cache.sample_ms = meta.sample_ms;
                    }
                    if (meta.score_ms && meta.score_ms.length) {
                        cache.score_ms = meta.score_ms;
                        cache.score_vals = meta.score_vals;
                    }
                    if (meta.score_threshold != null) {
                        cache.score_threshold = meta.score_threshold;
                    }
                    if (meta.hover_cache_seq != null) cache.seq = meta.hover_cache_seq;
                    window.__plotHoverCache = cache;
                    return cache;
                };
            }
            if (!window.__withCachedHoverMeta) {
                window.__withCachedHoverMeta = function(layout) {
                    if (!layout) return layout;
                    var meta = Object.assign({}, layout.meta || {});
                    window.__ingestHoverMeta(meta);
                    var cache = window.__plotHoverCache || {};
                    if (cache.sample_ms && cache.sample_ms.length
                        && !(meta.sample_ms && meta.sample_ms.length)) {
                        meta.sample_ms = cache.sample_ms;
                    }
                    if (cache.score_ms && cache.score_ms.length
                        && !(meta.score_ms && meta.score_ms.length)) {
                        meta.score_ms = cache.score_ms;
                        meta.score_vals = cache.score_vals;
                    }
                    if (cache.score_threshold != null && meta.score_threshold == null) {
                        meta.score_threshold = cache.score_threshold;
                    }
                    layout.meta = meta;
                    return layout;
                };
            }

            function afterGraphReady() {
                window.__clampPanToData(gd);
                window.__installSharedPanelSync(gd);
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
                // Ingest/preserve hover arrays: later figures often omit sample_ms
                // (one-shot push) and Plotly.react would wipe them from layout.meta
                // before the tip ever runs — tip then shows score "(없음)".
                if (window.__withCachedHoverMeta) {
                    layout = window.__withCachedHoverMeta(layout);
                } else if (window.__ingestHoverMeta && layout.meta) {
                    window.__ingestHoverMeta(layout.meta);
                }
                window.Plotly.react(gd, figure.data, layout, cfg).then(function() {
                    // Re-apply server axis ranges after react. Stable/layout races
                    // (e.g. anomaly select→줌) can leave the previous Y scale even
                    // when the figure JSON already has Y-auto bounds for the window.
                    try {
                        var patch = {};
                        var meta = layout.meta || {};
                        var zx = meta.zoom_x;
                        if (zx && zx.length === 2) {
                            patch['xaxis.autorange'] = false;
                            patch['xaxis.range'] = [zx[0], zx[1]];
                            patch['xaxis2.autorange'] = false;
                            patch['xaxis2.range'] = [zx[0], zx[1]];
                            patch['xaxis3.autorange'] = false;
                            patch['xaxis3.range'] = [zx[0], zx[1]];
                        }
                        ['yaxis', 'yaxis2', 'yaxis3'].forEach(function(key) {
                            var ax = layout[key];
                            if (!ax || !ax.range || ax.range.length !== 2) return;
                            if (ax.autorange) return;
                            patch[key + '.autorange'] = false;
                            patch[key + '.range'] = [ax.range[0], ax.range[1]];
                            patch[key + '.fixedrange'] = true;
                        });
                        if (Object.keys(patch).length) {
                            return window.Plotly.relayout(gd, patch);
                        }
                    } catch (err) {}
                }).then(function() {
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
