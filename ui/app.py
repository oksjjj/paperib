#!/usr/bin/env python3
"""Rank별 사업자 anomaly 구간 라벨링 앱.

실행:
    python ui/app.py
    # 또는
    cd labeling && python app.py

브라우저가 열리면 사업자를 선택하세요 (선택 즉시 로드).

새 날짜의 data/*.csv를 추가한 경우:
    python scripts/build_ib_cache.py
    python ui/app.py

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
REPO = os.path.dirname(ROOT)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(ROOT)

from core.features import PAPER_PLMNS, metric_columns  # noqa: E402

from ui.data_view import (  # noqa: E402
    chrono_split_bounds,
    display_metric,
    display_plmn,
    format_kst,
    load_or_build_ranking,
    load_plmn,
    parse_time,
    to_plot_time,
    to_plot_times,
)
from ui.plotting import (  # noqa: E402
    FIGURE_HEIGHT,
    build_figure,
    clamp_time_range,
    data_time_bounds,
    freeze_shape_editing,
    ranked_hover_html,
    ranked_metric_pairs,
)
from ui.ui_labels import (  # noqa: E402
    LABEL_HIGHLIGHT_END,
    LABEL_HIGHLIGHT_START,
    LABEL_REASON_DEFS,
    add_label,
    label_highlight_overlays,
    label_line,
    labels_missing_reasons,
    load_labels,
    pending_anchor_shape,
    pending_range_fill_shape,
    remove_label,
    save_labels,
    update_label,
    value_cursor_overlays,
)
from ui.ui_predictions import (  # noqa: E402
    ANOMALY_SCORE_KEY,
    anomaly_score_at_time,
    apply_threshold_mode,
    available_metrics_splits,
    enrich_predictions_for_ui,
    ensure_score_column,
    list_prediction_sources,
    load_predictions,
    pred_chrono_split_tag,
    pred_line,
    primary_threshold_label,
    score_at_row,
    score_hover_arrays,
)

MAX_POINTS = 1500
HOST = "127.0.0.1"
PORT = 8050

# 논문 평가 대상 10개 PLMN만 UI의 사업자 선택 목록에 노출한다.
rank_df = load_or_build_ranking(top_n=None, paper_only=True)
plmn_options = [
    {
        "label": f"#{int(r.rank):03d}  {display_plmn(r.PLMN)}  ({int(r.M971_sum):,})",
        "value": r.PLMN,
    }
    for r in rank_df.itertuples(index=False)
]
plmn_ids = [o["value"] for o in plmn_options]
_missing_paper = [p for p in PAPER_PLMNS if p not in plmn_ids]
if _missing_paper:
    raise RuntimeError(
        "paper PLMNs missing from ranking cache: " + ", ".join(_missing_paper)
    )

state: dict = {
    "df": None,
    "doc": None,
    "plmn": None,
    "rank": None,
    "metrics": [],
    # 현재 실제 그래프에 그리고 있는 metric 목록이다. 빈 목록이면 아무 metric도 그리지 않는다.
    "visible_metrics": [],
    # 현재 선택한 모델 결과에서 필터 대상으로 허용되는 metric 집합이다. None이면 전체를 허용한다.
    "metric_universe": None,
    "metric_rev": 0,
    "zoom_start": None,
    "zoom_end": None,
    # 전체 보기인지 확대 상태인지 명시적으로 기록해 브라우저/서버 x축 상태가 엇갈리는 race를 줄인다.
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
    # highlight overlay가 바뀔 때 revision 값을 올린다. 그렇지 않으면 Plotly의 안정된 uirevision 때문에
    # 이전 선택 shape가 화면에 남을 수 있다.
    "hi_rev": 0,
    "confirm_action": None,
    "value_cursor_pos": None,
    "hover_pos": None,
    "_last_click": (None, 0.0),
    "show_anomalies": True,
    "show_model_preds": True,
    "predictions": None,
    "pred_source": None,
    "selected_pred_id": None,
    "threshold_mode": "pot",  # threshold 표시 모드: primary / point best-F1 / PA best-F1
    "metrics_eval_split": "valid",  # metric을 계산해 보여줄 split: VALID 또는 TEST
    # 새 label에 적용할 anomaly 원인 입력 초안이다. 기존 label을 선택하면 그 값도 여기로 복사한다.
    "draft_reasons": [],
    "draft_fail_metrics": [],
    "draft_note": "",
    # 현재 원인 패널이 편집 중인 label ID다. 다른 label로 이동하기 전에 이 ID에 입력값을 저장한다.
    "reason_bound_id": None,
}


def _sample_plot_ms(df: pd.DataFrame) -> list[int]:
    """DataFrame의 KST wall-clock 시간을 Plotly x축 snapping에 사용할 epoch millisecond 배열로 변환한다."""
    vals = to_plot_times(df["time"])
    return vals.astype("datetime64[ms]").astype("int64").tolist()


def _empty_figure():
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        height=FIGURE_HEIGHT,
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
    """전문가 label의 시작시간을 기준으로 TRAIN / VALID / TEST 태그를 붙인다."""
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
    """fail_surge 원인에 연결할 수 있는 counter metric 다중선택 옵션을 만든다."""
    cols = list(state.get("metrics") or [])
    if not cols and state.get("df") is not None:
        cols = metric_columns(state["df"])
    opts = []
    for c in cols:
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


def _flush_reasons_to_label(
    label_id: str | None,
    *,
    reasons: list[str] | None,
    fail_metrics: list[str] | None,
    note: str | None,
) -> bool:
    """현재 원인 입력 UI의 값을 선택된 label에 저장한다. 실제로 변경되었으면 True를 반환한다."""
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
    """기여도/목록 표시용 feature 이름 규칙을 결정한다. comb_share도 UI에서는 comb와 같은 이름을 사용한다."""
    preds = preds if preds is not None else (state.get("predictions") or {})
    mode = preds.get("feature_mode")
    if mode == "comb_share":
        return "comb"
    return mode


def _metrics_split_wanted() -> str:
    """현재 Metrics radio에서 선택된 평가 split(valid 또는 test)을 반환한다."""
    return (
        state.get("metrics_eval_split")
        or (state.get("predictions") or {}).get("metrics_eval_split")
        or "valid"
    )


def _pred_matches_metrics_split(
    item: dict,
    pred_doc: dict | None = None,
    *,
    metrics_split: str | None = None,
) -> bool:
    """한 model prediction segment가 현재 선택된 VALID/TEST 평가 구간에 속하는지 확인한다."""
    wanted = (metrics_split or _metrics_split_wanted()).lower()
    tag = pred_chrono_split_tag(item, pred_doc or state.get("predictions"))
    if not tag:
        # split 메타정보가 없는 과거 export는 호환성을 위해 그대로 유지한다.
        return True
    if wanted == "test":
        return tag == "TEST"
    return tag == "VALID"


def _model_pred_items(*, metrics_split: str | None = None) -> list[dict]:
    pred = state.get("predictions") or {}
    wanted = metrics_split or _metrics_split_wanted()
    return [
        x
        for x in (pred.get("labels") or [])
        if _pred_matches_metrics_split(x, pred, metrics_split=wanted)
    ]


def _predictions_for_display() -> dict | None:
    """현재 Metrics split에 맞는 prediction label만 남긴 표시용 prediction 문서를 반환한다."""
    pred = state.get("predictions")
    if not pred:
        return None
    items = _model_pred_items()
    if len(items) == len(pred.get("labels") or []):
        return pred
    out = dict(pred)
    out["labels"] = items
    out["n_pred_segments"] = len(items)
    return out


def _model_pred_options():
    pred = state.get("predictions") or {}
    items = _model_pred_items()
    mode = state.get("threshold_mode") or pred.get("threshold_mode") or "pot"
    primary = primary_threshold_label(pred, source=state.get("pred_source"))
    if mode == "pot":
        mode_tag = primary["short"]
    elif mode == "best_f1_point":
        mode_tag = "best-F1·point"
    elif mode == "best_f1":
        mode_tag = "best-F1·PA"
    else:
        mode_tag = str(mode)
    # 드롭다운에는 짧은 이름만 표시하고 자세한 기여 feature 정보는 아래 패널에서 보여준다.
    # prediction segment가 수백 개여도 드롭다운이 너무 길어지지 않게 하기 위한 처리다.
    opts = []
    for x in items:
        split_tag = pred_chrono_split_tag(x, pred)
        line = pred_line(x, with_contrib=False, predictions=None)
        if split_tag:
            label = f"[{split_tag}] [{mode_tag}] {line}"
        else:
            label = f"[{mode_tag}] {line}"
        opts.append({"label": label, "value": x["id"]})
    return opts


def _pred_by_id(pred_id: str | None) -> dict | None:
    if not pred_id:
        return None
    for item in _model_pred_items():
        if item.get("id") == pred_id:
            return item
    # metrics split 변경 전에 선택했던 항목처럼 현재 필터 목록에 없으면 전체 prediction 목록에서 다시 찾는다.
    for item in (state.get("predictions") or {}).get("labels") or []:
        if item.get("id") == pred_id:
            return item
    return None


def _model_pred_source_options():
    plmn = state.get("plmn")
    if not plmn:
        return []
    return list_prediction_sources(plmn)


def _default_pred_source(options: list[dict] | None = None) -> str | None:
    opts = options if options is not None else _model_pred_source_options()
    values = [o["value"] for o in opts if o.get("value")]
    cur = state.get("pred_source")
    if cur and cur in values:
        return cur
    return values[0] if values else None


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


def _fmt_opt(v: Any, *, digits: int = 4) -> str:
    if isinstance(v, (int, float)) and v == v:
        return f"{float(v):.{digits}f}"
    return "—"


def _model_metrics_tag(pred: dict) -> tuple[str, str]:
    """모델별 detection metric 패널에 표시할 이름과 강조 색상을 결정한다."""
    model = str(pred.get("model") or pred.get("source") or "model").lower()
    src = str(pred.get("source") or "").lower()
    key = model if model not in ("", "none") else src
    if key.startswith("roca"):
        return "RoCA", "#0f766e"
    if key.startswith("lstm"):
        return "LSTM-AD", "#0369a1"
    if "anomalytransformer" in key or key.startswith("at"):
        return "AnomalyTransformer", "#7c3aed"
    # 표시 이름을 따로 찾지 못하면 source 파일 stem을 보기 좋게 변환해 사용한다.
    stem = (pred.get("model") or pred.get("source") or "Model")
    stem = str(stem).split("_")[0]
    return stem.upper() if len(stem) <= 6 else stem, "#334155"


def _threshold_evaluation_table(pred: dict) -> Any | None:
    block = pred.get("threshold_evaluation") or {}
    methods = list(block.get("methods") or [])
    if not methods:
        return None

    header = html.Tr(
        [
            html.Th("Method"),
            html.Th("Event-F1"),
            html.Th("Point-F1"),
            html.Th("Precision"),
            html.Th("Recall"),
            html.Th("FA/day"),
        ]
    )
    body = []
    for row in methods:
        body.append(
            html.Tr(
                [
                    html.Td(str(row.get("label") or row.get("key") or "—")),
                    html.Td(_fmt_opt(row.get("event_f1"))),
                    html.Td(_fmt_opt(row.get("f1"))),
                    html.Td(_fmt_opt(row.get("precision"))),
                    html.Td(_fmt_opt(row.get("recall"))),
                    html.Td(_fmt_opt(row.get("false_alarms_per_day"), digits=3)),
                ]
            )
        )

    return html.Div(
        [
            html.Div(
                "ContextAE threshold comparison (validation)",
                style={"fontWeight": "700", "marginBottom": "4px"},
            ),
            html.Div(
                "각 threshold는 context별 clean TRAIN reconstruction error에서 계산하며, "
                "validation label은 성능 평가에만 사용합니다.",
                style={"fontSize": "11px", "color": "#64748b", "marginBottom": "6px"},
            ),
            html.Table(
                [html.Thead(header), html.Tbody(body)],
                style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                    "fontFamily": "monospace",
                    "fontSize": "11px",
                },
            ),
        ],
        style={
            "marginTop": "10px",
            "paddingTop": "8px",
            "borderTop": "1px solid #cbd5e1",
        },
    )


def _detection_metrics_panel(pred: dict) -> list[Any]:
    """현재 선택된 모델의 validation/test detection metric을 사람이 읽기 쉬운 패널로 구성한다."""
    paper = pred.get("paper_metrics") or {}
    m = pred.get("metrics") or {}
    split = (
        paper.get("eval_split")
        or m.get("eval_split")
        or pred.get("metrics_eval_split")
        or pred.get("eval_split")
        or "valid"
    )
    vus_pr = paper.get("vus_pr", m.get("vus_pr"))
    vus_roc = paper.get("vus_roc", m.get("vus_roc"))
    r_auc_pr = paper.get("r_auc_pr", m.get("r_auc_pr"))
    auc_pr = paper.get("auc_pr", m.get("auc_pr"))
    vus_window = paper.get("vus_window", m.get("vus_window"))
    point = dict(paper.get("point") or {})
    if not any(point.get(k) is not None for k in ("precision", "recall", "f1", "TP", "FP")):
        point = {
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "TP": m.get("tp") if m.get("tp") is not None else m.get("TP"),
            "FP": m.get("fp") if m.get("fp") is not None else m.get("FP"),
            "FN": m.get("fn") if m.get("fn") is not None else m.get("FN"),
        }
    event = dict(paper.get("event") or {})
    if not any(event.get(k) is not None for k in ("f1", "precision", "recall", "n_gt_events")):
        event = {
            "f1": m.get("event_f1"),
            "precision": m.get("event_precision"),
            "recall": m.get("event_recall"),
            "n_gt_events": m.get("n_gt_events"),
            "n_pred_events": m.get("n_pred_events"),
            "n_hit_gt": m.get("n_hit_gt"),
        }
    op = dict(paper.get("operational") or {})
    if not any(
        op.get(k) is not None
        for k in ("false_alarms_per_day", "mean_detection_delay_min", "observation_days")
    ):
        op = {
            "false_alarms_per_day": m.get("false_alarms_per_day"),
            "n_false_alarm_events": m.get("n_false_alarm_events"),
            "observation_days": m.get("observation_days"),
            "mean_detection_delay_min": m.get("mean_detection_delay_min"),
            "median_detection_delay_min": m.get("median_detection_delay_min"),
        }
    tag, accent = _model_metrics_tag(pred)
    thr_method = (
        paper.get("threshold_method")
        or pred.get("threshold_method")
        or "percentile"
    )
    rows: list[Any] = [
        html.Div(
            f"{tag} metrics — VUS · Event-F1 · Point P/R/F1 · operational · {split}",
            style={
                "fontWeight": "700",
                "color": accent,
                "marginBottom": "6px",
            },
        ),
        html.Div(
            "VUS는 threshold-independent score ranking 평가이며, 아래 Event/Point/FA 지표는 현재 threshold 기준입니다.",
            style={"fontSize": "11px", "color": "#64748b", "marginBottom": "8px"},
        ),
        html.Div(
            f"VUS-PR = {_fmt_opt(vus_pr)}  "
            f"VUS-ROC = {_fmt_opt(vus_roc)}  "
            f"R-AUC-PR = {_fmt_opt(r_auc_pr)}  "
            f"AUC-PR = {_fmt_opt(auc_pr)}  "
            f"window = {vus_window if vus_window is not None else '—'}",
            style={
                "fontFamily": "monospace",
                "fontSize": "12px",
                "fontWeight": "600",
                "marginBottom": "6px",
                "color": "#1e293b",
            },
        ),
        html.Div(
            "Event-F1: "
            + _fmt_prf(
                event.get("precision"),
                event.get("recall"),
                event.get("f1"),
            )
            + f"  hit={event.get('n_hit_gt')} / gt={event.get('n_gt_events')}  "
            f"pred_events={event.get('n_pred_events')}",
            style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Div(
            "Point: "
            + _fmt_prf(
                point.get("precision"),
                point.get("recall"),
                point.get("f1"),
                tp=point.get("TP"),
                fp=point.get("FP"),
                fn=point.get("FN"),
            ),
            style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Div(
            f"FA/day = {_fmt_opt(op.get('false_alarms_per_day'), digits=3)}  "
            f"(FA events={op.get('n_false_alarm_events')} / "
            f"days={_fmt_opt(op.get('observation_days'), digits=2)})",
            style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "2px"},
        ),
        html.Div(
            f"Detection delay (min): mean={_fmt_opt(op.get('mean_detection_delay_min'), digits=2)}  "
            f"median={_fmt_opt(op.get('median_detection_delay_min'), digits=2)}",
            style={"fontFamily": "monospace", "fontSize": "12px", "marginBottom": "6px"},
        ),
        html.Div(
            f"예측 구간 {len(pred.get('labels') or [])}개 · thr={_fmt_opt(pred.get('threshold'))} "
            f"({thr_method})",
            style={"fontSize": "12px", "color": "#334155"},
        ),
    ]
    threshold_table = _threshold_evaluation_table(pred)
    if threshold_table is not None:
        rows.append(threshold_table)
    return rows


def _threshold_mode_options(pred: dict | None = None, source: str | None = None) -> list[dict]:
    """primary threshold와 분석용 best-F1 point threshold를 선택하는 radio 옵션을 만든다. PA-F1은 주요 UI에서 제외한다."""
    pred = pred if pred is not None else (state.get("predictions") or {})
    src = source if source is not None else state.get("pred_source")
    primary = primary_threshold_label(pred, source=src)
    return [
        {"label": primary["radio"], "value": "pot"},
        {"label": "best-F1 point (analysis only)", "value": "best_f1_point"},
    ]


def _threshold_mode_value_for_pred(
    pred: dict | None = None, source: str | None = None
) -> str:
    """현재 모델에서 선택되어야 할 threshold mode 값을 결정한다."""
    pred = pred if pred is not None else (state.get("predictions") or {})
    mode = state.get("threshold_mode") or pred.get("threshold_mode") or "pot"
    if mode in ("rule", "argos_rule", "best_f1"):
        return "pot"
    if mode in ("pot", "best_f1_point"):
        return mode
    return "pot"


def _detection_metrics_children() -> list[Any]:
    pred = state.get("predictions") or {}
    if not pred.get("labels") and not pred.get("thresholds") and not state.get("pred_source"):
        return [
            html.I(
                "모델 예측 파일이 없습니다. "
                "학습·export 후 다시 로드하세요."
            )
        ]
    return _detection_metrics_panel(pred)


def _metrics_split_options(pred: dict | None = None) -> list[dict]:
    splits = available_metrics_splits(
        pred if pred is not None else state.get("predictions")
    )
    return [{"label": s, "value": s} for s in splits]


def _model_attr_children(item: dict | None):
    if item is None:
        n = len(_model_pred_items())
        split_tag = _metrics_split_wanted()
        if n == 0:
            return html.I(
                f"모델 예측 파일이 없거나 {split_tag} 구간이 없습니다. "
                "`run_plmn.py`로 학습·export 후 다시 로드하세요."
            )
        return html.I(
            f"모델 예측 {n}개 ({split_tag}) — 보라색 구간/점을 클릭하거나 "
            "목록에서 고르면 이상으로 본 기여 metric(상위)이 표시됩니다."
        )

    pred_doc = state.get("predictions") or {}
    feat_mode = _ui_feature_mode_for_display(pred_doc)
    tops = item.get("top_metrics") or []
    if not tops:
        return html.Span(
            [
                html.B(
                    pred_line(
                        item,
                        feature_mode=feat_mode,
                        predictions=state.get("predictions"),
                    )
                ),
                html.Br(),
                html.I(
                    "기여 metric 정보가 없습니다. "
                    "(POT/percentile export 구간만 attribution이 포함됩니다. "
                    "best-F1 point·PA는 점수·threshold로 재구성한 구간입니다.)"
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
            "그래프·metric 필터는 comb와 동일한 원본 counter를 표시"
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
            html.Div(
                pred_line(
                    item,
                    feature_mode=feat_mode,
                    predictions=state.get("predictions"),
                ),
                style={"marginBottom": "4px"},
            ),
            html.Ul(rows, style={"margin": "0 0 4px 18px", "padding": 0}),
            html.Div(note, style={"fontSize": "11px", "color": "#64748b"}),
        ]
    )


def _plot_metrics() -> list[str]:
    """현재 metric universe 중 실제 그래프에 그려지고 있는 metric 목록을 반환한다."""
    universe = _metric_universe()
    visible = state.get("visible_metrics")
    if visible is None:
        return list(universe)
    selected = set(visible)
    return [m for m in universe if m in selected]


def _metrics_by_view_sum() -> list[str]:
    """현재 화면 구간의 합계를 기준으로 metric filter 후보를 정렬한다."""
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

    # 원본 metric은 현재 구간 합계가 큰 순서로 배치한다.
    # 현재 구간에서 0인 metric도 뒤에 남긴다.
    rest = list(universe)

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
    # metric universe의 모든 항목은 checklist에 유지하고 값이 0인 항목만 뒤로 보낸다.
    # 0인 항목을 숨기면 '전체 선택'이 일부 항목만 선택하는 것처럼 보여 혼란이 생긴다.
    return active + inactive


def _metric_filter_ui() -> tuple[list[dict], list[str]]:
    """metric universe 전체를 checklist options로 만들고 현재 선택값을 함께 반환한다. 확대 구간에서 값이 0이어도 options에서 제거하지 않는다."""
    ranked = _metrics_by_view_sum()
    selected = [m for m in (state.get("visible_metrics") or []) if m]
    keep = [m for m in selected if m not in ranked]
    ordered = ranked + keep
    opts = [{"label": display_metric(m), "value": m} for m in ordered]
    selected_set = set(selected)
    vals = [m for m in ordered if m in selected_set]
    return opts, vals


def _full_visible_metrics() -> list[str]:
    return list(state.get("metrics") or [])


def _pred_uses_feature_subset(preds: dict | None = None) -> bool:
    preds = preds if preds is not None else (state.get("predictions") or {})
    cols = list(preds.get("feature_columns") or [])
    mode = preds.get("feature_mode")
    source = str(preds.get("source") or state.get("pred_source") or "")
    # all/core/all_raw 표기는 현재 논문 feature universe를 뜻하며 지금은 core feature 집합으로 통일한다.
    return mode in ("comb", "comb_share", "selected") or (
        mode not in ("all", "core", "all_raw", None)
        and source != ""
        and bool(cols)
    )


def _comb_ui_metric_columns(preds: dict) -> list[str]:
    """comb/comb_share 계열에서 UI에 보여줄 원본 counter 목록을 반환한다."""
    raw = list(preds.get("comb_raw_counters") or [])
    if not raw:
        raw = [
            c
            for c in (preds.get("feature_columns") or [])
            if c
        ]
    cols: list[str] = []
    seen: set[str] = set()
    for c in raw:
        if c in seen:
            continue
        seen.add(c)
        cols.append(c)
    return cols


def _metric_universe_for_predictions(preds: dict | None = None) -> list[str]:
    """현재 모델 run의 feature 설정을 바탕으로 UI에서 선택 가능한 metric universe를 결정한다."""
    preds = preds if preds is not None else (state.get("predictions") or {})
    full = _full_visible_metrics()
    if not _pred_uses_feature_subset(preds):
        return full
    mode = preds.get("feature_mode")
    source = str(preds.get("source") or state.get("pred_source") or "")
    # comb_share 학습은 M971 대비 비율을 쓰지만 UI에서는 같은 원본 counter series 이름으로 보여준다.
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
    """현재 모델이 실제 사용한 feature만 metric filter에 남기고 처음에는 모두 보이도록 설정한다."""
    universe = _metric_universe_for_predictions(preds)
    state["metric_universe"] = universe
    state["visible_metrics"] = list(universe)
    state["metric_rev"] = int(state.get("metric_rev") or 0) + 1


def _visible_from_filter_selection(selected) -> list[str]:
    """checklist 선택값을 실제 그래프에 그릴 metric 목록으로 변환한다."""
    universe = _metric_universe()
    selected_set = set(selected or [])
    return [m for m in universe if m in selected_set]


def _label_by_id(label_id):
    for item in (state["doc"] or {}).get("labels", []):
        if item["id"] == label_id:
            return item
    return None


def _label_at_time(ts, *, point_tol_seconds: float | None = None):
    """주어진 시간에서 가장 적절한 전문가 label을 찾는다. 경계선을 우선하고 겹치면 더 좁은 range를 우선한다."""
    doc = state.get("doc") or {}
    return _interval_hit_at_time(
        doc.get("labels") or [],
        ts,
        point_tol_seconds=point_tol_seconds,
    )


def _pred_at_time(ts, *, point_tol_seconds: float | None = None):
    """주어진 시간에서 가장 적절한 model prediction segment를 찾는다. 전문가 label과 같은 hit 규칙을 사용한다."""
    return _interval_hit_at_time(
        _model_pred_items(),
        ts,
        point_tol_seconds=point_tol_seconds,
    )


def _interval_hit_at_time(
    items: list,
    ts,
    *,
    point_tol_seconds: float | None = None,
):
    """주어진 시간에 걸리는 point/range 항목 중 가장 적절한 하나를 선택한다."""
    zoom_start = state.get("zoom_start")
    zoom_end = state.get("zoom_end")
    if zoom_start is not None and zoom_end is not None:
        win_sec = max((zoom_end - zoom_start).total_seconds(), 1.0)
        edge_tol = max(15 * 60.0, win_sec * 0.012)
    else:
        edge_tol = 150.0
    if point_tol_seconds is not None:
        edge_tol = max(edge_tol, float(point_tol_seconds))

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
    """그래프에서 클릭한 전문가 label을 선택하거나 같은 항목을 다시 누르면 highlight를 해제한다. 구간 편집 모드에서는 edge drag가 끊기지 않도록 재클릭해도 선택을 유지한다."""
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
        _bump_hi_rev()
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
    _bump_hi_rev()
    # 지연되어 도착한 plotly_relayout 이벤트가 방금 만든 highlight를 되돌리지 못하게 막는다.
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
    """선택한 전문가 label 주변으로 x축을 확대하고 highlight를 표시한다."""
    start_ts = parse_time(item["start"])
    end_ts = parse_time(item["end"])
    kind = (item.get("kind") or "point").lower()
    is_point = kind == "point" or start_ts == end_ts
    # point label 선택 시 고정 폭으로 확대하되 이전보다 줌아웃 두 단계만큼 더 넓게 본다.
    # 즉 기본 폭에 (1/0.7)^9 배를 적용한다.
    # range label은 최소 30분 폭으로 label에 맞춘 뒤 (1/0.7)^7 배만큼 여유를 둔다.
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
    _bump_hi_rev()
    _reset_y()
    # 줌인/전체와 마찬가지로 layout uirevision을 갱신해야 Plotly.react가 새 x축과
    # _reset_y()에서 계산한 새 Y 자동 범위를 받아들인다. uirevision이 그대로면
    # 이전 metric/score y축 범위를 계속 유지할 수 있다.
    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["_force_ui_rev"] = time.time()
    _arm_select_guard()


def _zoom_to_model_pred_item(item: dict) -> None:
    """선택한 model prediction 주변으로 x축을 확대한다. 전문가 label highlight는 건드리지 않는다."""
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
    """TRAIN / VALID / TEST / 전체 구간 중 하나로 x축을 이동한다. 적용할 수 없으면 False를 반환한다."""
    df = state.get("df")
    if df is None or not len(df):
        return False
    if which == "full":
        state["zoom_start"], state["zoom_end"] = data_time_bounds(df)
        state["x_full_view"] = True
        state["label_range_anchor"] = None
        state.pop("_offscreen_cleared", None)
        _reset_y()
        state["x_gen"] = int(state.get("x_gen") or 0) + 1
        state["_force_ui_rev"] = time.time()
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
    # figure를 통째로 교체하는 동작은 새 uirevision이 필요하며 axis-cmd만 보내면 무시될 수 있다.
    state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["_force_ui_rev"] = time.time()
    return True


def _model_pred_ids() -> list[str]:
    return [str(x["id"]) for x in _model_pred_items() if x.get("id")]


def _neighbor_model_pred_id(current: str | None, delta: int) -> str | None:
    """현재 model prediction 기준으로 이전/다음 segment ID를 반환하고 범위를 벗어나지 않게 제한한다."""
    ids = _model_pred_ids()
    if not ids:
        return None
    if current is None or str(current) not in ids:
        return ids[0] if delta >= 0 else ids[-1]
    i = ids.index(str(current))
    j = max(0, min(len(ids) - 1, i + int(delta)))
    return ids[j]


def _snap_inspect_to_item(item: dict):
    """선택한 anomaly 시작시간에 값 탐색 cursor를 맞추고 해당 row와 정보 패널을 반환한다."""
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
    """range label 배치용 임시 overlay가 바뀔 때 Plotly revision 값을 증가시킨다."""
    state["place_rev"] = int(state.get("place_rev") or 0) + 1


def _bump_hi_rev() -> None:
    """전문가/model highlight overlay가 바뀔 때 Plotly revision 값을 증가시킨다."""
    state["hi_rev"] = int(state.get("hi_rev") or 0) + 1


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
    """선택된 label의 fill/edge/highlight shape 인덱스를 모아 실시간 edge drag에 사용한다."""
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
    """현재 prediction에 anomaly score series가 있는지 확인한다."""
    preds = state.get("predictions") or {}
    ss = preds.get("score_series") or {}
    return bool(ss.get("times") and ss.get("scores"))


def _score_subplot_active() -> bool:
    """score subplot은 항상 예약되어 있으므로 현재 figure에서 score 행을 사용한다고 본다."""
    return True


def _metric_axis_row() -> int:
    """원본 metric 패널의 행 번호를 반환한다."""
    return 2


def _label_axis_row() -> int:
    """전문가 label track의 행 번호를 반환한다."""
    return 3


def _model_label_axis_row() -> int:
    """모델 prediction track의 행 번호를 반환한다."""
    return 4


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
        show_labels=True,
        highlight_id=state.get("highlight_id"),
        predictions=_predictions_for_display(),
        show_predictions=True,
        highlight_pred_id=state.get("selected_pred_id"),
    )
    # 줌/이동 버튼 뒤 이전 WebGL buffer나 축 범위가 남지 않도록 revision을 한 번 더 올린다.
    # 서버가 새 figure를 그릴 때 브라우저가 예전 화면을 재사용하지 못하게 하는 목적이다.
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
        # figure 교체 시 새 x축과 Y 자동 범위를 확실히 적용한다.
        state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["zoom_rendered"] = rendered
    # 일반 줌 동안에는 layout.uirevision을 안정적으로 유지하고 metric filter 변경 때만 metric_rev를 올린다.
    # 그래야 Scattergl이 필터에서 제거된 series를 실제로 버린다.
    # 그렇지 않으면 전체 해제 후에도 사라져야 할 WebGL trace가 잔상처럼 남을 수 있다.
    metric_rev = int(state.get("metric_rev") or 0)
    label_rev = int(state.get("label_rev") or 0)
    place_rev = int(state.get("place_rev") or 0)
    cursor_rev = int(state.get("cursor_rev") or 0)
    hi_rev = int(state.get("hi_rev") or 0)
    x_gen = int(state.get("x_gen") or 0)
    force_ui = state.pop("_force_ui_rev", None)
    force_bit = f":f{force_ui}" if force_ui is not None else ""
    fig.update_layout(
        datarevision=(
            f"{fig.layout.datarevision}:{state['fig_gen']}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}:c{cursor_rev}"
            f":h{hi_rev}:x{x_gen}{force_bit}"
        ),
        # figure 재생성 시 새 시간 범위를 받아들이도록 x_gen을 uirevision 구성에 포함한다.
        # '전체' 동작은 한 번의 강제 UI reset을 위해 _force_ui_rev도 설정한다.
        # click/list highlight 변경도 hi_rev를 바꿔야 Plotly가 이전 선택 shape를
        # 그대로 남겨두지 않는다.
        uirevision=(
            f"{state.get('plmn') or 'labeling'}"
            f":m{metric_rev}:l{label_rev}:p{place_rev}:c{cursor_rev}"
            f":h{hi_rev}:x{x_gen}{force_bit}"
        ),
        # figure에는 metric, score, 아래쪽 전문가/모델 label track이 포함된다.
        height=FIGURE_HEIGHT,
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
    # 마우스 drag zoom/pan은 시간축만 움직이도록 y축은 항상 잠근다. 세로 scale은
    # Y+ / Y- / Y 자동 버튼으로만 조절하고 score 패널은
    # 현재 보이는 x 구간에 맞춰 자동으로 y축을 맞춘다.
    y_locked = True
    y_bounds = _effective_y_bounds()
    # score 패널은 항상 자리를 확보하며 score가 없으면 y 범위가 None일 수 있다.
    score_bounds = _visible_score_y_bounds() if _has_score_panel() else None
    state["y_rendered"] = y_bounds
    state["score_y_rendered"] = score_bounds
    metric_row = _metric_axis_row()
    # Y 자동/zoom 시 score 패널도 다시 그리도록 score 범위를 y_gen revision에 포함한다.
    score_rev = f"ys:{state.get('y_gen', 0)}"
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
    fig.update_yaxes(
        title_text="human",
        range=[0, 1],
        autorange=False,
        fixedrange=True,
        showticklabels=False,
        uirevision="human-label-track",
        row=_label_axis_row(),
        col=1,
    )
    fig.update_yaxes(
        title_text="model",
        range=[0, 1],
        autorange=False,
        fixedrange=True,
        showticklabels=False,
        uirevision="model-label-track",
        row=_model_label_axis_row(),
        col=1,
    )

    if state["label_range_anchor"] is not None:
        before_shapes = len(fig.layout.shapes or ())
        lab_row = _label_axis_row()
        fig.add_shape(
            **pending_range_fill_shape(state["label_range_anchor"], row=lab_row)
        )
        state["_pending_fill_index"] = before_shapes
        fig.add_shape(
            **pending_anchor_shape(state["label_range_anchor"], row=lab_row)
        )
        # label 배치 중에는 그래프 위 annotation tooltip을 띄우지 않는다. series를 가리기 때문이다.
    else:
        state["_pending_fill_index"] = None

    if state.get("highlight_id"):
        item = _label_by_id(state["highlight_id"])
        if item is not None:
            before = len(fig.layout.shapes or ())
            # label edge drag는 custom JS가 직접 hit-test하므로 Plotly 기본 shape edit 기능은 끈다.
            hs, hn = label_highlight_overlays(
                item,
                editable=False,
                line_width=4 if click_mode == "edit_range" else 1,
                row=_label_axis_row(),
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

    # pan/zoom 뒤 서버가 계산한 x축과 고해상도 trace를 Plotly가 확실히 적용하도록 한다.
    # 여기서는 축 range만 설정하고 각 축별 uirevision은 따로 주지 않는다.
    # 축별 uirevision을 주면 subplot의 matches='x'가 깨져 '전체'가 이전 범위에 고정될 수 있다.
    # 대신 layout.uirevision에 x_gen을 넣어 전체 figure 수준에서 상태를 관리한다.
    fig.update_xaxes(
        range=[x0_plot, x1_plot],
        autorange=False,
        # label 배치용 custom spike가 점선을 직접 그리므로 Plotly 기본 spike는 끈다.
        # 그렇지 않으면 zoom/pan/inspect에서 세로선이 두 개 겹쳐 보인다.
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

    # 브라우저 쪽 pan 제한과 label edge drag를 위해 edge index와 데이터 시간 범위를 meta에 넣는다.
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
        # sample_ms는 전체 PLMN 길이라 크므로 사업자 로드 때 한 번만 보내고 browser cache에서 재사용한다.
        # pan/zoom 때는 react merge로 유지한다. score 배열은 더 작지만 score 패널이 켜져 있으면 반드시
        # meta에 계속 포함해야 한다. 그렇지 않으면 이후 Plotly.react가
        # 값을 지워 hover tip이 score를 읽기 전에 layout.meta가 비어버릴 수 있다.
        # 그 경우 실제 VALID score가 있어도 '(없음)'으로 표시된다.
        # range 편집 모드에서는 edge를 실제 5분 sample에 맞추기 위해 sample_ms를 항상 전달한다.
        meta["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0)
        # clientside tooltip의 score 텍스트는 실제 score subplot이 있을 때만 표시한다.
        meta["score_panel"] = True
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
            meta["score_polarity"] = str(
                preds.get("score_polarity") or "higher_is_anomaly"
            )
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
    """임의의 클릭 시간을 가장 가까운 실제 5분 sample 시간에 맞춘다."""
    df = state["df"]
    if df is None or not len(df):
        return ts
    ts = pd.to_datetime(ts, utc=True)
    pos = int((df["time"] - ts).abs().to_numpy().argmin())
    return df.iloc[pos]["time"]


def _cancel_pending_range(click_mode: str):
    """진행 중인 range label 시작점을 취소하고 임시 overlay를 지운다."""
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
    """그래프 시간 클릭을 point/range 전문가 label 배치 동작으로 처리한다."""
    if click_mode == "label_point":
        state["label_range_anchor"] = None
        ts = _snap_to_data_time(ts)
        # 기존 label의 원인 입력은 그 label에 저장하고 새 label을 만들 때는 빈 원인으로 시작한다.
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


def _apply_label_edge_from_drag(payload) -> str | None:
    """가로 방향 edge drag 이벤트를 선택된 전문가 label의 시작/끝 수정으로 반영한다."""
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
    metrics = _plot_metrics()
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
    if not (ymin == ymin and ymax == ymax):  # NaN 여부 확인
        return None
    # count metric은 기본적으로 0 이상이므로 데이터가 음수가 아닌 한 y축 하단을 0에 고정한다.
    base = min(0.0, ymin)
    if ymax <= base:
        return base, base + 1.0
    return base, ymax + (ymax - base) * 0.05


def _visible_score_y_bounds() -> tuple[float, float] | None:
    """현재 보이는 x 구간에서 anomaly score subplot에 적절한 y축 범위를 계산한다."""
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


def _effective_y_bounds() -> tuple[float, float] | None:
    """Y 자동이면 현재 화면에 맞춘 범위를, 수동 고정이면 사용자가 고정한 y축 범위를 반환한다."""
    if state.get("y_auto") or state.get("y_min") is None or state.get("y_max") is None:
        return _visible_y_bounds()
    return float(state["y_min"]), float(state["y_max"])


def _scale_y(factor: float) -> bool:
    """y축 하단은 고정하고 상단만 조절한다. Y+는 범위를 줄이고 Y-는 넓힌다."""
    if state["df"] is None:
        return False
    bounds = _effective_y_bounds()
    if bounds is None:
        return False
    lo, hi = float(bounds[0]), float(bounds[1])
    # 사용자가 고정한 y 하단값은 유지하고, 자동 모드에서 처음 수동 전환할 때 음수가 아니면 0으로 맞춘다.
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
    # 새 값 탐색 cursor shape가 반영되도록 revision을 갱신한다.
    state["cursor_rev"] = int(state.get("cursor_rev") or 0) + 1
    row = df.iloc[pos]
    panel = _hover_panel_at_row(row)
    return row, panel


def _hover_panel_at_row(row: pd.Series):
    """현재 row의 보이는 metric 중 값이 0이 아닌 항목을 정렬해 '이 시점 특성값' 패널을 만든다."""
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
        higher = True
        pol = str(preds.get("score_polarity") or "")
        if pol in ("lower_is_anomaly", "low", "lt"):
            higher = False
        anomalous = (score > thr) if higher else (score < thr)
        flag = " · anomaly" if thr is not None and anomalous else ""
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
        ranked_hover_html(row["time"], row, cols, max_rows=2),
        dangerously_allow_html=True,
    )
    if score_block is None:
        return body
    return html.Div([score_block, body])


def _refresh_hover_panel():
    """마지막 hover/inspect sample을 기준으로 아래쪽 값 패널을 다시 만든다."""
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
    """x축 변경 전에 현재 y축을 고정해 다음 figure 재생성에서 자동 fitting이 덮어쓰지 않게 한다."""
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
    """현재 시간창을 좌우로 이동하되 데이터 시작/끝을 넘지 않게 한다."""
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
    """현재 x축 중심을 기준으로 시간창을 확대 또는 축소한다."""
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
    # 지나치게 확대되어 sample이 1~2개만 보이지 않도록 최소 표시 point 수를 보장한다.
    tmin, tmax = data_time_bounds(state["df"])
    min_half = pd.Timedelta(minutes=15)
    if half < min_half and factor < 1:
        half = min_half
    new_start, new_end = _clamp_zoom(mid - half, mid + half)

    if factor > 1:
        # 이미 거의 전체 범위라면 정확한 전체 범위로 한 번 맞춰 버튼이
        # 눌렸을 때 아무 반응이 없는 것처럼 보이지 않게 한다.
        if _same_time_window((start_ts, end_ts), (tmin, tmax)):
            exact = start_ts == tmin and end_ts == tmax and bool(
                state.get("x_full_view", True)
            )
            if exact:
                return False
            new_start, new_end = tmin, tmax
        elif _same_time_window((new_start, new_end), (start_ts, end_ts)):
            # 데이터 경계 clamp 때문에 폭이 변하지 않았다면 전체 보기로 이동한다.
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
    """왼쪽 경계는 고정한 채 시간창 폭만 줄인다."""
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
    """전문가 label 시간 구간이 현재 x축 [z0,z1]과 겹치는지 확인한다."""
    try:
        a = parse_time(item["start"])
        b = parse_time(item.get("end") or item["start"])
    except (ValueError, TypeError, KeyError):
        return False
    lo, hi = (a, b) if a <= b else (b, a)
    return lo <= z1 and hi >= z0


def _clear_highlight_if_offscreen() -> bool:
    """현재 선택 label이 화면 밖으로 나가면 highlight 선택을 해제한다."""
    hid = state.get("highlight_id")
    if hid is None or state.get("df") is None:
        return False
    item = _label_by_id(hid)
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if item is None or z0 is None or z1 is None:
        state["highlight_id"] = None
        state["_offscreen_cleared"] = True
        _bump_hi_rev()
        return True
    if _label_overlaps_x_window(item, z0, z1):
        return False
    state["highlight_id"] = None
    state["_offscreen_cleared"] = True
    _bump_hi_rev()
    return True


def _is_full_x_view() -> bool:
    """현재 시간창이 사실상 전체 데이터 범위인지 판별한다. 명시적 flag와 실제 범위를 함께 본다."""
    if bool(state.get("x_full_view", True)):
        return True
    if state.get("df") is None:
        return True
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if z0 is None or z1 is None:
        return True
    return _same_time_window((z0, z1), data_time_bounds(state["df"]))


def _note_x_window_changed() -> None:
    """zoom_start/end 변경 뒤 x_full_view 상태를 다시 계산한다."""
    if state.get("df") is None:
        state["x_full_view"] = True
        return
    z0, z1 = state.get("zoom_start"), state.get("zoom_end")
    if z0 is None or z1 is None:
        state["x_full_view"] = True
        return
    state["x_full_view"] = _same_time_window((z0, z1), data_time_bounds(state["df"]))


def _arm_zoom_guard(seconds: float = 2.0) -> None:
    """서버가 만든 zoom/pan 직후 지연된 브라우저 relayout echo를 잠깐 무시하도록 guard를 설정한다."""
    state["zoom_guard_until"] = time.monotonic() + seconds


def _arm_select_guard(seconds: float = 2.0) -> None:
    """label 선택/자동 zoom 직후 늦게 도착한 relayout이 highlight를 지우지 못하도록 guard를 설정한다."""
    until = time.monotonic() + seconds
    state["select_guard_until"] = until
    state["zoom_guard_until"] = until


def _arm_place_click_guard(seconds: float = 0.3) -> None:
    """label 배치 직후 뒤따라오는 중복 clickData echo를 잠깐 무시한다."""
    state["place_click_guard_until"] = time.monotonic() + seconds


def _place_click_guarded() -> bool:
    return time.monotonic() < float(state.get("place_click_guard_until") or 0)


def _make_axis_cmd(
    status: str = "",
    *,
    apply_x: bool = True,
    touch_y: bool = True,
    live_axes: bool = True,
    bump_x_gen: bool = True,
    guard_seconds: float = 4.0,
    rebuild_ms: int = 300,
) -> dict:
    """브라우저 축을 즉시 움직이고 이후 고해상도 데이터를 재생성하기 위한 axis command payload를 만든다."""
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
        # figure 재생성 시 새 x축을 받아들이도록 revision을 올린다. 안정된 layout
        # uirevision을 그대로 두면 이전 zoom을 복원해 '전체' 버튼을 두 번 눌러야 하는 문제가 생긴다.
        # pan에서는 x_gen을 유지해야 재생성 때 카메라가 이전 위치로 튀지 않는다.
        if bump_x_gen:
            state["x_gen"] = int(state.get("x_gen") or 0) + 1
    state["zoom_rendered"] = rendered
    _arm_zoom_guard(guard_seconds)
    score_payload = None
    if touch_y:
        yb = _effective_y_bounds()
        state["y_rendered"] = yb
        y_payload = [float(yb[0]), float(yb[1])] if yb else None
        if _score_subplot_active():
            sb = _visible_score_y_bounds()
            state["score_y_rendered"] = sb
            score_payload = [float(sb[0]), float(sb[1])] if sb else None
    else:
        # 현재 고정/렌더링된 y축을 유지하고 새 y 범위를 브라우저에 밀어넣지 않는다.
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
        # drag zoom/pan은 브라우저가 이미 x축을 바꿨으므로 서버가 x축을 다시 적용하면 사용자 gesture와 충돌한다.
        "x": [str(to_plot_time(x0)), str(to_plot_time(x1))] if apply_x else None,
        "y": y_payload,
        "y_score": score_payload,
        "touch_y": bool(touch_y),
        "live_axes": bool(live_axes),
        "status": status,
        "rebuild_ms": int(rebuild_ms),
    }


def _relayout_x_window(relayout) -> tuple[Any, Any] | str | None:
    """Plotly relayout payload에서 shared x축 범위를 추출한다. 여러 subplot 중 어느 xaxis key가 와도 같은 시간창으로 해석한다."""
    if not relayout:
        return None
    if (
        relayout.get("xaxis.autorange")
        or relayout.get("xaxis2.autorange")
        or relayout.get("xaxis3.autorange")
        or relayout.get("xaxis4.autorange")
    ):
        return "autorange"
    for prefix in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
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
            # Plotly 더블클릭으로 전체 보기로 돌아간 경우 서버 상태도 브라우저 전체 범위와 맞춘다.
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
        # 서버가 방금 그린 x 범위가 relayout echo로 다시 들어오면 중복 처리하지 않는다.
        # 이전 범위가 늦게 재생되는 경우 zoom_guard 동안만 무시하고, guard가 끝난 뒤에는
        # 사용자가 실제로 이전 구간으로 drag한 것으로 보고 Y 자동도 다시 계산한다.
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
            # 버튼 zoom 직후 자주 들어오는 이전 x 범위의 지연 echo다.
            # guard가 끝난 뒤 같은 범위가 다시 오면 실제 사용자 이동으로 처리해
            # Y 자동 fitting을 다시 적용한다.
            if time.monotonic() < float(state.get("zoom_guard_until") or 0):
                return False
            # guard 이후의 이전 범위 요청은 사용자가 의도적으로 다시 이동한 것으로 본다.
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
        # drag/box zoom으로 선택한 label이 화면 밖으로 나가면 highlight 선택을 해제한다.
        new_w = (incoming[1] - incoming[0]).total_seconds()
        if prev_w is not None and new_w < prev_w * 0.98:
            _clear_highlight_if_offscreen()
        return True

    if relayout.get("yaxis.autorange") or relayout.get("yaxis2.autorange"):
        _reset_y()
        return True
    return False


def _force_adopt_browser_x(view_range) -> bool:
    """유효한 브라우저 x축 범위를 조건 없이 서버 상태로 채택한다. drag-zoom 직후 stale 서버 상태를 덮어쓸 때 사용한다."""
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
    """브라우저가 서버보다 더 좁은 구간을 보고 있으면 실제 zoom으로 판단해 브라우저 x축을 채택한다."""
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
    # 브라우저가 서버보다 명확히 더 좁은 구간을 보여주면 실제 사용자 zoom으로 보고 브라우저 상태를 신뢰한다.
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
    # 사업자가 정해져야 해당 PLMN의 prediction 파일 목록을 알 수 있으므로 그 뒤 source를 선택한다.
    state["plmn"] = plmn
    src_opts = list_prediction_sources(plmn)
    pred_source = _default_pred_source(src_opts)
    preds = load_predictions(plmn, source=pred_source or "")
    df = ensure_score_column(df, preds, source=pred_source or "", plmn=plmn)
    metrics = metric_columns(df)
    tmin, tmax = data_time_bounds(df)
    mode = state.get("threshold_mode") or "pot"
    msplit = state.get("metrics_eval_split") or "valid"
    preds = enrich_predictions_for_ui(
        df,
        preds,
        doc.get("labels") or [],
        threshold_mode=mode,
        metrics_split=msplit,
    )
    state.update(
        df=df,
        doc=doc,
        predictions=preds,
        pred_source=pred_source,
        threshold_mode=preds.get("threshold_mode") or mode,
        metrics_eval_split=preds.get("metrics_eval_split") or msplit,
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
        # Plotly 축과 동일한 KST-naive 시간을 epoch millisecond로 저장한다.
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


# 첫 화면부터 데이터가 보이도록 1순위 사업자를 미리 로드한다. 초기 빈 화면 깜빡임을 줄이기 위한 처리다.
_START_PLMN = plmn_ids[0] if plmn_ids else None
_START_LABEL_OPTS: list = []
_START_LABEL_VALUE = None
_START_FIGURE = None
_START_METRIC_OPTS: list = []
_START_METRIC_VALS: list = []
_START_MODEL_PRED_OPTS: list = []
_START_MODEL_SRC_OPTS: list = []
_START_MODEL_SRC_VALUE = None
_START_THR_OPTS: list = [
    {"label": "POT (논문 primary)", "value": "pot"},
    {"label": "best-F1 point (oracle)", "value": "best_f1_point"},
    {"label": "best-F1 PA (oracle)", "value": "best_f1"},
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
                    clearable=True,
                    searchable=False,
                    placeholder="예측 파일 없음 — 학습 후 export",
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
                html.B("Metrics", style={"marginLeft": "12px", "marginRight": "6px"}),
                dcc.RadioItems(
                    id="metrics-eval-split",
                    options=[{"label": "valid", "value": "valid"}],
                    value="valid",
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
            style={"width": "100%", "height": f"{FIGURE_HEIGHT}px"},
        ),
        html.Div(id="status", style={"display": "none"}),
        html.B("이 시점 특성값 (값 내림차순)"),
        html.Div(
            id="hover-panel",
            children=html.I(
                "그래프에 커서를 올리면 이 시점의 특성값이 내림차순으로 표시됩니다."
            ),
            style={
                # hover 패널은 timestamp header와 두 줄의 metric grid로 구성한다.
                "height": "108px",
                "overflow": "hidden",
                "border": "1px solid #ddd",
                "padding": "6px 8px",
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
        # pan 중에는 graph.figure 자체를 교체하지 않고 데이터만 갱신해 카메라가 튀는 현상을 막는다.
        dcc.Store(id="pan-plot-apply"),
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
    preferred = _default_pred_source(opts)
    # refresh 때 현재 prediction source를 유지해 불필요한 source 변경 callback 연쇄가
    # metric filter 초기화와 빈 그래프로 이어지지 않게 한다.
    cur = state.get("pred_source")
    if cur and any(o.get("value") == cur for o in opts):
        return opts, cur
    return opts, preferred


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
    Output("threshold-mode", "value"),
    Output("metrics-eval-split", "options"),
    Output("metrics-eval-split", "value"),
    Output("metrics-eval-split", "style"),
    Input("model-pred-source", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _switch_model_pred_source(source, click_mode):
    plmn = state.get("plmn")
    if not plmn:
        return (no_update,) * 13
    source = source or None
    # startup sync처럼 같은 dropdown 값을 다시 쓰는 무의미한 reload는 건너뛴다.
    if (
        source == state.get("pred_source")
        and state.get("predictions") is not None
        and state.get("df") is not None
    ):
        return (no_update,) * 13
    state["pred_source"] = source
    preds = load_predictions(plmn, source=source or "")
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
        if mode in ("rule", "argos_rule"):
            mode = "pot"
        msplit = state.get("metrics_eval_split") or "valid"
        if msplit not in available_metrics_splits(preds):
            msplit = "valid"
        state["predictions"] = enrich_predictions_for_ui(
            state["df"],
            preds,
            human,
            threshold_mode=mode,
            metrics_split=msplit,
        )
        state["threshold_mode"] = (
            state["predictions"].get("threshold_mode") or mode
        )
        state["metrics_eval_split"] = (
            state["predictions"].get("metrics_eval_split") or msplit
        )
    else:
        state["predictions"] = preds
        state["threshold_mode"] = state.get("threshold_mode") or "pot"
    state["_score_hover"] = None
    state["_hover_arrays_pushed"] = False
    state["hover_cache_seq"] = int(state.get("hover_cache_seq") or 0) + 1
    _apply_pred_metric_visibility(state["predictions"])
    opts = _model_pred_options()
    m_opts, m_vals = _metric_filter_ui()
    thr_style = {"fontSize": "13px"}
    thr_opts = _threshold_mode_options(state.get("predictions"), source)
    thr_val = _threshold_mode_value_for_pred(state.get("predictions"), source)
    state["threshold_mode"] = thr_val
    split_opts = _metrics_split_options(state.get("predictions"))
    split_vals = {o["value"] for o in split_opts}
    split_val = state.get("metrics_eval_split") or "valid"
    if split_val not in split_vals:
        split_val = "valid"
    state["metrics_eval_split"] = split_val
    # 선택 가능한 metric split이 2개 미만이면 split 선택 UI를 비활성화된 회색 상태로 표시한다.
    split_style = (
        {"fontSize": "13px", "opacity": "0.35", "pointerEvents": "none"}
        if len(split_opts) < 2
        else {"fontSize": "13px"}
    )
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
        thr_val,
        split_opts,
        split_val,
        split_style,
    )


@app.callback(
    Output("metrics-eval-split", "options", allow_duplicate=True),
    Output("metrics-eval-split", "value", allow_duplicate=True),
    Output("metrics-eval-split", "style", allow_duplicate=True),
    Input("dd-plmn", "value"),
    prevent_initial_call=True,
)
def _sync_metrics_split_on_plmn(_plmn):
    pred = state.get("predictions") or {}
    opts = _metrics_split_options(pred)
    vals = {o["value"] for o in opts}
    cur = state.get("metrics_eval_split") or "valid"
    if cur not in vals:
        cur = "valid"
    state["metrics_eval_split"] = cur
    style = (
        {"fontSize": "13px", "opacity": "0.35", "pointerEvents": "none"}
        if len(opts) < 2
        else {"fontSize": "13px"}
    )
    return opts, cur, style


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
    preds = state.get("predictions") or {}
    mode = mode or "pot"
    if mode in ("rule", "argos_rule"):
        mode = "pot"
    state["threshold_mode"] = mode
    df = state.get("df")
    if df is None or not preds:
        return (no_update,) * 5
    msplit = state.get("metrics_eval_split") or "valid"
    if preds.get("labels_by_mode") and preds.get("thresholds"):
        state["predictions"] = apply_threshold_mode(
            df, preds, threshold_mode=mode
        )
        # threshold mode가 바뀌면 현재 metrics_split에 대한 metric을 다시 계산한다.
        human = ((state.get("doc") or {}).get("labels")) or []
        state["predictions"] = enrich_predictions_for_ui(
            df,
            state["predictions"],
            human,
            threshold_mode=mode,
            metrics_split=msplit,
        )
    else:
        human = ((state.get("doc") or {}).get("labels")) or []
        state["predictions"] = enrich_predictions_for_ui(
            df,
            preds,
            human,
            threshold_mode=mode,
            metrics_split=msplit,
        )
    state["threshold_mode"] = (
        state["predictions"].get("threshold_mode") or mode
    )
    state["metrics_eval_split"] = (
        state["predictions"].get("metrics_eval_split") or msplit
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
    Output("graph", "figure", allow_duplicate=True),
    Output("model-pred-list", "options", allow_duplicate=True),
    Output("model-pred-list", "value", allow_duplicate=True),
    Output("model-pred-attr", "children", allow_duplicate=True),
    Output("detection-metrics", "children", allow_duplicate=True),
    Input("metrics-eval-split", "value"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _switch_metrics_eval_split(msplit, click_mode):
    msplit = msplit or "valid"
    state["metrics_eval_split"] = msplit
    df = state.get("df")
    preds = state.get("predictions")
    if df is None or not preds:
        return (no_update,) * 5
    human = ((state.get("doc") or {}).get("labels")) or []
    mode = state.get("threshold_mode") or "pot"
    state["predictions"] = enrich_predictions_for_ui(
        df,
        preds,
        human,
        threshold_mode=mode,
        metrics_split=msplit,
    )
    state["metrics_eval_split"] = (
        state["predictions"].get("metrics_eval_split") or msplit
    )
    state["selected_pred_id"] = None
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
    """model prediction 목록 선택을 해당 segment zoom과 기여정보 표시로 연결한다."""
    prev = state.get("selected_pred_id")
    state["selected_pred_id"] = pred_id
    if prev != pred_id:
        _bump_hi_rev()
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
        f"{pred_line(item, feature_mode=_ui_feature_mode_for_display(), predictions=state.get('predictions'))}"
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
    """그래프의 보라색 model prediction을 클릭해 선택한다. 같은 위치에 전문가 label이 있으면 전문가 선택을 우선한다."""
    click_mode = click_mode or "zoom"
    if click_mode in ("label_range", "label_point", "edit_range"):
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

    # 전문가 label track click은 _main_body가 처리하고 모델 track click은 model prediction 선택에 사용한다.
    track = None
    if prop == "shape-click-event.data" and shape_click:
        track = shape_click.get("track")
    if track == "human":
        return no_update
    if track != "model":
        # 전문가/모델이 겹치거나 metric 영역 click이 애매하면 전문가 label 선택을 우선한다.
        if _label_at_time(ts) is not None:
            return no_update

    hit = _pred_at_time(ts)
    if hit is None:
        return no_update
    hit_id = hit.get("id")
    if not hit_id:
        return no_update
    # 이미 선택한 model segment를 다시 누르면 선택을 해제한다.
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
        f"{pred_line(item, feature_mode=_ui_feature_mode_for_display(), predictions=state.get('predictions'))}"
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
        # label 목록 변경 시 원인 입력 저장/불러오기는 _sync_label_reasons가 전담한다.
        # 새 label이 현재 입력한 원인 값을 이어받도록 draft와 UI 상태를 동기화한다.
        codes = list(reason_codes or [])
        mets = list(fail_metrics or []) if "fail_surge" in codes else []
        _store_draft_reasons(codes, mets, reason_note)
    result = _main_body(
        plmn,
        n_prev,
        n_next,
        _mode_change,
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
    # metric filter option 목록은 명시적인 PLMN 로드 때만 다시 만든다.
    # zoom/pan/overlay 재생성 때 options까지 다시 쓰면 checklist value와 callback race가 생겨
    # 모든 trace가 사라질 수 있다.
    prop = ""
    if callback_context.triggered:
        prop = callback_context.triggered[0].get("prop_id") or ""
    refresh_filter = prop in (
        "dd-plmn.value",
        "btn-prev.n_clicks",
        "btn-next.n_clicks",
    )
    if (
        refresh_filter
        and out[0] is not no_update
        and state.get("df") is not None
        and out[7] is no_update
    ):
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
        # axis-cmd와 충돌할 수 있는 불필요한 click-mode 재설정은 하지 않는다.
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
        # 구간 편집 모드에서 다른 label을 선택해도 편집 모드를 유지한다.
        if (click_mode or "") == "edit_range":
            return no_update
        # 메인 동작과 동일하게 전체 보기가 아니라 이미 확대된 상태에서만 자동 zoom한다.
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
    """원인 입력 UI와 선택된 label 또는 새 label 초안 상태를 서로 동기화한다."""
    tid = getattr(callback_context, "triggered_id", None)
    fail_opts = _fail_metric_options()
    list_opts = no_update

    if tid in ("label-list", "dd-plmn", None):
        # 다른 label로 이동하기 전에 현재 note/checklist 입력을 이전 label에 저장한다.
        # note는 저장 전까지 textarea에만 존재하므로 반드시 먼저 flush해야 한다.
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
        # 선택된 label이 없으면 원인 UI도 비워 다음 새 label이 빈 상태에서 시작하게 한다.
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
    # 전문가 label과 모델 prediction track은 항상 서로 다른 subplot row에 그린다.
    state["show_anomalies"] = True
    state["show_model_preds"] = True

    # ----- 초기 실행 / dropdown 변경 / 이전 / 다음 버튼에서 사업자 로드 -----
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
            # 이전/다음 버튼이 dropdown 값을 바꾼 뒤 같은 값이 다시 callback으로 들어온 echo는 무시한다.
            return (no_update,) * 7
        if already:
            # startup prefetch에서 이미 1순위 사업자를 읽었으면 만들어둔 figure를 그대로 재사용한다.
            # 여기서 metric filter options/value를 다시 쓰지 않는다. Dash가 잠깐
            # checklist를 비우면 visible_metrics도 비어 그래프 전체가 사라질 수 있다.
            opts = _label_options()
            return (
                no_update,
                opts,
                no_update,
                "",
                _cancel_style(click_mode),
                no_update,
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
            "사업자를 선택하세요.",
            no_update,
            no_update,
            no_update,
        )

    # drag zoom/pan의 상태 갱신은 _drag_axis_nav가 담당한다. 메인 callback에서
    # 매 이벤트마다 browser view-range Store를 다시 적용하지 않는다. 이 Store는
    # list/button zoom보다 늦게 갱신될 수 있어 서버 상태를 전체 범위로 되돌리는
    # 문제가 생길 수 있다.

    # ----- 현재 click mode별 동작 분기 -----
    if prop == "click-mode.value":
        state["label_range_anchor"] = None
        state.pop("place_click_guard_until", None)
        if click_mode != "inspect":
            if state.get("value_cursor_pos") is not None:
                state["value_cursor_pos"] = None
                # 값 탐색 cursor shape가 사라지도록 revision을 갱신한다.
                state["cursor_rev"] = int(state.get("cursor_rev") or 0) + 1
        else:
            # 새 값 탐색 세션의 첫 click이 이전 debounce에 막히지 않도록 click guard를 초기화한다.
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
        _bump_hi_rev()
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
            # highlight만 지우고 zoom은 그대로 유지한다.
            state["highlight_id"] = None
            state["label_range_anchor"] = None
            state.pop("_offscreen_cleared", None)
            _bump_hi_rev()
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
            _bump_hi_rev()
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
        # 자동 zoom 여부는 label 선택 여부가 아니라 현재 화면이 전체/확대 상태인지로 판단한다.
        # 전체 보기에서는 선택만 하고, 이미 확대되어 있으면 새 항목 선택과 함께 해당 구간으로 zoom한다.
        # 단 구간 편집 중에는 값 탐색 모드로 강제로 바꾸지 않는다.
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
            _bump_hi_rev()
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
        # Plotly shape 자체는 click event를 주지 않으므로 실제 plot 영역 click으로 label hit를 판별한다.
        if state.get("df") is None or state.get("doc") is None:
            return (no_update,) * 7
        try:
            ts = parse_time(shape_click["x"])
        except (ValueError, TypeError):
            return (no_update,) * 7
        last_ts, last_at = state.get("_last_click", (None, 0.0))
        now = time.monotonic()
        # shape-click과 clickData가 같은 sample에서 중복 발생한 경우에만 두 번째 echo를 버린다.
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        state["_last_click"] = (ts, now)

        # range/point 전문가 label 배치 처리.
        if click_mode in ("label_range", "label_point"):
            if _place_click_guarded():
                return (no_update,) * 7
            placed = _place_label_click(ts, click_mode)
            return placed if placed is not None else (no_update,) * 7

        # 전문가 anomaly 선택을 우선하고 모델 track click은
        # _pick_model_pred_on_graph가 처리한다. 얇은 label strip은 hit tolerance를 조금 넓게 잡는다.
        track = shape_click.get("track")
        on_label_track = track in ("human", "model") or bool(
            shape_click.get("label_track")
        )
        if track == "model":
            return (no_update,) * 7
        hit = _label_at_time(
            ts,
            point_tol_seconds=(45 * 60.0) if on_label_track else None,
        )
        if hit is not None:
            return _select_or_toggle_label_hit(hit, click_mode)

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

        return (no_update,) * 7

    if prop == "graph.relayoutData":
        # drag zoom/pan은 Y 자동 fitting과 고해상도 재생성을 _drag_axis_nav 경로로 처리한다.
        # toolbar 버튼과 같은 axis-cmd 경로를 써서 별도의 전체 figure 재생성이
        # 사용자 drag gesture와 충돌하지 않게 한다.
        return (no_update,) * 7

    if prop == "graph.clickData" and click_data and click_data.get("points"):
        x = click_data["points"][0].get("x")
        if x is None:
            return (no_update,) * 7
        ts = parse_time(x)
        last_ts, last_at = state.get("_last_click", (None, 0.0))
        now = time.monotonic()
        # shape-click과 clickData가 같은 sample에서 중복 발생한 경우에만 두 번째 echo를 버린다.
        if (
            last_ts is not None
            and abs((ts - last_ts).total_seconds()) < 0.1
            and now - last_at < 0.35
        ):
            return (no_update,) * 7
        state["_last_click"] = (ts, now)

        # 값 탐색 모드에서도 anomaly 구간을 누르면 label 선택을 우선하고 빈 공간/trace click은 cursor 이동으로 처리한다.
        hit = _label_at_time(ts)
        if hit is not None and click_mode not in ("label_range", "label_point"):
            return _select_or_toggle_label_hit(hit, click_mode)

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

        # label_range/label_point 모드에서는 trace clickData의 정확한 Plotly x 시간을 사용한다.
        if click_mode in ("label_range", "label_point"):
            if _place_click_guarded():
                return (no_update,) * 7
            placed = _place_label_click(ts, click_mode)
            return placed if placed is not None else (no_update,) * 7

        return (no_update,) * 7

    return (no_update,) * 7


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("metric-filter", "options", allow_duplicate=True),
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
    """현재 metric universe 안에서 실제로 그릴 metric을 변경한다. 전체 선택은 Y 자동, 전체 해제는 기존 Y를 유지한다."""
    if state.get("df") is None:
        return no_update, no_update, no_update, no_update, no_update
    universe = _metric_universe()
    if not universe:
        return no_update, no_update, no_update, no_update, no_update
    tid = getattr(callback_context, "triggered_id", None)
    click_mode = click_mode or "zoom"

    def _bump_metric_rev() -> None:
        state["metric_rev"] = int(state.get("metric_rev") or 0) + 1

    def _filter_options(ordered: list[str]) -> list[dict]:
        return [{"label": display_metric(m), "value": m} for m in ordered]

    if tid == "btn-metrics-all":
        # 현재 화면에서 값이 0이 아닌 항목만이 아니라 전체 metric universe를 기준으로 checklist를 구성한다.
        ordered = _metrics_by_view_sum()
        # ranking 계산에서 빠진 항목이 있어도 universe의 모든 metric을 options에 반드시 포함한다.
        missing = [m for m in universe if m not in ordered]
        ordered = ordered + missing
        state["visible_metrics"] = list(universe)
        _bump_metric_rev()
        _reset_y()
        return (
            _build_graph(click_mode),
            _filter_options(ordered),
            list(universe),
            f"metric 전체 선택 ({len(universe)})",
            _refresh_hover_panel(),
        )
    if tid == "btn-metrics-none":
        # 전체 metric을 해제해 빈 그래프가 되어도 기존 Y scale을 유지해 화면이 튀지 않게 한다.
        _pin_y_before_x_change()
        state["y_gen"] = int(state.get("y_gen") or 0) + 1
        state["visible_metrics"] = []
        _bump_metric_rev()
        return (
            _build_graph(click_mode),
            no_update,
            [],
            "metric 전체 해제 (Y축 유지)",
            _refresh_hover_panel(),
        )
    # 다른 callback이 checklist options/value를 다시 쓰는 순간 Dash가 중간 상태로
    # 빈 value를 잠깐 보낼 수 있으므로 무시한다. 실제 전체 해제는 전용 버튼으로만 처리한다.
    if tid == "metric-filter" and not selected:
        if state.get("visible_metrics"):
            return no_update, no_update, no_update, no_update, no_update
    new_vis = _visible_from_filter_selection(selected)
    if new_vis == list(state.get("visible_metrics") or []):
        return no_update, no_update, no_update, no_update, no_update
    state["visible_metrics"] = new_vis
    _bump_metric_rev()
    _reset_y()
    return (
        _build_graph(click_mode),
        no_update,
        no_update,
        f"표시 metric {len(new_vis)}/{len(universe)}",
        _refresh_hover_panel(),
    )


@app.callback(
    Output("axis-cmd", "data"),
    Input("btn-pan-left", "n_clicks"),
    Input("btn-pan-right", "n_clicks"),
    Input("btn-y-in", "n_clicks"),
    Input("btn-y-out", "n_clicks"),
    Input("btn-y-auto", "n_clicks"),
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
    click_mode,
    view_range,
):
    """축 이동 명령의 공통 helper다. 상태를 갱신한 뒤 브라우저 relayout/figure 재생성 흐름을 제어한다."""
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
    return no_update


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("btn-zoom-train", "n_clicks"),
    Input("btn-zoom-valid", "n_clicks"),
    Input("btn-zoom-test", "n_clicks"),
    Input("btn-zoom-full-split", "n_clicks"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _zoom_chrono_hard(_n_train, _n_valid, _n_test, _n_full, click_mode):
    """TRAIN / VALID / TEST / 전체 split으로 이동할 때 figure를 직접 교체해 Plotly uirevision race를 피한다."""
    if state.get("df") is None:
        return no_update, no_update
    prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    which = {
        "btn-zoom-train.n_clicks": "train",
        "btn-zoom-valid.n_clicks": "valid",
        "btn-zoom-test.n_clicks": "test",
        "btn-zoom-full-split.n_clicks": "full",
    }.get(prop)
    if not which:
        return no_update, no_update
    if not _zoom_to_chrono_split(which):
        return no_update, f"{which} 구간을 찾을 수 없습니다."
    status = {
        "train": "Train 구간",
        "valid": "Valid 구간",
        "test": "Test 구간",
        "full": "전체 구간",
    }.get(which, which)
    _arm_zoom_guard(4.0)
    return _build_graph(click_mode or "zoom"), status


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("zoom-gesture", "data"),
    State("click-mode", "value"),
    prevent_initial_call=True,
)
def _zoom_x_hard(gesture, click_mode):
    """현재 브라우저 x축을 기준으로 줌인/왼쪽 줌인/줌아웃한 뒤 figure를 직접 교체한다."""
    if state.get("df") is None or not gesture:
        return no_update, no_update
    action = gesture.get("action")
    mode = click_mode or "zoom"
    pan_keep_y = mode == "pan_keep_y"
    # '전체' 이후 drag-zoom처럼 서버 상태가 늦을 수 있으므로 먼저 브라우저의 현재 live view를 동기화한다.
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
    """'전체' 동작에서 axis-cmd race를 피하기 위해 figure를 직접 전체 범위로 교체한다."""
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
    """drag-zoom 또는 drag-pan이 끝난 뒤 필요하면 Y 자동 fitting을 수행하고 고해상도 trace를 지연 재생성한다."""
    if not relayout or state["df"] is None:
        return no_update
    has_x = _relayout_x_window(relayout) is not None
    if not has_x:
        return no_update
    click_mode = click_mode or "zoom"
    # 이동(Y고정)만 현재 y축을 유지하고 zoom/이동(Y자동) 등은 Y 자동 fitting을 사용한다.
    y_auto = click_mode != "pan_keep_y"
    is_pan = click_mode in ("pan", "pan_keep_y")
    # label 선택/자동 zoom 직후에는 zoom_guard를 적용해 늦게 도착한 plotly_relayout이
    # 새 x축이나 빨간 highlight를 덮어쓰지 못하게 한다.
    # guard 시간이 지나면 실제 drag는 정상적으로 반영되도록 한다.
    protect = time.monotonic() < float(state.get("select_guard_until") or 0)
    if not _apply_axes_from_relayout(
        relayout, ignore_guard=not protect, y_auto=y_auto
    ):
        return no_update
    # pan 중에는 축을 실시간으로 다시 patch하지 않는다. 사용자 gesture와 충돌하기 때문이다.
    # x_gen도 유지해 지연된 데이터 재생성이 현재 카메라 위치를 초기화하지 않게 한다.
    # 연속 drag가 끝난 뒤에만 고해상도 데이터를 갱신하도록 짧은 idle debounce를 둔다.
    return _make_axis_cmd(
        "",
        apply_x=False,
        touch_y=y_auto,
        live_axes=False,
        bump_x_gen=not is_pan,
        guard_seconds=0.5,
        rebuild_ms=250 if is_pan else 300,
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
    # zoom/pan 재생성에서는 metric filter options를 다시 쓰지 않는다.
    # 좁은 구간 기준으로 options를 다시 만들면 값이 0인 series가 checklist에서 빠져
    # 그래프가 비어버릴 수 있다.
    # zoom으로 화면 밖 selection이 해제된 경우 dropdown도 맞추되,
    # 지연 재생성 전에 사용자가 다른 항목을 다시 선택했다면 그 선택을 덮어쓰지 않는다.
    cleared = state.pop("_offscreen_cleared", False)
    list_val = (
        None if (cleared and state.get("highlight_id") is None) else no_update
    )
    mode = click_mode or "zoom"
    # pan에서는 layout.uirevision을 안정적으로 유지해
    # 고해상도 trace를 갱신해도 카메라가 초기화되지 않게 한다.
    # pan 모드에서는 clientside가 meta.zoom_x나 Plots.resize를 다시 적용하면 안 된다.
    return _build_graph(mode), no_update, list_val


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

        // figure 전체 교체와 충돌할 수 있는 예약된 재생성 작업은 먼저 취소한다.
        if (window.__rebuildTimer) {
            clearTimeout(window.__rebuildTimer);
            window.__rebuildTimer = null;
        }
        window.__ignoreDataXClampUntil = Date.now() + 5000;

        // drag-zoom 직후 서버 상태가 늦을 수 있으므로 브라우저가 실제로 보여주는 현재 범위를 읽는다.
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
        var hasYScore = cmd.y_score && cmd.y_score.length === 2;
        var touchY = (cmd.touch_y !== false);
        // pan 종료 직후에는 실시간 축 patch를 하지 않는다. 재생성 전에 x축이 되돌아갈 수 있기 때문이다.
        var liveAxes = (cmd.live_axes !== false);
        // pan 고해상도 갱신처럼 재생성만 필요한 명령은 항상 허용한다.
        // Y 범위가 없다는 이유로 조기 종료하면 pan 정밀 갱신 전체가 빠질 수 있다.
        var host = document.getElementById('graph');
        var gd = host && host.querySelector('.js-plotly-plot');
        if (!gd || !window.Plotly) {
            return window.dash_clientside.no_update;
        }
        if (hasX) {
            window.__ignoreDataXClampUntil = Date.now() + 4000;
        }
        var patch = {};
        if (liveAxes && hasX) {
            patch['xaxis.autorange'] = false;
            patch['xaxis.range'] = [cmd.x[0], cmd.x[1]];
            // score / metric / 전문가 / 모델 패널은 하나의 시간축을 공유한다.
            ['xaxis2', 'xaxis3', 'xaxis4'].forEach(function(ax) {
                patch[ax + '.autorange'] = false;
                patch[ax + '.range'] = [cmd.x[0], cmd.x[1]];
            });
        }
        if (liveAxes && touchY) {
            // 고정 4행 구성: score, metrics, 전문가 label, 모델 prediction.
            var scoreKey = 'yaxis';
            var metricsKey = 'yaxis2';
            if (hasY) {
                patch[metricsKey + '.autorange'] = false;
                patch[metricsKey + '.range'] = [cmd.y[0], cmd.y[1]];
                patch[metricsKey + '.fixedrange'] = true;
            } else {
                patch[metricsKey + '.autorange'] = true;
                patch[metricsKey + '.fixedrange'] = true;
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
        var scheduleRebuild = function() {
            if (window.__rebuildTimer) {
                clearTimeout(window.__rebuildTimer);
                window.__rebuildTimer = null;
            }
            var delay = (cmd.rebuild_ms != null) ? cmd.rebuild_ms : 300;
            // rebuild_ms < 0이면 Y축만 바꾸고 figure는 다시 mount하지 않아 높이와 x축을 유지한다.
            if (delay < 0) return;
            var seq = cmd.seq;
            var fire = function() {
                window.__rebuildTimer = null;
                // 새로운 pan이 시작되었으므로 pointer가 다시 올라올 때까지 재생성을 미룬다.
                if (window.__graphPointerDown) {
                    window.__rebuildTimer = setTimeout(fire, 120);
                    return;
                }
                window.dash_clientside.set_props('rebuild-trigger', {
                    data: {seq: seq, t: Date.now()}
                });
            };
            window.__rebuildTimer = setTimeout(fire, delay);
        };
        var apply = function() {
            scheduleRebuild();
            if (cmd.status) {
                window.dash_clientside.set_props('status', {children: cmd.status});
            }
        };
        if (Object.keys(patch).length) {
            window.__clampingDataX = true;
            // Y축만 바꿀 때는 layout 높이와 subplot domain을 그대로 유지한다.
            if (!hasX && (hasY || hasYScore)) {
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


app.clientside_callback(
    """
    function(payload) {
        // pan 정밀 갱신은 Dash graph.figure를 건드리지 않고 trace만 바꿔 카메라가 튀지 않게 한다.
        if (!payload || !window.Plotly) {
            return window.dash_clientside.no_update;
        }
        var fig = payload.figure || payload;
        if (!fig || !fig.data) {
            return window.dash_clientside.no_update;
        }
        var host = document.getElementById('graph');
        var gd = host && host.querySelector('.js-plotly-plot');
        if (!gd) return window.dash_clientside.no_update;

        window.__applyPanFigureSilent = function(figIn) {
            var host2 = document.getElementById('graph');
            var gd2 = host2 && host2.querySelector('.js-plotly-plot');
            if (!figIn || !figIn.data || !gd2 || !window.Plotly) return;
            if (window.__graphPointerDown) {
                window.__queuedPanFigure = figIn;
                return;
            }
            var liveX = null;
            try {
                var fl = gd2._fullLayout || {};
                var xr = (fl.xaxis && fl.xaxis.range)
                    || (fl.xaxis2 && fl.xaxis2.range)
                    || (fl.xaxis3 && fl.xaxis3.range);
                if (xr && xr.length === 2) liveX = [xr[0], xr[1]];
            } catch (err) {}
            var hostH = (host2 && host2.clientHeight) || 420;
            var wantH = Math.max(hostH > 40 ? hostH : 420, 260);
            var layout = Object.assign({}, figIn.layout || {}, {
                autosize: true,
                height: wantH
            });
            // 현재 카메라는 유지하고 datarevision만 올려 Scattergl buffer를 새 데이터로 교체한다.
            if (gd2.layout && gd2.layout.uirevision != null) {
                layout.uirevision = gd2.layout.uirevision;
            }
            layout.datarevision = String(
                (layout.datarevision != null ? layout.datarevision : 'pan')
                + ':' + Date.now()
            );
            if (liveX) {
                layout.xaxis = Object.assign({}, layout.xaxis || {}, {
                    range: liveX, autorange: false
                });
                ['xaxis2', 'xaxis3', 'xaxis4'].forEach(function(ax) {
                    if (layout[ax] || (gd2.layout && gd2.layout[ax])) {
                        layout[ax] = Object.assign({}, layout[ax] || {}, {
                            range: liveX, autorange: false
                        });
                    }
                });
                if (layout.meta) {
                    layout.meta = Object.assign({}, layout.meta, {zoom_x: liveX});
                }
            }
            if (window.__withCachedHoverMeta) {
                layout = window.__withCachedHoverMeta(layout);
            }
            var cfg = gd2.config || {responsive: true, displayModeBar: true};
            window.__clampingDataX = true;
            window.__ignoreDataXClampUntil = Date.now() + 1500;
            // Y축은 재생성된 figure의 자동 범위를 쓰고 react 이후 x축은 절대 다시 건드리지 않는다.
            var yPatch = {};
            ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'].forEach(function(key) {
                var ax = layout[key];
                if (!ax || !ax.range || ax.range.length !== 2) return;
                if (ax.autorange) return;
                yPatch[key + '.autorange'] = false;
                yPatch[key + '.range'] = [ax.range[0], ax.range[1]];
                yPatch[key + '.fixedrange'] = true;
            });
            window.Plotly.react(gd2, figIn.data, layout, cfg).then(function() {
                if (Object.keys(yPatch).length) {
                    return window.Plotly.relayout(gd2, yPatch);
                }
            }).finally(function() {
                window.__clampingDataX = false;
                if (window.__desiredDragmode !== undefined) {
                    var cur = gd2._fullLayout && gd2._fullLayout.dragmode;
                    if (cur !== window.__desiredDragmode) {
                        window.Plotly.relayout(gd2, {dragmode: window.__desiredDragmode});
                    }
                }
                if (window.__queuedPanFigure && !window.__graphPointerDown) {
                    var q = window.__queuedPanFigure;
                    window.__queuedPanFigure = null;
                    window.__applyPanFigureSilent(q);
                }
            });
        };

        window.__applyPanFigureSilent(fig);
        return {t: Date.now(), pan_apply: true};
    }
    """,
    Output("axis-cmd-ack", "data", allow_duplicate=True),
    Input("pan-plot-apply", "data"),
    prevent_initial_call=True,
)


# mode radio가 focus를 가진 상태에서 방향키를 누르면 브라우저가 radio 선택을 바꿀 수 있다.
# capture 단계에서 키 이벤트를 가로채고 focus를 해제해
# 방향키가 값 탐색 cursor 이동에만 사용되도록 한다.
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
        // 배치/편집/값 탐색/pan/zoom 모드에서는 custom tip 엔진을 사용한다.
        // 값 탐색 모드에서는 anomaly-score 패널 위에서만 floating tip을 띄운다.
        window.__showPlaceTimeTip = (
            mode === 'label_range' || mode === 'label_point'
            || mode === 'edit_range' || mode === 'inspect'
            || mode === 'pan' || mode === 'pan_keep_y'
            || mode === 'zoom'
        );
        // 값 탐색 모드를 벗어나면 파란 점선 cursor를 즉시 지운다.
        // 실선 value_cursor shape는 다음 서버 figure 재생성에서 함께 정리된다.
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
                    // 현재 figure 갱신이 끝난 뒤 누적된 키 이동 횟수를 한 번에 서버로 보낸다.
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
                // 서버가 no_update를 반환해 figure 이벤트가 없으면 잠깐 뒤 키 잠금을 해제한다.
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
                // Esc 키는 어떤 모드에서도 진행 중인 range label 배치를 취소한다.
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
                // 키 반복 입력은 하나의 step 묶음으로 합쳐 clientside cursor와 서버 상태가 서로 싸우지 않게 한다.
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
                // 안정된 layout.uirevision이 이전 dragmode를 유지할 수 있으므로 mode radio 값으로 dragmode를 강제한다.
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


# container 크기가 안정되기 전에 figure가 그려지면 WebGL trace는 이전 폭을 기준으로 렌더링되고
# SVG label overlay는 최종 폭을 써서 구간이 어긋나 보일 수 있다.
# figure update 직후 크기를 다시 측정하면 첫 zoom 전부터 정렬이 맞는다.
# 서버가 직접 figure를 다시 그리면 relayout 이벤트가 발생하지 않으므로 저장된 view도
# 여기서 함께 갱신해야 다음 동작이 오래된 x 범위를 재적용하지 않는다.
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

        // hot reload 후 좌표/snap 수정이 바로 반영되도록 helper를 매번 다시 정의한다.
        window.__plotPlaceUtils = function(gd) {
                function plotAreaRect() {
                    // clientX와 pixel 변환이 맞도록 가장 위쪽 x축 subplot을 우선 사용한다.
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
                    // 세로로 쌓인 모든 subplot 영역에서 click을 허용한다.
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
                    // 주의: JavaScript에서는 isFinite(null)이 true이므로 null은 반드시 별도로 검사한다.
                    if (snapped == null || !(typeof snapped === 'number' && isFinite(snapped))) {
                        // sample 목록이 없으면 5분 cadence 기준으로 시간을 반올림해 맞춘다.
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
                // 초기 placeholder figure의 기본 축 범위는 실제 데이터 범위가 아니므로
                // 이를 저장하면 나중에 실제 데이터 figure에 잘못된 범위가 재적용될 수 있다.
                if (!gd || !gd._fullData || !gd._fullData.length) return null;
                if (!layout) return null;
                var xr = (layout.xaxis && layout.xaxis.range)
                    || (layout.xaxis2 && layout.xaxis2.range)
                    || (layout.xaxis3 && layout.xaxis3.range)
                    || (layout.xaxis4 && layout.xaxis4.range)
;
                if (!xr) return null;
                var view = {
                    x: [String(xr[0]), String(xr[1])]
                };
                // metric subplot은 제목이 'value'인 축으로 찾고 전문가/모델 label track은 제외한다.
                var yAxis = null;
                ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'].forEach(function(key) {
                    var ax = layout[key];
                    if (!ax) return;
                    var t = '';
                    try { t = (ax.title && ax.title.text) || ''; } catch (err) {}
                    if (/value/i.test(String(t))) yAxis = ax;
                });
                if (!yAxis) {
                    // 축 제목이 없으면 고정 metric 패널을 사용한다.
                    yAxis = layout.yaxis2 || layout.yaxis;
                }
                if (yAxis && yAxis.range) {
                    view.y = [yAxis.range[0], yAxis.range[1]];
                }
                return view;
            };
        }

        // 위/아래 subplot의 실제 plot 영역을 합친 범위를 pointer hit 영역으로 사용한다.
        // dual-panel 수정이 hot reload 후에도 반영되도록 매번 helper를 다시 정의한다.
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
                if (!gd || gd.__sharedPanelSyncV5) return;
                gd.__sharedPanelSyncV5 = true;
                var X_KEYS = ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'];

                function xPairFromEvent(ed) {
                    if (!ed) return null;
                    for (var i = 0; i < X_KEYS.length; i++) {
                        var ax = X_KEYS[i];
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

                function rangeMs(v) {
                    if (v == null) return NaN;
                    if (typeof v === 'number') return v;
                    var t = Date.parse(v);
                    return isNaN(t) ? NaN : t;
                }

                // box-zoom 중에는 최종 동기화만 한다. 실시간 plotly_relayouting과 Plotly.relayout을 같이 쓰면
                // gesture가 취소되고 잘못된 범위로 튀어 선이 평평해지는 문제가 생길 수 있다.
                // pan 중에는 shared axis가 이미 따라가므로 별도 동기화를 건너뛴다.
                // mouseup에서 범위를 다시 적용하면 화면이 뒤로 갔다 앞으로 오는 현상이 생긴다.
                gd.on('plotly_relayout', function(ed) {
                    if (window.__syncingSharedX || window.__clampingDataX || !ed) return;
                    if (window.__graphPointerDown) return;
                    var mode = window.__currentClickMode;
                    if (mode === 'pan' || mode === 'pan_keep_y'
                        || window.__desiredDragmode === 'pan') return;
                    if (ed['xaxis.autorange'] || ed['xaxis2.autorange']
                        || ed['xaxis3.autorange'] || ed['xaxis4.autorange']) return;
                    var pair = xPairFromEvent(ed);
                    if (!pair) return;
                    var fl = gd._fullLayout || {};
                    var patch = {};
                    X_KEYS.forEach(function(ax) {
                        if (ax === pair.from) return;
                        var cur = fl[ax] && fl[ax].range;
                        if (cur && cur.length >= 2) {
                            var d0 = Math.abs(rangeMs(cur[0]) - rangeMs(pair.a));
                            var d1 = Math.abs(rangeMs(cur[1]) - rangeMs(pair.b));
                            if (d0 < 1 && d1 < 1) return;
                        }
                        patch[ax + '.autorange'] = false;
                        patch[ax + '.range'] = [pair.a, pair.b];
                    });
                    if (!Object.keys(patch).length) return;
                    window.__syncingSharedX = true;
                    window.Plotly.relayout(gd, patch).finally(function() {
                        window.__syncingSharedX = false;
                    });
                });

                // Plotly 기본 subplot별 zoom box는 숨기고 두 패널을 가로지르는 custom band를 그린다.
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

                // 두 패널 전체에 시각적 zoom band만 그리며 실제 축 값은 이 단계에서 바꾸지 않는다.
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
                // Plotly 기본 zoom gesture를 방해하지 않도록 bubble phase에서 pointer 이벤트를 처리한다.
                gd.addEventListener('mousedown', onDown, false);
                window.addEventListener('mousemove', onMove, true);
                window.addEventListener('mouseup', onUp, true);
                window.addEventListener('blur', onUp, true);
            };

        // 선택 label의 진홍색 edge에서만 ↔ cursor와 가로 drag 편집을 허용한다.
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
                    // clientX와 data 좌표 변환이 맞도록 위쪽 x축 plot rect를 우선 사용한다.
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
                    // y 방향 hit 영역은 쌓인 subplot 전체를 합친 영역으로 본다.
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
                        // 데이터 cadence에 맞춰 항상 5분 단위 시각으로 snap한다.
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

        // pan/zoom 시간창이 데이터 범위 [tmin,tmax] 밖으로 나가지 않도록 제한한다.
        // drag 중 매 tick마다가 아니라 최종 relayout에서만 보정해 사용자 gesture와 충돌하지 않게 한다.
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
                    // 서버 figure가 적용되는 중에는 보정을 건너뛰어 zoom-in과 충돌하지 않게 한다.
                    if (window.__ignoreDataXClampUntil &&
                        Date.now() < window.__ignoreDataXClampUntil) {
                        return;
                    }
                    // 진행 중인 pan gesture에는 서버가 개입하지 않는다.
                    if (window.__graphPointerDown) return;
                    // autorange echo가 사용자가 의도적으로 줄인 zoom 범위를 다시 넓히지 못하게 한다.
                    if (eventData['xaxis.autorange'] || eventData['xaxis2.autorange']
                        || eventData['xaxis3.autorange']
                        || eventData['xaxis4.autorange']) return;
                    var meta = (gd.layout && gd.layout.meta) || {};
                    var bounds = meta.data_x;
                    if (!bounds || bounds.length < 2) return;
                    var tmin = window.__ms(bounds[0]);
                    var tmax = window.__ms(bounds[1]);
                    if (!(tmax > tmin)) return;

                    var x0, x1;
                    var axKeys = ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'];
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
                        // 범위 안이면 shared-panel 동기화가 다른 subplot 축도 같은 값으로 맞춘다.
                        return;
                    }
                    // Plotly가 보낸 값의 타입(string/number)을 그대로 유지한다.
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
                    var clampPatch = {};
                    axKeys.forEach(function(ax) {
                        clampPatch[ax + '.autorange'] = false;
                        clampPatch[ax + '.range'] = [out0, out1];
                    });
                    window.Plotly.relayout(gd, clampPatch).finally(function() {
                        window.__clampingDataX = false;
                    });
                });
            };
        }

        // 빈 영역이나 label band click은 가장 가까운 sample 시간으로 변환한다. helper는 매번 다시 bind한다.
        window.__installPlotClickHost = function() {
                var host = document.getElementById('graph');
                if (!host) return;
                if (host.__plotClickHostBoundV6) return;
                host.__plotClickHostBoundV6 = true;
                var lastEmit = {x: null, t: 0};
                var ptrDown = null;

                function labelTrackInfo(gd, clientX, clientY) {
                    // label strip 위에 있으면 어떤 track인지와 해당 rect를 반환한다.
                    try {
                        var fl = gd._fullLayout;
                        var plots = fl && fl._plots;
                        if (!plots) return null;
                        var keys = Object.keys(plots);
                        for (var i = 0; i < keys.length; i++) {
                            var p = plots[keys[i]];
                            if (!p || !p.plot || !p.plot.getBoundingClientRect) continue;
                            var ya = p.yaxis;
                            var title = '';
                            try {
                                title = (ya && ya.title && ya.title.text) || '';
                            } catch (err) {}
                            var r = p.plot.getBoundingClientRect();
                            if (!(r.width > 2 && r.height > 2)) continue;
                            if (clientX < r.left || clientX > r.right
                                || clientY < r.top || clientY > r.bottom) {
                                continue;
                            }
                            var t = String(title);
                            if (/human|사람/i.test(t) || (/label/i.test(t) && !/model/i.test(t))) {
                                return {track: 'human', rect: r};
                            }
                            if (/model|모델/i.test(t)) {
                                return {track: 'model', rect: r};
                            }
                        }
                    } catch (err2) {}
                    return null;
                }

                function emitSnap(clientX, clientY, opts) {
                    opts = opts || {};
                    var gd = host.querySelector('.js-plotly-plot');
                    if (!gd || !gd._fullLayout || !window.__plotPlaceUtils) return false;
                    var plot = window.__plotPlaceUtils(gd);
                    var info = labelTrackInfo(gd, clientX, clientY);
                    var track = (info && info.track) || opts.track || null;
                    if (!info && !plot.inPlotArea(clientX, clientY)) return false;
                    var xVal = plot.snapClientX(clientX);
                    if (xVal == null) return false;
                    var now = Date.now();
                    if (lastEmit.x === xVal && now - lastEmit.t < 120) return false;
                    lastEmit = {x: xVal, t: now};
                    if (!window.dash_clientside || !window.dash_clientside.set_props) return false;
                    window.dash_clientside.set_props('shape-click-event', {
                        data: {
                            x: xVal,
                            sequence: now,
                            track: track,
                            label_track: !!track
                        }
                    });
                    return true;
                }

                host.addEventListener('pointerdown', function(ev) {
                    if (ev.button !== 0) return;
                    var gd0 = host.querySelector('.js-plotly-plot');
                    var info0 = gd0 && labelTrackInfo(gd0, ev.clientX, ev.clientY);
                    ptrDown = {
                        x: ev.clientX,
                        y: ev.clientY,
                        track: info0 && info0.track
                    };
                }, true);

                host.addEventListener('click', function(ev) {
                    if (ev.button !== 0) return;
                    // 값 탐색에서는 pointerup이 WebGL score-panel click을 처리하므로 중복 emit을 막는다.
                    if (window.__currentClickMode === 'inspect') return;
                    if (window.__skipNextShapeClick) {
                        window.__skipNextShapeClick = false;
                        return;
                    }
                    if (window.__edgeDragging) return;
                    var mode = window.__currentClickMode;
                    // trace를 직접 누른 경우 Plotly clickData의 정확한 x 시간을 label 배치에 사용한다.
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

                // 짧은 click은 Plotly zoom/pan이 이벤트를 소비하더라도 label 선택으로 해석할 수 있게 별도 처리한다.
                // label track은 얇고 pan 모드에서도 누르기 쉽도록 click 허용 이동거리를 조금 넓게 잡는다.
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
                    // label 배치 모드는 빈 영역 click과 trace clickData를 모두 사용한다.
                    if (mode === 'label_range' || mode === 'label_point') return;
                    var gd1 = host.querySelector('.js-plotly-plot');
                    var info1 = gd1 && labelTrackInfo(gd1, ev.clientX, ev.clientY);
                    var track = (down && down.track) || (info1 && info1.track) || null;
                    if (down) {
                        var dx = Math.abs(ev.clientX - down.x);
                        var dy = Math.abs(ev.clientY - down.y);
                        var maxMove = track ? 28 : 6;
                        if (dx > maxMove || dy > maxMove) return;
                    }
                    emitSnap(ev.clientX, ev.clientY, {track: track});
                }, true);
            };

        // 전문가/모델 label subplot은 pan/zoom하지 않고 click을 segment 선택에만 사용한다.
        window.__disableLabelTrackDrag = function(gd) {
                if (!gd || !gd._fullLayout) return;
                try {
                    var plots = gd._fullLayout._plots || {};
                    Object.keys(plots).forEach(function(key) {
                        var p = plots[key];
                        var ya = p && p.yaxis;
                        var title = '';
                        try {
                            title = (ya && ya.title && ya.title.text) || '';
                        } catch (err) {}
                        if (!/label|human|model|사람|모델/i.test(String(title))) return;
                        var nodes = gd.querySelectorAll(
                            '.draglayer .' + key + ' .nsewdrag, .draglayer > .' + key + ' > .nsewdrag'
                        );
                        for (var i = 0; i < nodes.length; i++) {
                            nodes[i].style.pointerEvents = 'none';
                            nodes[i].style.cursor = 'pointer';
                        }
                    });
                } catch (err2) {}
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
                    // 사용자가 두 번 클릭해 range를 만드는 중에는 임시 preview를 자동 정리하지 않는다.
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

        // pending preview 설치 여부와 관계없이 helper를 항상 다시 정의한다.
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
                        // custom tip 스타일은 fig.layout.hoverlabel의 흰 배경/고정폭 글꼴/11px 설정과 맞춘다.
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
                        // custom spike도 layout.xaxis의 dash/color/thickness 설정과 맞춘다.
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
                    // cursor가 여러 subplot을 자연스럽게 가로지르도록 위/아래 plot 영역의 합친 범위를 사용한다.
                    // 각 패널에서 같은 세로 cursor가 보이게 한다.
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

                // sample_ms가 없으면 5분 wall-clock 기준으로 시간을 반올림한다.
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
                    if (meta.score_polarity) {
                        cache.score_polarity = meta.score_polarity;
                    }
                    if (meta.hover_cache_seq != null) cache.seq = meta.hover_cache_seq;
                    window.__plotHoverCache = cache;
                    return cache;
                }
                // figure 업데이트는 Plotly.react 전에 이 함수를 호출해 hover용 배열을 cache한다.
                // 사용자가 아직 hover하지 않았어도 필요한 데이터가 준비되게 한다.
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
                    // score 패널은 첫 번째 행에 예약되어 있다.
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
                        // score row가 실제로 있을 때만 가장 위 nsewdrag 영역을 score 패널로 본다.
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

                // 시간 표시는 Plotly hoverformat과 동일한 한국식 연-월-일 시:분 형식을 사용한다.
                function formatHoverTime(ms) {
                    var d = new Date(ms);
                    function pad(n) { return (n < 10 ? '0' : '') + n; }
                    return d.getUTCFullYear() + '년 ' + pad(d.getUTCMonth() + 1) + '월 '
                        + pad(d.getUTCDate()) + '일 ' + pad(d.getUTCHours()) + ':'
                        + pad(d.getUTCMinutes());
                }

                function formatScoreNumber(v) {
                    if (v == null || !isFinite(v)) return String(v);
                    var a = Math.abs(v);
                    // 일부 모델 score는 1보다 매우 작을 수 있어 toFixed(2)만 쓰면 모두 0.00처럼 보일 수 있다.
                    if (a !== 0 && a < 0.01) return v.toPrecision(3);
                    if (a >= 1000) return v.toFixed(1);
                    return v.toFixed(4);
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
                        if (thr != null) text += '\\nthreshold = ' + formatScoreNumber(thr);
                        return text;
                    }
                    text += '\\nAnomaly score = ' + formatScoreNumber(score);
                    if (thr != null) {
                        text += '\\nthreshold = ' + formatScoreNumber(thr);
                        // paperib 기본 규칙은 score가 클수록 anomaly에 가깝다는 것이다.
                        var higher = true;
                        if (cache.score_polarity === 'lower_is_anomaly'
                            || cache.score_polarity === 'low'
                            || cache.score_polarity === 'lt') {
                            higher = false;
                        }
                        var hit = higher ? (score > thr) : (score < thr);
                        if (hit) text += ' · anomaly';
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
                    // box-zoom/pan/edge 편집 중에는 floating tip을 숨긴다.
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
                    // 값 탐색 모드에서는 anomaly-score 패널에서만 floating tip을 보이고 나머지 값은 아래 패널에서 보여준다.
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
                    // Plotly 기본 spike 대신 snap된 시간에 custom 세로선 하나만 그린다.
                    showSpikeAtMs(snapped, rect);
                    // anomaly-score 패널에서는 시간과 score를 tip으로 보여준다.
                    // 다른 패널의 빈 영역에서는 시간만 보여주고 series hover는 Plotly 기본 기능에 맡긴다.
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

            // tip 엔진이 아직 설치되지 않았어도 동작할 수 있도록 hover cache helper를 독립적으로 정의한다.
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
                    if (meta.score_polarity) {
                        cache.score_polarity = meta.score_polarity;
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

            // 연속 drag가 끝난 뒤에만 지연 재생성이 실행되도록 pan/zoom pointer 상태를 추적한다.
            // pointerup 즉시 재생성을 실행하면 plotly_relayout과 race가 생기므로 바로 실행하지 않는다.
            // 그렇지 않으면 실제 정밀 갱신 전에 pan 이전 화면이 잠깐 다시 보일 수 있다.
            if (!window.__installGraphPointerGuard) {
                window.__installGraphPointerGuard = function(root) {
                    if (!root || root.__graphPointerGuardBound) return;
                    root.__graphPointerGuardBound = true;
                    var down = function(ev) {
                        if (ev.button !== 0) return;
                        window.__graphPointerDown = true;
                        if (window.__rebuildTimer) {
                            clearTimeout(window.__rebuildTimer);
                            window.__rebuildTimer = null;
                            window.__pendingPanRebuild = true;
                        }
                    };
                    var up = function() {
                        if (!window.__graphPointerDown) return;
                        window.__graphPointerDown = false;
                        // drag 중 미뤄진 재생성이 있으면 pointerup 뒤 다시 예약한다.
                        // 정상 경로는 relayout 이후 axis-cmd이며 여기 처리는 fallback이다.
                        if (window.__pendingPanRebuild) {
                            window.__pendingPanRebuild = false;
                            // relayout → axis-cmd가 재생성을 예약하며, 이미
                            // pointer가 눌린 동안 실행되었다면 fire()가 재시도하므로
                            // 여기서는 보류 flag만 정리하면 된다.
                        }
                    };
                    root.addEventListener('pointerdown', down, true);
                    window.addEventListener('pointerup', up, true);
                    window.addEventListener('pointercancel', up, true);
                };
            }

            function afterGraphReady() {
                window.__clampPanToData(gd);
                window.__installSharedPanelSync(gd);
                window.__installGraphPointerGuard(host);
                window.__installPlotClickHost();
                if (window.__disableLabelTrackDrag) {
                    window.__disableLabelTrackDrag(gd);
                }
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
                // relayout 후 drag node가 다시 만들어질 수 있으므로 label track의 pan/zoom 비활성화를 다시 적용한다.
                if (window.__disableLabelTrackDrag) {
                    setTimeout(function() {
                        window.__disableLabelTrackDrag(gd);
                    }, 0);
                }
                window.__inspectInFlight = false;
                clearTimeout(window.__inspectInFlightWatchdog);
                if (window.__valueInspectMode && (window.__inspectPendingStep || 0) !== 0) {
                    if (!window.__inspectFlushTimer && window.__flushInspectKeys) {
                        window.__inspectFlushTimer = setTimeout(window.__flushInspectKeys, 0);
                    }
                }
            }

            // edge drag가 남긴 임시 relayout patch가 화면에 남지 않도록 전체 layout을 다시 동기화한다.
            // 서버가 수정된 label shape를 보낸 뒤 최종 상태로 맞추기 위한 처리다.
            if (figure && figure.data && figure.layout && !window.__edgeDragging) {
                var cfg = gd.config || {responsive: true, displayModeBar: true};
                var hostH = host.clientHeight || 420;
                var wantH = Math.max(hostH > 40 ? hostH : 420, 260);
                var layout = Object.assign({}, figure.layout, {
                    autosize: true,
                    height: wantH
                });
                // hover용 배열을 읽어 browser cache에 보존한다. 이후 figure는 sample_ms를 생략할 수 있기 때문이다.
                // sample_ms는 최초 한 번만 전달되므로 그대로 두면 Plotly.react가 layout.meta에서 지울 수 있다.
                // 그 전에 tip이 실행되면 실제 score가 있어도 '(없음)'으로 보일 수 있다.
                if (window.__withCachedHoverMeta) {
                    layout = window.__withCachedHoverMeta(layout);
                } else if (window.__ingestHoverMeta && layout.meta) {
                    window.__ingestHoverMeta(layout.meta);
                }
                var isPanMode = (
                    window.__currentClickMode === 'pan'
                    || window.__currentClickMode === 'pan_keep_y'
                    || window.__desiredDragmode === 'pan'
                );
                // pan 정밀 갱신에서는 Dash가 이미 Plotly.react를 수행했으므로 두 번째
                // react를 다시 호출하면 특히 setTimeout 뒤 pan 이전
                // trace가 잠깐 보였다가 바뀔 수 있다. 여기서는 Y만 맞추고 X/data는 건드리지 않는다.
                if (isPanMode && !window.__graphPointerDown && !window.__edgeDragging) {
                    try {
                        var yOnly = {};
                        ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'].forEach(function(key) {
                            var ax = layout[key];
                            if (!ax || !ax.range || ax.range.length !== 2) return;
                            if (ax.autorange) return;
                            yOnly[key + '.autorange'] = false;
                            yOnly[key + '.range'] = [ax.range[0], ax.range[1]];
                            yOnly[key + '.fixedrange'] = true;
                        });
                        var done = function() { afterGraphReady(); };
                        if (Object.keys(yOnly).length) {
                            window.__clampingDataX = true;
                            window.Plotly.relayout(gd, yOnly).finally(function() {
                                window.__clampingDataX = false;
                                done();
                            });
                        } else {
                            done();
                        }
                    } catch (errPan) {
                        afterGraphReady();
                    }
                    return;
                }
                // pointer가 눌린 동안에는 현재 브라우저 카메라를 유지하고, 손을 뗀 뒤에는
                // 방금 끝난 pan에 맞춰 재생성된 figure의 zoom_x를 사용한다.
                var keepX = null;
                try {
                    if (window.__graphPointerDown) {
                        var fl0 = gd._fullLayout || {};
                        var xr0 = (fl0.xaxis && fl0.xaxis.range)
                            || (fl0.xaxis2 && fl0.xaxis2.range)
                            || (fl0.xaxis3 && fl0.xaxis3.range);
                        if (xr0 && xr0.length === 2) keepX = [xr0[0], xr0[1]];
                    }
                } catch (errLive) {}
                if (!keepX && layout.meta && layout.meta.zoom_x
                    && layout.meta.zoom_x.length === 2) {
                    keepX = [layout.meta.zoom_x[0], layout.meta.zoom_x[1]];
                }
                if (keepX) {
                    layout = Object.assign({}, layout);
                    layout.xaxis = Object.assign({}, layout.xaxis || {}, {
                        range: keepX, autorange: false
                    });
                    ['xaxis2', 'xaxis3', 'xaxis4'].forEach(function(ax) {
                        if (layout[ax]) {
                            layout[ax] = Object.assign({}, layout[ax], {
                                range: keepX, autorange: false
                            });
                        }
                    });
                    if (layout.meta) {
                        layout.meta = Object.assign({}, layout.meta, {zoom_x: keepX});
                    }
                }
                window.Plotly.react(gd, figure.data, layout, cfg).then(function() {
                    // react 뒤 축 범위를 다시 적용해 layout race를 정리한다.
                    // 예를 들어 anomaly 선택→줌 과정에서 figure JSON에는 새 Y 자동 범위가 있어도
                    // 실제 화면에는 이전 Y scale이 남을 수 있기 때문이다.
                    try {
                        var patch = {};
                        var meta = layout.meta || {};
                        var zx = keepX || meta.zoom_x;
                        if (zx && zx.length === 2 && !window.__graphPointerDown) {
                            ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4'].forEach(function(ax) {
                                patch[ax + '.autorange'] = false;
                                patch[ax + '.range'] = [zx[0], zx[1]];
                            });
                        }
                        ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'].forEach(function(key) {
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
    print(f"논문 PLMN {len(rank_df)}개: {', '.join(plmn_ids)}")
    print(f"라벨링 앱 실행 중 → {url}")
    print("종료: Ctrl+C")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
