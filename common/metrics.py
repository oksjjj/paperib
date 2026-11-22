"""paperib 실험에서 공통으로 사용하는 평가 지표 모음.

핵심 지표: VUS-PR
구간 단위 지표: Event-F1
추가 지표: PR-AUC, Precision, Recall, F1
운영 지표: False Alarms / day, Detection Delay

Point-Adjusted F1과 Accuracy는 이상구간을 과도하게 유리하게 평가할 수 있으므로
논문의 primary reporting에서는 의도적으로 제외한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """연속된 True 구간을 (시작 인덱스, 끝 인덱스) 형태로 추출한다. 끝 인덱스도 구간에 포함된다."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    segs: list[tuple[int, int]] = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < n and m[j]:
            j += 1
        segs.append((i, j - 1))
        i = j
    return segs


def point_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=bool).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def event_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Event 단위 Precision/Recall/F1을 계산한다.

    GT 이상구간과 예측구간이 한 점이라도 겹치면 해당 event를 탐지한 것으로 본다.
    Point-adjustment처럼 GT 전체 구간을 자동으로 예측으로 채우지는 않는다.
    """
    gt = _segments(y_true)
    pr = _segments(y_pred)
    if not gt and not pr:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "n_gt_events": 0,
            "n_pred_events": 0,
            "n_hit_gt": 0,
            "n_hit_pred": 0,
        }
    hit_gt = 0
    for a0, a1 in gt:
        if any(not (b1 < a0 or b0 > a1) for b0, b1 in pr):
            hit_gt += 1
    hit_pred = 0
    for b0, b1 in pr:
        if any(not (a1 < b0 or a0 > b1) for a0, a1 in gt):
            hit_pred += 1
    prec = hit_pred / len(pr) if pr else (1.0 if not gt else 0.0)
    rec = hit_gt / len(gt) if gt else (1.0 if not pr else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_gt_events": len(gt),
        "n_pred_events": len(pr),
        "n_hit_gt": hit_gt,
        "n_hit_pred": hit_pred,
    }


def false_alarms_per_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_minutes: float = 5.0,
) -> dict[str, float | int]:
    """GT 이상구간과 전혀 겹치지 않는 예측 event 수를 관측 일수로 나누어 FA/day를 계산한다."""
    gt = _segments(y_true)
    pr = _segments(y_pred)
    fa = 0
    for b0, b1 in pr:
        if not any(not (a1 < b0 or a0 > b1) for a0, a1 in gt):
            fa += 1
    n = len(np.asarray(y_true).reshape(-1))
    days = max(n * sample_minutes / (60.0 * 24.0), 1e-12)
    return {
        "n_false_alarm_events": int(fa),
        "observation_days": float(days),
        "false_alarms_per_day": float(fa / days),
    }


def detection_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_minutes: float = 5.0,
) -> dict[str, float | int | None]:
    """GT event 시작부터 첫 예측 hit까지의 탐지 지연시간(분)을 계산한다.

    끝까지 탐지되지 않은 event는 평균 지연시간 계산에서는 제외하고,
    놓친 event 수는 n_missed로 따로 기록한다.
    """
    gt = _segments(y_true)
    pred = np.asarray(y_pred, dtype=bool).reshape(-1)
    delays: list[float] = []
    missed = 0
    for a0, a1 in gt:
        hit_idx = None
        for t in range(a0, a1 + 1):
            if pred[t]:
                hit_idx = t
                break
        if hit_idx is None:
            missed += 1
            continue
        delays.append((hit_idx - a0) * sample_minutes)
    return {
        "n_gt_events": len(gt),
        "n_detected": len(delays),
        "n_missed": missed,
        "mean_detection_delay_min": float(np.mean(delays)) if delays else None,
        "median_detection_delay_min": float(np.median(delays)) if delays else None,
    }


def pr_auc(y_true: np.ndarray, score_high_is_anomaly: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(score_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(s))
    y, s = y[:n], s[:n]
    ok = np.isfinite(s)
    if ok.sum() < 2 or y[ok].sum() == 0 or (~y[ok]).sum() == 0:
        return None
    return float(average_precision_score(y[ok].astype(int), s[ok]))


def _median_event_length(y_true: np.ndarray) -> int:
    """GT anomaly event 길이의 중앙값을 point 단위로 계산한다. 최소값은 1이다.

    VUS tolerance window를 모델의 입력 window와 연결하지 않고,
    전문가가 표시한 실제 anomaly 길이를 기준으로 정하기 위해 사용한다.
    """
    lengths = [b - a + 1 for a, b in _segments(np.asarray(y_true, dtype=bool))]
    if not lengths:
        return 1
    return max(int(round(float(np.median(lengths)))), 1)


def resolve_vus_window(
    y_true: np.ndarray,
    sliding_window: int | None = None,
) -> int:
    """모든 모델에 공통으로 사용할 VUS tolerance window를 결정한다.

    값이 없거나 0 이하이면 GT event 길이 중앙값을 자동으로 사용한다.
    """
    if sliding_window is None or int(sliding_window) <= 0:
        return _median_event_length(y_true)
    return max(int(sliding_window), 1)


def score_distribution(
    scores_high_is_anomaly: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """train과 validation score 분포가 크게 어긋나는지 확인하기 위한 간단한 분포 통계를 만든다."""
    s = np.asarray(scores_high_is_anomaly, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {
            "fraction_above_threshold": None,
            "q50": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    return {
        "fraction_above_threshold": float(np.mean(s > float(threshold))),
        "q50": float(np.quantile(s, 0.50)),
        "q90": float(np.quantile(s, 0.90)),
        "q95": float(np.quantile(s, 0.95)),
        "q99": float(np.quantile(s, 0.99)),
        "max": float(np.max(s)),
    }


def event_score_diagnostics(
    y_true: np.ndarray,
    scores_high_is_anomaly: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int | bool | None]]:
    """각 GT anomaly event마다 score와 전체 score 내 percentile을 진단용으로 계산한다."""
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(scores_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(s))
    y, s = y[:n], s[:n]
    finite_all = np.sort(s[np.isfinite(s)])

    def pct(v: float | None) -> float | None:
        if v is None or not np.isfinite(v) or finite_all.size == 0:
            return None
        return float(np.searchsorted(finite_all, v, side="right") / finite_all.size)

    rows: list[dict[str, float | int | bool | None]] = []
    for event_id, (a, b) in enumerate(_segments(y), start=1):
        ss = s[a : b + 1]
        finite = ss[np.isfinite(ss)]
        if finite.size:
            max_score = float(np.max(finite))
            mean_score = float(np.mean(finite))
            pred_points = int(np.sum(finite > float(threshold)))
        else:
            max_score = None
            mean_score = None
            pred_points = 0
        rows.append(
            {
                "event_id": event_id,
                "start_index": int(a),
                "end_index": int(b),
                "length_points": int(b - a + 1),
                "detected": bool(pred_points > 0),
                "n_pred_points": pred_points,
                "max_score": max_score,
                "mean_score": mean_score,
                "max_score_percentile": pct(max_score),
                "mean_score_percentile": pct(mean_score),
            }
        )
    return rows


def vus_pr(
    y_true: np.ndarray,
    score_high_is_anomaly: np.ndarray,
    *,
    sliding_window: int | None = None,
) -> dict[str, float | int | None]:
    """공식 vus 패키지를 이용해 VUS-PR 계열 지표를 계산한다.

    score_high_is_anomaly는 값이 클수록 더 anomalous해야 한다.
    예: AutoEncoder reconstruction MSE.
    """
    from vus.metrics import get_metrics

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(score_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(s))
    y, s = y[:n], s[:n]
    ok = np.isfinite(s)
    vus_window = resolve_vus_window(y, sliding_window)
    if ok.sum() < 8 or int(y[ok].sum()) == 0:
        return {
            "vus_pr": None,
            "auc_pr": None,
            "r_auc_pr": None,
            "vus_roc": None,
            "vus_window": vus_window,
        }
    # window 기반 모델의 초기 warm-up NaN 때문에 길이가 달라지지 않도록 첫 유효 score로 앞부분을 채운다.
    s2 = s.copy()
    if not ok.all():
        finite_idx = np.flatnonzero(ok)
        first = int(finite_idx[0])
        s2[:first] = s2[first]
        for i in range(first + 1, n):
            if not np.isfinite(s2[i]):
                s2[i] = s2[i - 1]
    # VUS 예제와 동일하게 score를 [0, 1] 범위로 min-max 정규화한다. 순위는 변하지 않는다.
    lo, hi = float(np.min(s2)), float(np.max(s2))
    if hi > lo:
        s2 = (s2 - lo) / (hi - lo)
    else:
        s2 = np.zeros_like(s2)
    try:
        res = get_metrics(
            s2,
            y.astype(int),
            metric="all",
            slidingWindow=vus_window,
        )
    except Exception:
        return {
            "vus_pr": None,
            "auc_pr": None,
            "r_auc_pr": None,
            "vus_roc": None,
            "vus_window": vus_window,
        }
    return {
        "vus_pr": _f(res.get("VUS_PR")),
        "auc_pr": _f(res.get("AUC_PR")),
        "r_auc_pr": _f(res.get("R_AUC_PR")),
        "vus_roc": _f(res.get("VUS_ROC")),  # VUS-ROC은 보조 지표로만 사용한다.
        "vus_window": vus_window,
    }


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def evaluate_valid(
    y_true: np.ndarray,
    scores_high_is_anomaly: np.ndarray,
    threshold: float,
    *,
    vus_window: int | None = None,
    sample_minutes: float = 5.0,
) -> dict[str, Any]:
    """한 사업자의 하나의 평가 split에 대해 논문에서 필요한 전체 metric 묶음을 계산한다.

    scores_high_is_anomaly는 값이 클수록 이상인 score이며,
    현재 binary prediction은 score > threshold일 때 anomaly로 판단한다.
    """
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(scores_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(scores))
    y, scores = y[:n], scores[:n]
    ok = np.isfinite(scores)
    pred = np.zeros(n, dtype=bool)
    pred[ok] = scores[ok] > float(threshold)

    # window 기반 모델의 warm-up 구간을 제외하기 위해 첫 유효 score부터 metric을 맞춰 계산한다.
    if ok.any():
        first = int(np.flatnonzero(ok)[0])
        y_e, pred_e, s_e = y[first:], pred[first:], scores[first:]
    else:
        y_e, pred_e, s_e = y, pred, scores

    point = point_prf(y_e, pred_e)
    event = event_f1(y_e, pred_e)
    fa = false_alarms_per_day(y_e, pred_e, sample_minutes=sample_minutes)
    delay = detection_delay(y_e, pred_e, sample_minutes=sample_minutes)
    vus = vus_pr(y_e, s_e, sliding_window=vus_window)
    ap = pr_auc(y_e, s_e)
    score_diag = score_distribution(s_e, threshold)
    event_diag = event_score_diagnostics(y_e, s_e, threshold)

    return {
        "primary": {"vus_pr": vus.get("vus_pr")},
        "event": event,
        "point": point,
        "pr_auc": ap if ap is not None else vus.get("auc_pr"),
        "vus": vus,
        "vus_window": vus.get("vus_window"),
        "score_distribution": score_diag,
        "event_score_diagnostics": event_diag,
        "operational": {
            **fa,
            **delay,
        },
        "threshold": float(threshold),
        "n_eval": int(len(y_e)),
        "n_pred_points": int(pred_e.sum()),
        "excluded_from_primary": ["point_adjusted_f1", "accuracy"],
        "auroc_policy": "auxiliary_only",
    }
