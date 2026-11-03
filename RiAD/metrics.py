"""Paperib evaluation metrics for RiAD (storyline 2026-11-03).

Primary: VUS-PR
Event: Event-F1
Additional: PR-AUC, Precision, Recall, F1
Operational: False Alarms / day, Detection Delay

Point-Adjusted F1 and Accuracy are intentionally omitted from primary reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of contiguous True runs."""
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
    """Event-level P/R/F1: overlap ⇒ hit (no point-adjustment fill)."""
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
    """Pred events with no GT overlap, normalized by observation days."""
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
    """Mean delay (minutes) from GT event start to first overlapping pred hit.

    Missed events are excluded from the mean; ``n_missed`` is reported separately.
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


def vus_pr(
    y_true: np.ndarray,
    score_high_is_anomaly: np.ndarray,
    *,
    sliding_window: int = 100,
) -> dict[str, float | None]:
    """VUS-PR via official ``vus`` package (Paparrizos et al.).

    ``score_high_is_anomaly`` must be higher-for-anomaly (e.g. reconstruction MSE).
    """
    from vus.metrics import get_metrics

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(score_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(s))
    y, s = y[:n], s[:n]
    ok = np.isfinite(s)
    if ok.sum() < 8 or int(y[ok].sum()) == 0:
        return {"vus_pr": None, "auc_pr": None, "r_auc_pr": None}
    # Fill leading NaNs (window warm-up) with nearest finite for VUS length match.
    s2 = s.copy()
    if not ok.all():
        finite_idx = np.flatnonzero(ok)
        first = int(finite_idx[0])
        s2[:first] = s2[first]
        for i in range(first + 1, n):
            if not np.isfinite(s2[i]):
                s2[i] = s2[i - 1]
    # Min-max to [0,1] as in VUS examples.
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
            slidingWindow=max(int(sliding_window), 1),
        )
    except Exception:
        return {"vus_pr": None, "auc_pr": None, "r_auc_pr": None}
    return {
        "vus_pr": _f(res.get("VUS_PR")),
        "auc_pr": _f(res.get("AUC_PR")),
        "r_auc_pr": _f(res.get("R_AUC_PR")),
        "vus_roc": _f(res.get("VUS_ROC")),  # auxiliary only
    }


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def evaluate_riad_valid(
    y_true: np.ndarray,
    scores_high_is_anomaly: np.ndarray,
    threshold: float,
    *,
    window: int = 100,
    sample_minutes: float = 5.0,
) -> dict[str, Any]:
    """Full metric bundle for one operator valid split.

    ``scores_high_is_anomaly``: reconstruction MSE (higher = more anomalous).
    Anomaly when ``score > threshold``.
    """
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(scores_high_is_anomaly, dtype=np.float64).reshape(-1)
    n = min(len(y), len(scores))
    y, scores = y[:n], scores[:n]
    ok = np.isfinite(scores)
    pred = np.zeros(n, dtype=bool)
    pred[ok] = scores[ok] > float(threshold)

    # Align metrics to first finite score (window warm-up).
    if ok.any():
        first = int(np.flatnonzero(ok)[0])
        y_e, pred_e, s_e = y[first:], pred[first:], scores[first:]
    else:
        y_e, pred_e, s_e = y, pred, scores

    point = point_prf(y_e, pred_e)
    event = event_f1(y_e, pred_e)
    fa = false_alarms_per_day(y_e, pred_e, sample_minutes=sample_minutes)
    delay = detection_delay(y_e, pred_e, sample_minutes=sample_minutes)
    vus = vus_pr(y_e, s_e, sliding_window=window)
    ap = pr_auc(y_e, s_e)

    return {
        "primary": {"vus_pr": vus.get("vus_pr")},
        "event": event,
        "point": point,
        "pr_auc": ap if ap is not None else vus.get("auc_pr"),
        "vus": vus,
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
