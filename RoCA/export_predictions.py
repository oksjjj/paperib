"""Build labeling-UI prediction JSON for RoCA."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from metrics import evaluate_valid

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED_DIR = os.path.join(_REPO, "data", "ib_data", "predictions")


def _mask_to_labels(
    times: pd.Series | np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    source: str = "roca",
) -> list[dict[str, Any]]:
    times = pd.to_datetime(pd.Series(times), utc=True)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = min(len(times), len(mask), len(scores))
    items: list[dict[str, Any]] = []
    i = 0
    k = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        sl = scores[i:j]
        finite = sl[np.isfinite(sl)]
        seg_score = float(np.max(finite)) if len(finite) else None
        k += 1
        items.append(
            {
                "id": f"roca_{k:06d}",
                "kind": "range",
                "start": times.iloc[i].isoformat(),
                "end": times.iloc[j - 1].isoformat(),
                "score": seg_score,
                "source": source,
            }
        )
        i = j
    return items


def write_roca_prediction(
    *,
    plmn: str,
    times_valid: pd.Series,
    scores_valid: np.ndarray,
    y_valid: np.ndarray,
    threshold: float,
    feature_columns: list[str],
    window: int,
    split: dict[str, str],
    metrics_extra: dict[str, Any] | None = None,
    source: str = "roca",
) -> str:
    os.makedirs(PRED_DIR, exist_ok=True)
    scores = np.asarray(scores_valid, dtype=np.float64)
    pred = np.isfinite(scores) & (scores > threshold)
    labels = _mask_to_labels(times_valid, pred, scores, source=source)

    eval_bundle = evaluate_valid(
        y_valid, scores, threshold, vus_window=None, sample_minutes=5.0
    )
    point = eval_bundle["point"]
    event = eval_bundle["event"]
    op = eval_bundle["operational"]
    vus = eval_bundle["vus"]

    times_iso = [
        pd.Timestamp(t).tz_convert("UTC").isoformat()
        if pd.Timestamp(t).tzinfo
        else pd.Timestamp(t, tz="UTC").isoformat()
        for t in times_valid
    ]
    payload: dict[str, Any] = {
        "plmn": plmn,
        "source": source,
        "model": "roca",
        "run_name": "paperib",
        "eval_split": "valid",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_method": "percentile",
        "window_length": int(window),
        "n_scores": int(np.isfinite(scores).sum()),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "feature_columns": list(feature_columns),
        "feature_mode": "core",
        "split": split,
        "labels": labels,
        "score_series": {
            "times": times_iso,
            "scores": [float(v) if np.isfinite(v) else None for v in scores],
            "threshold": float(threshold),
            "note": (
                "RoCA score = distance-to-center (loss_n; higher = more anomalous); "
                "primary thr = train top-percentile. "
                "Upstream: ruiking04/RoCA. Primary metric = VUS-PR."
            ),
        },
        "score_polarity": "higher_is_anomaly",
        "metrics": {
            "eval_split": "valid",
            "test_held_out": True,
            "primary_metric": "vus_pr",
            "score_polarity": "higher_is_anomaly",
            "vus_pr": eval_bundle["primary"]["vus_pr"],
            "pr_auc": eval_bundle["pr_auc"],
            "vus_window": eval_bundle.get("vus_window"),
            "r_auc_pr": vus.get("r_auc_pr"),
            "score_distribution": eval_bundle.get("score_distribution"),
            "event_score_diagnostics": eval_bundle.get("event_score_diagnostics"),
            "event_f1": event.get("f1"),
            "event_precision": event.get("precision"),
            "event_recall": event.get("recall"),
            "n_gt_events": event.get("n_gt_events"),
            "n_pred_events": event.get("n_pred_events"),
            "n_hit_gt": event.get("n_hit_gt"),
            "precision": point.get("precision"),
            "recall": point.get("recall"),
            "f1": point.get("f1"),
            "tp": point.get("tp"),
            "fp": point.get("fp"),
            "fn": point.get("fn"),
            "false_alarms_per_day": op.get("false_alarms_per_day"),
            "n_false_alarm_events": op.get("n_false_alarm_events"),
            "observation_days": op.get("observation_days"),
            "mean_detection_delay_min": op.get("mean_detection_delay_min"),
            "median_detection_delay_min": op.get("median_detection_delay_min"),
            "n_detected_events": op.get("n_detected"),
            "n_missed_events": op.get("n_missed"),
            "excluded_from_primary": eval_bundle["excluded_from_primary"],
            "auroc_policy": eval_bundle["auroc_policy"],
            **(metrics_extra or {}),
        },
        "paper_metrics": {
            "primary_metric": "vus_pr",
            "vus_pr": eval_bundle["primary"]["vus_pr"],
            "pr_auc": eval_bundle["pr_auc"],
            "event": {
                "f1": event.get("f1"),
                "precision": event.get("precision"),
                "recall": event.get("recall"),
                "n_gt_events": event.get("n_gt_events"),
                "n_pred_events": event.get("n_pred_events"),
                "n_hit_gt": event.get("n_hit_gt"),
            },
            "point": {
                "precision": point.get("precision"),
                "recall": point.get("recall"),
                "f1": point.get("f1"),
                "TP": point.get("tp"),
                "FP": point.get("fp"),
                "FN": point.get("fn"),
            },
            "score_distribution": eval_bundle.get("score_distribution"),
            "event_score_diagnostics": eval_bundle.get("event_score_diagnostics"),
            "operational": {
                "false_alarms_per_day": op.get("false_alarms_per_day"),
                "mean_detection_delay_min": op.get("mean_detection_delay_min"),
                "median_detection_delay_min": op.get("median_detection_delay_min"),
                "n_false_alarm_events": op.get("n_false_alarm_events"),
                "observation_days": op.get("observation_days"),
            },
            "threshold": float(threshold),
            "threshold_method": "percentile",
            "score_polarity": "higher_is_anomaly",
            "eval_split": "valid",
            "note": "PA-F1 / Accuracy excluded from primary reporting",
        },
    }
    path = os.path.join(PRED_DIR, f"{plmn}_{source}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
