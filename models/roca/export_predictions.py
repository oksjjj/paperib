"""RoCA validation 결과를 labeling UI와 공통 평가 코드가 읽는 JSON 형식으로 저장한다.

이 파일은 모델을 학습하지 않는다.
run_plmn.py에서 이미 계산된 score, threshold, validation label을 받아
연속 anomaly segment와 metric을 정리하여 prediction JSON을 만든다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import evaluate_valid

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED_DIR = os.path.join(_REPO, "data", "ib_data", "predictions")


def _mask_to_labels(
    times: pd.Series | np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    source: str = "roca",
) -> list[dict[str, Any]]:
    """point 단위 anomaly boolean mask를 연속된 range label 목록으로 변환한다.

    UI에서는 개별 point보다 연속 anomaly 구간을 하나의 prediction segment로
    표시하는 것이 편하므로 True가 이어지는 구간을 묶는다.
    각 segment의 대표 score는 해당 구간의 최대 anomaly score다.
    """
    times = pd.to_datetime(pd.Series(times), utc=True)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    # 시간, 예측 마스크, 점수의 공통 유효 길이까지만 사용한다.
    n = min(len(times), len(mask), len(scores))
    items: list[dict[str, Any]] = []
    # 연속 True 구간을 찾아 시작/끝 시각 및 score 정보를 하나의 prediction 이벤트로 묶는다.
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
    """RoCA validation score와 평가 결과를 prediction JSON으로 저장한다."""

    os.makedirs(PRED_DIR, exist_ok=True)
    scores = np.asarray(scores_valid, dtype=np.float64)

    # score가 유효하고 TRAIN에서 정한 threshold를 초과하면 anomaly point로 판단한다.
    pred = np.isfinite(scores) & (scores > threshold)
    labels = _mask_to_labels(times_valid, pred, scores, source=source)

    # ContextAE/HourAE와 동일한 공통 metric 함수를 사용해 비교 조건을 맞춘다.
    eval_bundle = evaluate_valid(
        y_valid,
        scores,
        threshold,
        vus_window=None,
        sample_minutes=5.0,
    )
    point = eval_bundle["point"]
    event = eval_bundle["event"]
    op = eval_bundle["operational"]
    vus = eval_bundle["vus"]

    # 다른 모델과 동일하게 UTC 시각 문자열로 저장하여 UI와 평가에서 원본 시계열에 정렬한다.
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
                "RoCA score는 hypersphere center 거리 기반 loss_n이며 "
                "값이 클수록 anomaly에 가깝다. "
                "기본 threshold는 TRAIN score q0.99이다. "
                "원본 구현: ruiking04/RoCA."
            ),
        },
        "score_polarity": "higher_is_anomaly",
        "metrics": {
            "eval_split": "valid",
            "test_held_out": True,
            "score_polarity": "higher_is_anomaly",
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
            "note": "PA-F1과 Accuracy는 primary reporting에서 제외함",
        },
    }
    path = os.path.join(PRED_DIR, f"{plmn}_{source}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
