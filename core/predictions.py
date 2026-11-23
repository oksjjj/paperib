"""모델 prediction export와 평가에서 공유하는 변환 유틸리티."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    """UTF-8 JSON 파일을 dict로 읽는다."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """연속된 True 구간을 inclusive (start, end) 인덱스 목록으로 변환한다."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < len(m) and m[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def mask_to_labels(
    times: pd.Series,
    mask: np.ndarray,
    scores: np.ndarray,
    prefix: str,
) -> list[dict[str, Any]]:
    """point mask를 prediction JSON의 range label 목록으로 변환한다."""
    t = pd.to_datetime(pd.Series(times), utc=True)
    items: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(segments(mask), start=1):
        ss = np.asarray(scores[a : b + 1], dtype=np.float64)
        finite = ss[np.isfinite(ss)]
        items.append(
            {
                "id": f"{prefix}_{i:06d}",
                "kind": "range",
                "start": t.iloc[a].isoformat(),
                "end": t.iloc[b].isoformat(),
                "score": float(np.max(finite)) if finite.size else None,
                "source": prefix,
            }
        )
    return items


def metric_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    """core.metrics.evaluate_valid() 결과를 prediction JSON용 flat payload로 변환한다."""
    event = bundle["event"]
    point = bundle["point"]
    op = bundle["operational"]
    vus = bundle["vus"]
    return {
        "primary_metric": "vus_pr",
        "vus_pr": bundle["primary"]["vus_pr"],
        "pr_auc": bundle["pr_auc"],
        "r_auc_pr": vus.get("r_auc_pr"),
        "vus_window": bundle.get("vus_window"),
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
        "score_distribution": bundle.get("score_distribution"),
        "event_score_diagnostics": bundle.get("event_score_diagnostics"),
    }
