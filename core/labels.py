"""전문가 GT label을 모델 평가에서 사용하기 위한 최소 공통 기능."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LABEL_DIR = REPO / "data" / "ib_data" / "labels"


def label_path(plmn: str) -> Path:
    """한 사업자의 전문가 label JSON 경로를 반환한다."""
    return LABEL_DIR / f"{plmn}_labels.json"


def empty_label_doc(plmn: str) -> dict[str, Any]:
    """label 파일이 없을 때 사용할 최소 문서 구조를 반환한다."""
    return {
        "plmn": plmn,
        "updated_at": None,
        "labels": [],
    }


def load_labels(plmn: str) -> dict[str, Any]:
    """전문가 label JSON을 읽는다.

    모델/평가 코드에서는 UI 전용 reason 정규화가 필요하지 않으므로
    학습 코드가 필요한 labels 필드만 안정적으로 보장한다.
    """
    path = label_path(plmn)
    if not path.exists():
        return empty_label_doc(plmn)
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("plmn", plmn)
    doc.setdefault("labels", [])
    return doc


def human_anomaly_mask(
    df: pd.DataFrame,
    labels: list[dict[str, Any]] | None,
) -> np.ndarray:
    """전문가 point/range label이 각 sample을 덮는지 boolean mask로 변환한다."""
    mask = np.zeros(len(df), dtype=bool)
    if not labels or not len(df):
        return mask

    times = pd.to_datetime(df["time"], utc=True)
    for item in labels:
        if not isinstance(item, dict) or not item.get("start"):
            continue
        start = pd.to_datetime(item["start"], utc=True)
        end = pd.to_datetime(item.get("end") or item["start"], utc=True)
        mask |= (times >= start) & (times <= end)
    return mask
