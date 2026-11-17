"""PLMN 데이터 로딩, 시간순 분할, 스케일링, 윈도우 생성을 함께 사용하는 공통 유틸리티."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LABELING = os.path.join(_REPO, "labeling")
if _LABELING not in sys.path:
    sys.path.insert(0, _LABELING)

from tool import (  # noqa: E402
    PAPER_PLMNS,
    chrono_split_bounds,
    human_anomaly_mask,
    is_rate_metric,
    load_labels,
    load_plmn,
    metric_columns,
)

DEFAULT_WINDOW = 100
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2


def feature_columns(df: pd.DataFrame) -> list[str]:
    """논문에서 사용하는 core counter만 반환한다. rate 계열 파생 지표는 모델 입력에서 제외한다."""
    return [c for c in metric_columns(df) if not is_rate_metric(c)]


def split_indices(
    n: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> tuple[int, int]:
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    valid_end = min(valid_end, n)
    return train_end, valid_end


def sliding_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """(T, F) 형태의 연속 시계열을 (N, window, F) 형태의 겹치는 윈도우로 변환한다."""
    t, f = arr.shape
    if t < window:
        raise ValueError(f"need T>={window}, got {t}")
    n = t - window + 1
    out = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(arr),
        shape=(n, window, f),
        strides=(arr.strides[0], arr.strides[0], arr.strides[1]),
    )
    return np.array(out, copy=True)


def prepare_plmn(
    plmn: str,
    *,
    window: int = DEFAULT_WINDOW,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict[str, Any]:
    """한 사업자를 불러와 시간순으로 분할하고, TRAIN에만 MinMaxScaler를 fit한 뒤 split 배열과 메타정보를 반환한다.

    주의:
    이 함수는 RoCA처럼 window 기반 baseline에서 편리하게 쓰기 위한 공통 함수다.
    ContextAE와 HourAE는 학습 중 label을 아예 읽지 않기 위해 각 파일의
    prepare_unlabeled_plmn()을 별도로 사용한다.
    """
    if plmn not in PAPER_PLMNS and plmn != PAPER_PLMNS[0]:
        # 논문 대상 사업자 외의 입력도 연구 도구에서는 실행 가능하도록 막지 않는다.
        pass
    df = load_plmn(plmn)
    cols = feature_columns(df)
    if not cols:
        raise SystemExit(f"{plmn}: no feature columns")
    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    times = pd.to_datetime(df["time"], utc=True)

    n = len(df)
    train_end, valid_end = split_indices(n, train_ratio, valid_ratio)
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(values[:train_end]).astype(np.float32)
    valid_s = scaler.transform(values[train_end:valid_end]).astype(np.float32)
    test_s = scaler.transform(values[valid_end:]).astype(np.float32)
    full_s = np.concatenate([train_s, valid_s, test_s], axis=0)

    doc = load_labels(plmn)
    y_full = human_anomaly_mask(df, doc.get("labels"))
    y_valid = y_full[train_end:valid_end]
    y_test = y_full[valid_end:]

    bounds = chrono_split_bounds(df, predictions=None)
    split_meta = {
        "train_time_end": str(times.iloc[train_end - 1]),
        "valid_time_start": str(times.iloc[train_end]),
        "valid_time_end": str(times.iloc[valid_end - 1]),
        "test_time_start": str(times.iloc[valid_end]) if valid_end < n else str(times.iloc[-1]),
        "test_time_end": str(times.iloc[-1]),
    }

    return {
        "plmn": plmn,
        "df": df,
        "cols": cols,
        "scaler": scaler,
        "window": int(window),
        "train_end": train_end,
        "valid_end": valid_end,
        "n": n,
        "train": train_s,
        "valid": valid_s,
        "test": test_s,
        "full": full_s,
        "times": times,
        "y_valid": y_valid.astype(bool),
        "y_test": y_test.astype(bool),
        "y_full": y_full.astype(bool),
        "split": split_meta,
        "bounds": bounds,
    }


def windows_for_split(block: np.ndarray, window: int) -> np.ndarray:
    return sliding_windows(block, window)
