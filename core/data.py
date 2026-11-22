"""UI와 독립적인 연구용 데이터 로딩, 시간순 분할, 스케일링 유틸리티.

이 파일은 모델 학습 코드가 labeling/tool.py를 import하지 않도록 하기 위한 공통 계층이다.
raw CSV를 PLMN별로 모으고 시간순으로 정렬한 뒤, 모델이 필요한 core feature만 제공한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.features import PAPER_PLMNS, feature_columns
from core.labels import human_anomaly_mask, load_labels

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "ib_data" / "raw"
CACHE_DIR = REPO / "data" / "ib_data" / "cache"

DEFAULT_WINDOW = 100
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2


def _data_signature(data_dir: Path = DATA_DIR) -> tuple[tuple[str, int, int], ...]:
    """raw CSV 목록/크기/수정시간으로 cache 유효성을 판단할 signature를 만든다."""
    if not data_dir.exists():
        return ()
    rows: list[tuple[str, int, int]] = []
    for path in sorted(data_dir.glob("*.csv")):
        stat = path.stat()
        rows.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(rows)


def load_plmn(
    plmn: str,
    *,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """여러 raw CSV에서 한 PLMN의 행만 모아 시간순 DataFrame으로 반환한다.

    이 함수는 모델 학습용이므로 UI용 rate column이나 display mapping을 만들지 않는다.
    cache에도 원래 raw counter와 time/PLMN column만 유지한다.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{plmn}.pkl"
    signature = _data_signature(data_dir)

    if use_cache and cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if cached.attrs.get("signature") == signature:
                return cached
        except Exception:
            pass

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"raw CSV가 없습니다: {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in csv_files:
        frame = pd.read_csv(path)
        if "PLMN" not in frame.columns:
            continue
        part = frame[frame["PLMN"].astype(str) == str(plmn)]
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError(f"PLMN={plmn} 데이터가 없습니다.")

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.attrs["signature"] = signature

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception:
            pass

    return df


def split_indices(
    n: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> tuple[int, int]:
    """전체 길이를 시간순 TRAIN / VALID / TEST 경계 인덱스로 변환한다."""
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    return train_end, min(valid_end, n)


def chrono_split_bounds(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None = None,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict[str, pd.Timestamp] | None:
    """시간순 split의 실제 timestamp 경계를 반환한다.

    prediction export에 split 정보가 있으면 이를 우선 사용하고,
    없으면 기본 60/20/20 비율로 다시 계산한다.
    """
    if df is None or not len(df):
        return None

    times = pd.to_datetime(df["time"], utc=True)
    t_min = times.iloc[0]
    t_max = times.iloc[-1]

    split = (predictions or {}).get("split") or {}
    valid_start_raw = split.get("valid_time_start")
    test_start_raw = split.get("test_time_start")

    if valid_start_raw and test_start_raw:
        valid_start = pd.to_datetime(valid_start_raw, utc=True)
        test_start = pd.to_datetime(test_start_raw, utc=True)
        train_end_raw = split.get("train_time_end")
        valid_end_raw = split.get("valid_time_end")
        return {
            "t_min": t_min,
            "train_end": pd.to_datetime(train_end_raw, utc=True) if train_end_raw else valid_start,
            "valid_start": valid_start,
            "valid_end": pd.to_datetime(valid_end_raw, utc=True) if valid_end_raw else test_start,
            "test_start": test_start,
            "t_max": t_max,
        }

    if len(df) < 3:
        return None

    train_end, valid_end = split_indices(
        len(df),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    if valid_end <= train_end or valid_end >= len(df):
        return None

    return {
        "t_min": t_min,
        "train_end": times.iloc[train_end - 1],
        "valid_start": times.iloc[train_end],
        "valid_end": times.iloc[valid_end - 1],
        "test_start": times.iloc[valid_end],
        "t_max": t_max,
    }


def sliding_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """(T, F) 연속 시계열을 (N, window, F) 겹치는 window 배열로 변환한다."""
    t, f = arr.shape
    if t < window:
        raise ValueError(f"window={window}을 만들기에는 T={t}가 너무 짧습니다.")
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
    """RoCA 같은 window baseline용 공통 데이터 묶음을 만든다.

    1. raw DataFrame 로딩
    2. 11개 core feature 선택
    3. NaN/±inf를 0으로 치환
    4. 시간순 split
    5. TRAIN에만 MinMaxScaler fit
    6. VALID/TEST는 같은 scaler로 transform
    7. 전문가 label은 평가용 mask로만 준비
    """
    df = load_plmn(plmn)
    cols = feature_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: 사용할 core feature가 없습니다.")

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

    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(df, label_doc.get("labels"))
    y_valid = y_full[train_end:valid_end]
    y_test = y_full[valid_end:]

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
        "bounds": chrono_split_bounds(df),
    }
