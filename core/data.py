"""UI와 독립적인 연구용 데이터 로딩, 시간순 분할, 스케일링 유틸리티.

이 파일은 모델 학습·평가·UI가 함께 사용하는 데이터 접근의 canonical 공통 계층이다.
raw CSV 로딩, 공통 cache, PLMN ranking, 시간순 분할과 모델용 전처리를 UI와 독립적으로 제공한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.features import PAPER_PLMNS, metric_columns
from core.labels import human_anomaly_mask, load_labels

REPO = Path(__file__).resolve().parents[1]
IB_DATA_DIR = REPO / "data" / "ib_data"
DATA_DIR = IB_DATA_DIR / "raw"
CACHE_DIR = IB_DATA_DIR / "cache"
LABEL_DIR = IB_DATA_DIR / "labels"
RANK_CACHE_PATH = LABEL_DIR / "plmn_rank.csv"
RANK_SIGNATURE_PATH = CACHE_DIR / "plmn_rank.signature"

DEFAULT_WINDOW = 100
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2


def data_signature(data_dir: Path = DATA_DIR) -> tuple[tuple[str, int, int], ...]:
    """raw CSV 목록/크기/수정시간으로 cache 유효성을 판단할 signature를 만든다."""
    # 디렉터리가 없으면 빈 signature → 캐시와도 절대 일치하지 않음
    if not data_dir.exists():
        return ()
    # 각 원소: (파일명, 바이트 크기, 수정시각 ns). 정렬해 순서를 고정한다.
    rows: list[tuple[str, int, int]] = []
    for path in sorted(data_dir.glob("*.csv")):
        stat = path.stat()
        # 내용 해시 대신 메타데이터만 모아 빠른 변경 감지에 쓴다.
        rows.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(rows)


def load_plmn(
    plmn: str,
    *,
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """여러 raw CSV에서 한 PLMN의 행만 모아 시간순 DataFrame으로 반환한다.

    이 함수는 모델 학습용이므로 UI 표시용 파생 데이터를 만들지 않는다.
    cache에도 원래 raw counter와 time/PLMN column만 유지한다.
    """
    # PLMN별 pickle 캐시: raw가 같으면 CSV 재스캔을 건너뛴다.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{plmn}.pkl"
    signature = data_signature(data_dir)

    # 캐시 hit: 저장된 signature가 현재 raw와 같으면 바로 반환
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if cached.attrs.get("signature") == signature:
                return cached
        except Exception:
            # 깨진/구버전 pickle은 무시하고 아래에서 다시 만든다.
            pass

    # --- 캐시 miss: raw CSV에서 해당 PLMN만 모아 재구성 ---
    # Path.glob: 디렉터리에서 패턴(*.csv)에 맞는 파일 경로를 순회한다.
    # sorted로 파일 처리 순서를 이름 기준 고정한다.
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"raw CSV가 없습니다: {data_dir}")

    parts: list[pd.DataFrame] = []
    for path in csv_files:
        frame = pd.read_csv(path)
        if "PLMN" not in frame.columns:
            continue
        # 타입 혼재(숫자/문자)에 대비해 문자열로 비교
        part = frame[frame["PLMN"].astype(str) == str(plmn)]
        if not part.empty:
            parts.append(part)

    if not parts:
        raise ValueError(f"PLMN={plmn} 데이터가 없습니다.")

    # 여러 CSV 조각을 하나로 합치고 시간순으로 정렬
    df = pd.concat(parts, ignore_index=True)
    # sort_values("time"): time 컬럼 기준으로 오름차순 정렬
    # reset_index(drop=True): 정렬로 헝클어진 인덱스를 0,1,2...로 다시 매기고
    #   옛 인덱스는 버리고(drop) 새 연속 인덱스만 남긴다.
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    # 다음에 캐시 유효성 비교할 때 쓸 signature를 DataFrame에 붙인다.
    df.attrs["signature"] = signature

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception:
            # 디스크 쓰기 실패해도 메모리 결과는 반환한다.
            pass

    return df


def data_signature_text(data_dir: Path = DATA_DIR) -> str:
    """raw CSV signature를 파일 저장/비교에 사용할 안정적인 문자열로 반환한다."""
    # json.dumps: Python 객체(튜플 목록) → JSON 문자열로 직렬화한다.
    #   예: (("a.csv", 10, 99),) → '[["a.csv",10,99]]'
    # ensure_ascii=False: 한글 등 비ASCII를 \uXXXX로 이스케이프하지 않음
    # separators=(",", ":"): 공백 없는 콤팩트 JSON → 비교/저장 결과가 항상 동일
    return json.dumps(data_signature(data_dir), ensure_ascii=False, separators=(",", ":"))


def clear_plmn_cache(plmn: str | None = None) -> int:
    """공통 PLMN pickle cache를 삭제하고 삭제된 파일 수를 반환한다."""
    if not CACHE_DIR.exists():
        return 0
    # plmn이 주어지면 그 파일만, 아니면 cache 디렉터리의 모든 .pkl
    targets = [CACHE_DIR / f"{plmn}.pkl"] if plmn else list(CACHE_DIR.glob("*.pkl"))
    removed = 0
    for path in targets:
        # UI score cache는 별도 이름 규칙이므로 전체 삭제 시에도 건드리지 않는다.
        if "_score_" in path.name:
            continue
        if path.exists():
            path.unlink()  # 파일 삭제
            removed += 1
    return removed


def build_plmn_ranking(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """raw CSV 전체에서 PLMN별 M971 합계를 계산해 내림차순 순위를 만든다."""
    # PLMN → M971 누적합. 여러 CSV에 걸친 합을 모은다.
    sums: dict[str, int] = {}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            # 필요한 두 컬럼만 읽어 메모리를 줄인다.
            chunk = pd.read_csv(path, usecols=["PLMN", "M971"])
        except ValueError:
            # PLMN 또는 M971이 없는 CSV는 건너뛴다.
            continue
        # groupby("PLMN")["M971"].sum():
        #   같은 PLMN 행들의 M971을 더해 Series(index=PLMN, value=합)를 만든다.
        #   예: PLMN A가 10,20 → part["A"] == 30
        part = chunk.groupby("PLMN")["M971"].sum()
        # 파일마다 나온 부분합을 전역 dict에 더한다.
        # sums.get(key, 0): 처음 본 PLMN이면 0부터 시작.
        for plmn, value in part.items():
            key = str(plmn)
            sums[key] = sums.get(key, 0) + int(value)

    # sums dict → 2열 DataFrame(PLMN, M971_sum)으로 만든다.
    # keys()/values()는 같은 삽입 순서를 따르므로 행이 짝을 맞춘다.
    # sort_values(..., ascending=False): M971_sum 큰 쪽이 위로 (내림차순)
    # reset_index(drop=True): 정렬 후 인덱스를 0,1,2...로 다시 매긴다.
    rank_df = (
        pd.DataFrame({"PLMN": list(sums.keys()), "M971_sum": list(sums.values())})
        .sort_values("M971_sum", ascending=False)
        .reset_index(drop=True)
    )
    # insert(0, ...): 맨 앞 열에 rank를 넣는다. index+1 → 1등, 2등, ...
    rank_df.insert(0, "rank", rank_df.index + 1)
    return rank_df


def save_ranking_cache(rank_df: pd.DataFrame, signature: str | None = None) -> None:
    """PLMN 순위와 raw 데이터 signature를 공통 data 영역에 저장한다."""
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 순위 표 + 그 순위가 어떤 raw snapshot으로 만들어졌는지 signature를 함께 저장
    rank_df.to_csv(RANK_CACHE_PATH, index=False)
    RANK_SIGNATURE_PATH.write_text(
        signature or data_signature_text(DATA_DIR),
        encoding="utf-8",
    )


def load_or_build_ranking(top_n: int | None = None) -> pd.DataFrame:
    """현재 raw snapshot과 일치하는 순위 cache를 읽고 필요하면 다시 계산한다."""
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    signature = data_signature_text(DATA_DIR)
    cached_signature = None
    if RANK_SIGNATURE_PATH.exists():
        try:
            cached_signature = RANK_SIGNATURE_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    if RANK_CACHE_PATH.exists() and cached_signature == signature:
        # 캐시 hit: raw가 안 바뀌었으면 저장된 순위 CSV를 그대로 사용
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    elif any(DATA_DIR.glob("*.csv")):
        # raw는 있는데 캐시가 없거나 오래됨 → 다시 계산 후 저장
        rank_df = build_plmn_ranking(DATA_DIR)
        save_ranking_cache(rank_df, signature)
    elif RANK_CACHE_PATH.exists():
        # raw CSV가 없어도 예전 캐시가 있으면 그걸로 fallback
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    else:
        raise FileNotFoundError(f"No source CSV or ranking cache found under {DATA_DIR}")

    # top_n이 있으면 상위 N개만 잘라 반환
    if top_n is not None:
        rank_df = rank_df.head(top_n).copy()
    return rank_df.reset_index(drop=True)


def split_indices(
    n: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> tuple[int, int]:
    """전체 길이를 시간순 TRAIN / VALID / TEST 경계 인덱스로 변환한다."""
    # 예: n=100, 0.6/0.2 → train_end=60, valid_end=80
    #   [0:train_end)=TRAIN, [train_end:valid_end)=VALID, [valid_end:)=TEST
    # max(..., 1) / train_end+1: 구간이 비지 않도록 최소 길이 보장
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
    """시간순 TRAIN/VALID/TEST 구간의 '시각(timestamp)' 경계를 돌려준다.

    split_indices는 행 번호(인덱스)만 주지만, UI·그래프에서는
    "몇 시부터 VALID인가?"처럼 실제 시각이 필요하다. 이 함수가 그 변환을 담당한다.

    반환 예 (시간축 위 경계점들):
      t_min ── TRAIN ── train_end | valid_start ── VALID ── valid_end | test_start ── TEST ── t_max

    prediction export에 split 정보가 있으면 이를 우선 사용하고,
    없으면 기본 60/20/20 비율로 다시 계산한다.
    """
    if df is None or not len(df):
        return None

    # df는 이미 시간순이라고 가정. 첫/마지막 행 = 전체 구간 시작·끝
    times = pd.to_datetime(df["time"], utc=True)
    t_min = times.iloc[0]
    t_max = times.iloc[-1]

    # --- 경로 1: 모델이 학습 때 썼던 split 시각이 export에 있으면 그대로 사용 ---
    # (나중에 다시 비율로 나누면 경계가 어긋날 수 있어 재현성을 위해 우선)
    split = (predictions or {}).get("split") or {}
    valid_start_raw = split.get("valid_time_start")
    test_start_raw = split.get("test_time_start")

    if valid_start_raw and test_start_raw:
        valid_start = pd.to_datetime(valid_start_raw, utc=True)
        test_start = pd.to_datetime(test_start_raw, utc=True)
        train_end_raw = split.get("train_time_end")
        valid_end_raw = split.get("valid_time_end")
        # 일부 필드가 없으면 인접 경계로 대체 (train_end ≈ valid_start 직전 개념)
        return {
            "t_min": t_min,
            "train_end": pd.to_datetime(train_end_raw, utc=True) if train_end_raw else valid_start,
            "valid_start": valid_start,
            "valid_end": pd.to_datetime(valid_end_raw, utc=True) if valid_end_raw else test_start,
            "test_start": test_start,
            "t_max": t_max,
        }

    # --- 경로 2: export에 split이 없으면 비율로 인덱스 경계를 잡고 시각으로 변환 ---
    # 행이 3개 미만이면 TRAIN/VALID/TEST를 나눌 수 없음
    if len(df) < 3:
        return None

    train_end, valid_end = split_indices(
        len(df),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    # 경계가 뒤집히거나 TEST 구간이 비면 실패
    if valid_end <= train_end or valid_end >= len(df):
        return None

    # 인덱스 → timestamp:
    #   train_end(인덱스)는 "VALID가 시작되는 행"이므로
    #   TRAIN의 마지막 시각 = times[train_end - 1]
    #   VALID의 시작 시각   = times[train_end]
    # VALID/TEST도 같은 방식
    return {
        "t_min": t_min,
        "train_end": times.iloc[train_end - 1],
        "valid_start": times.iloc[train_end],
        "valid_end": times.iloc[valid_end - 1],
        "test_start": times.iloc[valid_end],
        "t_max": t_max,
    }


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
    cols = metric_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: 사용할 core feature가 없습니다.")

    # df[cols]: 선택한 metric 컬럼만 뽑은 DataFrame (행=시점, 열=feature)
    # to_numpy: pandas → 2차원 numpy 배열로 변환. 모델/스케일러 입력 형태.
    #   shape 예: (시점 수 n, feature 수)  ← float64
    values = df[cols].to_numpy(dtype=np.float64)
    # nan_to_num: 학습이 깨지지 않게 비정상 숫자를 0으로 치환
    #   NaN(결측) → 0,  +inf → 0,  -inf → 0
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    # 시간 컬럼도 UTC datetime으로 맞춰 둔다 (split 경계·라벨 매칭용)
    times = pd.to_datetime(df["time"], utc=True)

    n = len(df)
    train_end, valid_end = split_indices(n, train_ratio, valid_ratio)

    # 스케일은 TRAIN에만 fit → VALID/TEST에 같은 변환을 적용 (정보 누수 방지)
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(values[:train_end]).astype(np.float32)
    valid_s = scaler.transform(values[train_end:valid_end]).astype(np.float32)
    test_s = scaler.transform(values[valid_end:]).astype(np.float32)
    full_s = np.concatenate([train_s, valid_s, test_s], axis=0)

    # 사람(전문가) 라벨 → 시점별 anomaly bool mask
    # load_labels: PLMN별 label JSON을 읽는다.
    #   예: {"plmn": "...", "labels": [{"start": "...", "end": "..."}, ...]}
    #   파일이 없으면 labels=[] 인 빈 문서를 준다.
    # human_anomaly_mask: 각 label 구간 [start, end]에 들어가는 행을 True로 표시.
    #   결과 shape = (n,) bool 배열. True = 그 시점이 anomaly로 표시됨.
    # 주의: 학습 손실의 정답(y)으로 쓰지 않고, VALID/TEST 평가용으로만 쓴다.
    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(df, label_doc.get("labels"))
    y_valid = y_full[train_end:valid_end]
    y_test = y_full[valid_end:]

    # export/UI에서 쓸 수 있게 split 경계를 문자열 timestamp로도 남긴다.
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
