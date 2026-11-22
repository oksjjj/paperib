"""이상구간 라벨링 UI가 사용하는 공통 기능 모음.

이 파일은 크게 다음 역할을 담당한다.
- PLMN/feature 이름과 논문용 core feature 정의
- 원본 데이터와 캐시 로딩
- 전문가 label 저장/수정
- 모델 prediction 및 anomaly score 정렬
- 평가 metric 계산
- Plotly 그래프와 overlay 생성

즉 app.py가 화면 이벤트를 담당한다면, tool.py는 실제 데이터 처리와 시각화 로직을 담당한다.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

KST = ZoneInfo("Asia/Seoul")
# 화면에 표시할 날짜는 Asia/Seoul 기준의 한국식 형식을 사용한다.
KST_FMT = "%Y년 %m월 %d일 %H:%M"
KST_FMT_SEC = "%Y년 %m월 %d일 %H:%M:%S"
_KST_PARSE_FMTS = (
    "%Y년 %m월 %d일 %H:%M:%S",
    "%Y년 %m월 %d일 %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ib_data", "raw")
IB_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ib_data")
LABEL_DIR = os.path.join(IB_DATA_DIR, "labels")
PRED_DIR = os.path.join(IB_DATA_DIR, "predictions")
RANK_CACHE_PATH = os.path.join(LABEL_DIR, "plmn_rank.csv")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RANK_SIGNATURE_PATH = os.path.join(CACHE_DIR, "plmn_rank.signature")

# 논문 평가에 사용하는 10개 사업자 목록이다. UI 순서도 이 순서를 그대로 사용한다.
PAPER_PLMNS: tuple[str, ...] = (
    "P0480",
    "P0714",
    "P1888",
    "P0975",
    "P2309",
    "P1921",
    "P1575",
    "P0139",
    "P2291",
    "P0110",
)
PAPER_PLMN_SET: frozenset[str] = frozenset(PAPER_PLMNS)

# 전문가 라벨링과 모델 학습에 공통으로 사용하는 핵심 counter 목록이다.
# 이 순서는 그래프와 모델 입력에서 사용하는 표준 feature 순서이므로 임의로 바꾸지 않는다.
CORE_FEATURES: tuple[str, ...] = (
    "M971",  # COMB_ATTEMPT
    "M185",  # COMB_AUC_FAIL
    "M855",  # COMB_AUC_TO
    "M965",  # COMB_DNS_FAIL
    "M162",  # COMB_DNS_TO
    "M843",  # COMB_GW_FAIL
    "M430",  # COMB_GW_TO
    "M874",  # COMB_HSS_FAIL
    "M037",  # COMB_HSS_TO
    "M419",  # COMB_SM_FAIL
    "M618",  # COMB_SM_TO
)
CORE_FEATURE_SET: frozenset[str] = frozenset(CORE_FEATURES)
CORE_FEATURE_MODE = "core"

# 그래프는 metric 패널 높이를 고정하고 anomaly score 영역을 항상 위쪽에 확보한다.
# 사람 label과 모델 prediction은 서로 겹치지 않도록 아래쪽에 각각 별도 track을 사용한다.
# 이렇게 분리해야 metric/score 선과 label overlay가 서로 가리지 않는다.
FIGURE_METRICS_HEIGHT = 320
FIGURE_SCORE_HEIGHT = 100
FIGURE_LABEL_HEIGHT = 48  # per track (human + model)
FIGURE_HEIGHT = (
    FIGURE_METRICS_HEIGHT
    + FIGURE_SCORE_HEIGHT
    + FIGURE_LABEL_HEIGHT
    + FIGURE_LABEL_HEIGHT
)

ANOMALY_FILL = "rgba(201, 162, 39, 0.28)"
ANOMALY_LINE = "#c9a227"
# 모델 prediction은 전문가 label과 시각적으로 구분해서 표시한다.
MODEL_FILL = "rgba(124, 58, 237, 0.22)"
MODEL_LINE = "#7c3aed"
ANOMALY_SCORE_KEY = "__anomaly_score__"
ANOMALY_SCORE_COLOR = "#7c3aed"
ANOMALY_SCORE_THRESHOLD_COLOR = "#c026d3"

# 시간순 TRAIN / VALID / TEST 구간 배경색 정의.
# 패널 배경 위에서도 split 구간이 보이도록 적당한 투명도를 사용한다.
# split 배경색은 전문가 anomaly(금색)와 모델 prediction(보라색)과 구분되게 한다.
SPLIT_TRAIN_BAND = "rgba(14, 165, 233, 0.14)"   # sky blue
SPLIT_VALID_BAND = "rgba(16, 185, 129, 0.14)"   # emerald
SPLIT_TEST_BAND = "rgba(244, 114, 182, 0.12)"   # pink
SPLIT_BOUNDARY = "#0e7490"
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2

# 흰 배경에서 여러 시계열을 구분하기 쉽도록 중간 밝기의 색상들을 사용한다.
SERIES_COLORWAY = (
    "#3d7ab5",  # blue
    "#d9655a",  # coral red
    "#3da86a",  # green
    "#9b6bb8",  # purple
    "#d4a017",  # gold
    "#4a90a4",  # steel
    "#c06a5a",  # brick
    "#2e9a85",  # teal
    "#8e6ba8",  # soft purple
    "#d4833a",  # orange
    "#5b8fbf",  # sky
    "#c97b72",  # rose
    "#4caf77",  # mint green
    "#a07cbc",  # lilac
    "#b8a03a",  # olive gold
    "#5b9bd5",  # light steel
    "#b8875a",  # tan
    "#3cb09a",  # sea green
    "#9a7aab",  # plum
    "#c97a55",  # rust
)

# 아래 이름이 붙은 shape/annotation은 실제 label이 아니라 라벨링 작업 중 잠깐 표시하는 안내선이다.
PENDING_ANCHOR_NAME = "pending_range_anchor"
PENDING_RANGE_FILL_NAME = "pending_range_fill"
LABEL_HIGHLIGHT_NAME = "label_highlight"
LABEL_HIGHLIGHT_START = "label_highlight_start"
LABEL_HIGHLIGHT_END = "label_highlight_end"
VALUE_CURSOR_NAME = "value_cursor"


PLMN_MAPPING_PATH = os.path.join(DATA_DIR, "plmn_mapping.txt")
METRIC_MAPPING_PATH = os.path.join(DATA_DIR, "metric_mapping.txt")

_PLMN_ORIGINAL: dict[str, str] | None = None
_METRIC_ORIGINAL: dict[str, str] | None = None
_MAPPING_ENABLED = True


def set_mapping_enabled(enabled: bool) -> None:
    """PLMN/metric 원래 이름을 보여주는 mapping lookup 사용 여부를 설정한다. 공유용 환경에서는 비활성화하면 비식별화 ID만 표시된다."""
    global _MAPPING_ENABLED
    _MAPPING_ENABLED = bool(enabled)


def mapping_enabled() -> bool:
    return _MAPPING_ENABLED


def _load_mapping_tables() -> None:
    """로컬 mapping 파일이 존재하면 처음 필요할 때만 읽어 메모리에 캐시한다."""
    global _PLMN_ORIGINAL, _METRIC_ORIGINAL
    if _PLMN_ORIGINAL is not None and _METRIC_ORIGINAL is not None:
        return

    _PLMN_ORIGINAL = {}
    _METRIC_ORIGINAL = {}

    if os.path.exists(PLMN_MAPPING_PATH):
        plmn_df = pd.read_csv(PLMN_MAPPING_PATH, sep="\t")
        _PLMN_ORIGINAL = {
            str(row["masked_plmn"]): str(row["original_plmn"])
            for _, row in plmn_df.iterrows()
        }

    if os.path.exists(METRIC_MAPPING_PATH):
        metric_df = pd.read_csv(METRIC_MAPPING_PATH, sep="\t")
        _METRIC_ORIGINAL = {
            str(row["masked_metric"]): str(row["original_metric"])
            for _, row in metric_df.iterrows()
        }


def reload_mappings() -> None:
    """mapping 파일을 교체한 뒤 다시 읽을 수 있도록 기존 캐시를 비우고 재로딩한다."""
    global _PLMN_ORIGINAL, _METRIC_ORIGINAL
    _PLMN_ORIGINAL = None
    _METRIC_ORIGINAL = None
    _load_mapping_tables()


def plmn_original(plmn: str) -> str | None:
    if not _MAPPING_ENABLED:
        return None
    _load_mapping_tables()
    assert _PLMN_ORIGINAL is not None
    return _PLMN_ORIGINAL.get(plmn)


def metric_original(metric: str) -> str | None:
    if not _MAPPING_ENABLED:
        return None
    _load_mapping_tables()
    assert _METRIC_ORIGINAL is not None
    return _METRIC_ORIGINAL.get(metric)


def plmn_original_short(plmn: str) -> str | None:
    """원래 PLMN 문자열에서 'PLMN_' 접두어를 제거한 짧은 값을 반환한다."""
    original = plmn_original(plmn)
    if original is None:
        return None
    if original.startswith("PLMN_"):
        return original[len("PLMN_") :]
    return original


def display_plmn(plmn: str) -> str:
    """mapping이 있으면 비식별화 PLMN과 원래 PLMN을 함께 표시하고, 없으면 비식별화 값만 반환한다."""
    short = plmn_original_short(plmn)
    return f"{plmn} ({short})" if short else plmn


M658_COL = "M658"
M696_COL = "M696"
M971_COL = "M971"
S_RATE_KEY = "S_RATE"
A_RATE_KEY = "A_RATE"
S_RATE_COLOR = "#00BFFF"
A_RATE_COLOR = "#e8590c"
REF_LINE_WIDTH = 2
REF_LINE_DASH = "3px,2px"
REF_LINE = dict(width=REF_LINE_WIDTH, dash=REF_LINE_DASH)
RATE_METRICS = frozenset({S_RATE_KEY, A_RATE_KEY})
RATE_Y_RANGE = (0.0, 1.0)


def is_rate_metric(metric: str) -> bool:
    return metric in RATE_METRICS


def rate_metrics_available(
    df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
) -> list[str]:
    """현재 데이터에 계산 가능한 rate metric 목록을 반환한다."""
    cols: set[str] = set(metrics or [])
    if df is not None:
        cols.update(metric_columns(df))
    return [key for key in (S_RATE_KEY, A_RATE_KEY) if key in cols]


def overlay_metrics_available(
    df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
) -> list[str]:
    """현재 데이터에서 UI overlay로 추가할 수 있는 파생 metric 목록을 반환한다."""
    return list(rate_metrics_available(df, metrics))


def metric_division_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.astype("float64")
    den = denominator.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def add_rate_columns(df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """필요한 원본 counter가 있으면 UI용 SUCCESS/ACCEPT rate column을 추가한다. 이미 있으면 기본적으로 다시 계산하지 않는다."""
    if force or S_RATE_KEY not in df.columns:
        if M658_COL in df.columns and M971_COL in df.columns:
            df[S_RATE_KEY] = metric_division_rate(df[M658_COL], df[M971_COL])
    if force or A_RATE_KEY not in df.columns:
        if M696_COL in df.columns and M971_COL in df.columns:
            df[A_RATE_KEY] = metric_division_rate(df[M696_COL], df[M971_COL])
    return df


def load_plmn(
    plmn: str,
    data_dir: str = DATA_DIR,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """한 사업자의 시계열을 여러 CSV에서 모아 시간순으로 정렬해 반환한다. 반복 로딩을 줄이기 위해 pickle cache를 사용한다."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{plmn}.pkl")
    signature = _data_signature(data_dir)

    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            if cached.attrs.get("signature") == signature:
                # 최신 pickle에는 rate가 저장되어 있으므로, 오래된 캐시에 rate가 없을 때만 다시 계산한다.
                return add_rate_columns(cached)
        except Exception:
            pass

    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    parts = []
    for file in csv_files:
        temp = pd.read_csv(os.path.join(data_dir, file))
        temp = temp[temp["PLMN"] == plmn]
        if not temp.empty:
            parts.append(temp)
    if not parts:
        raise ValueError(f"No rows found for PLMN={plmn}")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = add_rate_columns(df, force=True)
    df.attrs["signature"] = signature

    if use_cache:
        try:
            df.to_pickle(cache_path)
        except Exception:
            pass
    return df


def display_training_feature(
    metric: str,
    *,
    feature_mode: str | None = None,
) -> str:
    """모델 입력 feature를 UI에서 읽기 쉬운 이름으로 표시한다. comb_share 모드에서는 M971 대비 비율 표현을 붙인다."""
    name = display_metric(metric)
    if (
        feature_mode == "comb_share"
        and metric != M971_COL
        and not is_rate_metric(metric)
    ):
        return f"{name} ÷ M971"
    return name


def display_metric(metric: str) -> str:
    """metric ID를 화면용 이름으로 바꾼다. 로컬 mapping이 있으면 원래 이름도 함께 표시한다."""
    if metric == S_RATE_KEY:
        return "SUCCESS rate"
    if metric == A_RATE_KEY:
        return "ACCEPT rate"
    original = metric_original(metric)
    return f"{metric} ({original})" if original else metric


def window_plot_indices(
    values: np.ndarray,
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    max_points: int,
) -> np.ndarray:
    """전체 데이터 범위와 현재 확대 구간을 고려해 Plotly에 그릴 sample 인덱스를 계산한다."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)
    detail_start, detail_end = detail_window(df, start, end)
    pos = window_positions(df, detail_start, detail_end)
    ends = np.array([0, n - 1], dtype=int)
    if pos is None:
        return minmax_indices(values, max_points)
    return np.union1d(ends, pos[minmax_indices(values[pos], max_points)])


def to_kst(ts) -> pd.Timestamp:
    """입력 timestamp를 Asia/Seoul timezone으로 정규화한다."""
    t = pd.to_datetime(ts, utc=True)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(KST)


def format_kst(ts, *, seconds: bool = False) -> str:
    """timestamp를 한국식 KST 날짜 문자열로 변환한다."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return ""
    t = to_kst(ts)
    return t.strftime(KST_FMT_SEC if seconds else KST_FMT)


def to_plot_time(ts) -> pd.Timestamp:
    """Plotly 축 표시용으로 KST wall-clock 값을 timezone 없는 Timestamp로 바꾼다."""
    return to_kst(ts).tz_localize(None)


def to_plot_times(values) -> np.ndarray:
    """여러 timestamp를 한 번에 Plotly 표시용 KST-naive datetime 배열로 변환한다."""
    s = pd.to_datetime(pd.Series(values), utc=True)
    return s.dt.tz_convert(KST).dt.tz_localize(None).to_numpy()


def chrono_split_bounds(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None = None,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict[str, pd.Timestamp] | None:
    """시간순 TRAIN/VALID/TEST 경계를 계산한다. prediction export에 저장된 split 정보가 있으면 이를 우선 사용한다."""
    if df is None or not len(df):
        return None
    times = pd.to_datetime(df["time"], utc=True)
    t_min = times.iloc[0]
    t_max = times.iloc[-1]

    split = (predictions or {}).get("split") or {}
    v0 = split.get("valid_time_start")
    t0 = split.get("test_time_start")
    if v0 and t0:
        valid_start = pd.to_datetime(v0, utc=True)
        test_start = pd.to_datetime(t0, utc=True)
        train_end = split.get("train_time_end")
        valid_end = split.get("valid_time_end")
        return {
            "t_min": t_min,
            "train_end": pd.to_datetime(train_end, utc=True) if train_end else valid_start,
            "valid_start": valid_start,
            "valid_end": pd.to_datetime(valid_end, utc=True) if valid_end else test_start,
            "test_start": test_start,
            "t_max": t_max,
        }

    n = len(df)
    if n < 3:
        return None
    train_end_i = max(int(n * train_ratio), 1)
    valid_end_i = max(int(n * (train_ratio + valid_ratio)), train_end_i + 1)
    valid_end_i = min(valid_end_i, n - 1)
    if valid_end_i <= train_end_i:
        return None
    return {
        "t_min": t_min,
        "train_end": times.iloc[train_end_i - 1],
        "valid_start": times.iloc[train_end_i],
        "valid_end": times.iloc[valid_end_i - 1],
        "test_start": times.iloc[valid_end_i],
        "t_max": t_max,
    }


def add_split_region_overlays(
    fig: go.Figure,
    bounds: dict[str, pd.Timestamp] | None,
) -> None:
    """그래프 뒤쪽에 TRAIN/VALID/TEST 배경 band와 경계선을 추가한다."""
    if not bounds:
        return
    t_min = bounds["t_min"]
    valid_start = bounds["valid_start"]
    test_start = bounds["test_start"]
    t_max = bounds["t_max"]

    def _band(x0, x1, fill: str) -> dict[str, Any]:
        return dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=to_plot_time(x0),
            x1=to_plot_time(x1),
            y0=0,
            y1=1,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
            editable=False,
        )

    def _vline(x) -> dict[str, Any]:
        xp = to_plot_time(x)
        return dict(
            type="line",
            xref="x",
            yref="paper",
            x0=xp,
            x1=xp,
            y0=0,
            y1=1,
            line=dict(color=SPLIT_BOUNDARY, width=3.0, dash="dash"),
            layer="above",
            editable=False,
        )

    def _label(x0, x1, text: str, bgcolor: str, border: str) -> dict[str, Any]:
        mid = x0 + (x1 - x0) / 2
        return dict(
            x=to_plot_time(mid),
            y=1.0,
            xref="x",
            yref="paper",
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=11, color="#1e293b"),
            bgcolor=bgcolor,
            bordercolor=border,
            borderwidth=1,
            borderpad=3,
            yanchor="bottom",
        )

    new_shapes = [
        _band(t_min, valid_start, SPLIT_TRAIN_BAND),
        _band(valid_start, test_start, SPLIT_VALID_BAND),
        _band(test_start, t_max, SPLIT_TEST_BAND),
        _vline(valid_start),
        _vline(test_start),
    ]
    new_ann = [
        _label(t_min, valid_start, "TRAIN", "rgba(241, 245, 249, 0.9)", "#94a3b8"),
        _label(valid_start, test_start, "VALID", "rgba(237, 233, 254, 0.92)", "#8b5cf6"),
        _label(test_start, t_max, "TEST", "rgba(254, 249, 195, 0.92)", "#ca8a04"),
    ]
    existing_shapes = list(fig.layout.shapes or ())
    existing_ann = list(fig.layout.annotations or ())
    # panel 배경을 먼저 만든 뒤 split tint를 추가해야 아래 layer에서도 구간 색이 보인다.
    fig.layout.shapes = tuple(existing_shapes + new_shapes)
    fig.layout.annotations = tuple(new_ann + existing_ann)


def parse_time(value) -> pd.Timestamp:
    """사용자/UI에서 들어온 시간을 UTC timestamp로 변환한다. timezone이 없으면 KST로 해석한다."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        raise ValueError("empty timestamp")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        s = str(value).strip()
        ts = None
        if "년" in s:
            for fmt in _KST_PARSE_FMTS:
                try:
                    ts = pd.Timestamp(datetime.strptime(s, fmt))
                    break
                except ValueError:
                    continue
        if ts is None:
            ts = pd.to_datetime(s, utc=False)
    if getattr(ts, "tzinfo", None) is None or ts.tzinfo is None:
        ts = ts.tz_localize(KST)
    return ts.tz_convert("UTC")


def ensure_label_dir() -> str:
    os.makedirs(LABEL_DIR, exist_ok=True)
    return LABEL_DIR


def filter_paper_plmns(rank_df: pd.DataFrame) -> pd.DataFrame:
    """전체 사업자 순위에서 논문 평가 대상 10개만 정해진 순서대로 추린다."""
    if rank_df is None or rank_df.empty or "PLMN" not in rank_df.columns:
        return rank_df
    order = {plmn: i for i, plmn in enumerate(PAPER_PLMNS)}
    out = rank_df[rank_df["PLMN"].astype(str).isin(order)].copy()
    out["_paper_ord"] = out["PLMN"].astype(str).map(order)
    out = (
        out.sort_values("_paper_ord", kind="stable")
        .drop(columns=["_paper_ord"])
        .reset_index(drop=True)
    )
    if "rank" in out.columns:
        out["rank"] = range(1, len(out) + 1)
    return out


def load_or_build_ranking(
    top_n: int | None = None,
    *,
    paper_only: bool = False,
) -> pd.DataFrame:
    """현재 raw 데이터 snapshot과 맞는 PLMN 순위 cache를 읽고, 오래되었으면 다시 계산한다."""
    ensure_label_dir()
    os.makedirs(CACHE_DIR, exist_ok=True)
    signature = _data_signature(DATA_DIR)
    cached_signature = None
    if os.path.exists(RANK_SIGNATURE_PATH):
        try:
            with open(RANK_SIGNATURE_PATH, encoding="utf-8") as f:
                cached_signature = f.read().strip()
        except OSError:
            pass

    if os.path.exists(RANK_CACHE_PATH) and cached_signature == signature:
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    elif any(f.endswith(".csv") for f in os.listdir(DATA_DIR)):
        rank_df = _build_ranking_from_data()
        save_ranking_cache(rank_df, signature)
    elif os.path.exists(RANK_CACHE_PATH):
        rank_df = pd.read_csv(RANK_CACHE_PATH)
    else:
        raise FileNotFoundError(f"No source CSV or ranking cache found under {DATA_DIR}")

    if top_n is not None:
        rank_df = rank_df.head(top_n).copy()

    if paper_only:
        rank_df = filter_paper_plmns(rank_df)

    rank_df = rank_df.reset_index(drop=True)
    if "PLMN" in rank_df.columns:
        rank_df["PLMN_name"] = [plmn_original_short(p) or "" for p in rank_df["PLMN"]]
        rank_df["PLMN_display"] = [display_plmn(p) for p in rank_df["PLMN"]]
    return rank_df


def save_ranking_cache(rank_df: pd.DataFrame, signature: str | None = None) -> None:
    """PLMN 순위와 그 순위를 만들 때 사용한 raw 데이터 signature를 저장한다."""
    ensure_label_dir()
    os.makedirs(CACHE_DIR, exist_ok=True)
    rank_df.to_csv(RANK_CACHE_PATH, index=False)
    with open(RANK_SIGNATURE_PATH, "w", encoding="utf-8") as f:
        f.write(signature or _data_signature(DATA_DIR))


def _build_ranking_from_data() -> pd.DataFrame:
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    sums: dict[str, int] = {}
    for file in csv_files:
        chunk = pd.read_csv(os.path.join(DATA_DIR, file), usecols=["PLMN", "M971"])
        part = chunk.groupby("PLMN")["M971"].sum()
        for plmn, value in part.items():
            sums[plmn] = sums.get(plmn, 0) + int(value)
    rank_df = (
        pd.DataFrame({"PLMN": list(sums.keys()), "M971_sum": list(sums.values())})
        .sort_values("M971_sum", ascending=False)
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", rank_df.index + 1)
    return rank_df


def _data_signature(data_dir: str) -> str:
    """raw 데이터 파일 목록/크기/수정시간으로 cache 유효성을 판단할 fingerprint를 만든다."""
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    newest = 0.0
    total = 0
    for file in csv_files:
        stat = os.stat(os.path.join(data_dir, file))
        newest = max(newest, stat.st_mtime)
        total += stat.st_size
    return f"{len(csv_files)}-{int(newest)}-{total}"


def clear_plmn_cache(plmn: str | None = None) -> int:
    """사업자별 pickle cache를 삭제하고 삭제된 파일 수를 반환한다."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    targets = (
        [f"{plmn}.pkl"]
        if plmn
        else [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    )
    removed = 0
    for name in targets:
        path = os.path.join(CACHE_DIR, name)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    return removed


def all_metric_columns(df: pd.DataFrame) -> list[str]:
    """time/PLMN/anomaly score를 제외한 모든 수치형 series column을 반환한다."""
    return [
        c
        for c in df.columns
        if c not in ("time", "PLMN", ANOMALY_SCORE_KEY)
    ]


def metric_columns(df: pd.DataFrame) -> list[str]:
    """전문가가 사용하는 core feature만 CORE_FEATURES의 고정 순서대로 반환한다. 모델 학습과 UI가 동일한 feature 공간을 사용하도록 하는 핵심 함수다."""
    present = set(df.columns)
    return [c for c in CORE_FEATURES if c in present]


def label_path(plmn: str) -> str:
    ensure_label_dir()
    return os.path.join(LABEL_DIR, f"{plmn}_labels.json")


def empty_label_doc(plmn: str, rank: int | None = None) -> dict[str, Any]:
    return {
        "plmn": plmn,
        "rank": rank,
        "updated_at": None,
        "labels": [],
    }


def load_labels(plmn: str, rank: int | None = None) -> dict[str, Any]:
    path = label_path(plmn)
    if not os.path.exists(path):
        return empty_label_doc(plmn, rank)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("plmn", plmn)
    if rank is not None:
        doc["rank"] = rank
    doc.setdefault("labels", [])
    for item in doc["labels"]:
        if isinstance(item, dict):
            item.pop("metrics", None)
            item.pop("tag", None)
            item.update(
                normalize_label_reason_fields(
                    reasons=item.get("reasons"),
                    fail_metrics=item.get("fail_metrics"),
                    note=item.get("note"),
                )
            )
    return doc


def save_labels(doc: dict[str, Any]) -> str:
    doc = dict(doc)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    cleaned = []
    for item in doc.get("labels", []):
        item = dict(item)
        item.pop("metrics", None)
        item.pop("tag", None)
        item.update(
            normalize_label_reason_fields(
                reasons=item.get("reasons"),
                fail_metrics=item.get("fail_metrics"),
                note=item.get("note"),
            )
        )
        cleaned.append(item)
    doc["labels"] = cleaned
    path = label_path(doc["plmn"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def pred_path(plmn: str, source: str = "") -> str:
    return os.path.join(PRED_DIR, f"{plmn}_{source}.json")


def list_prediction_sources(plmn: str) -> list[dict[str, str]]:
    """현재 사업자에 대해 UI에서 선택 가능한 model prediction JSON 목록을 만든다."""
    options: list[dict[str, str]] = []
    if not os.path.isdir(PRED_DIR):
        return []
    prefix = f"{plmn}_"
    for name in sorted(os.listdir(PRED_DIR)):
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        source = name[len(prefix) : -len(".json")]
        # 비교표나 보고서 JSON은 모델 prediction source 목록에서 제외한다.
        if source == "context_vs_hour24":
            continue
        path = os.path.join(PRED_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            doc = {}
        doc.pop("score_series", None)
        model = str(doc.get("model") or source)
        run = str(doc.get("run_name") or "paperib")
        n_seg = len(doc.get("labels") or [])
        cols = doc.get("feature_columns") or []
        label = f"{model} · {run} ({len(cols)}d, {n_seg}구간)"
        family = 0 if model.lower().startswith("context") else (
            1 if model.lower().startswith("hour") else (
                2 if model.lower().startswith("roca") else 9
            )
        )
        options.append(
            {
                "label": label,
                "value": source,
                "_sort": (family, source),
            }
        )
    options.sort(key=lambda o: o.pop("_sort"))
    return options

def empty_pred_doc(plmn: str, source: str = "") -> dict[str, Any]:
    return {
        "plmn": plmn,
        "source": source or "",
        "updated_at": None,
        "threshold": None,
        "labels": [],
    }


def load_predictions(
    plmn: str,
    *,
    source: str = "",
) -> dict[str, Any]:
    """선택한 model prediction JSON을 읽어 UI에서 사용할 수 있는 형태로 반환한다."""
    if not source:
        return empty_pred_doc(plmn, source="")
    path = pred_path(plmn, source=source)
    if not os.path.isfile(path):
        return empty_pred_doc(plmn, source=source)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("plmn", plmn)
    doc.setdefault("source", source)
    doc.setdefault("labels", [])
    _migrate_anomalytransformer_score_scale(doc)
    # hover마다 ISO 시간을 다시 파싱하지 않도록 prediction을 읽을 때 score 시간 인덱스를 한 번만 만든다.
    _ensure_score_index(doc)
    return doc


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN → None


def human_anomaly_mask(
    df: pd.DataFrame,
    labels: list[dict[str, Any]] | None,
) -> np.ndarray:
    """전문가 point/range label이 각 sample 시간을 덮는지 계산해 boolean mask로 반환한다."""
    mask = np.zeros(len(df), dtype=bool)
    if not labels or not len(df):
        return mask
    times = pd.to_datetime(df["time"], utc=True)
    for item in labels:
        start = pd.to_datetime(item["start"], utc=True)
        end = pd.to_datetime(item.get("end") or item["start"], utc=True)
        mask |= (times >= start) & (times <= end)
    return mask


def _eval_region_mask(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    *,
    metrics_split: str = "valid",
) -> np.ndarray:
    """VALID 또는 TEST 평가 구간과 유효 score 구간의 교집합 mask를 만든다."""
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool)
    times = pd.to_datetime(df["time"], utc=True)
    split = predictions.get("split") or {}
    which = "test" if metrics_split == "test" else "valid"
    if which == "test":
        t0 = split.get("test_time_start")
        t1 = split.get("test_time_end") or split.get("time_end")
    else:
        t0 = split.get("valid_time_start")
        t1 = split.get("valid_time_end")
    if t0 and t1:
        a = pd.to_datetime(t0, utc=True)
        b = pd.to_datetime(t1, utc=True)
        region = (times >= a) & (times <= b)
    elif which == "valid" and split.get("valid_time_start") and split.get("valid_time_end"):
        a = pd.to_datetime(split["valid_time_start"], utc=True)
        b = pd.to_datetime(split["valid_time_end"], utc=True)
        region = (times >= a) & (times <= b)
    else:
        region = np.ones(n, dtype=bool) if which == "valid" else np.zeros(n, dtype=bool)
    if ANOMALY_SCORE_KEY in df.columns:
        scored = np.isfinite(df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64))
        region = region & scored
    return np.asarray(region, dtype=bool)


def available_metrics_splits(predictions: dict[str, Any] | None) -> list[str]:
    """현재 prediction에서 metric을 표시할 수 있는 eval split 목록을 반환한다."""
    if not predictions:
        return ["valid"]
    out = ["valid"]
    m = predictions.get("metrics") or {}
    es = str(predictions.get("eval_split") or "")
    split = predictions.get("split") or {}
    has_test_meta = (
        "test" in es
        or m.get("test-f1") is not None
        or m.get("test-precision") is not None
        or m.get("test-n-pred") is not None
        or (split.get("test_time_start") and split.get("test_time_end"))
    )
    if has_test_meta:
        out.append("test")
    return out


def paper_metrics_from_ib_saved(
    metrics: dict[str, Any] | None,
    *,
    metrics_split: str = "valid",
) -> dict[str, dict[str, Any]]:
    """저장된 IB prediction 형식에서 논문용 metric 묶음을 읽어 UI 표준 형식으로 변환한다."""
    m = metrics or {}
    out: dict[str, dict[str, Any]] = {}
    split = "test" if metrics_split == "test" else "valid"

    if split == "valid":
        if (
            _as_float(m.get("valid-pot-f1")) is not None
            or _as_float(m.get("pot-threshold")) is not None
        ):
            out["pot"] = {
                "precision": _as_float(m.get("valid-pot-precision")),
                "recall": _as_float(m.get("valid-pot-recall")),
                "f1": _as_float(m.get("valid-pot-f1")),
                "TP": m.get("valid-pot-TP"),
                "FP": m.get("valid-pot-FP"),
                "FN": m.get("valid-pot-FN"),
                "threshold": _as_float(
                    m.get("valid-pot-threshold") or m.get("pot-threshold")
                ),
                "note": "primary · POT from train scores (valid eval, PA)",
            }
        if (
            _as_float(m.get("valid-bf-f1")) is not None
            or _as_float(m.get("valid-bf-threshold")) is not None
        ):
            out["best_f1"] = {
                "precision": _as_float(m.get("valid-bf-precision")),
                "recall": _as_float(m.get("valid-bf-recall")),
                "f1": _as_float(m.get("valid-bf-f1")),
                "threshold": _as_float(m.get("valid-bf-threshold")),
                "note": "oracle · best-F1 PA on valid labels (saved)",
            }
        if "pot" not in out and _as_float(m.get("valid-f1")) is not None:
            ratio = m.get("anormly_ratio")
            try:
                ratio_note = f"anormly_ratio={float(ratio):g}%"
            except (TypeError, ValueError):
                ratio_note = "train energy / residual percentile"
            out["pot"] = {
                "precision": _as_float(m.get("valid-precision")),
                "recall": _as_float(m.get("valid-recall")),
                "f1": _as_float(m.get("valid-f1")),
                "threshold": _as_float(
                    m.get("energy_threshold") or m.get("threshold")
                ),
                "note": f"primary thr on valid · {ratio_note}",
            }
    else:
        # TEST에서는 학습/개발 단계에서 정한 primary threshold를 그대로 적용한다.
        if _as_float(m.get("test-f1")) is not None:
            out["pot"] = {
                "precision": _as_float(m.get("test-precision")),
                "recall": _as_float(m.get("test-recall")),
                "f1": _as_float(m.get("test-f1")),
                "TP": m.get("test-TP"),
                "FP": m.get("test-FP"),
                "FN": m.get("test-FN"),
                "threshold": _as_float(
                    m.get("pot-threshold")
                    or m.get("energy_threshold")
                    or m.get("threshold")
                ),
                "note": (
                    "primary thr (train) applied on **test** · point-wise "
                    "(not re-fit on test)"
                ),
            }
            if _as_float(m.get("test-pa-f1")) is not None:
                out["pot"]["note"] += (
                    f"; saved PA F1={float(m['test-pa-f1']):.4f}"
                )

    return out


def _point_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    tp = int(np.sum(pred & actual))
    tn = int(np.sum(~pred & ~actual))
    fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def adjust_predicts(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """point-adjustment 분석용 함수. 한 GT anomaly 구간에서 한 점이라도 탐지되면 GT 구간 전체를 탐지로 채운다."""
    predict = np.asarray(pred, dtype=bool).copy()
    actual = np.asarray(label, dtype=bool)
    anomaly_state = False
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if not actual[j]:
                    break
                predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict


def score_higher_is_anomaly(
    predictions: dict[str, Any] | None = None,
    *,
    source: str | None = None,
) -> bool:
    """현재 prediction score가 클수록 이상인지 작을수록 이상인지 판별한다."""
    if predictions is None and source is None:
        return True
    pol = str((predictions or {}).get("score_polarity") or "")
    if pol in ("higher_is_anomaly", "high", "gt"):
        return True
    if pol in ("lower_is_anomaly", "low", "lt"):
        return False
    return True


def is_anomalous_score(
    score: float | np.ndarray,
    threshold: float,
    *,
    higher_is_anomaly: bool = True,
) -> bool | np.ndarray:
    """현재 score polarity를 고려해 한 score가 threshold를 넘는 anomaly인지 판단한다."""
    if higher_is_anomaly:
        return score > threshold
    return score < threshold


def compute_detection_metrics(
    score: np.ndarray,
    y_true: np.ndarray,
    threshold: float | None,
    *,
    higher_is_anomaly: bool = True,
) -> dict[str, Any]:
    """현재 threshold에서 point-wise와 point-adjusted precision/recall/F1을 계산한다."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    n = min(len(score), len(y))
    score, y = score[:n], y[:n]
    out: dict[str, Any] = {
        "threshold": threshold,
        "n_scored": int(np.isfinite(score).sum()),
        "n_gt": int(y.sum()),
        "higher_is_anomaly": bool(higher_is_anomaly),
    }
    if threshold is None or not np.isfinite(score).any():
        out["point"] = None
        out["point_adjust"] = None
        return out
    ok = np.isfinite(score)
    pred = np.zeros(n, dtype=bool)
    pred[ok] = is_anomalous_score(
        score[ok], float(threshold), higher_is_anomaly=higher_is_anomaly
    )
    if ok.any():
        first = int(np.flatnonzero(ok)[0])
        pred_e, y_e = pred[first:], y[first:]
    else:
        pred_e, y_e = pred, y
    out["point"] = _point_metrics(pred_e, y_e)
    out["point_adjust"] = _point_metrics(adjust_predicts(pred_e, y_e), y_e)
    out["n_pred"] = int(pred_e.sum())
    return out


SCORE_THRESHOLD_MODES = ("pot", "best_f1_point", "best_f1")


def best_f1_threshold(
    score: np.ndarray,
    y_true: np.ndarray,
    *,
    n_steps: int = 200,
    objective: str = "point_adjust",
    higher_is_anomaly: bool = True,
) -> tuple[float | None, dict[str, Any] | None]:
    """주어진 평가 구간에서 F1이 최대가 되는 oracle threshold를 찾는다. 논문 primary가 아니라 분석용이다."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    n = min(len(score), len(y))
    score, y = score[:n], y[:n]
    ok = np.isfinite(score)
    if not ok.any() or not y[ok].any():
        return None, None
    s = score[ok]
    yy = y[ok]
    metric_key = "point" if objective == "point" else "point_adjust"
    lo, hi = float(np.min(s)), float(np.max(s))
    if not np.isfinite(lo) or lo >= hi:
        thr = lo if np.isfinite(lo) else None
        live = (
            compute_detection_metrics(
                s, yy, thr, higher_is_anomaly=higher_is_anomaly
            )
            if thr is not None
            else None
        )
        return thr, live
    best_f1 = -1.0
    best_thr = lo
    best_live: dict[str, Any] | None = None
    for thr in np.linspace(lo, hi, max(2, int(n_steps))):
        live = compute_detection_metrics(
            s, yy, float(thr), higher_is_anomaly=higher_is_anomaly
        )
        f1 = float((live.get(metric_key) or {}).get("f1") or -1.0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_live = live
    return best_thr, best_live


def is_roca_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """현재 prediction 문서가 RoCA 결과인지 판별한다."""
    src = str(source or (predictions or {}).get("source") or "")
    model = str((predictions or {}).get("model") or "")
    return src.startswith("roca") or model == "roca"


def is_paper_backbone_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """현재 유지 중인 contamination-robust backbone 계열 prediction인지 판별한다."""
    return is_roca_prediction(predictions, source=source)


def has_vus_paper_metrics(predictions: dict[str, Any] | None) -> bool:
    """prediction 문서에 공통 VUS metric schema가 이미 들어 있는지 확인한다."""
    if not predictions:
        return False
    paper = predictions.get("paper_metrics") or {}
    metrics = predictions.get("metrics") or {}
    if paper.get("vus_pr") is not None or metrics.get("vus_pr") is not None:
        return True
    if paper.get("primary_metric") == "vus_pr" or metrics.get("primary_metric") == "vus_pr":
        return True
    if isinstance(paper.get("event"), dict) and paper.get("operational") is not None:
        return True
    return False


def paper_metrics_from_flat_metrics(
    metrics: dict[str, Any] | None,
    *,
    eval_split: str = "valid",
    threshold: float | None = None,
    threshold_method: str | None = None,
) -> dict[str, Any] | None:
    """과거 flat metric 형식을 현재 UI용 paper_metrics 구조로 변환한다."""
    m = metrics or {}
    if m.get("vus_pr") is None and m.get("primary_metric") != "vus_pr":
        return None
    return {
        "primary_metric": "vus_pr",
        "vus_pr": m.get("vus_pr"),
        "vus_roc": m.get("vus_roc"),
        "pr_auc": m.get("pr_auc"),
        "auc_pr": m.get("auc_pr"),
        "vus_window": m.get("vus_window"),
        "r_auc_pr": m.get("r_auc_pr"),
        "event": {
            "f1": m.get("event_f1"),
            "precision": m.get("event_precision"),
            "recall": m.get("event_recall"),
            "n_gt_events": m.get("n_gt_events"),
            "n_pred_events": m.get("n_pred_events"),
            "n_hit_gt": m.get("n_hit_gt"),
        },
        "point": {
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "TP": m.get("tp", m.get("TP")),
            "FP": m.get("fp", m.get("FP")),
            "FN": m.get("fn", m.get("FN")),
        },
        "operational": {
            "false_alarms_per_day": m.get("false_alarms_per_day"),
            "mean_detection_delay_min": m.get("mean_detection_delay_min"),
            "median_detection_delay_min": m.get("median_detection_delay_min"),
            "n_false_alarm_events": m.get("n_false_alarm_events"),
            "observation_days": m.get("observation_days"),
        },
        "threshold": threshold if threshold is not None else m.get("threshold"),
        "threshold_method": threshold_method or m.get("threshold_method"),
        "score_polarity": m.get("score_polarity") or "higher_is_anomaly",
        "eval_split": m.get("eval_split") or eval_split,
        "note": "PA-F1 / Accuracy excluded from primary reporting",
    }


def _load_evaluate_valid():
    """common/metrics.py의 evaluate_valid()를 지연 로딩한다."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "common" / "metrics.py"
    spec = importlib.util.spec_from_file_location("paperib_metrics_ui", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_valid


def paper_metrics_from_vus_bundle(
    bundle: dict[str, Any],
    *,
    eval_split: str = "valid",
    threshold: float | None = None,
    threshold_method: str | None = None,
) -> dict[str, Any]:
    """공통 evaluate_valid() 반환값을 UI가 사용하는 paper_metrics 구조로 바꾼다."""
    point = bundle.get("point") or {}
    event = bundle.get("event") or {}
    op = bundle.get("operational") or {}
    vus = bundle.get("vus") or {}
    thr = threshold if threshold is not None else bundle.get("threshold")
    return {
        "primary_metric": "vus_pr",
        "vus_pr": (bundle.get("primary") or {}).get("vus_pr", vus.get("vus_pr")),
        "vus_roc": vus.get("vus_roc"),
        "pr_auc": bundle.get("pr_auc"),
        "auc_pr": vus.get("auc_pr"),
        "vus_window": bundle.get("vus_window", vus.get("vus_window")),
        "r_auc_pr": vus.get("r_auc_pr"),
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
            "TP": point.get("tp", point.get("TP")),
            "FP": point.get("fp", point.get("FP")),
            "FN": point.get("fn", point.get("FN")),
        },
        "operational": {
            "false_alarms_per_day": op.get("false_alarms_per_day"),
            "mean_detection_delay_min": op.get("mean_detection_delay_min"),
            "median_detection_delay_min": op.get("median_detection_delay_min"),
            "n_false_alarm_events": op.get("n_false_alarm_events"),
            "observation_days": op.get("observation_days"),
        },
        "threshold": thr,
        "threshold_method": threshold_method,
        "score_polarity": "higher_is_anomaly",
        "eval_split": eval_split,
        "note": "PA-F1 / Accuracy excluded from primary reporting",
    }


def compute_vus_paper_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    higher_is_anomaly: bool = True,
    eval_split: str = "valid",
    threshold_method: str | None = None,
    window: int | None = None,
) -> dict[str, Any] | None:
    """현재 score와 label로 VUS/Event/운영 metric을 실시간 계산한다."""
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = min(len(y), len(s))
    if n == 0:
        return None
    y, s = y[:n], s[:n]
    thr = float(threshold)
    if not higher_is_anomaly:
        s = -s
        thr = -thr
    try:
        evaluate = _load_evaluate_valid()
        bundle = evaluate(
            y,
            s,
            thr,
            vus_window=window,
            sample_minutes=5.0,
        )
    except Exception:
        return None
    return paper_metrics_from_vus_bundle(
        bundle,
        eval_split=eval_split,
        threshold=float(threshold),
        threshold_method=threshold_method,
    )


def is_anomalytransformer_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """현재 prediction이 Anomaly Transformer 계열인지 판별한다."""
    src = str(source or (predictions or {}).get("source") or "")
    model = str((predictions or {}).get("model") or "")
    method = str((predictions or {}).get("threshold_method") or "")
    return (
        src.startswith("anomalytransformer")
        or model == "anomalytransformer"
    )


def is_baseline_percentile_prediction(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> bool:
    """TRAIN percentile threshold를 사용하는 baseline prediction인지 판별한다."""
    src = str(source or (predictions or {}).get("source") or "")
    model = str((predictions or {}).get("model") or "")
    return (
        src.startswith("roca")
        or src.startswith("classical")
        or src.startswith("lstm_ad")
        or model in ("roca", "classical", "lstm_ad")
    )


def primary_threshold_label(
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
) -> dict[str, str]:
    """현재 모델의 primary threshold 방식을 UI에 표시할 짧은 이름으로 만든다."""
    src = source or (predictions or {}).get("source")
    if is_baseline_percentile_prediction(predictions, source=src):
        model = str((predictions or {}).get("model") or "")
        if str(src).startswith("classical") or model == "classical":
            return {
                "short": "percentile",
                "title": "[percentile]",
                "radio": "percentile (train · classical)",
                "note": (
                    "primary · train score percentile on −#fires; "
                    "binary overlay = OR(IQR,EWMA) (not POT)"
                ),
                "heading": (
                    "Detection metrics — classical IQR+EWMA vs best-F1 "
                    "point / PA (valid)"
                ),
                "overlay": "percentile / OR",
            }
        if str(src).startswith("roca") or model == "roca":
            return {
                "short": "percentile",
                "title": "[percentile]",
                "radio": "percentile (train · RoCA)",
                "note": (
                    "primary thr · train top-percentile on RoCA loss_n "
                    "(higher = more anomalous). "
                    "Validation evaluation with train-score percentile threshold."
                ),
                "heading": "RoCA metrics — Event-F1 / Point P-R-F1 / operational",
                "overlay": "percentile",
            }
        return {
            "short": "percentile",
            "title": "[percentile]",
            "radio": "percentile (train residual)",
            "note": (
                "primary · LSTM-AD train residual percentile on −MSE (not POT)"
            ),
            "heading": (
                "Detection metrics — LSTM-AD percentile vs best-F1 "
                "point / PA (valid)"
            ),
            "overlay": "percentile",
        }
    if is_anomalytransformer_prediction(predictions, source=src):
        ratio = (predictions or {}).get("anormly_ratio")
        if ratio is None:
            ratio = ((predictions or {}).get("metrics") or {}).get("anormly_ratio")
        try:
            ratio_s = f"{float(ratio):g}%"
        except (TypeError, ValueError):
            ratio_s = "1%"
        return {
            "short": "percentile",
            "title": "[percentile]",
            "radio": f"percentile (train energy · {ratio_s})",
            "note": (
                f"primary · train energy percentile "
                f"(100 − anormly_ratio={ratio_s}; not POT)"
            ),
            "heading": (
                "Detection metrics — percentile vs best-F1 point / PA (valid)"
            ),
            "overlay": "percentile",
        }
    return {
        "short": "POT",
        "title": "[POT]",
        "radio": "POT (논문 primary)",
        "note": "primary · POT on train scores (SPOT; 오버레이 기본)",
        "heading": "Detection metrics — POT vs best-F1 point / PA (valid)",
        "overlay": "POT",
    }


def thresholds_from_pred_doc(predictions: dict[str, Any]) -> dict[str, float | None]:
    """prediction JSON에서 primary threshold와 분석용 oracle threshold들을 안전하게 추출한다."""
    m = predictions.get("metrics") or {}
    ss = predictions.get("score_series") or {}
    method = str(predictions.get("threshold_method") or "pot")
    cached = predictions.get("thresholds") or {}

    # primary threshold는 우선 export된 label-free threshold 정보를 사용한다.
    pot = (
        _as_float(m.get("pot-threshold"))
        or _as_float(m.get("valid-pot-threshold"))
    )
    if pot is None and method in ("percentile", "energy_percentile", "pot"):
        # 일부 기존 baseline은 TRAIN percentile threshold를 metrics.threshold에 저장한다.
        pot = _as_float(m.get("threshold"))
    if pot is None and predictions.get("score_scale") == "log10_thr_over_energy":
        pot = 0.0
    if pot is None and method in ("percentile", "energy_percentile"):
        pot = _as_float(m.get("energy_threshold"))
    # 이전 enrich 단계에서 캐시한 primary threshold가 있으면 mode 변경에도 그대로 유지한다.
    if pot is None:
        pot = _as_float(cached.get("pot"))
    # 처음 파일을 읽는 경우에만 top-level 또는 score_series의 threshold를 fallback으로 사용한다.
    if pot is None and not cached:
        pot = _as_float(ss.get("threshold")) or _as_float(predictions.get("threshold"))

    best_pa = _as_float(m.get("valid-bf-threshold"))
    if best_pa is None:
        best_pa = _as_float(cached.get("best_f1"))
    return {
        "pot": pot,
        "best_f1_point": _as_float(cached.get("best_f1_point")),
        "best_f1": best_pa,
    }


def mask_to_pred_labels(
    mask: np.ndarray,
    times: pd.Series | np.ndarray,
    *,
    scores: np.ndarray | None = None,
    id_prefix: str = "pred",
    source: str = "model",
) -> list[dict[str, Any]]:
    """boolean anomaly mask를 UI에서 그릴 연속 prediction segment 목록으로 변환한다."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    times = pd.to_datetime(pd.Series(times), utc=True)
    n = min(len(mask), len(times))
    items: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        seg_score = None
        if scores is not None:
            sl = np.asarray(scores[i:j], dtype=np.float64)
            finite = sl[np.isfinite(sl)]
            if len(finite):
                # paperib 기본 score 방향은 값이 클수록 더 anomalous한 것이다.
                seg_score = float(np.max(finite))
        kind = "point" if j == i + 1 else "range"
        items.append(
            {
                "id": f"{id_prefix}_{uuid.uuid4().hex[:8]}",
                "kind": kind,
                "start": times.iloc[i].isoformat(),
                "end": times.iloc[j - 1].isoformat(),
                "score": seg_score,
                "source": source,
            }
        )
        i = j
    return items


def rebuild_pred_labels_for_threshold(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    threshold: float,
    *,
    id_prefix: str = "pred",
) -> list[dict[str, Any]]:
    """정렬된 anomaly score에 새 threshold를 적용해 prediction segment를 다시 만든다."""
    if ANOMALY_SCORE_KEY not in df.columns or not len(df):
        return []
    score = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
    ok = np.isfinite(score)
    mask = np.zeros(len(df), dtype=bool)
    higher = score_higher_is_anomaly(predictions)
    mask[ok] = is_anomalous_score(
        score[ok], float(threshold), higher_is_anomaly=higher
    )
    return mask_to_pred_labels(
        mask,
        df["time"],
        scores=score,
        id_prefix=id_prefix,
        source=str(predictions.get("source") or "model"),
    )


def enrich_predictions_for_ui(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    human_labels: list[dict[str, Any]] | None,
    *,
    threshold_mode: str = "pot",
    metrics_split: str = "valid",
) -> dict[str, Any]:
    """prediction 문서를 UI 표시용으로 보강한다. threshold mode, split별 metric, overlay label을 한 번에 정리한다."""
    doc = dict(predictions)
    split_choice = "test" if metrics_split == "test" else "valid"
    if split_choice == "test" and "test" not in available_metrics_splits(doc):
        split_choice = "valid"
    doc["metrics_eval_split"] = split_choice

    mode = (
        threshold_mode
        if threshold_mode in SCORE_THRESHOLD_MODES
        else "pot"
    )
    thresholds = thresholds_from_pred_doc(doc)
    export_paper = doc.get("paper_metrics")
    # 가능하면 현재 모델 export가 사용하는 공통 metric schema를 우선 사용한다.
    
    flat_vus = paper_metrics_from_flat_metrics(
        doc.get("metrics") or {},
        eval_split=split_choice,
        threshold=_as_float(doc.get("threshold")),
        threshold_method=str(doc.get("threshold_method") or "") or None,
    )
    if isinstance(export_paper, dict) and (
        export_paper.get("vus_pr") is not None
        or export_paper.get("event") is not None
        or export_paper.get("primary_metric") == "vus_pr"
    ):
        paper = dict(export_paper)
        paper["eval_split"] = split_choice
        paper.setdefault("primary_metric", "vus_pr")
    elif flat_vus is not None:
        paper = flat_vus
    else:
        paper = paper_metrics_from_ib_saved(
            doc.get("metrics") or {}, metrics_split=split_choice
        )

    # oracle threshold는 분석용으로 VALID에서만 찾고 TEST에서는 절대 다시 맞추지 않는다.
    region_valid = _eval_region_mask(df, doc, metrics_split="valid")
    region = _eval_region_mask(df, doc, metrics_split=split_choice)
    y_full = human_anomaly_mask(df, human_labels)
    if ANOMALY_SCORE_KEY in df.columns and region.any():
        score = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
        score_e = np.where(region, score, np.nan)
    else:
        score_e = np.array([], dtype=np.float64)

    higher = score_higher_is_anomaly(doc)
    if ANOMALY_SCORE_KEY in df.columns and region_valid.any() and y_full[region_valid].any():
        score_all = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
        s_val = score_all[region_valid]
        y_val = y_full[region_valid]
        # point-F1 oracle은 VALID에서 실시간 계산하고, PA oracle은 저장된 값이 있으면 이를 사용한다.
        bf_pt, _ = best_f1_threshold(
            s_val, y_val, objective="point", higher_is_anomaly=higher
        )
        thresholds["best_f1_point"] = bf_pt
        if thresholds.get("best_f1") is None:
            bf_pa, _ = best_f1_threshold(
                s_val, y_val, objective="point_adjust", higher_is_anomaly=higher
            )
            thresholds["best_f1"] = bf_pa

    live_by_mode: dict[str, Any] = {}
    if region.any() and len(score_e):
        for mkey, mthr in thresholds.items():
            if mthr is None:
                continue
            live_by_mode[mkey] = compute_detection_metrics(
                score_e[region],
                y_full[region],
                mthr,
                higher_is_anomaly=higher,
            )

    # threshold mode별 prediction label은 전체 시계열 기준으로 유지하며 metric split에 맞춰 잘라내지 않는다.
    labels_by_mode: dict[str, list] = {}
    pot_thr = thresholds.get("pot")
    method = str(doc.get("threshold_method") or "pot")
    prev_by_mode = doc.get("labels_by_mode") or {}
    # export 원본 label을 한 번 보관해두어 threshold mode 전환이 primary overlay를 덮어쓰지 못하게 한다.
    # 예를 들어 best-F1 segment를 보고 다시 primary mode로 돌아올 때 원본 segment를 복원할 수 있다.
    export_labels = doc.get("export_labels")
    if export_labels is None:
        if prev_by_mode.get("pot") is not None and doc.get("threshold_mode") in (
            None,
            "pot",
        ):
            export_labels = list(prev_by_mode.get("pot") or [])
        elif not prev_by_mode:
            export_labels = list(doc.get("labels") or [])
        else:
            export_labels = list(prev_by_mode.get("pot") or [])
        doc["export_labels"] = export_labels
    else:
        export_labels = list(export_labels)

    if pot_thr is not None and method in (
        "pot",
        "percentile",
        "energy_percentile",
    ) and export_labels and split_choice == "valid":
        # VALID 화면에서는 기여 feature 정보가 들어 있는 export label을 우선 사용한다.
        labels_by_mode["pot"] = export_labels
    elif pot_thr is not None:
        labels_by_mode["pot"] = rebuild_pred_labels_for_threshold(
            df, doc, float(pot_thr), id_prefix="pot"
        )
    else:
        labels_by_mode["pot"] = export_labels

    for mode_key, prefix in (
        ("best_f1_point", "bfpt"),
        ("best_f1", "bf"),
    ):
        mthr = thresholds.get(mode_key)
        if mthr is not None:
            labels_by_mode[mode_key] = rebuild_pred_labels_for_threshold(
                df, doc, float(mthr), id_prefix=prefix
            )
        else:
            labels_by_mode[mode_key] = []

    if mode not in labels_by_mode or (
        mode in ("best_f1", "best_f1_point")
        and not labels_by_mode.get(mode)
        and labels_by_mode.get("pot")
    ):
        if thresholds.get(mode) is None:
            mode = "pot" if thresholds.get("pot") is not None else mode

    thr = thresholds.get(mode)
    if thr is None:
        thr = (
            thresholds.get("pot")
            or thresholds.get("best_f1_point")
            or thresholds.get("best_f1")
        )
        if (
            thr is not None
            and mode in ("best_f1", "best_f1_point")
            and thresholds.get(mode) is None
        ):
            mode = "pot"

    active_labels = labels_by_mode.get(mode) or []
    doc["threshold"] = thr
    doc["threshold_mode"] = mode
    doc["thresholds"] = thresholds
    # 현재 선택된 split과 threshold에 대해 VUS, Event metric, FA/day를 실시간 재계산한다.
    if thr is not None and region.any() and ANOMALY_SCORE_KEY in df.columns:
        score_all = df[ANOMALY_SCORE_KEY].to_numpy(dtype=np.float64)
        s_reg = score_all[region]
        y_reg = y_full[region]
        win = None
        try:
            win = int(doc.get("window_length") or doc.get("window") or 0) or None
        except (TypeError, ValueError):
            win = None
        live_paper = compute_vus_paper_metrics(
            y_reg,
            s_reg,
            float(thr),
            higher_is_anomaly=higher,
            eval_split=split_choice,
            threshold_method=str(doc.get("threshold_method") or mode),
            window=win,
        )
        if live_paper is not None:
            paper = live_paper
            # UI의 이전 형식 fallback도 동작하도록 flat metric key를 함께 갱신한다.
            mflat = dict(doc.get("metrics") or {})
            mflat.update(
                {
                    "primary_metric": "vus_pr",
                    "vus_pr": live_paper.get("vus_pr"),
                    "pr_auc": live_paper.get("pr_auc"),
                    "vus_window": live_paper.get("vus_window"),
                    "event_f1": (live_paper.get("event") or {}).get("f1"),
                    "event_precision": (live_paper.get("event") or {}).get("precision"),
                    "event_recall": (live_paper.get("event") or {}).get("recall"),
                    "n_gt_events": (live_paper.get("event") or {}).get("n_gt_events"),
                    "n_pred_events": (live_paper.get("event") or {}).get("n_pred_events"),
                    "n_hit_gt": (live_paper.get("event") or {}).get("n_hit_gt"),
                    "precision": (live_paper.get("point") or {}).get("precision"),
                    "recall": (live_paper.get("point") or {}).get("recall"),
                    "f1": (live_paper.get("point") or {}).get("f1"),
                    "tp": (live_paper.get("point") or {}).get("TP"),
                    "fp": (live_paper.get("point") or {}).get("FP"),
                    "fn": (live_paper.get("point") or {}).get("FN"),
                    "false_alarms_per_day": (live_paper.get("operational") or {}).get(
                        "false_alarms_per_day"
                    ),
                    "n_false_alarm_events": (live_paper.get("operational") or {}).get(
                        "n_false_alarm_events"
                    ),
                    "observation_days": (live_paper.get("operational") or {}).get(
                        "observation_days"
                    ),
                    "mean_detection_delay_min": (live_paper.get("operational") or {}).get(
                        "mean_detection_delay_min"
                    ),
                    "median_detection_delay_min": (
                        live_paper.get("operational") or {}
                    ).get("median_detection_delay_min"),
                    "eval_split": split_choice,
                }
            )
            doc["metrics"] = mflat
        elif has_vus_paper_metrics({"paper_metrics": paper, "metrics": doc.get("metrics")}):
            # vus 패키지 누락 등으로 실시간 계산이 실패하면 export 시 저장된 VUS 값을 유지한다.
            paper = dict(paper)
            paper["eval_split"] = split_choice
    doc["paper_metrics"] = paper
    doc["live_by_mode"] = live_by_mode
    doc["labels_by_mode"] = labels_by_mode
    doc["labels"] = active_labels
    ss = dict(doc.get("score_series") or {})
    if thr is not None:
        ss["threshold"] = float(thr)
    doc["score_series"] = ss
    doc["n_pred_segments"] = len(active_labels)
    doc["n_pred_points"] = int(
        sum(
            1
            if it.get("kind") == "point"
            else max(
                1,
                int(
                    (
                        pd.to_datetime(it.get("end"), utc=True)
                        - pd.to_datetime(it.get("start"), utc=True)
                    ).total_seconds()
                    / 60
                )
                + 1,
            )
            for it in active_labels
        )
    ) if active_labels else 0
    return doc


def apply_threshold_mode(
    df: pd.DataFrame,
    predictions: dict[str, Any],
    *,
    threshold_mode: str,
) -> dict[str, Any]:
    """이미 보강된 prediction에서 threshold mode만 바꿔 overlay와 metric을 갱신한다."""
    doc = dict(predictions)
    mode = (
        threshold_mode
        if threshold_mode in SCORE_THRESHOLD_MODES
        else "pot"
    )
    thresholds = dict(doc.get("thresholds") or thresholds_from_pred_doc(doc))
    labels_by_mode = dict(doc.get("labels_by_mode") or {})
    if not labels_by_mode:
        # 아직 UI용 보강 처리가 되지 않은 prediction이면 enrich_predictions_for_ui()를 먼저 호출해야 한다.
        return enrich_predictions_for_ui(
            df,
            predictions,
            None,
            threshold_mode=mode,
        )
    prefix = {"best_f1_point": "bfpt", "best_f1": "bf"}.get(mode, "pred")
    if mode in ("best_f1", "best_f1_point") and not labels_by_mode.get(mode):
        thr = thresholds.get(mode)
        if thr is not None and ANOMALY_SCORE_KEY in df.columns:
            labels_by_mode[mode] = rebuild_pred_labels_for_threshold(
                df, doc, float(thr), id_prefix=prefix
            )
        else:
            mode = "pot"
    thr = thresholds.get(mode)
    if thr is None:
        mode = "pot"
        thr = thresholds.get("pot")
    active = labels_by_mode.get(mode) or []
    doc["threshold_mode"] = mode
    doc["threshold"] = thr
    doc["thresholds"] = thresholds
    doc["labels_by_mode"] = labels_by_mode
    doc["labels"] = active
    ss = dict(doc.get("score_series") or {})
    if thr is not None:
        ss["threshold"] = float(thr)
    doc["score_series"] = ss
    doc["n_pred_segments"] = len(active)
    return doc


def _migrate_anomalytransformer_score_scale(doc: dict[str, Any]) -> None:
    """과거 Anomaly Transformer score 형식을 현재 UI score scale로 변환한다."""
    source = str(doc.get("source") or "")
    is_at = doc.get("model") == "anomalytransformer" or source.startswith(
        "anomalytransformer"
    )
    if not is_at:
        return
    if doc.get("score_scale") == "log10_thr_over_energy":
        return
    ss = doc.get("score_series") or {}
    scores_raw = ss.get("scores") or []
    if not scores_raw:
        return
    energy_thr = doc.get("energy_threshold")
    thr = doc.get("threshold")
    if energy_thr is None and thr is not None:
        # 과거 Anomaly Transformer 형식에서는 UI threshold가 -energy_threshold로 저장되었다.
        energy_thr = abs(float(thr))
    if energy_thr is None or not (float(energy_thr) > 0):
        return
    energy_thr = float(energy_thr)
    # 과거 형식의 score는 -energy이므로 현재 score 방향과 맞추기 위한 변환이 필요하다.
    energies: list[float] = []
    for v in scores_raw:
        if v is None:
            energies.append(float("nan"))
        else:
            energies.append(max(-float(v), 0.0))
    eps = 1e-15
    thr_c = max(energy_thr, eps)
    new_scores: list[float | None] = []
    for e in energies:
        if e != e:  # NaN
            new_scores.append(None)
        else:
            new_scores.append(float(np.log10(thr_c) - np.log10(max(e, eps))))
    ss = dict(ss)
    ss["scores"] = new_scores
    ss["threshold"] = 0.0
    ss["note"] = (
        "score = log10(energy_threshold) - log10(energy); "
        "higher = more normal; anomaly when score < 0"
    )
    doc["score_series"] = ss
    doc["threshold"] = 0.0
    doc["energy_threshold"] = energy_thr
    doc["score_polarity"] = "log10_thr_over_energy"
    doc["score_scale"] = "log10_thr_over_energy"
    # score scale을 바꾼 뒤 기존 segment 대표 score도 새 scale에 맞춰 다시 계산한다.
    times = ss.get("times") or []
    if times and new_scores:
        t_ns = (
            pd.to_datetime(times, utc=True, errors="coerce")
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64, copy=False)
        )
        sc = np.asarray(
            [np.nan if v is None else float(v) for v in new_scores],
            dtype=np.float64,
        )
        for item in doc.get("labels") or []:
            try:
                s = int(pd.to_datetime(item["start"], utc=True).value)
                e = int(pd.to_datetime(item["end"], utc=True).value)
            except (ValueError, TypeError, KeyError):
                continue
            mask = (t_ns >= s) & (t_ns <= e) & np.isfinite(sc)
            if np.any(mask):
                item["score"] = float(np.min(sc[mask]))
    # score migration 뒤 이전 시간 인덱스는 무효이므로 삭제해 다음 접근 때 다시 만든다.
    doc.pop("_score_index", None)


def _ensure_score_index(predictions: dict[str, Any] | None) -> dict[str, Any] | None:
    """prediction의 시간과 score를 빠르게 찾을 수 있도록 정렬된 timestamp index를 캐시한다."""
    if not predictions:
        return None
    cached = predictions.get("_score_index")
    if isinstance(cached, dict) and "ns" in cached and "scores" in cached:
        return cached
    ss = predictions.get("score_series") or {}
    times_raw = ss.get("times") or []
    scores_raw = ss.get("scores") or []
    if not times_raw or not scores_raw:
        predictions["_score_index"] = {"ns": np.array([], dtype=np.int64), "scores": np.array([], dtype=np.float64)}
        return predictions["_score_index"]
    times = pd.to_datetime(times_raw, utc=True, errors="coerce")
    scores = np.asarray(scores_raw, dtype=np.float64)
    mask = ~pd.isna(times)
    times = times[mask]
    scores = scores[mask]
    if len(times) == 0:
        predictions["_score_index"] = {"ns": np.array([], dtype=np.int64), "scores": np.array([], dtype=np.float64)}
        return predictions["_score_index"]
    # pandas 내부 datetime 단위가 달라도 비교를 안정적으로 하기 위해 항상 epoch nanosecond로 변환한다.
    ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    order = np.argsort(ns, kind="mergesort")
    index = {
        "ns": np.asarray(ns[order], dtype=np.int64),
        "scores": np.asarray(scores[order], dtype=np.float64),
    }
    predictions["_score_index"] = index
    return index


def prediction_score_lookup(
    predictions: dict[str, Any] | None,
) -> dict[pd.Timestamp, float]:
    """특정 UTC sample time의 anomaly score를 찾는 호환용 helper다."""
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return {}
    out: dict[pd.Timestamp, float] = {}
    for ns, score in zip(index["ns"], index["scores"]):
        ts = pd.Timestamp(int(ns), unit="ns", tz="UTC")
        out[ts] = float(score)
    return out


def align_scores_to_df(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None,
) -> np.ndarray:
    """prediction score를 DataFrame 각 row 시간에 맞춰 1:1 배열로 정렬한다."""
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0 or n == 0 or "time" not in df.columns:
        return out
    sample_ns = (
        pd.to_datetime(df["time"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )
    score_ns = index["ns"]
    scores = index["scores"]
    pos = np.searchsorted(score_ns, sample_ns)
    tol = np.int64(5 * 60 * 1_000_000_000)
    # 가장 가까운 시점을 찾을 때 삽입 위치의 바로 앞/뒤 두 후보만 비교하면 된다.
    left = np.clip(pos - 1, 0, len(score_ns) - 1)
    right = np.clip(pos, 0, len(score_ns) - 1)
    dt_l = np.abs(score_ns[left] - sample_ns)
    dt_r = np.abs(score_ns[right] - sample_ns)
    use_left = dt_l <= dt_r
    best_i = np.where(use_left, left, right)
    best_dt = np.where(use_left, dt_l, dt_r)
    ok = best_dt <= tol
    out[ok] = scores[best_i[ok]]
    return out


def anomaly_score_at_time(
    predictions: dict[str, Any] | None,
    ts,
    *,
    lookup: dict[pd.Timestamp, float] | None = None,
) -> float | None:
    """주어진 sample 시간의 score를 정확히 또는 5분 이내 최근접 값으로 찾는다."""
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return None
    target = pd.to_datetime(ts, utc=True)
    if getattr(target, "tzinfo", None) is None:
        target = target.tz_localize("UTC")
    sn = int(target.value)
    score_ns = index["ns"]
    scores = index["scores"]
    p = int(np.searchsorted(score_ns, sn))
    tol = 5 * 60 * 1_000_000_000
    best = None
    best_dt = None
    for cand in (p - 1, p):
        if 0 <= cand < len(score_ns):
            dt = abs(int(score_ns[cand]) - sn)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = cand
    if best is not None and best_dt is not None and best_dt <= tol:
        return float(scores[best])
    # 과거 호출부와의 호환을 위해 dict 입력 경로도 유지한다.
    if lookup:
        if target in lookup:
            return lookup[target]
        floored = target.floor("min")
        if floored in lookup:
            return lookup[floored]
    return None


def score_hover_arrays(
    predictions: dict[str, Any] | None,
) -> tuple[list[int], list[float]] | None:
    """브라우저 hover가 사용할 Plotly 시간축 millisecond 배열과 score 배열을 만든다."""
    index = _ensure_score_index(predictions)
    if not index or len(index["ns"]) == 0:
        return None
    # score 시간을 UTC에서 Plotly 표시용 KST wall-clock millisecond로 바꿔 sample_ms와 같은 좌표계로 맞춘다.
    utc_times = pd.to_datetime(index["ns"], unit="ns", utc=True)
    plot_ms = to_plot_times(utc_times).astype("datetime64[ms]").astype("int64")
    scores = np.asarray(index["scores"], dtype=np.float64)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return None
    return plot_ms[finite].tolist(), scores[finite].tolist()


def score_cache_path(plmn: str, source: str = "model") -> str:
    safe = str(source).replace("/", "_")
    return os.path.join(CACHE_DIR, f"{plmn}_score_{safe}.pkl")


def _df_time_ns(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_datetime(df["time"], utc=True)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64, copy=False)
    )


def write_score_cache(
    plmn: str,
    source: str,
    *,
    values: np.ndarray,
    time_ns: np.ndarray,
    threshold: float | None = None,
    pred_mtime: float | None = None,
    score_scale: str | None = None,
) -> str:
    """DataFrame row에 맞춰 정렬된 anomaly score를 별도 cache 파일로 저장한다."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = score_cache_path(plmn, source)
    payload = {
        "plmn": plmn,
        "source": source,
        "n_rows": int(len(values)),
        "time_ns": np.asarray(time_ns, dtype=np.int64),
        "values": np.asarray(values, dtype=np.float64),
        "threshold": None if threshold is None else float(threshold),
        "pred_mtime": pred_mtime,
        "score_scale": score_scale,
    }
    pd.to_pickle(payload, path)
    return path


def ensure_score_column(
    df: pd.DataFrame,
    predictions: dict[str, Any] | None,
    *,
    source: str | None = None,
    plmn: str | None = None,
) -> pd.DataFrame:
    """prediction score를 현재 DataFrame 행에 맞춰 ANOMALY_SCORE_KEY column으로 붙인다."""
    out = df
    if ANOMALY_SCORE_KEY in out.columns:
        out = out.drop(columns=[ANOMALY_SCORE_KEY])
    if not predictions or not len(out):
        return out
    plmn = str(plmn or predictions.get("plmn") or "")
    source = str(source or predictions.get("source") or "model")
    score_scale = predictions.get("score_scale")
    if not plmn:
        values = align_scores_to_df(out, predictions)
        out = out.copy()
        out[ANOMALY_SCORE_KEY] = values
        return out

    ppath = pred_path(plmn, source)
    pred_mtime = os.path.getmtime(ppath) if os.path.isfile(ppath) else None
    time_ns = _df_time_ns(out)
    cpath = score_cache_path(plmn, source)
    thr = None
    ss = predictions.get("score_series") or {}
    if ss.get("threshold") is not None:
        thr = float(ss["threshold"])
    elif predictions.get("threshold") is not None:
        thr = float(predictions["threshold"])
    if os.path.isfile(cpath):
        try:
            cached = pd.read_pickle(cpath)
            if (
                cached.get("pred_mtime") == pred_mtime
                and cached.get("score_scale") == score_scale
                and cached.get("threshold") == thr
                and int(cached.get("n_rows") or -1) == len(out)
                and np.array_equal(
                    np.asarray(cached.get("time_ns"), dtype=np.int64), time_ns
                )
            ):
                out = out.copy()
                out[ANOMALY_SCORE_KEY] = np.asarray(cached["values"], dtype=np.float64)
                return out
        except Exception:
            pass

    values = align_scores_to_df(out, predictions)
    try:
        write_score_cache(
            plmn,
            source,
            values=values,
            time_ns=time_ns,
            threshold=thr,
            pred_mtime=pred_mtime,
            score_scale=str(score_scale) if score_scale is not None else None,
        )
    except Exception:
        pass
    out = out.copy()
    out[ANOMALY_SCORE_KEY] = values
    return out


def score_at_row(row: pd.Series) -> float | None:
    """DataFrame 한 row에 이미 붙어 있는 anomaly score를 읽는다."""
    if ANOMALY_SCORE_KEY not in row.index:
        return None
    try:
        v = float(row[ANOMALY_SCORE_KEY])
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def add_model_indicator(
    fig: go.Figure,
    item: dict[str, Any],
    *,
    highlight: bool = False,
    draw_edges: bool = True,
    row: int | None = None,
) -> bool:
    """읽기 전용 모델 prediction overlay를 Plotly figure에 추가한다."""
    s_utc = pd.to_datetime(item["start"], utc=True)
    e_utc = pd.to_datetime(item["end"], utc=True)
    s = to_plot_time(s_utc)
    e = to_plot_time(e_utc)
    is_range = item.get("kind") == "range" or s_utc != e_utc
    fill = "rgba(124, 58, 237, 0.45)" if highlight else MODEL_FILL
    line = "#4c1d95" if highlight else MODEL_LINE
    edge_w = 3 if highlight else 2
    lid = str(item.get("id") or "")

    def _add(shape: dict[str, Any]) -> None:
        if row is not None:
            fig.add_shape(shape, row=row, col=1)
        else:
            shape = {
                **shape,
                "xref": shape.get("xref", "x"),
                "yref": "paper",
            }
            fig.add_shape(shape)

    def _edge(x, which: str) -> None:
        _add(
            dict(
                type="line",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                line=dict(color=line, width=edge_w, dash="dot"),
                layer="above",
                editable=False,
                name=f"model_edge_{which}:{lid}" if lid else None,
            )
        )

    if is_range:
        _add(
            dict(
                type="rect",
                x0=s,
                x1=e,
                y0=0,
                y1=1,
                fillcolor=fill,
                line=dict(width=0),
                layer="above",
                editable=False,
                name=f"model_fill:{lid}" if lid else None,
            )
        )
        if draw_edges or highlight:
            _edge(s, "start")
            _edge(e, "end")
    else:
        _edge(s, "start")
    return is_range


# 예측 segment가 너무 많으면 Plotly가 느려지므로 화면에 그릴 shape 수를 제한한다.
MAX_MODEL_OVERLAYS = 48


def model_overlays_for_plot(
    items: list[dict[str, Any]] | None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    highlight_id: str | None = None,
    max_items: int = MAX_MODEL_OVERLAYS,
) -> list[dict[str, Any]]:
    """현재 보이는 구간에서 그릴 model prediction overlay를 선택하고 너무 많으면 제한한다."""
    if not items:
        return []
    t0 = pd.to_datetime(start, utc=True) if start is not None else None
    t1 = pd.to_datetime(end, utc=True) if end is not None else None
    hi = str(highlight_id) if highlight_id is not None else None

    ranked: list[tuple[float, float, dict[str, Any]]] = []
    selected_item: dict[str, Any] | None = None
    for item in items:
        try:
            s = pd.to_datetime(item["start"], utc=True)
            e = pd.to_datetime(item["end"], utc=True)
        except (ValueError, TypeError, KeyError):
            continue
        if hi is not None and str(item.get("id") or "") == hi:
            selected_item = item
        if t0 is not None and t1 is not None and (e < t0 or s > t1):
            continue
        dur = max((e - s).total_seconds(), 60.0)
        try:
            scv = float(item["score"]) if item.get("score") is not None else 0.0
        except (TypeError, ValueError):
            scv = 0.0
        # 겹치는 prediction 후보 중 긴 구간을 우선하고, 그다음 더 anomalous한 score를 우선한다.
        ranked.append((dur, -scv, item))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if selected_item is not None:
        sid = str(selected_item.get("id") or "")
        out.append(selected_item)
        if sid:
            seen.add(sid)
    for _, _, item in ranked:
        iid = str(item.get("id") or "")
        if iid and iid in seen:
            continue
        out.append(item)
        if iid:
            seen.add(iid)
        if len(out) >= max_items:
            break
    return out


def labels_to_frame(doc: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in doc.get("labels", []):
        kind = item.get("kind", "point")
        start = format_kst(item.get("start"))
        end = format_kst(item.get("end"))
        interval = start if kind == "point" else f"{start} → {end}"
        rows.append(
            {
                "id": item.get("id"),
                "형식": "점" if kind == "point" else "구간",
                "시각 / 구간 (KST)": interval,
                "판정 기준": label_reason_summary(item),
                "updated_at": format_kst(item.get("updated_at"), seconds=True),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "형식",
                "시각 / 구간 (KST)",
                "판정 기준",
                "updated_at",
            ]
        )
    return pd.DataFrame(rows)


LABEL_REASON_DEFS: list[tuple[str, str]] = [
    ("attempt_drop", "M971 급락"),
    ("attempt_spike", "M971 급증"),
    ("fail_surge", "FAIL/TO 등 실패·타임아웃 폭증"),
    ("other", "기타 (메모 권장)"),
]
LABEL_REASON_CODES = frozenset(code for code, _ in LABEL_REASON_DEFS)
LABEL_REASON_LABELS = dict(LABEL_REASON_DEFS)


def normalize_label_reason_fields(
    *,
    reasons: list[str] | None = None,
    fail_metrics: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """전문가 label의 reason/fail_metrics/note 필드를 일관된 형식으로 정리한다."""
    codes = []
    for raw in reasons or []:
        code = str(raw).strip()
        if code in LABEL_REASON_CODES and code not in codes:
            codes.append(code)
    metrics: list[str] = []
    if "fail_surge" in codes:
        for raw in fail_metrics or []:
            m = str(raw).strip()
            if m and m not in metrics:
                metrics.append(m)
    note_s = (note or "").strip()
    out: dict[str, Any] = {"reasons": codes}
    if metrics:
        out["fail_metrics"] = metrics
    else:
        out["fail_metrics"] = []
    out["note"] = note_s
    return out


def label_reason_summary(item: dict[str, Any]) -> str:
    """저장된 anomaly 원인 정보를 label 목록에 표시할 짧은 한국어 문장으로 만든다."""
    codes = [c for c in (item.get("reasons") or []) if c in LABEL_REASON_CODES]
    if not codes:
        return "기준 미입력"
    parts: list[str] = []
    for c in codes:
        if c == "fail_surge":
            mets = [str(m) for m in (item.get("fail_metrics") or []) if m]
            if mets:
                shown = ", ".join(mets[:4])
                if len(mets) > 4:
                    shown += "…"
                parts.append(f"실패 폭증({shown})")
            else:
                parts.append(LABEL_REASON_LABELS.get(c, c))
        else:
            parts.append(LABEL_REASON_LABELS.get(c, c))
    note = (item.get("note") or "").strip()
    text = " · ".join(parts)
    if note:
        text = f"{text} · {note[:40]}{'…' if len(note) > 40 else ''}"
    return text


def labels_missing_reasons(doc: dict[str, Any] | None) -> list[str]:
    """원인 코드가 비어 있는 전문가 label ID 목록을 반환한다."""
    missing = []
    for item in (doc or {}).get("labels") or []:
        codes = [c for c in (item.get("reasons") or []) if c in LABEL_REASON_CODES]
        if not codes:
            missing.append(str(item.get("id")))
    return missing


def add_label(
    doc: dict[str, Any],
    *,
    kind: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    label_id: str | None = None,
    reasons: list[str] | None = None,
    fail_metrics: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    kind = kind.lower()
    if kind not in {"point", "range"}:
        raise ValueError("kind must be 'point' or 'range'")

    start_ts = parse_time(start)
    end_ts = start_ts if kind == "point" else parse_time(end)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    item = {
        "id": label_id or str(uuid.uuid4())[:8],
        "kind": kind,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    item.update(
        normalize_label_reason_fields(
            reasons=reasons,
            fail_metrics=fail_metrics,
            note=note,
        )
    )

    labels = [x for x in doc.get("labels", []) if x.get("id") != item["id"]]
    labels.append(item)
    labels.sort(key=lambda x: x["start"])
    doc["labels"] = labels
    return doc


def remove_label(doc: dict[str, Any], label_id: str) -> dict[str, Any]:
    doc["labels"] = [x for x in doc.get("labels", []) if x.get("id") != label_id]
    return doc


def update_label(doc: dict[str, Any], label_id: str, **fields: Any) -> dict[str, Any]:
    found = None
    for item in doc.get("labels", []):
        if item.get("id") == label_id:
            found = item
            break
    if found is None:
        raise KeyError(f"label id not found: {label_id}")

    if "kind" in fields:
        found["kind"] = fields["kind"]
    if "start" in fields:
        found["start"] = parse_time(fields["start"]).isoformat()
    if "end" in fields:
        found["end"] = parse_time(fields["end"]).isoformat()
    if any(k in fields for k in ("reasons", "fail_metrics", "note")):
        found.update(
            normalize_label_reason_fields(
                reasons=fields["reasons"]
                if "reasons" in fields
                else found.get("reasons"),
                fail_metrics=fields["fail_metrics"]
                if "fail_metrics" in fields
                else found.get("fail_metrics"),
                note=fields["note"] if "note" in fields else found.get("note"),
            )
        )
    if found["kind"] == "point":
        found["end"] = found["start"]
    else:
        start_ts = parse_time(found["start"])
        end_ts = parse_time(found["end"])
        if end_ts < start_ts:
            found["start"], found["end"] = end_ts.isoformat(), start_ts.isoformat()
    found["updated_at"] = datetime.now(timezone.utc).isoformat()

    doc["labels"] = sorted(doc["labels"], key=lambda x: x["start"])
    return doc


def ranked_metric_pairs(
    row: pd.Series, cols: list[str], *, nonzero_only: bool = False
) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for col in cols:
        val = row[col]
        if pd.isna(val):
            continue
        fval = float(val)
        if nonzero_only and fval == 0:
            continue
        pairs.append((col, fval))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def format_metric_value(
    value: float,
    *,
    metric: str | None = None,
    attempt: float | None = None,
) -> str:
    """metric 값을 보기 좋게 포맷하고 가능한 경우 M971 대비 비율도 함께 표시한다."""
    if metric and is_rate_metric(metric):
        return f"{float(value) * 100.0:.1f}%"
    base = f"{round(float(value)):,}"
    if attempt is None:
        return base
    try:
        att = float(attempt)
    except (TypeError, ValueError):
        return base
    if not (att == att) or att == 0:  # NaN or zero
        return base
    return f"{base} ({float(value) / att * 100.0:.1f}%)"


# 아래 설정은 '이 시점 특성값' 패널에서 순위·이름·값을 강조할 feature에 사용한다.
HOVER_RED_METRICS = frozenset({"M855", "M037", "M430", "M162", "M618", "M520"})
HOVER_ORANGE_METRICS = frozenset({"M874", "M185", "M965", "M843", "M419"})


# feature 순위 tooltip은 확대된 구간에서만 넣는다. 전체 구간에서는 데이터가 너무 커지기 때문이다.
# 전체 보기에서는 무거운 tooltip 문자열을 생략해 figure JSON 크기를 줄인다.
# 대신 일반 hover와 아래쪽 값 패널은 그대로 동작한다.
HOVER_RANKED_MAX_FRAC = 0.45


def _row_attempt(row: pd.Series) -> float | None:
    if M971_COL not in row.index:
        return None
    try:
        att = float(row[M971_COL])
    except (TypeError, ValueError):
        return None
    if not (att == att):
        return None
    return att


def _should_embed_ranked_hover(
    df: pd.DataFrame,
    start,
    end,
) -> bool:
    if start is None or end is None or not len(df):
        return False
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    full = (tmax - tmin).total_seconds()
    if full <= 0:
        return False
    win = (
        pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
    ).total_seconds()
    return 0 < win < full * HOVER_RANKED_MAX_FRAC


def ranked_hover_text(
    row: pd.Series,
    cols: list[str],
    *,
    sep: str = "<br>",
    top_n: int | None = None,
    extra_pairs: list[tuple[str, float]] | None = None,
) -> str:
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    if extra_pairs:
        pairs = pairs + list(extra_pairs)
        pairs.sort(key=lambda x: x[1], reverse=True)
    attempt = _row_attempt(row)
    shown = pairs if top_n is None else pairs[:top_n]
    lines = [
        f"{display_metric(name)}="
        f"{format_metric_value(val, metric=name, attempt=attempt)}"
        for name, val in shown
    ]
    if top_n is not None and len(pairs) > top_n:
        lines.append(f"... +{len(pairs) - top_n} more (아래 패널 스크롤)")
    return sep.join(lines)


def _ranked_hover_texts_matrix(
    view: pd.DataFrame,
    cols: list[str],
    idx: np.ndarray,
    *,
    top_n: int | None = None,
    extra_cols: list[str] | None = None,
    extra_mat: np.ndarray | None = None,
) -> list[str]:
    """여러 sample의 feature 순위 hover 문자열을 벡터화해 생성한다."""
    use_cols = [c for c in cols if c in view.columns]
    n = len(idx)
    if n == 0:
        return []
    if not use_cols and (extra_mat is None or extra_cols is None):
        return ["(값 없음)"] * n

    labels = [display_metric(c) for c in use_cols]
    is_rate = [is_rate_metric(c) for c in use_cols]
    metric_ids = list(use_cols)
    mat = (
        view.loc[:, use_cols].to_numpy(dtype=np.float64, copy=False)[idx]
        if use_cols
        else np.zeros((n, 0), dtype=np.float64)
    )
    attempt_vals = (
        view[M971_COL].to_numpy(dtype=np.float64, copy=False)[idx]
        if M971_COL in view.columns
        else None
    )

    extra_labels: list[str] = []
    extra_is_rate: list[bool] = []
    extra_ids: list[str] = []
    if (
        extra_cols is not None
        and extra_mat is not None
        and len(extra_cols)
        and extra_mat.shape[0] == n
    ):
        extra_labels = [display_metric(c) for c in extra_cols]
        extra_is_rate = [is_rate_metric(c) for c in extra_cols]
        extra_ids = list(extra_cols)

    texts: list[str] = []
    for i in range(n):
        pairs: list[tuple[str, float, bool, str]] = []
        row = mat[i] if mat.shape[1] else None
        if row is not None:
            for j, lab in enumerate(labels):
                v = float(row[j])
                if v == v and v != 0:  # finite & nonzero
                    pairs.append((lab, v, is_rate[j], metric_ids[j]))
        if extra_labels:
            erow = extra_mat[i]
            for j, lab in enumerate(extra_labels):
                v = float(erow[j])
                if v == v and v != 0:
                    pairs.append((lab, v, extra_is_rate[j], extra_ids[j]))
        if not pairs:
            texts.append("(값 없음)")
            continue
        pairs.sort(key=lambda p: p[1], reverse=True)
        total = len(pairs)
        shown = pairs if top_n is None else pairs[:top_n]
        att = float(attempt_vals[i]) if attempt_vals is not None else None
        if att is not None and not (att == att):
            att = None
        lines = []
        for lab, v, rate, mid in shown:
            lines.append(
                f"{lab}={format_metric_value(v, metric=mid, attempt=att)}"
            )
        if top_n is not None and total > top_n:
            lines.append(f"... +{total - top_n} more")
        texts.append("<br>".join(lines))
    return texts


def _hover_sample_indices(
    view: pd.DataFrame,
    y_cols: list[str],
    *,
    start,
    end,
    max_points: int,
    filter_data: bool,
) -> np.ndarray:
    """보이지 않는 hover hit-target용 sample 인덱스를 다운샘플링한다."""
    top = view[y_cols].max(axis=1)
    if filter_data:
        return minmax_indices(top.to_numpy(), max_points)
    detail_start, detail_end = detail_window(view, start, end)
    pos = window_positions(view, detail_start, detail_end)
    if pos is None:
        return minmax_indices(top.to_numpy(), max_points)
    return pos[minmax_indices(top.to_numpy()[pos], max_points)]


def _add_ranked_hover_anchor(
    fig: go.Figure,
    view: pd.DataFrame,
    *,
    cols: list[str],
    row: int,
    uid: str,
    start,
    end,
    max_points: int,
    filter_data: bool,
    extra_cols: list[str] | None = None,
    extra_series: dict[str, pd.Series] | None = None,
    top_n: int | None = None,
) -> None:
    """feature 순위 tooltip을 띄우기 위한 보이지 않는 marker trace를 추가한다."""
    if not len(view) or not cols:
        return
    y_cols = [c for c in cols if c in view.columns]
    if not y_cols:
        return
    idx = _hover_sample_indices(
        view,
        y_cols,
        start=start,
        end=end,
        max_points=max_points,
        filter_data=filter_data,
    )
    if not len(idx):
        return
    top = view[y_cols].max(axis=1).to_numpy()[idx]
    extra_mat = None
    use_extra_cols = None
    if extra_cols and extra_series:
        cols_ok = [c for c in extra_cols if c in extra_series]
        if cols_ok:
            use_extra_cols = cols_ok
            rows_idx = view.index.to_numpy()[idx]
            extra_mat = np.column_stack(
                [
                    np.asarray(extra_series[c].reindex(rows_idx), dtype=np.float64)
                    for c in cols_ok
                ]
            )
    texts = _ranked_hover_texts_matrix(
        view,
        cols,
        idx,
        top_n=top_n,
        extra_cols=use_extra_cols,
        extra_mat=extra_mat,
    )
    fig.add_trace(
        go.Scattergl(
            x=to_plot_times(view["time"].to_numpy()[idx]),
            y=top,
            mode="markers",
            marker=dict(size=14, opacity=0),
            name=uid,
            showlegend=False,
            uid=uid,
            text=texts,
            hovertemplate="%{x|%Y년 %m월 %d일 %H:%M}<br>%{text}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def ranked_hover_html(
    ts,
    row: pd.Series,
    cols: list[str],
    *,
    cols_per_row: int = 4,
    max_rows: int = 2,
    extra_pairs: list[tuple[str, float]] | None = None,
) -> str:
    """한 시점의 metric 값을 큰 순서대로 HTML grid로 만든다."""
    pairs = ranked_metric_pairs(row, cols, nonzero_only=True)
    if extra_pairs:
        pairs = pairs + list(extra_pairs)
        pairs.sort(key=lambda x: x[1], reverse=True)
    total = len(pairs)
    cap = max(1, int(cols_per_row) * max(1, int(max_rows)))
    shown = pairs[:cap]
    items = []
    for i, (name, val) in enumerate(shown, start=1):
        bg = "#f6f8fa" if ((i - 1) // cols_per_row) % 2 else "#ffffff"
        if name in HOVER_RED_METRICS and val != 0:
            color, weight = "crimson", "700"
        elif name in HOVER_ORANGE_METRICS and val != 0:
            color, weight = "darkorange", "700"
        else:
            color, weight = "inherit", "400"
        items.append(
            "<div style='display:flex;justify-content:space-between;gap:6px;"
            "padding:3px 6px;background:{bg};min-width:0;"
            "color:{color};font-weight:{weight}'>"
            "<span style='min-width:22px;flex:0 0 auto;"
            "color:{rank_color}'>{i}</span>"
            "<span style='flex:1 1 auto;white-space:nowrap;overflow:hidden;"
            "text-overflow:ellipsis' title='{label}'>{label}</span>"
            "<span style='font-variant-numeric:tabular-nums;flex:0 0 auto'>"
            "{val}</span>"
            "</div>".format(
                bg=bg,
                color=color,
                weight=weight,
                rank_color="#888" if color == "inherit" else color,
                i=i,
                label=display_metric(name),
                val=format_metric_value(
                    val, metric=name, attempt=_row_attempt(row)
                ),
            )
        )
    more = total - len(shown)
    more_txt = f" · 상위 {len(shown)}/{total}" if more > 0 else f" · {total}개 특성"
    return (
        "<div style='font-family:sans-serif;font-size:12px;width:100%'>"
        f"<div style='font-weight:600;margin-bottom:4px'>{format_kst(ts)}"
        f"<span style='color:#666;font-weight:400'> (KST){more_txt} · 값 내림차순</span>"
        "</div>"
        f"<div style='display:grid;grid-template-columns:repeat({cols_per_row},minmax(0,1fr));"
        f"gap:2px 10px'>" + "".join(items) + "</div>"
        "</div>"
    )


def data_time_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """현재 로드된 시계열의 실제 시작/끝 시간을 반환한다."""
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    return tmin, tmax


def clamp_time_range(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """요청된 x축 구간이 데이터 범위를 벗어나지 않도록 폭을 최대한 유지하며 보정한다."""
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    width = end_ts - start_ts
    full = tmax - tmin
    if width >= full:
        return tmin, tmax
    if start_ts < tmin:
        start_ts = tmin
        end_ts = tmin + width
    if end_ts > tmax:
        end_ts = tmax
        start_ts = tmax - width
    return start_ts, end_ts


def _filter_df(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["time"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        out = out[out["time"] <= pd.to_datetime(end, utc=True)]
    return out


def minmax_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    """긴 시계열을 bucket별 min/max envelope로 줄여 spike를 보존하면서 point 수를 줄인다."""
    n = int(values.shape[0])
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    buckets = max(1, max_points // 2)
    step = int(np.ceil(n / buckets))
    if step <= 1:
        return np.arange(n)

    v = np.asarray(values, dtype="float64")
    pad = buckets * step - n
    if pad > 0:
        v = np.concatenate([v, np.full(pad, np.nan)])
    grid = v.reshape(buckets, step)
    filled = ~np.all(np.isnan(grid), axis=1)
    base = (np.arange(buckets) * step)[filled]
    sub = grid[filled]
    imax = np.argmax(np.where(np.isnan(sub), -np.inf, sub), axis=1)
    imin = np.argmin(np.where(np.isnan(sub), np.inf, sub), axis=1)
    idx = np.concatenate([base + imax, base + imin, [0, n - 1]])
    idx = np.unique(idx)
    return idx[idx < n]


def plot_series(
    view: pd.DataFrame,
    cols: list[str],
    max_points: int = 0,
) -> dict[str, tuple[Any, Any]]:
    """각 metric을 최대 point 수에 맞춰 downsample한 x/y 배열로 만든다."""
    times = to_plot_times(view["time"])
    out: dict[str, tuple[Any, Any]] = {}
    for col in cols:
        values = view[col].to_numpy()
        idx = minmax_indices(values, max_points)
        out[col] = (times[idx], values[idx])
    return out


def window_positions(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> np.ndarray | None:
    """현재 확대 구간의 DataFrame row 위치 범위를 구한다."""
    n = len(df)
    if n == 0:
        return None
    mask = np.ones(n, dtype=bool)
    if start is not None:
        mask &= (df["time"] >= pd.to_datetime(start, utc=True)).to_numpy()
    if end is not None:
        mask &= (df["time"] <= pd.to_datetime(end, utc=True)).to_numpy()
    pos = np.flatnonzero(mask)
    return pos if 0 < len(pos) < n else None


def detail_window(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    pad_ratio: float = 0.35,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """확대 구간 주변에 여유 폭을 둔 고해상도 렌더링 범위를 계산한다."""
    if start is None or end is None or not len(df):
        return start, end
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts <= start_ts:
        return start_ts, end_ts
    pad = (end_ts - start_ts) * pad_ratio
    tmin = pd.to_datetime(df["time"].min(), utc=True)
    tmax = pd.to_datetime(df["time"].max(), utc=True)
    return max(tmin, start_ts - pad), min(tmax, end_ts + pad)


def window_indices(
    values: np.ndarray,
    pos: np.ndarray | None,
    max_points: int,
    context_ratio: float = 0.3,
) -> np.ndarray:
    """확대 구간은 고해상도로 유지하고 필요하면 바깥쪽은 거칠게 줄인 인덱스를 만든다."""
    if pos is None:
        return minmax_indices(values, max_points)
    ctx_points = max(200, int(max_points * context_ratio))
    return np.union1d(
        minmax_indices(values, ctx_points),
        pos[minmax_indices(values[pos], max_points)],
    )


def plot_series_window(
    df: pd.DataFrame,
    cols: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_points: int = 1500,
    *,
    include_context: bool = False,
    pad_ratio: float = 0.35,
) -> dict[str, tuple[Any, Any]]:
    """현재 보이는 구간 주변을 고해상도로 유지한 metric별 x/y 배열을 만든다."""
    if not len(df):
        return {col: ([], []) for col in cols}
    times = to_plot_times(df["time"])
    out: dict[str, tuple[Any, Any]] = {}
    for col in cols:
        values = df[col].to_numpy()
        if include_context:
            detail_start, detail_end = detail_window(df, start, end, pad_ratio=pad_ratio)
            pos = window_positions(df, detail_start, detail_end)
            idx = window_indices(values, pos, max_points)
        else:
            idx = window_plot_indices(values, df, start, end, max_points)
        out[col] = (times[idx], values[idx])
    return out


def add_label_indicator(
    fig: go.Figure,
    item: dict[str, Any],
    *,
    highlight: bool = False,
    row: int | None = None,
) -> bool:
    """전문가 anomaly label overlay를 그래프에 추가하고 range 여부를 반환한다."""
    s_utc = pd.to_datetime(item["start"], utc=True)
    e_utc = pd.to_datetime(item["end"], utc=True)
    s = to_plot_time(s_utc)
    e = to_plot_time(e_utc)
    is_range = item.get("kind") == "range" or s_utc != e_utc

    if highlight:
        color = "rgba(220, 20, 60, 0.42)"
        line = "crimson"
        line_w = 3
    else:
        color = ANOMALY_FILL
        line = ANOMALY_LINE
        line_w = 2

    def _add(shape: dict[str, Any]) -> None:
        if row is not None:
            fig.add_shape(shape, row=row, col=1)
        else:
            fig.add_shape(
                {
                    **shape,
                    "xref": shape.get("xref", "x"),
                    "yref": "paper",
                }
            )

    def _edge(x, which: str, label_id: str) -> None:
        _add(
            dict(
                type="line",
                x0=x,
                x1=x,
                y0=0,
                y1=1,
                line=dict(color=line, width=line_w),
                layer="above",
                editable=False,
                name=f"label_edge_{which}:{label_id}" if label_id else None,
            )
        )

    if is_range:
        lid = str(item.get("id") or "")
        _add(
            dict(
                type="rect",
                x0=s,
                x1=e,
                y0=0,
                y1=1,
                fillcolor=color,
                line=dict(width=0),
                layer="above",
                editable=False,
                name=f"label_fill:{lid}" if lid else None,
            )
        )
        _edge(s, "start", lid)
        _edge(e, "end", lid)
    else:
        lid = str(item.get("id") or "")
        _edge(s, "start", lid)
    return is_range


def freeze_shape_editing(fig: go.Figure) -> go.Figure:
    """기존 Plotly shape가 통째로 드래그되지 않도록 모두 non-editable로 고정한다."""
    shapes = fig.layout.shapes
    if not shapes:
        return fig
    locked = []
    for shape in shapes:
        data = shape.to_plotly_json() if hasattr(shape, "to_plotly_json") else dict(shape)
        data["editable"] = False
        locked.append(data)
    fig.layout.shapes = tuple(locked)
    return fig


def label_line(item: dict[str, Any]) -> str:
    """저장된 전문가 label 하나를 목록에 표시할 한 줄 설명으로 만든다."""
    kind = (item.get("kind") or "point").lower()
    start = format_kst(item.get("start"))
    end = format_kst(item.get("end"))
    reason = label_reason_summary(item)
    if kind == "point":
        text = f"[점] anomaly · {start} · {reason}"
    else:
        text = f"[구간] anomaly · {start} → {end} · {reason}"
    return f"{text}  ({item.get('id')})"


def pred_chrono_split_tag(
    item: dict[str, Any],
    predictions: dict[str, Any] | None,
) -> str:
    """model prediction segment 시작시간 기준으로 TRAIN/VALID/TEST 태그를 붙인다."""
    if not item:
        return ""
    split = (predictions or {}).get("split") or {}
    v0 = split.get("valid_time_start")
    t0 = split.get("test_time_start")
    if not v0 or not t0:
        return ""
    try:
        start = pd.to_datetime(item.get("start"), utc=True)
        valid_start = pd.to_datetime(v0, utc=True)
        test_start = pd.to_datetime(t0, utc=True)
    except (TypeError, ValueError):
        return ""
    if start >= test_start:
        return "TEST"
    if start >= valid_start:
        return "VALID"
    return "TRAIN"


def pred_line(
    item: dict[str, Any],
    *,
    feature_mode: str | None = None,
    with_contrib: bool = True,
    predictions: dict[str, Any] | None = None,
) -> str:
    """model prediction 하나를 목록에 표시할 한 줄 설명으로 만든다."""
    kind = (item.get("kind") or "point").lower()
    start = format_kst(item.get("start"))
    end = format_kst(item.get("end"))
    if kind == "point":
        text = f"[점] model · {start}"
    else:
        text = f"[구간] model · {start} → {end}"
    split_tag = pred_chrono_split_tag(item, predictions)
    if split_tag:
        text = f"[{split_tag}] {text}"
    if with_contrib:
        tops = item.get("top_metrics") or []
        if tops:
            names = ", ".join(
                display_training_feature(str(t.get("metric")), feature_mode=feature_mode)
                for t in tops[:3]
                if t.get("metric")
            )
            if names:
                text = f"{text} · 기여 {names}"
    return f"{text}  ({item.get('id')})"


def predictions_to_frame(doc: dict[str, Any] | None) -> pd.DataFrame:
    """model prediction segment 목록을 표 형태 DataFrame으로 변환한다."""
    rows = []
    for item in (doc or {}).get("labels") or []:
        tops = item.get("top_metrics") or []
        top_str = ", ".join(
            f"{display_metric(str(t.get('metric')))}"
            f"({float(t.get('contribution', 0)):.1f})"
            for t in tops[:5]
            if t.get("metric")
        )
        kind = item.get("kind", "point")
        start = format_kst(item.get("start"))
        end = format_kst(item.get("end"))
        interval = start if kind == "point" else f"{start} → {end}"
        rows.append(
            {
                "id": item.get("id"),
                "형식": "점" if kind == "point" else "구간",
                "시각 / 구간 (KST)": interval,
                "기여 metric (상위)": top_str,
                "score": item.get("score"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "형식",
                "시각 / 구간 (KST)",
                "기여 metric (상위)",
                "score",
            ]
        )
    return pd.DataFrame(rows)


def label_highlight_overlays(
    item: dict[str, Any],
    *,
    editable: bool = False,
    line_width: int = 1,
    row: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """선택된 전문가 label의 시작/끝을 강조하는 얇은 빨간 edge overlay를 만든다."""
    s = to_plot_time(pd.to_datetime(item["start"], utc=True))
    e = to_plot_time(pd.to_datetime(item["end"], utc=True))
    is_range = item.get("kind") == "range" or s != e
    if row is None:
        xref, yref = "x", "paper"
    elif row <= 1:
        xref, yref = "x", "y"
    else:
        xref, yref = f"x{row}", f"y{row}"

    def _edge(x, name: str) -> dict[str, Any]:
        return dict(
            type="line",
            xref=xref,
            yref=yref,
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(color="crimson", width=int(line_width)),
            layer="above",
            editable=False,
            name=name,
        )

    if is_range:
        return [_edge(s, LABEL_HIGHLIGHT_START), _edge(e, LABEL_HIGHLIGHT_END)], []
    return [_edge(s, LABEL_HIGHLIGHT_START)], []


def value_cursor_overlays(ts) -> dict[str, Any]:
    """값 탐색 모드에서 선택된 시점을 보여주는 세로 cursor overlay를 만든다."""
    x = to_plot_time(ts)
    return dict(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="royalblue", width=2, dash="dot"),
        layer="above",
        name=VALUE_CURSOR_NAME,
    )


def pending_anchor_shape(ts, *, row: int | None = None) -> dict[str, Any]:
    """range label 첫 클릭 위치를 보여주는 점선 안내선을 만든다."""
    x = to_plot_time(ts)
    if row is None:
        xref, yref = "x", "paper"
    elif row <= 1:
        xref, yref = "x", "y"
    else:
        xref, yref = f"x{row}", f"y{row}"
    return dict(
        type="line",
        xref=xref,
        yref=yref,
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        line=dict(color="royalblue", width=3, dash="dot"),
        layer="above",
        editable=False,
        name=PENDING_ANCHOR_NAME,
    )


def pending_anchor_annotation(ts) -> dict[str, Any]:
    return dict(
        x=to_plot_time(ts),
        y=1.0,
        xref="x",
        yref="paper",
        text="① 시작(왼쪽) — 오른쪽으로 이동 후 끝점 클릭",
        showarrow=False,
        yshift=-8,
        font=dict(size=11, color="white"),
        bgcolor="royalblue",
        borderpad=3,
        name=PENDING_ANCHOR_NAME,
    )


def pending_range_fill_shape(
    start_ts, end_ts=None, *, row: int | None = None
) -> dict[str, Any]:
    """range label 시작점과 현재 마우스 위치 사이의 임시 preview band를 만든다."""
    x0 = to_plot_time(start_ts)
    x1 = to_plot_time(end_ts) if end_ts is not None else x0
    show = end_ts is not None and end_ts > start_ts
    if row is None:
        xref, yref = "x", "paper"
    elif row <= 1:
        xref, yref = "x", "y"
    else:
        xref, yref = f"x{row}", f"y{row}"
    return dict(
        type="rect",
        xref=xref,
        yref=yref,
        x0=x0,
        x1=x1 if show else x0,
        y0=0,
        y1=1,
        fillcolor=ANOMALY_FILL if show else "rgba(0,0,0,0)",
        line=dict(width=0),
        layer="above",
        editable=False,
        name=PENDING_RANGE_FILL_NAME,
    )


def build_figure(
    df: pd.DataFrame,
    doc: dict[str, Any],
    *,
    metrics: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    title: str | None = None,
    max_metrics: int = 46,
    filter_data: bool = False,
    hover_values: bool = False,
    width: int | None = None,
    max_points: int = 1500,
    show_labels: bool = True,
    color_metrics: list[str] | None = None,
    highlight_id: str | None = None,
    highlight_pred_id: str | None = None,
    predictions: dict[str, Any] | None = None,
    show_predictions: bool = False,
) -> go.Figure:
    """현재 데이터, label, prediction, zoom 상태를 이용해 전체 시계열 Plotly figure를 구성한다."""
    view = _filter_df(df, start, end) if filter_data else df
    # metrics=[]는 '아무 것도 표시하지 않음'이라는 의미이므로 falsy라고 해서 전체 feature로 대체하면 안 된다.
    cols = metric_columns(df) if metrics is None else list(metrics)
    if max_metrics is not None and max_metrics >= 0:
        cols = cols[:max_metrics]
    plot_cols = list(cols)
    primary_cols = [c for c in plot_cols if not is_rate_metric(c)]
    available_rates = rate_metrics_available(df)
    if metrics is None:
        rate_cols = available_rates
    else:
        selected = set(cols)
        rate_cols = [c for c in available_rates if c in selected]
    color_source = list(color_metrics) if color_metrics else list(cols)

    series_cols = list(dict.fromkeys([*primary_cols, *rate_cols]))
    has_score_col = bool(
        show_predictions and ANOMALY_SCORE_KEY in view.columns and len(view)
    )
    if has_score_col:
        series_cols = list(dict.fromkeys([*series_cols, ANOMALY_SCORE_KEY]))
    if not len(view):
        series = {}
    elif filter_data:
        series = plot_series(view, series_cols, max_points)
    else:
        series = plot_series_window(
            view, series_cols, start, end, max_points, include_context=False
        )

    threshold = None
    if show_predictions and predictions:
        ss = predictions.get("score_series") or {}
        if ss.get("threshold") is not None:
            threshold = float(ss["threshold"])
        elif predictions.get("threshold") is not None:
            threshold = float(predictions["threshold"])

    if has_score_col:
        score_x, score_y = series.get(ANOMALY_SCORE_KEY, ([], []))
        show_score = bool(len(score_y) and np.any(np.isfinite(np.asarray(score_y, dtype=float))))
    else:
        # DataFrame에 미리 score column을 붙이지 못한 경우를 위한 fallback 경로다.
        score_vals_all = (
            align_scores_to_df(view, predictions)
            if show_predictions and predictions
            else np.full(0, np.nan, dtype=np.float64)
        )
        show_score = bool(len(score_vals_all) and np.any(np.isfinite(score_vals_all)))
        score_x, score_y = [], []
        if show_score and len(view):
            idx = window_plot_indices(score_vals_all, view, start, end, max_points)
            scored_pos = np.flatnonzero(np.isfinite(score_vals_all))
            if len(scored_pos):
                if start is not None and end is not None:
                    t0 = pd.to_datetime(start, utc=True)
                    t1 = pd.to_datetime(end, utc=True)
                    times_ns = _df_time_ns(view)
                    t0n, t1n = int(t0.value), int(t1.value)
                    in_win = scored_pos[
                        (times_ns[scored_pos] >= t0n) & (times_ns[scored_pos] <= t1n)
                    ]
                    pool = in_win if len(in_win) else scored_pos
                else:
                    pool = scored_pos
                if len(pool) > max_points:
                    step = max(1, len(pool) // max_points)
                    pool = pool[::step]
                idx = pool
            times_plot = to_plot_times(view["time"])
            score_x, score_y = times_plot[idx], score_vals_all[idx]

    rate_colors = {S_RATE_KEY: S_RATE_COLOR, A_RATE_KEY: A_RATE_COLOR}
    rate_hover = "값=%{y:.1%}<extra></extra>"
    count_hover = "값=%{y:,.0f}<extra></extra>"
    # 무거운 feature 순위 tooltip은 확대 상태에서만 생성한다.
    # 전체/거의 전체 구간에서는 일반 hover만 유지해 그래프 재생성을 가볍게 한다.
    embed_ranked = bool(hover_values) and _should_embed_ranked_hover(df, start, end)
    series_hoverinfo = "skip" if embed_ranked else None

    show_rate = bool(rate_cols)
    # score 영역과 사람/모델 label track은 데이터가 없어도 항상 자리를 확보한다.
    reserve_score = True
    reserve_labels = True
    has_score_data = bool(show_score and len(score_x))
    score_frac = FIGURE_SCORE_HEIGHT / FIGURE_HEIGHT
    metrics_frac = FIGURE_METRICS_HEIGHT / FIGURE_HEIGHT
    label_frac = FIGURE_LABEL_HEIGHT / FIGURE_HEIGHT
    if reserve_score and show_rate and reserve_labels:
        n_rows = 5
        row_heights = [
            score_frac,
            0.12,
            metrics_frac * 0.78,
            label_frac,
            label_frac,
        ]
        s = sum(row_heights)
        row_heights = [h / s for h in row_heights]
        titles = (
            "Anomaly score",
            "SUCCESS / COMB rate",
            "Metric 값",
            "사람 라벨",
            "모델 예측",
        )
        rate_row, metric_row = 2, 3
        human_label_row, model_label_row = 4, 5
        spacing = 0.05
    elif reserve_score and reserve_labels:
        n_rows = 4
        row_heights = [score_frac, metrics_frac, label_frac, label_frac]
        s = sum(row_heights)
        row_heights = [h / s for h in row_heights]
        titles = ("Anomaly score", "Metric 값", "사람 라벨", "모델 예측")
        rate_row, metric_row = None, 2
        human_label_row, model_label_row = 3, 4
        spacing = 0.05
    elif reserve_score:
        n_rows = 2
        row_heights = [
            score_frac / (score_frac + metrics_frac),
            metrics_frac / (score_frac + metrics_frac),
        ]
        titles = ("Anomaly score", "Metric 값")
        rate_row, metric_row = None, 2
        human_label_row, model_label_row = None, None
        spacing = 0.08
    else:
        n_rows = 1
        row_heights = [1.0]
        titles = ("Metric 값",)
        rate_row, metric_row = None, 1
        human_label_row, model_label_row = None, None
        spacing = 0.08

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=spacing,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    if has_score_data:
        fig.add_trace(
            go.Scattergl(
                x=score_x,
                y=score_y,
                mode="lines",
                name="Anomaly score",
                opacity=0.9,
                line=dict(color=ANOMALY_SCORE_COLOR, width=1.5),
                # 라벨 배치용 커스텀 tooltip이 hover를 담당하므로 값 탐색 click은 가능하도록 trace를 유지한다.
                hoverinfo="none",
                uid=ANOMALY_SCORE_KEY,
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line=dict(
                    color=ANOMALY_SCORE_THRESHOLD_COLOR,
                    width=1.5,
                    dash="dot",
                ),
                annotation_text="threshold",
                annotation_position="top left",
                annotation_font=dict(size=10, color=ANOMALY_SCORE_THRESHOLD_COLOR),
                row=1,
                col=1,
            )

    if rate_row is not None:
        for col in rate_cols:
            x, y = series.get(col, ([], []))
            metric_name = display_metric(col)
            rate_kw: dict[str, Any] = dict(
                x=x,
                y=y,
                mode="lines",
                name=metric_name,
                opacity=0.72,
                line=dict(
                    color=rate_colors.get(col, "#888"),
                    width=1,
                ),
                uid=col,
            )
            if series_hoverinfo:
                rate_kw["hoverinfo"] = series_hoverinfo
            else:
                rate_kw["hovertemplate"] = (
                    f"<b>{metric_name}</b><br>"
                    "%{x|%Y년 %m월 %d일 %H:%M}<br>"
                    f"{rate_hover}"
                )
            fig.add_trace(go.Scattergl(**rate_kw), row=rate_row, col=1)

    for i, col in enumerate(primary_cols):
        x, y = series.get(col, ([], []))
        metric_name = display_metric(col)
        try:
            color_i = color_source.index(col)
        except ValueError:
            color_i = i
        color = SERIES_COLORWAY[color_i % len(SERIES_COLORWAY)]
        metric_kw: dict[str, Any] = dict(
            x=x,
            y=y,
            mode="lines",
            name=metric_name,
            opacity=0.72,
            line=dict(width=1, color=color),
            uid=col,
        )
        if series_hoverinfo:
            metric_kw["hoverinfo"] = series_hoverinfo
        else:
            metric_kw["hovertemplate"] = (
                f"<b>{metric_name}</b><br>"
                "%{x|%Y년 %m월 %d일 %H:%M}<br>"
                f"{count_hover}"
            )
        fig.add_trace(go.Scattergl(**metric_kw), row=metric_row, col=1)

    data_rev = str(doc.get("plmn") or "labeling")
    if start is not None and end is not None:
        data_rev = (
            f"{data_rev}:{pd.to_datetime(start, utc=True).value}:"
            f"{pd.to_datetime(end, utc=True).value}"
        )
    if reserve_score:
        data_rev = f"{data_rev}:score"

    if embed_ranked and len(view):
        # rate 패널은 SUCC/COMB 계열을 큰 값부터 그리며 지정된 최대 sample 밀도를 사용한다.
        if rate_cols and rate_row is not None:
            _add_ranked_hover_anchor(
                fig,
                view,
                cols=list(rate_cols),
                row=rate_row,
                uid="rate-values",
                start=start,
                end=end,
                max_points=max_points,
                filter_data=filter_data,
            )

        # count metric 패널도 값이 큰 series부터 그린다.
        metric_hover_cols = list(primary_cols)
        if metric_hover_cols:
            _add_ranked_hover_anchor(
                fig,
                view,
                cols=metric_hover_cols,
                row=metric_row,
                uid="metric-values",
                start=start,
                end=end,
                max_points=max_points,
                filter_data=filter_data,
            )
    elif len(view) and (primary_cols or rate_cols) and not hover_values:
        # 순위 tooltip을 끈 경우에도 click/inspect가 가능하도록 보이지 않는 hit target을 남긴다.
        hover_y_cols = primary_cols or rate_cols
        hover_row = metric_row if primary_cols else (rate_row or 1)
        idx = _hover_sample_indices(
            view,
            hover_y_cols,
            start=start,
            end=end,
            max_points=max_points,
            filter_data=filter_data,
        )
        fig.add_trace(
            go.Scattergl(
                x=to_plot_times(view["time"].to_numpy()[idx]),
                y=view[hover_y_cols].max(axis=1).to_numpy()[idx],
                mode="markers",
                marker=dict(size=10, opacity=0),
                name="values",
                showlegend=False,
                uid="values",
                hoverinfo="skip",
            ),
            row=hover_row,
            col=1,
        )

    def _keep_label_track(row: int | None, *, uid: str, name: str) -> None:
        if row is None:
            return
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(opacity=0, size=1),
                name=name,
                showlegend=False,
                hoverinfo="skip",
                uid=uid,
            ),
            row=row,
            col=1,
        )

    _keep_label_track(human_label_row, uid="__human_label_track__", name="human-labels")
    _keep_label_track(model_label_row, uid="__model_label_track__", name="model-labels")

    if show_labels and human_label_row is not None:
        hi = str(highlight_id) if highlight_id is not None else None
        for item in doc.get("labels", []):
            label_id = item.get("id", "")
            selected = hi is not None and str(label_id) == hi
            add_label_indicator(
                fig, item, highlight=selected, row=human_label_row
            )

    if show_predictions and predictions and model_label_row is not None:
        hid_pred = str(highlight_pred_id) if highlight_pred_id is not None else None
        overlay_items = model_overlays_for_plot(
            predictions.get("labels") or [],
            start=start,
            end=end,
            highlight_id=hid_pred,
        )
        # prediction이 많을 때 모든 시작/끝 선을 그리면 Plotly가 느려지므로 선택된 구간 위주로 edge를 그린다.
        # 특히 수백 개 segment가 생기는 경우 edge shape 수를 제한해야 UI가 멈추지 않는다.
        draw_all_edges = len(overlay_items) <= 16
        for item in overlay_items:
            is_hi = bool(hid_pred and str(item.get("id") or "") == hid_pred)
            add_model_indicator(
                fig,
                item,
                highlight=is_hi,
                draw_edges=draw_all_edges or is_hi,
                row=model_label_row,
            )

    fig.update_layout(
        title=title or f"{display_plmn(str(doc.get('plmn')))} anomaly labeling",
        # metric 패널 높이는 고정하고 score/label 영역은 별도 높이로 추가한다.
        height=FIGURE_HEIGHT if reserve_score else FIGURE_METRICS_HEIGHT,
        autosize=True,
        # feature 순위 hover를 쓸 때는 같은 x의 여러 series를 함께 보여주고, 아니면 가장 가까운 한 series만 보여준다.
        hovermode="x" if embed_ranked else "closest",
        dragmode="zoom",
        showlegend=False,
        margin=dict(l=52, r=24, t=60, b=40),
        uirevision=str(doc.get("plmn") or "labeling"),
        datarevision=data_rev,
        colorway=list(SERIES_COLORWAY),
        plot_bgcolor="#ddd8cf",
        paper_bgcolor="#f3f1ec",
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="monospace",
            align="left",
            namelength=-1,
        ),
    )
    if width:
        fig.update_layout(width=width, autosize=False)
    else:
        fig.update_layout(autosize=True)

    if reserve_score:
        fig.update_yaxes(
            title_text="score",
            autorange=True,
            fixedrange=True,
            showgrid=True,
            showline=True,
            mirror=True,
            linewidth=1.5,
            linecolor="#6d28d9",
            gridcolor="rgba(124,58,237,0.22)",
            zeroline=False,
            title_font=dict(size=11, color="#5b21b6"),
            tickfont=dict(size=10, color="#5b21b6"),
            row=1,
            col=1,
        )

    if rate_row is not None:
        fig.update_yaxes(
            title_text="rate",
            autorange=True,
            fixedrange=True,
            showgrid=True,
            showline=True,
            mirror=True,
            linewidth=1.5,
            linecolor="#5f8499",
            gridcolor="rgba(95,132,153,0.28)",
            zeroline=False,
            title_font=dict(size=11, color="#3d6578"),
            tickfont=dict(size=10, color="#3d6578"),
            row=rate_row,
            col=1,
        )
    fig.update_yaxes(
        title_text="value",
        fixedrange=True,
        showline=True,
        mirror=True,
        linewidth=1.5,
        linecolor="#8a7f6e",
        gridcolor="rgba(138,127,110,0.28)",
        zeroline=False,
        title_font=dict(size=11, color="#5c5346"),
        tickfont=dict(size=10, color="#5c5346"),
        row=metric_row,
        col=1,
    )
    def _style_label_yaxis(row: int | None, *, title: str, line: str, font: str) -> None:
        if row is None:
            return
        fig.update_yaxes(
            title_text=title,
            range=[0, 1],
            autorange=False,
            fixedrange=True,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linewidth=1.5,
            linecolor=line,
            title_font=dict(size=11, color=font),
            row=row,
            col=1,
        )

    _style_label_yaxis(human_label_row, title="human", line="#a16207", font="#854d0e")
    _style_label_yaxis(model_label_row, title="model", line="#7c3aed", font="#5b21b6")

    tmin, tmax = data_time_bounds(view if len(view) else df)
    x0 = pd.to_datetime(start, utc=True) if start is not None else tmin
    x1 = pd.to_datetime(end, utc=True) if end is not None else tmax
    x0, x1 = clamp_time_range(x0, x1, tmin, tmax)
    x_range = [to_plot_time(x0), to_plot_time(x1)]
    x_axis_kwargs = dict(
        range=x_range,
        autorange=False,
        type="date",
        tickformat="%Y년 %m월 %d일\n%H:%M",
        hoverformat="%Y년 %m월 %d일 %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="royalblue",
        spikethickness=1,
        showline=True,
        mirror=True,
        linewidth=1.5,
    )
    for row_i in range(1, n_rows + 1):
        kw = dict(x_axis_kwargs)
        if n_rows == 1:
            kw["matches"] = None
            kw["title_text"] = "시간 (KST)"
            kw["linecolor"] = "#8a7f6e"
            kw["tickfont"] = dict(size=10, color="#5c5346")
            kw["title_font"] = dict(size=11, color="#5c5346")
        elif row_i == 1:
            kw["matches"] = None
            kw["linecolor"] = "#6d28d9" if reserve_score else "#5f8499"
            kw["tickfont"] = dict(
                size=10, color="#5b21b6" if reserve_score else "#3d6578"
            )
            kw["showticklabels"] = False
        elif row_i == n_rows:
            kw["matches"] = "x"
            kw["title_text"] = "시간 (KST)"
            if model_label_row is not None and row_i == model_label_row:
                kw["linecolor"] = "#7c3aed"
                kw["tickfont"] = dict(size=10, color="#5b21b6")
                kw["title_font"] = dict(size=11, color="#5b21b6")
            elif human_label_row is not None and row_i == human_label_row:
                kw["linecolor"] = "#a16207"
                kw["tickfont"] = dict(size=10, color="#854d0e")
                kw["title_font"] = dict(size=11, color="#854d0e")
            else:
                kw["linecolor"] = "#8a7f6e"
                kw["tickfont"] = dict(size=10, color="#5c5346")
                kw["title_font"] = dict(size=11, color="#5c5346")
        elif human_label_row is not None and row_i == human_label_row:
            kw["matches"] = "x"
            kw["linecolor"] = "#a16207"
            kw["tickfont"] = dict(size=10, color="#854d0e")
            kw["showticklabels"] = False
        elif metric_row is not None and row_i == metric_row:
            kw["matches"] = "x"
            kw["linecolor"] = "#8a7f6e"
            kw["tickfont"] = dict(size=10, color="#5c5346")
            kw["showticklabels"] = False
        else:
            kw["matches"] = "x"
            kw["linecolor"] = "#5f8499"
            kw["tickfont"] = dict(size=10, color="#3d6578")
            kw["showticklabels"] = False
        fig.update_xaxes(**kw, row=row_i, col=1)

    # 패널 배경을 먼저 그리고 그 위에 TRAIN/VALID/TEST tint를 같은 below layer로 추가한다.
    _style_panel_separation(
        fig,
        n_rows=n_rows,
        show_score=reserve_score,
        show_rate=show_rate,
        show_labels=reserve_labels,
    )
    add_split_region_overlays(fig, chrono_split_bounds(df, predictions))
    return fig


def _style_panel_separation(
    fig: go.Figure,
    *,
    n_rows: int = 2,
    show_score: bool = False,
    show_rate: bool = True,
    show_labels: bool = False,
) -> None:
    """subplot별 배경색, 테두리, subtitle 스타일을 적용해 패널을 시각적으로 구분한다."""
    domains = []
    for i in range(1, n_rows + 1):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        axis = getattr(fig.layout, key, None)
        dom = getattr(axis, "domain", None) if axis is not None else None
        domains.append(dom)
    if show_score and show_rate and show_labels and n_rows == 5:
        # subplot 순서: score / rate / metrics / 전문가 label / 모델 prediction
        defaults = [
            (0.84, 1.0),
            (0.70, 0.80),
            (0.22, 0.66),
            (0.11, 0.18),
            (0.0, 0.08),
        ]
    elif show_score and show_labels and n_rows == 4:
        # rate 패널이 없을 때 subplot 순서: score / metrics / 전문가 label / 모델 prediction
        defaults = [(0.82, 1.0), (0.22, 0.78), (0.11, 0.18), (0.0, 0.08)]
    elif n_rows == 4:
        defaults = [(0.82, 1.0), (0.66, 0.78), (0.14, 0.62), (0.0, 0.10)]
    elif show_score and show_labels and n_rows == 3:
        defaults = [(0.82, 1.0), (0.14, 0.78), (0.0, 0.10)]
    elif n_rows == 3:
        defaults = [(0.72, 1.0), (0.38, 0.64), (0.0, 0.30)]
    elif n_rows == 1:
        defaults = [(0.0, 1.0)]
    elif show_score and not show_rate:
        defaults = [(0.78, 1.0), (0.0, 0.70)]
    else:
        defaults = [(0.565, 1.0), (0.0, 0.435)]
    for i, dom in enumerate(domains):
        if not dom:
            domains[i] = defaults[i] if i < len(defaults) else (0.0, 1.0)

    x_key = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    x_axis = getattr(fig.layout, x_key, None) or fig.layout.xaxis
    x_dom = getattr(x_axis, "domain", None) or (0.0, 1.0)

    styles = []
    if show_score and show_rate and show_labels and n_rows == 5:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
            ("#fef9c3", "#a16207", "panel_bg_human_labels"),
            ("#f3e8ff", "#7c3aed", "panel_bg_model_labels"),
        ]
    elif show_score and show_labels and n_rows == 4:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
            ("#fef9c3", "#a16207", "panel_bg_human_labels"),
            ("#f3e8ff", "#7c3aed", "panel_bg_model_labels"),
        ]
    elif show_score and show_rate and show_labels and n_rows == 4:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
            ("#fef9c3", "#a16207", "panel_bg_labels"),
        ]
    elif show_score and show_labels and n_rows == 3:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
            ("#fef9c3", "#a16207", "panel_bg_labels"),
        ]
    elif show_score and show_rate and n_rows == 3:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]
    elif show_score and n_rows == 2:
        styles = [
            ("#efe7ff", "#6d28d9", "panel_bg_score"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]
    elif show_rate and n_rows == 2:
        styles = [
            ("#e7f1f7", "#5f8499", "panel_bg_rate"),
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]
    else:
        styles = [
            ("#f7f5f1", "#8a7f6e", "panel_bg_metrics"),
        ]

    for i, (fill, border, name) in enumerate(styles):
        if i >= len(domains):
            break
        y0, y1 = domains[i]
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x_dom[0],
            x1=x_dom[1],
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line=dict(color=border, width=1.5),
            layer="below",
            editable=False,
            name=name,
        )

    for i in range(len(domains) - 1):
        gap_mid = (domains[i + 1][1] + domains[i][0]) / 2
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.01,
            x1=0.99,
            y0=gap_mid,
            y1=gap_mid,
            line=dict(color="#9a9286", width=2),
            layer="above",
            editable=False,
            name=f"panel_divider_{i}",
        )

    score_title = dict(
        font=dict(size=12, color="#5b21b6", family="sans-serif"),
        bgcolor="rgba(239,231,255,0.96)",
        bordercolor="#6d28d9",
        borderwidth=1,
        borderpad=4,
    )
    rate_title = dict(
        font=dict(size=12, color="#1f4f63", family="sans-serif"),
        bgcolor="rgba(231,241,247,0.96)",
        bordercolor="#5f8499",
        borderwidth=1,
        borderpad=4,
    )
    metrics_title = dict(
        font=dict(size=12, color="#4a4338", family="sans-serif"),
        bgcolor="rgba(247,245,241,0.96)",
        bordercolor="#8a7f6e",
        borderwidth=1,
        borderpad=4,
    )
    human_labels_title = dict(
        font=dict(size=12, color="#854d0e", family="sans-serif"),
        bgcolor="rgba(254,249,195,0.96)",
        bordercolor="#a16207",
        borderwidth=1,
        borderpad=4,
    )
    model_labels_title = dict(
        font=dict(size=12, color="#5b21b6", family="sans-serif"),
        bgcolor="rgba(243,232,255,0.96)",
        bordercolor="#7c3aed",
        borderwidth=1,
        borderpad=4,
    )
    if show_score and show_rate and show_labels:
        title_styles = (
            score_title,
            rate_title,
            metrics_title,
            human_labels_title,
            model_labels_title,
        )
    elif show_score and show_labels:
        title_styles = (
            score_title,
            metrics_title,
            human_labels_title,
            model_labels_title,
        )
    elif show_score and show_rate:
        title_styles = (score_title, rate_title, metrics_title)
    elif show_score:
        title_styles = (score_title, metrics_title)
    elif show_rate:
        title_styles = (rate_title, metrics_title)
    else:
        title_styles = (metrics_title,)
    anns = list(fig.layout.annotations or ())
    for i, style in enumerate(title_styles):
        if i >= len(anns):
            break
        data = anns[i].to_plotly_json() if hasattr(anns[i], "to_plotly_json") else dict(anns[i])
        data.update(style)
        anns[i] = data
    if anns:
        fig.layout.annotations = tuple(anns)


def build_metric_figure(
    df: pd.DataFrame,
    doc: dict[str, Any],
    metric: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    filter_data: bool = False,
    width: int | None = None,
    max_points: int = 3000,
) -> go.Figure:
    view = _filter_df(df, start, end) if filter_data else df
    if not len(view):
        idx = []
    elif filter_data:
        idx = minmax_indices(view[metric].to_numpy(), max_points)
    else:
        idx = window_indices(
            view[metric].to_numpy(), window_positions(view, start, end), max_points
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=to_plot_times(view["time"].to_numpy()[idx]),
            y=view[metric].to_numpy()[idx],
            mode="lines",
            name=display_metric(metric),
            line=dict(width=1.5, color="steelblue"),
            hovertemplate="%{x|%Y년 %m월 %d일 %H:%M}<br>%{y:,.0f}<extra></extra>",
        )
    )
    for item in doc.get("labels", []):
        add_label_indicator(fig, item)
    fig.update_layout(
        title=f"{display_plmn(str(doc.get('plmn')))} — {display_metric(metric)}",
        height=216,
        dragmode="zoom",
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="시간 (KST)",
        yaxis_title=display_metric(metric),
        uirevision=f"{doc.get('plmn')}-{metric}",
    )
    if width:
        fig.update_layout(width=width, autosize=False)
    else:
        fig.update_layout(autosize=True)
    fig.update_yaxes(fixedrange=True)
    tmin, tmax = data_time_bounds(view if len(view) else df)
    x0 = pd.to_datetime(start, utc=True) if start is not None else tmin
    x1 = pd.to_datetime(end, utc=True) if end is not None else tmax
    x0, x1 = clamp_time_range(x0, x1, tmin, tmax)
    fig.update_xaxes(
        range=[to_plot_time(x0), to_plot_time(x1)],
        autorange=False,
        type="date",
        tickformat="%Y년 %m월 %d일\n%H:%M",
        hoverformat="%Y년 %m월 %d일 %H:%M",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="royalblue",
        spikethickness=1,
    )
    return fig


def suggest_zscore_points(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """참고용으로 각 metric의 절대 z-score가 threshold를 넘는 시점을 제안한다."""
    cols = metric_columns(df)
    mask = pd.Series(False, index=df.index)
    hit_metrics = [[] for _ in range(len(df))]
    for col in cols:
        std = df[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        z = ((df[col] - df[col].mean()) / std).abs() > threshold
        for i in df.index[z]:
            hit_metrics[i].append(col)
        mask |= z
    out = df.loc[mask, ["time"]].copy()
    out["time"] = [format_kst(t) for t in out["time"]]
    out["n_metrics"] = [len(hit_metrics[i]) for i in out.index]
    out["metrics"] = [
        ", ".join(display_metric(m) for m in hit_metrics[i][:8]) for i in out.index
    ]
    return out.reset_index(drop=True)
