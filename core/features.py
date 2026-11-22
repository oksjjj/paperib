"""논문 실험에서 공통으로 사용하는 사업자 목록과 feature 정의."""

from __future__ import annotations

import pandas as pd

# 논문 평가 대상 10개 사업자. 순서는 결과 표와 UI에서도 공통으로 사용한다.
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

# 전문가가 실제 anomaly 판단에 사용하는 11개 core counter.
# 모델 입력 순서도 이 순서를 그대로 사용한다.
CORE_FEATURES: tuple[str, ...] = (
    "M971",
    "M185",
    "M855",
    "M965",
    "M162",
    "M843",
    "M430",
    "M874",
    "M037",
    "M419",
    "M618",
)
CORE_FEATURE_SET: frozenset[str] = frozenset(CORE_FEATURES)

# UI에서만 사용하는 파생 rate 이름. 논문 모델 입력에서는 제외한다.
RATE_FEATURES: frozenset[str] = frozenset({"S_RATE", "A_RATE"})


def is_rate_metric(metric: str) -> bool:
    """주어진 이름이 UI용 파생 rate인지 확인한다."""
    return str(metric) in RATE_FEATURES


def metric_columns(df: pd.DataFrame) -> list[str]:
    """DataFrame에 실제로 존재하는 11개 core feature를 고정 순서대로 반환한다."""
    present = set(df.columns)
    return [c for c in CORE_FEATURES if c in present]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 학습에 사용할 raw core counter만 반환한다."""
    return [c for c in metric_columns(df) if not is_rate_metric(c)]
