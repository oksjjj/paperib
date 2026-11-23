"""HourAE의 데이터 준비, 정규화, decontamination, AE 학습과 scoring 로직."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from core.data import prepare_unlabeled_base
from models.pointae import (
    DecontaminatedAEArtifacts,
    build_point_autoencoder_from_state,
    device_for_run,
    fit_decontaminated_group,
    reconstruction_mse,
)

# @dataclass: 필드를 선언하면 __init__/비교 등을 자동 생성해 주는 데코레이터.
#   예: Config(timezone="UTC") 처럼 인자로 넘기며 설정 객체를 만들 수 있다.
@dataclass
class Config:
    # UTC 시각을 로컬 시각으로 변환해 매일 반복되는 시간대(hour)를 식별한다.
    timezone: str = "Asia/Seoul"
    # 1차 AE의 TRAIN 복원오차 상위 5%를 잠재 이상치로 제외한다(동점에 따라 실제 비율은 달라질 수 있다).
    stage1_remove_pct: float = 5.0
    # 최종 clean TRAIN 오차의 상위 0.01% 기준 분위수를 진단용으로 기록한다. 추가 제거 단계는 아니다.
    final_contamination_pct: float = 0.01
    # 표준화된 복원오차 z의 운영 임계치. 기본 z>3은 해당 모델의 평균+3표준편차와 같다.
    threshold_sigma: float = 3.0
    # AE 차원: 입력 F → hidden_dim → latent_dim → hidden_dim → F.
    hidden_dim: int = 32
    latent_dim: int = 8
    # Stage-1과 Final AE는 서로 독립적으로 초기화하여 각각 다음 epoch만큼 학습한다.
    stage1_epochs: int = 50
    final_epochs: int = 100
    # 한 번의 gradient 갱신에 사용하는 시점 수(데이터가 적으면 실제 크기를 줄인다).
    batch_size: int = 128
    # Adam 학습률과 가중치 감쇠 계수. anomaly score의 threshold와는 별개이다.
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # 난수 초기값 및 평가에서 하루당 오경보 수 등을 환산하기 위한 표본 간격(분).
    seed: int = 2
    sample_minutes: float = 5.0

# 원본 데이터에서 학습/평가 배열을 만든다. 이 과정에는 전문가 라벨이 입력되지 않는다.
def prepare_unlabeled_plmn(plmn: str, tz_name: str) -> dict[str, Any]:
    """공통 라벨 없는 데이터 준비 후 local hour만 HourAE용으로 추가한다."""
    prep = prepare_unlabeled_base(plmn, tz_name)
    hours = prep["local_times"].dt.hour.to_numpy(dtype=np.int16)
    a, b = prep["train_end"], prep["valid_end"]
    prep.update(
        {
            "train_hours": hours[:a],
            "valid_hours": hours[a:b],
            "test_hours": hours[b:],
        }
    )
    return prep


# 각 시점을 해당 시간대(hour) 모델로 추론한 뒤 원래 시간 순서에 MSE와 z-score를 배치한다.
def score_with_hour_models(
    values_raw: np.ndarray,
    hours: np.ndarray,
    artifacts: dict[int, DecontaminatedAEArtifacts],
    *,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    # 길이 T인 배열을 NaN으로 초기화한다. 점수가 할당되지 않은 위치를 0과 구분한다.
    raw_mse = np.full(len(values_raw), np.nan, dtype=np.float64)
    z_scores = np.full(len(values_raw), np.nan, dtype=np.float64)

    # 같은 그룹의 시점 인덱스를 모아 해당 그룹에서 학습한 모델만 적용한다.
    for hour in range(24):
        idx = np.flatnonzero(hours == hour)
        if idx.size == 0:
            continue

        art = artifacts[hour]
        model = build_point_autoencoder_from_state(
            art.final_state,
            n_features=n_features,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            device=device,
        )
        # VALID/TEST에도 clean TRAIN에서 fit한 scaler를 transform만 한다(재학습 및 통계 누출 방지).
        x = art.final_scaler.transform(values_raw[idx]).astype(np.float32)
        mse = reconstruction_mse(
            model,
            x,
            batch_size=cfg.batch_size,
            device=device,
        )
        # 그룹별 순서로 계산한 점수를 원본 시계열의 인덱스 위치에 되돌려 놓는다.
        raw_mse[idx] = mse
        z_scores[idx] = (
            (mse - art.final_score_mean)
            / max(art.final_score_std, 1e-12)
        )

    # 첫 번째는 임계치와 비교할 표준화 점수, 두 번째는 진단용 원래 MSE이다.
    return z_scores, raw_mse
