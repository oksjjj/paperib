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
    PointAutoEncoder,
    build_point_autoencoder_from_state,
    cpu_state_dict,
    device_for_run,
    fit_robust_scaler,
    quantile_for_contamination,
    reconstruction_mse,
    set_seed,
    train_point_autoencoder,
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

@dataclass
class HourArtifacts:
    hour: int
    # Stage-1은 오염 가능 TRAIN 전체, Final은 Stage-1에서 남긴 원본 데이터로 각각 fit한 scaler.
    stage1_scaler: RobustScaler
    final_scaler: RobustScaler
    # 두 모델의 state_dict(가중치/버퍼)를 CPU에 복사해 추론 및 checkpoint에 저장한다.
    stage1_state: dict[str, torch.Tensor]
    final_state: dict[str, torch.Tensor]
    # 1차 제거 기준 MSE, 최종 clean TRAIN MSE의 평균/표준편차 및 참고 분위수.
    stage1_threshold_mse: float
    final_score_mean: float
    final_score_std: float
    final_reference_quantile_mse: float
    final_reference_quantile_z: float
    # 원래 학습 시점 수, 제거 시점 수, 각 AE 마지막 epoch의 평균 배치 손실.
    n_train: int
    n_removed: int
    stage1_final_loss: float
    final_final_loss: float


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


# 하나의 시간대(hour)에 해당하는 TRAIN 행만으로 1차 학습 → 오염 후보 제거 → 최종 학습을 수행한다.
def fit_hour(
    x_hour_raw: np.ndarray,
    *,
    hour: int,
    cfg: Config,
    device: torch.device,
) -> HourArtifacts:
    # 지나치게 적은 표본으로 불안정한 모델을 학습하지 않도록 최소 20개를 요구한다.
    n = len(x_hour_raw)
    if n < 20:
        raise RuntimeError(f"hour {hour:02d}: too few train points ({n})")

    # [Stage-1] 잠재 오염을 포함한 TRAIN 그룹으로 scaler와 첫 번째 AE를 학습한다.
    stage1_scaler = fit_robust_scaler(x_hour_raw)
    x1 = stage1_scaler.transform(x_hour_raw).astype(np.float32)

    stage1_model, stage1_hist = train_point_autoencoder(
        x1,
        n_features=x1.shape[1],
        cfg=cfg,
        epochs=cfg.stage1_epochs,
        device=device,
        seed=cfg.seed + hour,
    )
    # Stage-1 TRAIN의 모든 시점에 복원오차를 부여한다. 높은 오차는 제거 후보이지 정답 이상치가 아니다.
    stage1_scores = reconstruction_mse(
        stage1_model,
        x1,
        batch_size=cfg.batch_size,
        device=device,
    )

    # 기본 95% 분위수보다 큰 MSE를 제거한다. 경계와 같은 값은 clean_mask에 남는다.
    stage1_cut = quantile_for_contamination(
        stage1_scores,
        cfg.stage1_remove_pct,
    )
    clean_mask = stage1_scores <= stage1_cut

    # 제거 후 충분한 학습 행이 남았는지 확인한다.
    if int(clean_mask.sum()) < max(20, int(0.5 * n)):
        raise RuntimeError(
            f"hour {hour:02d}: stage1 retained only {int(clean_mask.sum())}/{n}"
        )

    # [Final] 원본 단위의 행을 추출한 뒤 scaler도 새로 fit한다. 1차 정규화값을 재사용하지 않는다.
    clean_raw = x_hour_raw[clean_mask]
    final_scaler = fit_robust_scaler(clean_raw)
    x2 = final_scaler.transform(clean_raw).astype(np.float32)

    # clean TRAIN으로 새로운 AE를 처음부터 학습한다.
    final_model, final_hist = train_point_autoencoder(
        x2,
        n_features=x2.shape[1],
        cfg=cfg,
        epochs=cfg.final_epochs,
        device=device,
        seed=cfg.seed + 1000 + hour,
    )
    # 운영 임계치와 점수 표준화의 기준은 Final 모델의 clean TRAIN 복원오차이다.
    final_clean_scores = reconstruction_mse(
        final_model,
        x2,
        batch_size=cfg.batch_size,
        device=device,
    )

    # 그룹마다 MSE 분포가 다르므로 이후 z=(MSE-mean)/std로 공통 척도에서 비교한다.
    score_mean = float(np.mean(final_clean_scores))
    score_std = float(np.std(final_clean_scores, ddof=0))
    # 분산이 0에 가까울 때 0으로 나누는 문제를 방지한다.
    if not np.isfinite(score_std) or score_std < 1e-12:
        score_std = 1e-12

    # 99.99% 참고 분위수는 진단 및 저장용이다. 여기서 데이터를 추가로 제거하지 않는다.
    ref_mse = quantile_for_contamination(
        final_clean_scores,
        cfg.final_contamination_pct,
    )
    ref_z = float((ref_mse - score_mean) / score_std)

    # 재학습 없이 추론과 임계치 분석을 수행할 수 있도록 모든 필요한 산출물을 보존한다.
    return HourArtifacts(
        hour=hour,
        stage1_scaler=stage1_scaler,
        final_scaler=final_scaler,
        stage1_state=cpu_state_dict(stage1_model),
        final_state=cpu_state_dict(final_model),
        stage1_threshold_mse=float(stage1_cut),
        final_score_mean=score_mean,
        final_score_std=score_std,
        final_reference_quantile_mse=ref_mse,
        final_reference_quantile_z=ref_z,
        n_train=n,
        n_removed=int((~clean_mask).sum()),
        stage1_final_loss=float(stage1_hist[-1]),
        final_final_loss=float(final_hist[-1]),
    )


# 각 시점을 해당 시간대(hour) 모델로 추론한 뒤 원래 시간 순서에 MSE와 z-score를 배치한다.
def score_with_hour_models(
    values_raw: np.ndarray,
    hours: np.ndarray,
    artifacts: dict[int, HourArtifacts],
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
