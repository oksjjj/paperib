"""HourAE의 데이터 준비, 정규화, decontamination, AE 학습과 scoring 로직."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.data import load_plmn
from core.features import metric_columns
from models.hourae.model import PointAutoEncoder

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


# Python/NumPy/PyTorch 난수원을 고정해 실험 간 차이를 줄인다(완전한 결정성 보장은 아님).
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Apple Silicon은 MPS, NVIDIA는 CUDA를 우선 사용하고 둘 다 없으면 CPU를 사용한다.
def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 시계열 순서를 유지한 60/20/20 분할 인덱스: TRAIN[:a], VALID[a:b], TEST[b:].
def split_indices(
    n: int,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> tuple[int, int]:
    # 작은 데이터에서도 TRAIN과 VALID에 최소 한 시점을 배정하도록 경계를 계산한다.
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    return train_end, min(valid_end, n)



# 원본 데이터에서 학습/평가 배열을 만든다. 이 과정에는 전문가 라벨이 입력되지 않는다.
def prepare_unlabeled_plmn(plmn: str, tz_name: str) -> dict[str, Any]:
    # 공통 로더가 시간순 원본 데이터를 제공하고 공통 함수가 모델 입력 순서를 보장한다.
    df = load_plmn(plmn)
    cols = metric_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: no raw feature columns")

    # (T,F) 수치 행렬: 행은 5분 시점, 열은 core counter. 결측/무한대는 0으로 대체한다.
    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # 저장 및 평가 기준은 UTC로 유지하고 시간대 분류에만 로컬 시각을 사용한다.
    times = pd.to_datetime(df["time"], utc=True)
    local_times = times.dt.tz_convert(tz_name)
    # 날짜와 관계없이 같은 시각의 hour(0~23)를 동일한 모델에 대응시킨다.
    hours = local_times.dt.hour.to_numpy(dtype=np.int16)

    # 정규화 전에 시간순 분할하여 VALID/TEST 통계가 TRAIN에 유입되지 않도록 한다.
    train_end, valid_end = split_indices(len(df))

    return {
        "df": df,
        "cols": cols,
        "times": times,
        "local_times": local_times,
        "train_end": train_end,
        "valid_end": valid_end,
        # 원본 scale 배열을 그대로 반환하며 scaler는 각 TRAIN 그룹 내부에서 별도로 fit한다.
        "train": values[:train_end],
        "valid": values[train_end:valid_end],
        "test": values[valid_end:],
        "train_hours": hours[:train_end],
        "valid_hours": hours[train_end:valid_end],
        "test_hours": hours[valid_end:],
        # UTC 시간 경계는 UI 및 후속 평가에서 동일한 split을 재현하는 데 사용한다.
        "split": {
            "train_time_end": str(times.iloc[train_end - 1]),
            "valid_time_start": str(times.iloc[train_end]),
            "valid_time_end": str(times.iloc[valid_end - 1]),
            "test_time_start": str(times.iloc[valid_end]),
            "test_time_end": str(times.iloc[-1]),
        },
    }


# 특성별 RobustScaler: (x-median)/(Q75-Q25). 평균/표준편차보다 극단값에 덜 민감하다.
def fit_scaler(x: np.ndarray) -> RobustScaler:
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    # 입력 x의 열별 통계만 추정한다. 호출자가 TRAIN 또는 clean TRAIN만 전달해야 한다.
    scaler.fit(x)
    return scaler


# 정규화된 (N,F) point 배열에서 AE를 학습하고 epoch별 손실 기록을 반환한다.
def train_ae(
    x_scaled: np.ndarray,
    *,
    n_features: int,
    cfg: Config,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[PointAutoEncoder, list[float]]:
    # 각 단계/그룹마다 지정된 seed로 독립적인 파라미터 초기화를 수행한다.
    set_seed(seed)

    # 새 AE를 생성한다. Final 학습은 Stage-1 가중치를 이어서 사용하지 않는다.
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    ).to(device)

    # 역전파로 얻은 gradient를 이용해 모델 파라미터만 갱신한다.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    # 학습 손실은 배치와 특성 전체의 제곱오차 평균: 1/(B*F) sum (x_hat-x)^2.
    criterion = nn.MSELoss(reduction="mean")

    # float32 텐서로 변환한다. point-wise AE이므로 TRAIN 내부의 시점 순서를 섞어도 된다.
    tensor = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(cfg.batch_size, len(tensor)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 각 epoch의 mini-batch 평균 loss를 저장한다(마지막 작은 배치도 같은 비중으로 평균).
    history: list[float] = []
    for _epoch in range(int(epochs)):
        # 학습 모드를 설정한다.
        model.train()
        batch_losses: list[float] = []

        for (xb,) in loader:
            xb = xb.to(device)
            # PyTorch의 gradient는 누적되므로 새 mini-batch마다 초기화한다.
            optimizer.zero_grad()
            # (B,F) 입력을 동일 shape로 복원하고 정답 라벨 없이 자기 자신과 비교한다.
            recon = model(xb)
            loss = criterion(recon, xb)
            # chain rule로 각 파라미터의 gradient를 구하고 Adam으로 한 번 갱신한다.
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        # GPU/MPS 텐서를 CPU 실수로 꺼낸 loss들의 평균을 기록한다.
        history.append(float(np.mean(batch_losses)))

    return model, history


# 점수 계산은 가중치를 바꾸지 않으므로 gradient 저장을 비활성화한다.
@torch.no_grad()
def reconstruction_mse(
    model: PointAutoEncoder,
    x_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    # 입력 순서대로 점수를 반환해야 하므로 추론용 DataLoader는 shuffle하지 않는다.
    x = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(batch_size, max(len(x), 1)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # 평가 모드에서는 입력을 복원하기만 한다.
    model.eval()
    out: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        recon = model(xb)
        # dim=1은 F개 특성 방향의 평균이다. (B,F) → (B,)로 시점별 MSE 하나씩 계산한다.
        mse = torch.mean((recon - xb) ** 2, dim=1)
        out.append(mse.detach().cpu().numpy().astype(np.float64))

    # 배치별 (B,) 점수를 원래 시점 순서대로 연결하여 (N,) 배열을 만든다.
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)

# 학습된 파라미터를 gradient 연결 없이 CPU에 독립 복사해 기기 간 저장/로드에 사용한다.
def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# p% 상위 점수 경계는 q=1-p/100 분위수이다. p=5이면 q=0.95이다.
def quantile_for_contamination(
    scores: np.ndarray,
    contamination_pct: float,
) -> float:
    q = 1.0 - float(contamination_pct) / 100.0
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(scores, q))


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
    stage1_scaler = fit_scaler(x_hour_raw)
    x1 = stage1_scaler.transform(x_hour_raw).astype(np.float32)

    stage1_model, stage1_hist = train_ae(
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
    final_scaler = fit_scaler(clean_raw)
    x2 = final_scaler.transform(clean_raw).astype(np.float32)

    # clean TRAIN으로 새로운 AE를 처음부터 학습한다.
    final_model, final_hist = train_ae(
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


# 저장한 Final AE 가중치를 동일한 구조에 복원하고 추론 모드로 전환한다.
def build_final_model(
    artifacts: HourArtifacts,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> PointAutoEncoder:
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    )
    model.load_state_dict(artifacts.final_state)
    model.to(device)
    model.eval()
    return model


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
        model = build_final_model(
            art,
            n_features=n_features,
            cfg=cfg,
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
