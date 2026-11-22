"""paperib 실험에서 사용하는 RoCA 기본 하이퍼파라미터.

원 논문의 다변량 설정 중 SWaT 계열 구성을 바탕으로 하되,
paperib의 11개 core feature와 실험 환경에 맞게 일부 값을 덮어쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Augmentations:
    """RoCA가 contrastive/robust 학습에 사용하는 두 데이터 증강의 강도."""

    # scaling()에서 사용하는 랜덤 scale의 표준편차 역할을 한다.
    scale_ratio: float = 0.8
    # jitter()에서 원 신호에 더하는 Gaussian noise의 표준편차다.
    jitter_ratio: float = 0.2


@dataclass
class RoCAConfig:
    """원본 RoCA의 SWaT 설정을 paperib 실험용으로 옮긴 설정 클래스.

    중요한 항목은 크게 네 그룹으로 나눠서 보면 된다.

    1. 입력/window 구조: input_channels, window_size, time_step
    2. backbone 구조: CNN/LSTM/project 관련 차원
    3. optimizer: lr, weight, beta1, beta2
    4. contaminated-training: warmup, nu, mu
    """

    # network.py에서 어떤 CNN 경로를 사용할지 결정한다.
    # paperib는 다변량 데이터이므로 SWaT 계열 경로를 사용한다.
    dataset: str = "SWaT"

    # 실제 실행 시 run_plmn.py에서 현재 core feature 수로 다시 설정된다.
    input_channels: int = 46
    # convolution kernel 길이와 stride, CNN 출력 channel 및 LSTM hidden 차원을 설정한다.
    kernel_size: int = 8
    stride: int = 1
    final_out_channels: int = 32
    hidden_size: int = 64
    num_layers: int = 3
    # 두 projection과 hypersphere center가 공유할 표현 공간 차원이다.
    project_channels: int = 64
    dropout: float = 0.45
    # 한 학습 sample에 포함할 시점 수.
    # 5분 데이터에서 window=32이면 약 160분의 구간이다.
    window_size: int = 32

    # 학습 window의 시작점을 몇 point씩 이동할지 결정한다.
    # 학습은 기본 16칸 stride지만 평가 score는 VUS 비교를 위해 stride=1로 다시 계산한다.
    time_step: int = 16

    # 전체 epoch 수, 초기 center 갱신 기준 및 0에 가까운 center 성분 보정값이다.
    num_epoch: int = 20
    change_center_epoch: int = 1
    center_eps: float = 0.01
    # 최종 loss=loss_oc+omega*loss_sigam에서 분산 regularizer의 가중치이다.
    omega: float = 0.02

    # Adam의 1차/2차 모멘트 계수, 학습률과 weight decay를 정한다.
    beta1: float = 0.9
    beta2: float = 0.99
    lr: float = 3e-4
    weight: float = 5e-4

    drop_last: bool = False
    batch_size: int = 256

    # warmup 이전에는 모든 sample을 정상으로 보고 center에 가깝게 학습한다.
    warmup: int = 10

    # contaminated 모드에서는 score가 큰 일부 sample을 잠재 anomaly로 취급한다.
    train_method: str = "contaminated"

    # 한 batch 안에서 잠재 anomaly로 취급할 비율.
    # loss.py의 top-k 분할에 직접 사용된다.
    nu: float = 0.01

    # 잠재 anomaly 쪽 loss_a에 곱하는 가중치.
    mu: float = 0.85

    detect_nu: float = 0.0005
    threshold_determine: str = "floating"
    # 원본 설정과의 호환을 위해 남겨둔 값.
    # paperib의 실제 detection threshold는 run_plmn.py의 train score quantile로 계산한다.
    anomaly_ratio: float = 1.0

    augmentation: Augmentations = field(default_factory=Augmentations)
