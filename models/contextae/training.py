"""ContextAE의 설정, 데이터 준비, decontamination, AE 학습과 scoring 로직."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # UTC 시각을 로컬 시각으로 변환해 매일 반복되는 context를 식별한다.
    timezone: str = "Asia/Seoul"
    # k=0이면 TRAIN silhouette로 context 수를 선택하고 양수면 고정한다.
    k: int = 0
    k_min: int = 2
    k_max: int = 24
    # 연속 구간 최소 길이: 3 slot이면 15분이다. MAD는 시간대별 산포 정보의 포함 여부이다.
    min_segment_slots: int = 3
    profile_use_mad: bool = True
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
