"""HourAE가 사용하는 point-wise AutoEncoder 구조."""

import torch
from torch import nn

# HourAE는 각 시간대(hour)마다 독립적인 가중치를 학습한다.
# 모델 자체는 시간축을 직접 입력받지 않는 point-wise AE이다.
# B: mini-batch 크기, F: 입력 특성 수(현재 11), H: 은닉층 차원, L: 잠재 차원.
class PointAutoEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
        # nn.Module을 초기화해야 아래 계층의 파라미터가 PyTorch에 등록된다.
        super().__init__()
        # 인코더: (B,F) → (B,H) → (B,L). 기본 설정은 11 → 32 → 8이다.
        # Linear는 각 시점의 특성 간 관계를 학습하고 ReLU는 비선형성을 부여한다.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # 디코더: (B,L) → (B,H) → (B,F). 압축된 표현에서 입력을 복원한다.
        # 마지막 Linear 뒤에 활성화 함수를 두지 않는다. RobustScaler 결과에는 음수도 있기 때문이다.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    # x와 복원값의 shape는 모두 (B,F)이다. 학습 시 정답은 외부 라벨이 아닌 x 자체이다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder(x)로 잠재 표현을 만든 후 decoder로 복원한다.
        # 이 함수는 복원값만 반환하고 MSE 및 anomaly score는 training.py에서 계산한다.
        return self.decoder(self.encoder(x))
