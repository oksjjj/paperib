"""RoCA의 핵심 학습 목적함수.

원본:
ruiking04/RoCA의 models/RoCA/roca_trainer/trainer.py에 있는
train(), center_c() 로직에서 가져왔다.

핵심 아이디어는 representation을 하나의 hypersphere center 주변에 모으되,
contaminated training에서는 score가 큰 일부 sample을 잠재 anomaly로 분리하여
정상 sample과 다른 방향의 목적함수를 적용하는 것이다.

MIT License는 LICENSE를 참고한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def roca_loss(feature1, feature_dec1, center, epoch, config, device):
    """RoCA의 one-class loss와 variance regularizer를 계산한다.

    반환값:
        loss_oc    : 실제 학습에 사용되는 one-class 목적함수
        loss_sigam : representation collapse를 막기 위한 분산 regularizer
        loss_n     : 정상이라고 가정했을 때의 anomaly score
        loss_a     : 잠재 anomaly에 사용하는 반대 방향 loss

    paperib에서 최종 anomaly score로 사용하는 값은 loss_n이다.
    loss_n이 클수록 center에서 멀어져 더 anomalous한 sample로 본다.
    """

    # cosine distance를 계산하기 전에 center와 두 representation을 단위벡터로 정규화한다.
    center = center.unsqueeze(0)
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    feature_dec1 = F.normalize(feature_dec1, dim=1)

    # 원본 representation과 reconstruction representation 각각이
    # hypersphere center에서 얼마나 떨어져 있는지를 cosine distance로 계산한다.
    # 각 거리는 1-cos(theta)로 [0,2] 범위이며 두 거리의 합은 [0,4] 범위이다.
    distance1 = 1 - F.cosine_similarity(feature1, center, eps=1e-6)
    distance_dec1 = 1 - F.cosine_similarity(feature_dec1, center, eps=1e-6)

    # 모든 representation이 한 점으로 collapse되는 것을 막기 위해
    # 각 representation 차원의 표준편차가 너무 작으면 penalty를 준다.
    # var([0])는 batch 축의 분산이다. projection 차원별 산포를 계산한다.
    sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

    # loss_n: 정상 sample이라면 작아져야 하는 값.
    # 두 representation 모두 center에 가까울수록 loss_n이 작다.
    loss_n = distance1 + distance_dec1

    # loss_a: 잠재 anomaly에 대해서는 loss_n과 반대 방향의 목적을 주기 위한 값.
    # loss_n이 커질수록 loss_a는 작아진다.
    loss_a = 4 - loss_n

    # sample별 anomaly ranking에도 이 score를 그대로 사용한다.
    score = loss_n

    if epoch < config.warmup:
        # warmup 동안에는 contaminated 여부를 아직 분리하지 않고
        # 모든 sample이 center에 가까워지도록 학습한다.
        loss_oc = torch.mean(loss_n)
    else:
        if config.train_method == "contaminated":
            # warmup 이후에는 현재 batch에서 score가 가장 큰 nu 비율을
            # 잠재 anomaly 후보로 간주한다. 실제 anomaly label은 사용하지 않는다.
            n = int(score.shape[0])
            # score 길이는 원본 및 증강을 합친 batch 크기이다. 최소 한 개를 잠재 이상 후보로 선택한다.
            n_a = max(1, int(n * config.nu))
            n_n = max(1, n - n_a)

            # score가 낮은 대부분의 sample은 normal candidate.
            _, idx_n = torch.topk(score, n_n, largest=False, sorted=False)

            # score가 가장 높은 일부 sample은 anomaly candidate.
            _, idx_a = torch.topk(score, n_a, largest=True, sorted=False)

            # normal candidate에는 center로 가까워지는 loss_n,
            # anomaly candidate에는 반대 방향인 loss_a를 사용한다.
            # 실제 데이터 행을 삭제하는 것이 아니라 후보별로 다른 목적함수를 부여해 한 번에 최적화한다.
            loss_oc = torch.cat(
                [loss_n[idx_n], config.mu * loss_a[idx_a]],
                0,
            ).mean()
        else:
            loss_oc = torch.mean(loss_n)

    return loss_oc, loss_sigam, loss_n, loss_a


def center_c(train_loader, model, device, center, config, eps=0.1):
    """현재 train representation 전체의 평균으로 hypersphere center를 계산한다.

    원본 window뿐 아니라 weak/strong augmentation까지 함께 forward하여 center를 만든다.
    너무 0에 가까운 center 성분은 eps 크기로 밀어 0으로 붕괴되는 것을 방지한다.
    """
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data, _target, aug1, aug2 in train_loader:
            data = data.float().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            # 한 batch에서 원본, weak view, strong view를 세로로 붙여
            # 동일한 backbone으로 representation을 계산한다.
            all_data = torch.cat((data, aug1, aug2), dim=0)
            outputs, dec = model(all_data)
            n_samples += outputs.shape[0]
            # (3B,P) 원본 projection과 복원 projection을 이어 (6B,P)로 만든 후 P차원 합을 누적한다.
            all_feature = torch.cat((outputs, dec), dim=0)
            c = c + torch.sum(all_feature, dim=0)
    # n_samples는 view 수를 포함한 projection 행 수이다. 양쪽 표현을 합산했으므로 2배로 나눈다.
    c = c / (2 * n_samples)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def get_radius(dist: torch.Tensor, nu: float):
    """거리 분포의 상위 nu 비율을 제외하는 quantile radius를 계산한다.

    현재 paperib wrapper의 주 실행 경로에서는 직접 사용하지 않지만,
    원본 RoCA 코드와의 대응을 위해 유지한다.
    """

    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)
