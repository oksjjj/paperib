"""RoCA 학습에서 사용하는 데이터 증강 함수.

현재 paperib 실행 경로에서 실제로 사용하는 증강은 다음 두 가지다.

- weak augmentation: scaling
- strong augmentation: jitter

rotation, magnitude_warp, permutation은 원본 구현에 포함되어 있지만
현재 DataTransform에서는 기본적으로 호출되지 않는다.
"""

import numpy as np
import torch


def DataTransform(sample, config):
    """한 원본 window batch에서 weak/strong 두 증강 view를 만든다.

    RoCA 학습 시 원본, weak augmentation, strong augmentation을 하나로 합쳐
    같은 backbone에 통과시킨다.
    """

    # weak augmentation: sample별·시간 위치별 랜덤 배율을 모든 feature에 공통 적용한다.
    weak_aug = scaling(sample, config.augmentation.scale_ratio)

    # 원본 코드에는 magnitude_warp, permutation, rotation을 조합한
    # 다른 증강 후보도 있었지만 현재 paperib 실행 경로에서는 사용하지 않는다.
    # 필요하면 아래에 정의된 각 함수를 DataTransform에서 직접 호출해 비교할 수 있다.

    # strong augmentation: 원 신호에 Gaussian noise를 직접 더한다.
    strong_aug = jitter(sample, config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    """각 원소에 평균 0의 Gaussian noise를 더한다."""

    # 참고 아이디어 출처: https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def rotation(x):
    """feature 축의 부호와 순서를 랜덤하게 바꾸는 원본 후보 증강이다."""

    x = x.permute(0, 2, 1)
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    x = x.detach().numpy()
    ret = flip[:,np.newaxis,:] * x[:,:,rotate_axis]
    ret = ret.transpose((0, 2, 1))
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    """시간축을 따라 부드럽게 변하는 scale curve를 곱하는 원본 후보 증강이다."""

    x = x.permute(0, 2, 1)
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    ret = ret.transpose((0, 2, 1))
    return ret


def scaling(x, sigma=1.1):
    """각 sample/time 위치에 랜덤 scale factor를 곱한다.

    현재 DataTransform에서 weak augmentation으로 사용한다.
    """

    # 참고 아이디어 출처: https://arxiv.org/pdf/1706.00527.pdf
    # factor shape는 (N,T)이다. 원본의 평균 2, 표준편차 sigma인 정규분포를 그대로 사용한다.
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    # 모든 channel에 같은 sample/time factor를 적용하여 channel 간 상대 크기를 보존한다.
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    """시간축을 여러 구간으로 나눈 뒤 순서를 섞는 원본 후보 증강이다."""

    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

