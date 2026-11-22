"""RoCA용 sliding-window 데이터셋.

입력 시계열 (T, F)을 여러 window로 자른 뒤,
PyTorch Conv1d가 기대하는 channel-first 형태 (N, C, T_window)로 바꾼다.
각 원본 window에 대해 RoCA 학습용 weak/strong augmentation도 함께 생성한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.roca.official.augmentations import DataTransform


def subsequences(sequence: np.ndarray, window_size: int, time_step: int) -> np.ndarray:
    """원본 RoCA의 subsequences 로직을 옮긴 함수.

    sequence를 window_size 길이의 연속 구간으로 자르고,
    다음 window 시작점은 time_step만큼 이동한다.

    예:
        window_size=4, time_step=2
        [0,1,2,3], [2,3,4,5], [4,5,6,7], ...

    반환 shape는 입력이 2차원일 때 (N, window, F)다.
    """
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    if sq.ndim == 1:
        if a != 0:
            sq = sq[: sq.shape[0] - a]
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
    elif sq.ndim == 2:
        if a != 0:
            sq = sq[: sq.shape[0] - a, :]
        shape = (
            int((sq.shape[0] - window_size + time_step) / time_step),
            window_size,
            sq.shape[1],
        )
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
    else:
        raise ValueError(f"expected 1D/2D, got ndim={sq.ndim}")
    return np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride).copy()


class RoCAWindowDataset(Dataset):
    """RoCA 학습 batch에 필요한 원본/증강 window를 제공한다.

    각 item:
        x      : 원본 window, shape=(C, T)
        y      : 현재 paperib에서는 학습에 사용하지 않는 dummy label
        aug1   : weak augmentation
        aug2   : strong augmentation

    RoCA는 실제 anomaly label 없이 학습하므로 train_y는 모두 0으로 넣는다.
    """

    def __init__(self, samples_nct: np.ndarray, labels: np.ndarray, config: Any):
        if samples_nct.ndim != 3:
            raise ValueError("samples must be (N, C, T)")
        self.x_data = torch.from_numpy(samples_nct.astype(np.float32))
        self.y_data = torch.from_numpy(labels.astype(np.int64))
        self.len = self.x_data.shape[0]
        self.aug1, self.aug2 = DataTransform(self.x_data, config)
        if isinstance(self.aug1, np.ndarray):
            self.aug1 = torch.from_numpy(self.aug1.astype(np.float32))
        if isinstance(self.aug2, np.ndarray):
            self.aug2 = torch.from_numpy(self.aug2.astype(np.float32))

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]

    def __len__(self) -> int:
        return self.len


def block_to_nct(
    block: np.ndarray,
    *,
    window: int,
    time_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """연속 block을 RoCA 입력 window로 바꾸고 각 window의 마지막 point 위치를 반환한다.

    입력:
        block: (T, F)

    출력:
        nct : (N, C=F, T_window)
        ends: 각 window가 원본 block의 어느 point에서 끝나는지 나타내는 인덱스

    RoCA score는 window 하나당 하나가 나오므로, 나중에 이 end index 위치에
    해당 score를 배치해 point-wise score series로 변환한다.
    """
    # 먼저 사람이 보기 쉬운 (N, window, F) 형태의 window를 만든다.
    wins = subsequences(block, window, time_step)
    n = wins.shape[0]

    # i번째 window가 원본 block에서 끝나는 위치를 계산한다. 인덱스는 0부터 시작한다.
    ends = np.arange(window - 1, window - 1 + n * time_step, time_step, dtype=np.int64)
    if len(ends) > n:
        ends = ends[:n]
    # Conv1d는 channel-first 입력을 사용하므로 feature 축과 시간 축을 바꾼다.
    # 최종 shape: (N, F, window)
    nct = wins.transpose(0, 2, 1).astype(np.float32)
    return nct, ends
