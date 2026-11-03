"""Window dataset with RoCA augmentations (channel-first B×C×T)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from official.augmentations import DataTransform


def subsequences(sequence: np.ndarray, window_size: int, time_step: int) -> np.ndarray:
    """Official ``utils.ad_utils.subsequences`` (contiguous strided windows)."""
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
    """``(x, y, aug1, aug2)`` with x shaped (C, T) — official layout."""

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
    """(T, F) → windows (N, C, T_win) and end indices in the block."""
    wins = subsequences(block, window, time_step)  # (N, window, F)
    n = wins.shape[0]
    # end index of window i (0-based in block)
    ends = np.arange(window - 1, window - 1 + n * time_step, time_step, dtype=np.int64)
    if len(ends) > n:
        ends = ends[:n]
    nct = wins.transpose(0, 2, 1).astype(np.float32)  # (N, F, window)
    return nct, ends
