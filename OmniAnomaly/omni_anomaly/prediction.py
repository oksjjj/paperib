# -*- coding: utf-8 -*-
"""Predictor — PyTorch port of omni_anomaly.prediction.Predictor."""
import time

import numpy as np
import torch

from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Predictor']


class Predictor:
    """OmniAnomaly predictor (reconstruction probability)."""

    def __init__(self, model, device, batch_size=32, n_z=1, last_point_only=True):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_z = n_z
        self.last_point_only = last_point_only

    def get_score(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError('`values` must be a 2-D array')

        self.model.eval()
        sw = BatchSlidingWindow(
            array_size=len(values),
            window_size=self.model.window_length,
            batch_size=self.batch_size,
        )

        scores = []
        z_infos = []
        pred_times = []

        print('-' * 30, 'testing', '-' * 30)
        with torch.no_grad():
            for batch_x, in sw.get_iterator([values]):
                start = time.time()
                x = torch.from_numpy(batch_x).to(self.device)
                score, z_info = self.model.get_score(
                    x, n_z=self.n_z, last_point_only=self.last_point_only,
                )
                scores.append(score.cpu().numpy())
                z_infos.append(z_info.cpu().numpy())
                pred_times.append(time.time() - start)

        return (
            np.concatenate(scores, axis=0),
            np.concatenate(z_infos, axis=0),
            float(np.mean(pred_times)),
        )
