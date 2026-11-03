"""RoCA training objectives (from upstream trainer.py; Merlion-free).

Source: ruiking04/RoCA ``models/RoCA/roca_trainer/trainer.py``
(``train``, ``center_c``). MIT License — see ``LICENSE``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def roca_loss(feature1, feature_dec1, center, epoch, config, device):
    """Composite OC + variance regularizer; returns (loss_oc, loss_sigma, loss_n, loss_a)."""
    center = center.unsqueeze(0)
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    feature_dec1 = F.normalize(feature_dec1, dim=1)

    distance1 = 1 - F.cosine_similarity(feature1, center, eps=1e-6)
    distance_dec1 = 1 - F.cosine_similarity(feature_dec1, center, eps=1e-6)

    sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

    loss_n = distance1 + distance_dec1
    loss_a = 4 - loss_n

    score = loss_n
    if epoch < config.warmup:
        loss_oc = torch.mean(loss_n)
    else:
        if config.train_method == "contaminated":
            n = int(score.shape[0])
            n_a = max(1, int(n * config.nu))
            n_n = max(1, n - n_a)
            _, idx_n = torch.topk(score, n_n, largest=False, sorted=False)
            _, idx_a = torch.topk(score, n_a, largest=True, sorted=False)
            loss_oc = torch.cat([loss_n[idx_n], config.mu * loss_a[idx_a]], 0).mean()
        else:
            loss_oc = torch.mean(loss_n)

    return loss_oc, loss_sigam, loss_n, loss_a


def center_c(train_loader, model, device, center, config, eps=0.1):
    """Hypersphere center from a forward pass (official ``center_c``)."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data, _target, aug1, aug2 in train_loader:
            data = data.float().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            all_data = torch.cat((data, aug1, aug2), dim=0)
            outputs, dec = model(all_data)
            n_samples += outputs.shape[0]
            all_feature = torch.cat((outputs, dec), dim=0)
            c = c + torch.sum(all_feature, dim=0)
    c = c / (2 * n_samples)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def get_radius(dist: torch.Tensor, nu: float):
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)
