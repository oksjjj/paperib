# -*- coding: utf-8 -*-
"""Distribution helpers (faithful to tfsnippet / OmniAnomaly numerics)."""
import math

import torch


# -0.5 * log(2 * pi)  (same constant as RecurrentDistribution.log_prob_step)
LOG_2PI_HALF = 0.9189385332046727
LOG_2PI = math.log(2 * math.pi)


def truncated_normal_like(tensor, mean=0.0, std=1.0, trunc_abs=2.0):
    """Sample matching ``tf.truncated_normal`` (default: clip at ±2 std)."""
    out = torch.empty_like(tensor)
    torch.nn.init.trunc_normal_(out, mean=mean, std=std, a=-trunc_abs, b=trunc_abs)
    return out


def gaussian_log_prob(x, mean, std, group_ndims=1):
    """
    Diagonal Gaussian log-density.

    Matches OmniAnomaly RecurrentDistribution / zhusuan Normal style:
      log p = -0.9189385332046727 - log(std)
              - 0.5 * exp(-2*log(std)) * min(|x-mean|, 1e8)^2
    """
    logstd = torch.log(std + 1e-12)
    precision = torch.exp(-2.0 * logstd)
    diff = torch.clamp(torch.abs(x - mean), max=1e8)
    log_prob = -LOG_2PI_HALF - logstd - 0.5 * precision * (diff ** 2)
    if group_ndims == 1:
        log_prob = log_prob.sum(dim=-1)
    elif group_ndims == 2:
        log_prob = log_prob.sum(dim=-1).sum(dim=-1)
    return log_prob


class DiagonalNormal:
    """Diagonal Normal (reparameterized)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def log_prob(self, x, group_ndims=1):
        return gaussian_log_prob(x, self.mean, self.std, group_ndims=group_ndims)

    def sample(self, n_samples=None):
        if n_samples is None:
            noise = torch.randn_like(self.mean)
            return self.mean + self.std * noise
        noise = torch.randn(
            n_samples, *self.mean.shape,
            device=self.mean.device, dtype=self.mean.dtype,
        )
        return self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise


class GaussianStateSpacePrior:
    """
    Linear Gaussian state-space prior with identity transition / observation,
    matching TFP ``LinearGaussianStateSpaceModel`` used in OmniAnomaly:

        z_0 ~ N(0, I),  z_t ~ N(z_{t-1}, I)
    """

    def __init__(self, z_dim, window_length):
        self.z_dim = z_dim
        self.window_length = window_length

    def log_prob(self, z, group_ndims=1, reduce_time=False):
        """
        Args:
            z: (batch, T, z_dim) or (n_samples, batch, T, z_dim)
            reduce_time: if True, sum over time → shape (...,);
                if False (default for SGVB), return per-timestep (..., T)
        """
        z0 = gaussian_log_prob(
            z[..., 0, :],
            torch.zeros_like(z[..., 0, :]),
            torch.ones_like(z[..., 0, :]),
            group_ndims=1,
        )
        if z.shape[-2] == 1:
            log_p = z0.unsqueeze(-1)
        else:
            trans = gaussian_log_prob(
                z[..., 1:, :],
                z[..., :-1, :],
                torch.ones_like(z[..., 1:, :]),
                group_ndims=1,
            )
            # (..., T): [log p(z0), log p(z1|z0), ..., log p(zT|zT-1)]
            log_p = torch.cat([z0.unsqueeze(-1), trans], dim=-1)

        if reduce_time:
            return log_p.sum(dim=-1)
        return log_p
