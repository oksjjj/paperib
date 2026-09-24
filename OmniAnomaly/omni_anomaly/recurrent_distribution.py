# -*- coding: utf-8 -*-
"""RecurrentDistribution — PyTorch port of the TF original."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_anomaly.distributions import LOG_2PI_HALF, truncated_normal_like


class SoftplusStdLinear(nn.Module):
    """``softplus(dense(x)) + epsilon`` matching ``softplus_std``."""

    def __init__(self, in_features, out_features, epsilon=1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.epsilon = epsilon
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.softplus(self.linear(x)) + self.epsilon


class RecurrentDistribution(nn.Module):
    """
    Multi-variable distribution with recurrent structure.

    Sampling (matches TF ``sample_step``):
        z_t ~ N(μ([input_q_t, z_{t-1}]), σ([input_q_t, z_{t-1}]))

    Note on TF ``log_prob_step``:
        The published TensorFlow code concatenates ``[z_t, input_q_t]``, which
        does **not** match the sampling process and makes SGVB numerically
        unstable.  Here ``log_prob`` uses the same conditioning as sampling
        (``[input_q_t, z_{t-1}]``), which is the density of the generative
        process actually used to draw z — required for a correct SGVB estimator.
    """

    def __init__(self, z_dim, window_length, input_q_dim, std_epsilon=1e-4):
        super().__init__()
        self.z_dim = z_dim
        self.window_length = window_length
        self.input_q_dim = input_q_dim
        concat_dim = input_q_dim + z_dim

        self.mean_q_mlp = nn.Linear(concat_dim, z_dim)
        nn.init.xavier_uniform_(self.mean_q_mlp.weight)
        nn.init.zeros_(self.mean_q_mlp.bias)

        self.std_q_mlp = SoftplusStdLinear(concat_dim, z_dim, epsilon=std_epsilon)

    def sample(self, input_q, n_samples=None):
        """
        Args:
            input_q: (batch, window_length, input_q_dim)
            n_samples: ``None`` → squeeze sample dim (TF reduce_mean over axis 0)
        """
        batch_size = input_q.shape[0]
        device = input_q.device
        dtype = input_q.dtype

        squeeze = n_samples is None
        if n_samples is None:
            n_samples = 1

        # tf.scan(..., back_prop=False): no gradient through recurrent state
        z_prev = torch.zeros(n_samples, batch_size, self.z_dim, device=device, dtype=dtype)
        z_steps = []

        for t in range(self.window_length):
            input_q_t = input_q[:, t, :].unsqueeze(0).expand(n_samples, -1, -1)
            inp = torch.cat([input_q_t, z_prev], dim=-1)
            mu = self.mean_q_mlp(inp)
            std = self.std_q_mlp(inp)
            noise = truncated_normal_like(mu)
            z_t = mu + noise * std
            z_steps.append(z_t)
            z_prev = z_t.detach()

        samples = torch.stack(z_steps, dim=2)  # (n_samples, batch, T, z_dim)
        if squeeze:
            return samples.mean(dim=0)
        return samples

    def log_prob(self, given, input_q, group_ndims=0):
        """
        Density under the same transition used in ``sample``.

        Args:
            given: (batch, T, z_dim) or (n_samples, batch, T, z_dim)
            input_q: (batch, T, input_q_dim)
        """
        has_sample_dim = given.dim() > 3
        if not has_sample_dim:
            given = given.unsqueeze(0)

        n_samples, batch_size, window_length, _ = given.shape
        device, dtype = given.device, given.dtype
        z_prev = torch.zeros(n_samples, batch_size, self.z_dim, device=device, dtype=dtype)
        log_probs = []

        for t in range(window_length):
            given_t = given[:, :, t, :]
            input_q_t = input_q[:, t, :].unsqueeze(0).expand(n_samples, -1, -1)
            # Match sample_step conditioning: [input_q_t, z_{t-1}]
            inp = torch.cat([input_q_t, z_prev], dim=-1)
            mu = self.mean_q_mlp(inp)
            std = self.std_q_mlp(inp)
            logstd = torch.log(std + 1e-12)
            precision = torch.exp(-2.0 * logstd)
            diff = torch.clamp(torch.abs(given_t - mu), max=1e8)
            lp = -LOG_2PI_HALF - logstd - 0.5 * precision * (diff ** 2)
            log_probs.append(lp)
            # condition next step on observed z_t (standard density)
            z_prev = given_t

        log_prob = torch.stack(log_probs, dim=2)  # (n_samples, batch, T, z_dim)
        if not has_sample_dim:
            log_prob = log_prob.squeeze(0)

        if group_ndims == 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob
