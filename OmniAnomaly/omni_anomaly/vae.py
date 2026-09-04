# -*- coding: utf-8 -*-
"""VAE module — PyTorch port of omni_anomaly.vae (tfsnippet VAE)."""
import torch
import torch.nn as nn

from omni_anomaly.distributions import DiagonalNormal, GaussianStateSpacePrior, gaussian_log_prob
from omni_anomaly.recurrent_distribution import RecurrentDistribution


class VAE(nn.Module):
    """
    Variational auto-encoder used by OmniAnomaly.

    Mirrors tfsnippet ``VAE.variational`` / ``model`` / ``chain``.
    """

    def __init__(
        self,
        config,
        h_for_p_x,
        h_for_q_z,
        use_connected_z_p=True,
        use_connected_z_q=True,
        include_prior_in_loss=True,
        z_group_ndims=1,
        x_group_ndims=1,
    ):
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        self.x_dim = config.x_dim
        self.window_length = config.window_length
        self.use_connected_z_p = use_connected_z_p
        self.use_connected_z_q = use_connected_z_q
        self.include_prior_in_loss = include_prior_in_loss
        self.z_group_ndims = z_group_ndims
        self.x_group_ndims = x_group_ndims

        self.h_for_p_x = h_for_p_x
        self.h_for_q_z = h_for_q_z

        if use_connected_z_p:
            self.p_z = GaussianStateSpacePrior(config.z_dim, config.window_length)
        else:
            self.p_z = None

        if use_connected_z_q:
            self.recurrent_q = RecurrentDistribution(
                z_dim=config.z_dim,
                window_length=config.window_length,
                input_q_dim=config.dense_dim,
                std_epsilon=config.std_epsilon,
            )
        else:
            self.recurrent_q = None

    def variational(self, x, n_z=None, posterior_flow=None):
        """
        Derive q(z|h(x)), optionally transformed by ``posterior_flow``.

        Matches ``VAE.variational`` + ``FlowDistribution.sample``:
            z = flow(z0),  log q(z) = log q0(z0) - log|det|
        """
        z_params = self.h_for_q_z(x)

        if self.use_connected_z_q:
            input_q = z_params['input_q']
            z0 = self.recurrent_q.sample(input_q, n_samples=n_z)
            # base log_prob with group_ndims=z_group_ndims (default 1 → sum z_dim)
            log_q = self.recurrent_q.log_prob(
                z0, input_q, group_ndims=self.z_group_ndims,
            )
        else:
            normal = DiagonalNormal(z_params['mean'], z_params['std'])
            z0 = normal.sample(n_samples=n_z)
            log_q = normal.log_prob(z0, group_ndims=self.z_group_ndims)

        z = z0
        if posterior_flow is not None:
            # Flow over last dim; log_det shape == z.shape[:-1]
            # After group_ndims=1, log_q has already dropped z_dim, so we must
            # apply flow first then reduce, OR reduce log_det consistently.
            # FlowDistribution (v0.2.0-alpha1) samples base with group_ndims,
            # then subtracts log_det from the already-reduced log_prob.
            # PlanarNF log_det is per-feature-vector (no z_dim left).
            #
            # Our log_q is already reduced over z_dim → shape (..., T).
            # log_det from flow(z0) → shape (..., T). Elementwise subtract.
            z, log_det = posterior_flow(z0)
            log_q = log_q - log_det

        return {'z': z, 'z0': z0, 'log_q': log_q, 'z_params': z_params}

    def model_net(self, z, x=None, n_z=None):
        """
        Derive p(x|h(z)) and p(z).

        RNN encoder averages over n_z when z is 4-D (matches ``wrapper.rnn``).
        ``log_p_z`` / ``log_p_x`` are per-timestep (..., T) for SGVB.
        """
        if self.use_connected_z_p:
            # GSSM: z0 ~ N(0,I), zt ~ N(z_{t-1}, I)  → shape (..., T)
            log_p_z = self.p_z.log_prob(z, group_ndims=1, reduce_time=False)
        else:
            # Independent N(0,I) at each timestep → shape (..., T)
            log_p_z = gaussian_log_prob(
                z,
                torch.zeros_like(z),
                torch.ones_like(z),
                group_ndims=1,
            )

        x_params = self.h_for_p_x(z)
        p_x = DiagonalNormal(x_params['mean'], x_params['std'])

        log_p_x = None
        log_p_x_ungrouped = None
        if x is not None:
            # group_ndims=0 → keep feature dim, then sum features → (..., T)
            x_obs = x
            log_p_x_ungrouped = p_x.log_prob(x_obs, group_ndims=0)
            log_p_x = log_p_x_ungrouped.sum(dim=-1)

        return {
            'log_p_x': log_p_x,
            'log_p_x_ungrouped': log_p_x_ungrouped,
            'log_p_z': log_p_z,
            'x_params': x_params,
            'p_x': p_x,
        }

    def get_training_loss(self, x, posterior_flow=None, n_z=None):
        """
        SGVB training loss.

        Default (``include_prior_in_loss=True``):
            log_joint = log p(x|z) + log p(z)
            loss = mean(log q(z|x) - log_joint)

        Official TF OmniAnomaly (``include_prior_in_loss=False``):
            log_joint = log p(x|z)   # prior term omitted
            loss = mean(log q(z|x) - log_joint)
        """
        q_out = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
        z = q_out['z']
        log_q = q_out['log_q']  # (batch, T) or (n_z, batch, T)

        p_out = self.model_net(z, x=x, n_z=n_z)
        log_p_x = p_out['log_p_x']  # (batch, T)
        if self.include_prior_in_loss:
            log_joint = log_p_x + p_out['log_p_z']
        else:
            log_joint = log_p_x

        # SGVB: latent_log_prob - log_joint ; mean over sample dim then all elems
        sgvb = log_q - log_joint
        if n_z is not None and sgvb.dim() == 3:
            sgvb = sgvb.mean(dim=0)
        return sgvb.mean()

    def get_reconstruction_log_prob(self, x, n_z=None, posterior_flow=None,
                                    last_point_only=True, per_dim=False):
        """
        Reconstruction log-probability for anomaly scoring.

        Matches ``OmniAnomaly.get_score``:
          * q samples z (with optional flow)
          * p(x|z) uses RNN on z (4-D → mean over n_z inside encoder)
          * group_ndims = 0 if per_dim else 1
        """
        q_out = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
        z = q_out['z']

        # z mean / std over sample axis (TF get_score)
        if z.dim() == 4:
            z_mean = z.mean(dim=0)
            if n_z is not None and n_z > 1:
                # TF: sqrt(sum((z - mean)^2, 0) / (n_z - 1))
                z_std = torch.sqrt(
                    torch.sum((z - z_mean.unsqueeze(0)) ** 2, dim=0) / (n_z - 1)
                )
            else:
                z_std = torch.zeros_like(z_mean)
            z_info = torch.cat([z_mean, z_std], dim=-1)
        else:
            z_info = torch.cat([z, torch.zeros_like(z)], dim=-1)

        x_params = self.h_for_p_x(z)  # 4-D → mean over n_z inside encoder
        group_ndims = 0 if per_dim else 1
        log_prob = gaussian_log_prob(
            x, x_params['mean'], x_params['std'], group_ndims=group_ndims,
        )

        if last_point_only:
            # TF: r_prob[:, -1]  (last time step; keeps feature dim if per_dim)
            log_prob = log_prob[:, -1]

        return log_prob, z_info
