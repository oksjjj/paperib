# -*- coding: utf-8 -*-
"""OmniAnomaly model — PyTorch port of omni_anomaly.model."""
import torch.nn as nn

from omni_anomaly.flows import PlanarNormalizingFlows
from omni_anomaly.vae import VAE
from omni_anomaly.wrapper import GaussianParamNet, RecurrentGaussianParamNet


class OmniAnomaly(nn.Module):
    """
    Stochastic recurrent VAE (GRU + VAE + planar NF).

    Architecture mirrors the official TensorFlow implementation:
    https://github.com/NetManAIOps/OmniAnomaly
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._window_length = config.window_length
        self._x_dims = config.x_dim
        self._z_dims = config.z_dim

        if config.posterior_flow_type == 'nf':
            self._posterior_flow = PlanarNormalizingFlows(
                config.z_dim, config.nf_layers,
            )
        else:
            self._posterior_flow = None

        # p(x|z) head
        h_for_p_x = GaussianParamNet(
            input_dim=config.z_dim,
            output_dim=config.x_dim,
            rnn_num_hidden=config.rnn_num_hidden,
            dense_dim=config.dense_dim,
            rnn_cell=config.rnn_cell,
            hidden_dense=2,
            std_epsilon=config.std_epsilon,
        )

        # q(z|x) head
        if config.use_connected_z_q:
            h_for_q_z = RecurrentGaussianParamNet(
                input_dim=config.x_dim,
                rnn_num_hidden=config.rnn_num_hidden,
                dense_dim=config.dense_dim,
                rnn_cell=config.rnn_cell,
                hidden_dense=2,
            )
        else:
            h_for_q_z = GaussianParamNet(
                input_dim=config.x_dim,
                output_dim=config.z_dim,
                rnn_num_hidden=config.rnn_num_hidden,
                dense_dim=config.dense_dim,
                rnn_cell=config.rnn_cell,
                hidden_dense=2,
                std_epsilon=config.std_epsilon,
            )

        self._vae = VAE(
            config=config,
            h_for_p_x=h_for_p_x,
            h_for_q_z=h_for_q_z,
            use_connected_z_p=config.use_connected_z_p,
            use_connected_z_q=config.use_connected_z_q,
            include_prior_in_loss=getattr(config, 'include_prior_in_loss', True),
        )

    @property
    def x_dims(self):
        return self._x_dims

    @property
    def z_dims(self):
        return self._z_dims

    @property
    def window_length(self):
        return self._window_length

    @property
    def vae(self):
        return self._vae

    @property
    def posterior_flow(self):
        return self._posterior_flow

    def get_training_loss(self, x, n_z=None):
        """SGVB loss as in the official ``OmniAnomaly.get_training_loss``."""
        return self._vae.get_training_loss(
            x, posterior_flow=self._posterior_flow, n_z=n_z,
        )

    def get_score(self, x, n_z=None, last_point_only=True):
        """Reconstruction probability (official ``OmniAnomaly.get_score``)."""
        return self._vae.get_reconstruction_log_prob(
            x,
            n_z=n_z,
            posterior_flow=self._posterior_flow,
            last_point_only=last_point_only,
            per_dim=bool(self.config.get_score_on_dim),
        )
