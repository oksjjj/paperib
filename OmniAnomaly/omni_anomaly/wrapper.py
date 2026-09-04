# -*- coding: utf-8 -*-
"""Network building blocks — PyTorch port of omni_anomaly.wrapper."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class SoftplusStd(nn.Module):
    """``softplus(dense(inputs)) + epsilon`` (matches ``softplus_std``)."""

    def __init__(self, in_features, out_features, epsilon=1e-4):
        super().__init__()
        self.linear = _xavier_linear(in_features, out_features)
        self.epsilon = epsilon

    def forward(self, x):
        return F.softplus(self.linear(x)) + self.epsilon


class StackedRNNEncoder(nn.Module):
    """
    Port of ``wrapper.rnn``:

    * if input is 4-D → mean over first axis (n_z)
    * RNN over time
    * ``hidden_dense`` successive dense layers without activation
    """

    def __init__(
        self,
        input_dim,
        rnn_num_hidden=500,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=500,
    ):
        super().__init__()
        cell = (rnn_cell or 'GRU').upper()
        if cell == 'LSTM':
            self.rnn = nn.LSTM(input_dim, rnn_num_hidden, batch_first=True)
        elif cell == 'GRU':
            self.rnn = nn.GRU(input_dim, rnn_num_hidden, batch_first=True)
        elif cell == 'BASIC':
            self.rnn = nn.RNN(input_dim, rnn_num_hidden, batch_first=True)
        else:
            raise ValueError('rnn_cell must be LSTM, GRU, or Basic')

        layers = []
        in_dim = rnn_num_hidden
        for _ in range(hidden_dense):
            layers.append(_xavier_linear(in_dim, dense_dim))
            in_dim = dense_dim
        self.dense_layers = nn.ModuleList(layers)

    def forward(self, x):
        if x.dim() == 4:
            # wrapper.rnn: reduce_mean over n_z axis
            x = x.mean(dim=0)
        elif x.dim() != 3:
            logging.error('rnn input shape error')
        outputs, _ = self.rnn(x)
        for layer in self.dense_layers:
            outputs = layer(outputs)
        return outputs


class GaussianParamNet(nn.Module):
    """``wrap_params_net``: RNN → mean / std."""

    def __init__(self, input_dim, output_dim, rnn_num_hidden, dense_dim,
                 rnn_cell='GRU', hidden_dense=2, std_epsilon=1e-4):
        super().__init__()
        self.encoder = StackedRNNEncoder(
            input_dim, rnn_num_hidden, rnn_cell, hidden_dense, dense_dim,
        )
        self.mean_layer = _xavier_linear(dense_dim, output_dim)
        self.std_layer = SoftplusStd(dense_dim, output_dim, epsilon=std_epsilon)

    def forward(self, x):
        h = self.encoder(x)
        return {'mean': self.mean_layer(h), 'std': self.std_layer(h)}


class RecurrentGaussianParamNet(nn.Module):
    """``wrap_params_net_srnn`` / connected q: RNN → ``input_q``."""

    def __init__(self, input_dim, rnn_num_hidden, dense_dim,
                 rnn_cell='GRU', hidden_dense=2):
        super().__init__()
        self.encoder = StackedRNNEncoder(
            input_dim, rnn_num_hidden, rnn_cell, hidden_dense, dense_dim,
        )

    def forward(self, x):
        return {'input_q': self.encoder(x)}
