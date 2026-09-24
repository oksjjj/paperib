# -*- coding: utf-8 -*-
"""Planar normalizing flows (faithful port of tfsnippet.layers.planar_nf)."""
import torch
import torch.nn as nn


class PlanarNormalizingFlow(nn.Module):
    """
    Single planar NF layer (Danilo Rezende & Mohamed, 2015) with the
    invertible ``u_hat`` reparameterization used by tfsnippet.

    y = x + u_hat * tanh(w^T x + b)
    """

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
        self.u = nn.Parameter(torch.randn(1, dim) * 0.01)

    def _u_hat(self):
        # wu: (1, 1)
        wu = torch.matmul(self.w, self.u.t())
        # u_hat = u + (softplus(wu) - 1 - wu) * w / ||w||^2
        return self.u + (-1.0 + torch.nn.functional.softplus(wu) - wu) * (
            self.w / torch.sum(self.w ** 2)
        )

    def forward(self, z):
        """
        Args:
            z: (..., dim)
        Returns:
            y, log_det with shape matching z[..., 0] (i.e. without last dim)
        """
        u_hat = self._u_hat()
        # Flatten to 2-D for matmul, matching tfsnippet flatten_to_ndims
        flat = z.reshape(-1, z.shape[-1])
        wxb = torch.matmul(flat, self.w.t()) + self.b  # (N, 1)
        tanh_wxb = torch.tanh(wxb)
        y_flat = flat + u_hat * tanh_wxb
        y = y_flat.reshape(z.shape)

        grad = 1.0 - tanh_wxb ** 2
        phi = grad * self.w  # (N, dim)
        det_jac = 1.0 + torch.matmul(phi, u_hat.t())  # (N, 1)
        log_det = torch.log(torch.abs(det_jac) + 1e-8).squeeze(-1)
        log_det = log_det.reshape(z.shape[:-1])
        return y, log_det


class PlanarNormalizingFlows(nn.Module):
    """Stack of planar flows (matches ``planar_normalizing_flows(n_layers)``)."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.flows = nn.ModuleList(
            [PlanarNormalizingFlow(dim) for _ in range(n_layers)]
        )

    def forward(self, z):
        log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, ld = flow(z)
            log_det = log_det + ld
        return z, log_det
