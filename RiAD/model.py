"""RiAD-style contamination-robust reconstruction model (paperib).

Faithful to the *roles* in Chen et al., Information Sciences 2024 (RiAD):
- uncertainty-aware reconstruction
- adaptive loss that down-weights likely contaminated windows
- anomaly injection + memory of normality

This is a practical PyTorch backbone for IB experiments, not a line-by-line
reimplementation of the authors' unpublished code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AwarenessMemory(nn.Module):
    """FIFO bank of normal latent vectors (Awareness Memory Module, simplified)."""

    def __init__(self, dim: int, size: int = 256):
        super().__init__()
        self.size = int(size)
        self.register_buffer("bank", torch.zeros(self.size, dim), persistent=True)
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("filled", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def write(self, z: torch.Tensor, keep: torch.Tensor | None = None) -> None:
        """Write rows of ``z`` (B, D) marked by ``keep`` (B,) into the bank."""
        if z.ndim != 2:
            raise ValueError("z must be (B, D)")
        if keep is None:
            keep = torch.ones(z.size(0), dtype=torch.bool, device=z.device)
        rows = z.detach()[keep.bool()]
        if rows.numel() == 0:
            return
        for row in rows:
            i = int(self.ptr.item()) % self.size
            self.bank[i] = row
            self.ptr += 1
            self.filled = torch.clamp(self.filled + 1, max=self.size)

    def contrast_loss(self, z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Pull ``z`` toward nearest memory prototype when bank is warm."""
        n = int(self.filled.item())
        if n < 8:
            return z.new_zeros(())
        proto = self.bank[:n]
        # (B, N)
        sim = F.cosine_similarity(z.unsqueeze(1), proto.unsqueeze(0), dim=-1)
        max_sim, _ = sim.max(dim=1)
        return (1.0 - max_sim).mean() / max(temperature, 1e-3)


class RiADModel(nn.Module):
    """LSTM encoder–decoder with uncertainty head + awareness memory."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 64,
        layers: int = 1,
        latent: int = 32,
        memory_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(self.hidden, self.latent)
        self.from_latent = nn.Linear(self.latent, self.hidden)
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.recon_mu = nn.Linear(self.hidden, self.n_features)
        self.recon_logvar = nn.Linear(self.hidden, self.n_features)
        self.memory = AwarenessMemory(self.latent, size=memory_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        h0 = self.from_latent(z).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        # Teacher-forced decode from input windows (reconstruction AE).
        out, _ = self.decoder(x, (h0, c0))
        mu = self.recon_mu(out)
        logvar = self.recon_logvar(out).clamp(-8.0, 8.0)
        return mu, logvar

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        mu, logvar = self.decode(x, z)
        return {"z": z, "mu": mu, "logvar": logvar}

    def recon_error(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Per-window mean squared error → (B,)."""
        return ((x - mu) ** 2).mean(dim=(1, 2))

    def nll(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL per window → (B,)."""
        inv = torch.exp(-logvar)
        return (0.5 * (inv * (x - mu) ** 2 + logvar)).mean(dim=(1, 2))


def inject_anomalies(
    x: torch.Tensor,
    *,
    p: float = 0.15,
    scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anomaly Injection Module (simplified).

    Returns ``(x_aug, is_injected)`` where ``is_injected`` is (B,) bool.
    """
    if p <= 0:
        return x, torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    b, t, f = x.shape
    inj = torch.rand(b, device=x.device) < p
    if not inj.any():
        return x, inj
    out = x.clone()
    # Global spike on a random subset of features / timesteps.
    for i in torch.where(inj)[0].tolist():
        ti = int(torch.randint(0, t, (1,), device=x.device).item())
        fi = int(torch.randint(0, f, (1,), device=x.device).item())
        sign = 1.0 if torch.rand((), device=x.device) < 0.5 else -1.0
        # Relative to batch feature std for scale-free injection.
        feat_std = x[:, :, fi].std().clamp_min(1e-3)
        out[i, ti, fi] = out[i, ti, fi] + sign * scale * feat_std
    return out, inj


def adaptive_weights(
    loss_vec: torch.Tensor,
    *,
    temperature: float = 1.0,
    hard_frac: float = 0.2,
) -> torch.Tensor:
    """Down-weight high-loss (likely contaminated) windows.

    Soft weights via ``softmax(-loss/T)``, then zero the top ``hard_frac`` losses.
    """
    w = torch.softmax(-loss_vec / max(temperature, 1e-3), dim=0)
    if hard_frac > 0 and loss_vec.numel() > 4:
        k = max(1, int(loss_vec.numel() * hard_frac))
        topk = torch.topk(loss_vec, k=k, largest=True).indices
        w = w.clone()
        w[topk] = 0.0
        s = w.sum()
        if float(s) > 0:
            w = w / s
        else:
            w = torch.ones_like(loss_vec) / loss_vec.numel()
    return w


def riad_train_step(
    model: RiADModel,
    batch: torch.Tensor,
    *,
    inject_p: float = 0.15,
    mem_weight: float = 0.05,
    temperature: float = 1.0,
    hard_frac: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One training step with adaptive NLL + injection + memory contrast."""
    x_aug, injected = inject_anomalies(batch, p=inject_p)
    out = model(x_aug)
    nll = model.nll(batch, out["mu"], out["logvar"])
    # Treat injected windows as known-contaminated → force low weight.
    loss_for_w = nll.detach().clone()
    loss_for_w[injected] = loss_for_w[injected] + 10.0
    w = adaptive_weights(loss_for_w, temperature=temperature, hard_frac=hard_frac)
    recon = (w * nll).sum()
    # Write non-injected, low-loss windows into memory.
    with torch.no_grad():
        keep = (~injected) & (nll <= nll.median())
        model.memory.write(out["z"], keep=keep)
    mem = model.memory.contrast_loss(out["z"][~injected]) if (~injected).any() else batch.new_zeros(())
    loss = recon + mem_weight * mem
    stats = {
        "loss": float(loss.detach()),
        "nll": float(nll.mean().detach()),
        "mem": float(mem.detach()) if torch.is_tensor(mem) else 0.0,
        "inject_frac": float(injected.float().mean().detach()),
    }
    return loss, stats
