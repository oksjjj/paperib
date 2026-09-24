"""Train / score Anomaly Transformer on in-memory PLMN arrays.

Scoring follows the official association-discrepancy energy
(https://github.com/thuml/Anomaly-Transformer): higher energy = more anomalous.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.AnomalyTransformer import AnomalyTransformer


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, lr_: float) -> None:
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}", flush=True)


class WindowDataset(Dataset):
    """Sliding windows over a (T, D) array."""

    def __init__(
        self,
        data: np.ndarray,
        win_size: int,
        step: int = 1,
        labels: np.ndarray | None = None,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.win_size = int(win_size)
        self.step = max(1, int(step))
        if labels is None:
            self.labels = np.zeros(len(self.data), dtype=np.float32)
        else:
            self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
            if len(self.labels) != len(self.data):
                raise ValueError("labels length must match data length")
        n = len(self.data)
        if n < self.win_size:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(0, n - self.win_size + 1, self.step, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int):
        s = int(self.starts[index])
        e = s + self.win_size
        return self.data[s:e], self.labels[s:e]


class EarlyStopping:
    def __init__(self, patience: int = 3, verbose: bool = True, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf

    def __call__(
        self,
        val_loss: float,
        val_loss2: float,
        model: nn.Module,
        path: str,
        filename: str,
    ) -> None:
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, filename)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, filename)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        val_loss2: float,
        model: nn.Module,
        path: str,
        filename: str,
    ) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                "Saving model ...",
                flush=True,
            )
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, filename))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


def _association_losses(
    series: list[torch.Tensor],
    prior: list[torch.Tensor],
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
        prior_u = prior[u] / torch.unsqueeze(
            torch.sum(prior[u], dim=-1), dim=-1
        ).repeat(1, 1, 1, win_size)
        series_loss = series_loss + (
            torch.mean(my_kl_loss(series[u], prior_u.detach()))
            + torch.mean(my_kl_loss(prior_u.detach(), series[u]))
        )
        prior_loss = prior_loss + (
            torch.mean(my_kl_loss(prior_u, series[u].detach()))
            + torch.mean(my_kl_loss(series[u].detach(), prior_u))
        )
    series_loss = series_loss / len(prior)
    prior_loss = prior_loss / len(prior)
    return series_loss, prior_loss


def _window_energy(
    model: AnomalyTransformer,
    batch: torch.Tensor,
    *,
    win_size: int,
    temperature: float = 50.0,
    return_dim_mse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Official cri = softmax(-series_loss - prior_loss) * rec_mse (B, L)."""
    criterion = nn.MSELoss(reduction="none")
    output, series, prior, _ = model(batch)
    loss = torch.mean(criterion(batch, output), dim=-1)  # B, L
    series_loss = None
    prior_loss = None
    for u in range(len(prior)):
        prior_u = prior[u] / torch.unsqueeze(
            torch.sum(prior[u], dim=-1), dim=-1
        ).repeat(1, 1, 1, win_size)
        s = my_kl_loss(series[u], prior_u.detach()) * temperature
        p = my_kl_loss(prior_u, series[u].detach()) * temperature
        series_loss = s if series_loss is None else series_loss + s
        prior_loss = p if prior_loss is None else prior_loss + p
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    cri = metric * loss
    dim_mse = None
    if return_dim_mse:
        dim_mse = criterion(batch, output)  # B, L, D
    return cri, dim_mse


class PlmnAnomalyTransformer:
    """Thin trainer/scorer for paperib PLMN splits."""

    def __init__(
        self,
        *,
        win_size: int = 100,
        input_c: int,
        lr: float = 1e-4,
        k: float = 3.0,
        batch_size: int = 64,
        num_epochs: int = 10,
        e_layers: int = 3,
        d_model: int = 512,
        device: torch.device | None = None,
        anormly_ratio: float = 1.0,
        train_step: int = 1,
        temperature: float = 50.0,
    ):
        self.win_size = int(win_size)
        self.input_c = int(input_c)
        self.lr = float(lr)
        self.k = float(k)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.anormly_ratio = float(anormly_ratio)
        self.train_step = max(1, int(train_step))
        self.temperature = float(temperature)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = AnomalyTransformer(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.input_c,
            d_model=d_model,
            e_layers=e_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def _loader(
        self,
        data: np.ndarray,
        *,
        step: int,
        shuffle: bool,
        labels: np.ndarray | None = None,
    ) -> DataLoader:
        ds = WindowDataset(data, self.win_size, step=step, labels=labels)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def _vali(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        loss_1: list[float] = []
        loss_2: list[float] = []
        with torch.no_grad():
            for input_data, _ in loader:
                inp = input_data.float().to(self.device)
                output, series, prior, _ = self.model(inp)
                series_loss, prior_loss = _association_losses(series, prior, self.win_size)
                rec_loss = self.criterion(output, inp)
                loss_1.append((rec_loss - self.k * series_loss).item())
                loss_2.append((rec_loss + self.k * prior_loss).item())
        if not loss_1:
            return float("inf"), float("inf")
        return float(np.average(loss_1)), float(np.average(loss_2))

    def train(
        self,
        train: np.ndarray,
        valid: np.ndarray,
        *,
        model_dir: str,
        checkpoint_name: str = "checkpoint.pth",
    ) -> dict[str, Any]:
        print("======================TRAIN MODE======================", flush=True)
        train_loader = self._loader(train, step=self.train_step, shuffle=True)
        # Official uses test loader as vali; we use chronological valid.
        vali_loader = self._loader(valid, step=max(1, self.win_size // 2), shuffle=False)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train_steps = max(len(train_loader), 1)
        metrics: dict[str, Any] = {}

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list: list[float] = []
            epoch_time = time.time()
            self.model.train()
            time_now = time.time()
            for i, (input_data, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                inp = input_data.float().to(self.device)
                output, series, prior, _ = self.model(inp)
                series_loss, prior_loss = _association_losses(series, prior, self.win_size)
                rec_loss = self.criterion(output, inp)
                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                if (i + 1) % 50 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * (
                        (self.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s",
                        flush=True,
                    )
                    iter_count = 0
                    time_now = time.time()
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print(
                f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s",
                flush=True,
            )
            train_loss = float(np.average(loss1_list)) if loss1_list else float("nan")
            vali_loss1, vali_loss2 = self._vali(vali_loader)
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss1:.7f}",
                flush=True,
            )
            early_stopping(
                vali_loss1, vali_loss2, self.model, model_dir, checkpoint_name
            )
            if early_stopping.early_stop:
                print("Early stopping", flush=True)
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        ckpt = os.path.join(model_dir, checkpoint_name)
        if os.path.isfile(ckpt):
            self.load(ckpt)
        metrics["train_loss"] = train_loss if loss1_list else None
        metrics["vali_loss"] = vali_loss1
        return metrics

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def energy_series(
        self,
        data: np.ndarray,
        *,
        step: int = 1,
        with_dim_mse: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Point-aligned energy (T,) by averaging overlapping window scores.

        Also optionally returns per-dim MSE (T, D) averaged the same way.
        """
        data = np.asarray(data, dtype=np.float32)
        n, d = data.shape
        energy_sum = np.zeros(n, dtype=np.float64)
        energy_cnt = np.zeros(n, dtype=np.float64)
        dim_sum = np.zeros((n, d), dtype=np.float64) if with_dim_mse else None
        loader = self._loader(data, step=step, shuffle=False)
        self.model.eval()
        ds: WindowDataset = loader.dataset  # type: ignore[assignment]
        offset = 0
        for input_data, _ in loader:
            bsz = input_data.shape[0]
            starts = ds.starts[offset : offset + bsz]
            offset += bsz
            inp = input_data.float().to(self.device)
            cri, dim_mse = _window_energy(
                self.model,
                inp,
                win_size=self.win_size,
                temperature=self.temperature,
                return_dim_mse=with_dim_mse,
            )
            cri_np = cri.detach().cpu().numpy()
            dim_np = (
                dim_mse.detach().cpu().numpy() if dim_mse is not None else None
            )
            for bi, s in enumerate(starts):
                e = int(s) + self.win_size
                energy_sum[s:e] += cri_np[bi]
                energy_cnt[s:e] += 1.0
                if dim_sum is not None and dim_np is not None:
                    dim_sum[s:e] += dim_np[bi]
        mask = energy_cnt > 0
        out = np.full(n, np.nan, dtype=np.float64)
        out[mask] = energy_sum[mask] / energy_cnt[mask]
        dim_out = None
        if dim_sum is not None:
            dim_out = np.full((n, d), np.nan, dtype=np.float64)
            dim_out[mask] = dim_sum[mask] / energy_cnt[mask, None]
        return out, dim_out

    def threshold_from_train(self, train_energy: np.ndarray) -> float:
        finite = train_energy[np.isfinite(train_energy)]
        if len(finite) == 0:
            return float("inf")
        # Official: percentile(combined, 100 - anormly_ratio)
        return float(np.percentile(finite, 100.0 - self.anormly_ratio))
