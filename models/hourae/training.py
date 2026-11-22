"""HourAE의 데이터 준비, 정규화, decontamination, AE 학습과 scoring 로직."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.data import load_plmn
from core.features import is_rate_metric, metric_columns
from models.hourae.model import PointAutoEncoder

@dataclass
class Config:
    timezone: str = "Asia/Seoul"
    stage1_remove_pct: float = 5.0
    final_contamination_pct: float = 0.01
    threshold_sigma: float = 3.0
    hidden_dim: int = 32
    latent_dim: int = 8
    stage1_epochs: int = 50
    final_epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 2
    sample_minutes: float = 5.0

@dataclass
class HourArtifacts:
    hour: int
    stage1_scaler: RobustScaler
    final_scaler: RobustScaler
    stage1_state: dict[str, torch.Tensor]
    final_state: dict[str, torch.Tensor]
    stage1_threshold_mse: float
    final_score_mean: float
    final_score_std: float
    final_reference_quantile_mse: float
    final_reference_quantile_z: float
    n_train: int
    n_removed: int
    stage1_final_loss: float
    final_final_loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_indices(
    n: int,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> tuple[int, int]:
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    return train_end, min(valid_end, n)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in metric_columns(df) if not is_rate_metric(c)]


def prepare_unlabeled_plmn(plmn: str, tz_name: str) -> dict[str, Any]:
    df = load_plmn(plmn)
    cols = feature_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: no raw feature columns")

    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    times = pd.to_datetime(df["time"], utc=True)
    local_times = times.dt.tz_convert(tz_name)
    hours = local_times.dt.hour.to_numpy(dtype=np.int16)

    train_end, valid_end = split_indices(len(df))

    return {
        "df": df,
        "cols": cols,
        "times": times,
        "local_times": local_times,
        "train_end": train_end,
        "valid_end": valid_end,
        "train": values[:train_end],
        "valid": values[train_end:valid_end],
        "test": values[valid_end:],
        "train_hours": hours[:train_end],
        "valid_hours": hours[train_end:valid_end],
        "test_hours": hours[valid_end:],
        "split": {
            "train_time_end": str(times.iloc[train_end - 1]),
            "valid_time_start": str(times.iloc[train_end]),
            "valid_time_end": str(times.iloc[valid_end - 1]),
            "test_time_start": str(times.iloc[valid_end]),
            "test_time_end": str(times.iloc[-1]),
        },
    }


def fit_scaler(x: np.ndarray) -> RobustScaler:
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    scaler.fit(x)
    return scaler


def train_ae(
    x_scaled: np.ndarray,
    *,
    n_features: int,
    cfg: Config,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[PointAutoEncoder, list[float]]:
    set_seed(seed)

    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.MSELoss(reduction="mean")

    tensor = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(cfg.batch_size, len(tensor)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    history: list[float] = []
    for _epoch in range(int(epochs)):
        model.train()
        batch_losses: list[float] = []

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        history.append(float(np.mean(batch_losses)))

    return model, history


@torch.no_grad()
def reconstruction_mse(
    model: PointAutoEncoder,
    x_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(batch_size, max(len(x), 1)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model.eval()
    out: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        recon = model(xb)
        mse = torch.mean((recon - xb) ** 2, dim=1)
        out.append(mse.detach().cpu().numpy().astype(np.float64))

    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)

def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def quantile_for_contamination(
    scores: np.ndarray,
    contamination_pct: float,
) -> float:
    q = 1.0 - float(contamination_pct) / 100.0
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(scores, q))


def fit_hour(
    x_hour_raw: np.ndarray,
    *,
    hour: int,
    cfg: Config,
    device: torch.device,
) -> HourArtifacts:
    n = len(x_hour_raw)
    if n < 20:
        raise RuntimeError(f"hour {hour:02d}: too few train points ({n})")

    stage1_scaler = fit_scaler(x_hour_raw)
    x1 = stage1_scaler.transform(x_hour_raw).astype(np.float32)

    stage1_model, stage1_hist = train_ae(
        x1,
        n_features=x1.shape[1],
        cfg=cfg,
        epochs=cfg.stage1_epochs,
        device=device,
        seed=cfg.seed + hour,
    )
    stage1_scores = reconstruction_mse(
        stage1_model,
        x1,
        batch_size=cfg.batch_size,
        device=device,
    )

    stage1_cut = quantile_for_contamination(
        stage1_scores,
        cfg.stage1_remove_pct,
    )
    clean_mask = stage1_scores <= stage1_cut

    if int(clean_mask.sum()) < max(20, int(0.5 * n)):
        raise RuntimeError(
            f"hour {hour:02d}: stage1 retained only {int(clean_mask.sum())}/{n}"
        )

    clean_raw = x_hour_raw[clean_mask]
    final_scaler = fit_scaler(clean_raw)
    x2 = final_scaler.transform(clean_raw).astype(np.float32)

    final_model, final_hist = train_ae(
        x2,
        n_features=x2.shape[1],
        cfg=cfg,
        epochs=cfg.final_epochs,
        device=device,
        seed=cfg.seed + 1000 + hour,
    )
    final_clean_scores = reconstruction_mse(
        final_model,
        x2,
        batch_size=cfg.batch_size,
        device=device,
    )

    score_mean = float(np.mean(final_clean_scores))
    score_std = float(np.std(final_clean_scores, ddof=0))
    if not np.isfinite(score_std) or score_std < 1e-12:
        score_std = 1e-12

    ref_mse = quantile_for_contamination(
        final_clean_scores,
        cfg.final_contamination_pct,
    )
    ref_z = float((ref_mse - score_mean) / score_std)

    return HourArtifacts(
        hour=hour,
        stage1_scaler=stage1_scaler,
        final_scaler=final_scaler,
        stage1_state=cpu_state_dict(stage1_model),
        final_state=cpu_state_dict(final_model),
        stage1_threshold_mse=float(stage1_cut),
        final_score_mean=score_mean,
        final_score_std=score_std,
        final_reference_quantile_mse=ref_mse,
        final_reference_quantile_z=ref_z,
        n_train=n,
        n_removed=int((~clean_mask).sum()),
        stage1_final_loss=float(stage1_hist[-1]),
        final_final_loss=float(final_hist[-1]),
    )


def build_final_model(
    artifacts: HourArtifacts,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> PointAutoEncoder:
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    )
    model.load_state_dict(artifacts.final_state)
    model.to(device)
    model.eval()
    return model


def score_with_hour_models(
    values_raw: np.ndarray,
    hours: np.ndarray,
    artifacts: dict[int, HourArtifacts],
    *,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mse = np.full(len(values_raw), np.nan, dtype=np.float64)
    z_scores = np.full(len(values_raw), np.nan, dtype=np.float64)

    for hour in range(24):
        idx = np.flatnonzero(hours == hour)
        if idx.size == 0:
            continue

        art = artifacts[hour]
        model = build_final_model(
            art,
            n_features=n_features,
            cfg=cfg,
            device=device,
        )
        x = art.final_scaler.transform(values_raw[idx]).astype(np.float32)
        mse = reconstruction_mse(
            model,
            x,
            batch_size=cfg.batch_size,
            device=device,
        )
        raw_mse[idx] = mse
        z_scores[idx] = (
            (mse - art.final_score_mean)
            / max(art.final_score_std, 1e-12)
        )

    return z_scores, raw_mse
