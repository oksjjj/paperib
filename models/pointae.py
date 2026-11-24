"""HourAE와 ContextAE가 공유하는 point-wise AutoEncoder 및 학습 유틸리티."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PointAutoEncoder(nn.Module):
    """한 시점의 F개 feature를 잠재공간으로 압축했다가 다시 복원하는 공통 AE."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def set_seed(seed: int) -> None:
    """Python/NumPy/PyTorch 난수원을 같은 seed로 맞춘다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for_run() -> torch.device:
    """Apple Silicon MPS → CUDA → CPU 순서로 사용할 장치를 선택한다."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_robust_scaler(x: np.ndarray) -> RobustScaler:
    """feature별 median/IQR 기반 RobustScaler를 TRAIN 배열에 fit한다."""
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    scaler.fit(x)
    return scaler


def train_point_autoencoder(
    x_scaled: np.ndarray,
    *,
    n_features: int,
    cfg: Any,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[PointAutoEncoder, list[float]]:
    """정규화된 (N,F) point 배열로 공통 PointAutoEncoder를 학습한다."""
    set_seed(seed)
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=int(cfg.hidden_dim),
        latent_dim=int(cfg.latent_dim),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    criterion = nn.MSELoss(reduction="mean")

    tensor = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    if len(tensor) == 0:
        raise ValueError("AutoEncoder 학습 데이터가 비어 있습니다.")
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=min(int(cfg.batch_size), len(tensor)),
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
    model: nn.Module,
    x_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """입력 순서를 유지하면서 시점별 feature 평균 reconstruction MSE를 계산한다."""
    x = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(int(batch_size), max(len(x), 1)),
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
    """모델 state_dict를 장치와 분리된 CPU 복사본으로 반환한다."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def quantile_for_contamination(
    scores: np.ndarray,
    contamination_pct: float,
) -> float:
    """상위 contamination_pct%를 자르는 empirical quantile 경계를 반환한다."""
    q = 1.0 - float(contamination_pct) / 100.0
    q = float(np.clip(q, 0.0, 1.0))
    # np.quantile(scores, q): scores를 작은→큰 순으로 볼 때 하위 q*100% 지점의 값.
    #   예: contamination_pct=5 → q=0.95 → 상위 5%가 시작되는 점수(=경계 임계값).
    #   이 값보다 큰 score를 "제거할 후보"로 본다.
    return float(np.quantile(scores, q))


@dataclass
class DecontaminatedAEArtifacts:
    """한 시간 그룹/context의 2단계 AE 학습 산출물."""

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


def fit_decontaminated_group(
    x_raw: np.ndarray,
    *,
    cfg: Any,
    device: torch.device,
    group_seed: int,
    group_name: str,
    min_train_points: int = 20,
) -> DecontaminatedAEArtifacts:
    """한 그룹의 contaminated TRAIN으로 Stage-1 제거 후 fresh Final AE를 학습한다."""
    n = len(x_raw)
    if n < min_train_points:
        raise RuntimeError(f"{group_name}: too few train points ({n})")

    stage1_scaler = fit_robust_scaler(x_raw)
    x1 = stage1_scaler.transform(x_raw).astype(np.float32)
    # Stage-1은 그룹마다 서로 다른 난수 초기화를 사용한다.
    # cfg.seed는 전체 실험의 기준 seed이고, group_seed는 HourAE에서는 hour,
    # ContextAE에서는 context_id이다. 따라서 서로 다른 그룹의 AE가 우연히
    # 완전히 같은 초기 weight와 DataLoader shuffle 순서에서 시작하지 않도록 한다.
    # 예: cfg.seed=2, hour=5이면 Stage-1 seed는 7이다.
    stage1_seed = int(cfg.seed) + int(group_seed)
    stage1_model, stage1_hist = train_point_autoencoder(
        x1,
        n_features=x1.shape[1],
        cfg=cfg,
        epochs=int(cfg.stage1_epochs),
        device=device,
        seed=stage1_seed,
    )
    stage1_scores = reconstruction_mse(
        stage1_model,
        x1,
        batch_size=int(cfg.batch_size),
        device=device,
    )
    stage1_cut = quantile_for_contamination(
        stage1_scores,
        float(cfg.stage1_remove_pct),
    )
    clean_mask = stage1_scores <= stage1_cut

    if int(clean_mask.sum()) < max(min_train_points, int(0.5 * n)):
        raise RuntimeError(
            f"{group_name}: stage1 retained only {int(clean_mask.sum())}/{n}"
        )

    clean_raw = x_raw[clean_mask]
    final_scaler = fit_robust_scaler(clean_raw)
    x2 = final_scaler.transform(clean_raw).astype(np.float32)
    # Final AE는 Stage-1 가중치를 이어서 학습하는 모델이 아니라,
    # Stage-1이 골라낸 clean candidate TRAIN으로 처음부터 다시 만드는 fresh model이다.
    # 그래서 Final 단계에서도 새 PointAutoEncoder를 생성하며, seed까지 Stage-1과
    # 다르게 해 초기 weight와 shuffle 순서를 독립적으로 만든다.
    #
    # +1000은 통계적/이론적 의미가 있는 값이 아니다.
    # Stage-1 seed 영역과 Final seed 영역이 겹치지 않도록 분리하기 위한 단순 offset이다.
    # 같은 seed를 다시 써도 새 모델 객체이므로 'fresh model'이긴 하지만,
    # 그 경우 초기 weight가 Stage-1과 동일하게 재현될 수 있다.
    # 현재 구현은 완전히 독립적인 재초기화를 명확히 하기 위해 다른 seed를 사용한다.
    # 예: cfg.seed=2, hour/context=5이면 Final seed는 1007이다.
    final_seed = int(cfg.seed) + 1000 + int(group_seed)
    final_model, final_hist = train_point_autoencoder(
        x2,
        n_features=x2.shape[1],
        cfg=cfg,
        epochs=int(cfg.final_epochs),
        device=device,
        seed=final_seed,
    )
    final_clean_scores = reconstruction_mse(
        final_model,
        x2,
        batch_size=int(cfg.batch_size),
        device=device,
    )

    score_mean = float(np.mean(final_clean_scores))
    score_std = float(np.std(final_clean_scores, ddof=0))
    if not np.isfinite(score_std) or score_std < 1e-12:
        score_std = 1e-12

    ref_mse = quantile_for_contamination(
        final_clean_scores,
        float(cfg.final_contamination_pct),
    )
    ref_z = float((ref_mse - score_mean) / score_std)

    return DecontaminatedAEArtifacts(
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


def score_grouped_models(
    values_raw: np.ndarray,
    group_ids: np.ndarray,
    artifacts: dict[int, DecontaminatedAEArtifacts],
    *,
    n_features: int,
    cfg: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """각 시점을 자신의 group 모델로 추론해 z-score와 raw MSE를 원래 순서로 반환한다."""
    raw_mse = np.full(len(values_raw), np.nan, dtype=np.float64)
    z_scores = np.full(len(values_raw), np.nan, dtype=np.float64)

    for group_id, art in artifacts.items():
        idx = np.flatnonzero(group_ids == group_id)
        if idx.size == 0:
            continue

        model = build_point_autoencoder_from_state(
            art.final_state,
            n_features=n_features,
            hidden_dim=int(cfg.hidden_dim),
            latent_dim=int(cfg.latent_dim),
            device=device,
        )
        x = art.final_scaler.transform(values_raw[idx]).astype(np.float32)
        mse = reconstruction_mse(
            model,
            x,
            batch_size=int(cfg.batch_size),
            device=device,
        )
        raw_mse[idx] = mse
        z_scores[idx] = (
            (mse - art.final_score_mean)
            / max(art.final_score_std, 1e-12)
        )

    return z_scores, raw_mse


def build_point_autoencoder_from_state(
    state: dict[str, torch.Tensor],
    *,
    n_features: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
) -> PointAutoEncoder:
    """저장된 state_dict로 공통 PointAutoEncoder를 복원해 eval 모드로 반환한다."""
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
