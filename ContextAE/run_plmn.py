#!/usr/bin/env python3
"""Data-driven contextual AutoEncoder with K train-only context models.

This experiment changes only the context partitioning relative to HourAE.

1. Split data chronologically into train/valid/test.
2. Using TRAIN only, map each local 5-minute slot to one robust temporal profile.
3. Partition the 288 daily slots into K contiguous temporal regimes.
4. For each K, find the minimum-SSE contiguous segmentation by dynamic programming.
5. Select K using train-only silhouette score, unless --k explicitly fixes K.
6. Train one iterative-decontamination AutoEncoder per discovered context.
7. Score each point with the AutoEncoder assigned to its local 5-minute slot.
8. Load human labels only after all models and scores are fixed.
9. Train the fixed 24-hour HourAE baseline with the same hyperparameters and compare validation metrics.

Directional scoring is intentionally NOT included here. This script isolates the
effect of replacing fixed 24 hourly models with data-driven K context models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LABELING = REPO / "labeling"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"
MODEL_DIR = ROOT / "model"

if str(LABELING) not in sys.path:
    sys.path.insert(0, str(LABELING))

from tool import (
    human_anomaly_mask,
    is_rate_metric,
    load_labels,
    load_plmn,
    metric_columns,
)


def _load_metrics():
    path = REPO / "RiAD" / "metrics.py"
    spec = importlib.util.spec_from_file_location("paperib_metrics_contextae", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paper_metrics = _load_metrics()
evaluate_valid = paper_metrics.evaluate_riad_valid
resolve_vus_window = paper_metrics.resolve_vus_window


def _load_hourae():
    """Load the existing fixed 24-hour HourAE implementation."""
    path = REPO / "HourAE" / "run_plmn.py"
    spec = importlib.util.spec_from_file_location("paperib_hourae_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Config:
    timezone: str = "Asia/Seoul"
    k: int = 0
    k_min: int = 2
    k_max: int = 24
    min_segment_slots: int = 3
    profile_use_mad: bool = True
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


class PointAutoEncoder(nn.Module):
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


@dataclass
class ContextArtifacts:
    context_id: int
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


def local_slot_ids(local_times: pd.Series) -> np.ndarray:
    """5-minute slot index in local day: 00:00=0, ..., 23:55=287."""
    hour = local_times.dt.hour.to_numpy(dtype=np.int16)
    minute = local_times.dt.minute.to_numpy(dtype=np.int16)
    return (hour * 12 + minute // 5).astype(np.int16)


def slot_label(slot: int) -> str:
    hour = int(slot) // 12
    minute = (int(slot) % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def prepare_unlabeled_plmn(plmn: str, tz_name: str) -> dict[str, Any]:
    df = load_plmn(plmn)
    cols = feature_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: no raw feature columns")

    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    times = pd.to_datetime(df["time"], utc=True)
    local_times = times.dt.tz_convert(tz_name)
    slots = local_slot_ids(local_times)

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
        "train_slots": slots[:train_end],
        "valid_slots": slots[train_end:valid_end],
        "test_slots": slots[valid_end:],
        "split": {
            "train_time_end": str(times.iloc[train_end - 1]),
            "valid_time_start": str(times.iloc[train_end]),
            "valid_time_end": str(times.iloc[valid_end - 1]),
            "test_time_start": str(times.iloc[valid_end]),
            "test_time_end": str(times.iloc[-1]),
        },
    }


def robust_mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis)
    return np.median(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)


def build_train_slot_profiles(
    train_raw: np.ndarray,
    train_slots: np.ndarray,
    *,
    use_mad: bool,
) -> dict[str, Any]:
    """Build one robust profile for every 5-minute slot using TRAIN only.

    Raw features are robust-scaled from TRAIN only. For each slot we summarize
    across days with median and optionally MAD. If a slot is absent from TRAIN,
    its profile is copied from the nearest observed slot in circular clock time.
    Validation/test data are never used to construct or fill profiles.
    """
    clustering_input_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    train_scaled = clustering_input_scaler.fit_transform(train_raw)

    observed_slots = np.unique(train_slots).astype(int)
    if observed_slots.size == 0:
        raise RuntimeError("no train time slots observed")

    observed_profiles: dict[int, np.ndarray] = {}
    observed_counts: dict[int, int] = {}

    for slot in observed_slots:
        block = train_scaled[train_slots == slot]
        med = np.median(block, axis=0)
        if use_mad:
            mad = np.median(np.abs(block - med), axis=0)
            profile = np.concatenate([med, mad], axis=0)
        else:
            profile = med
        observed_profiles[int(slot)] = profile.astype(np.float64)
        observed_counts[int(slot)] = int(len(block))

    all_slots = np.arange(288, dtype=int)
    missing_slots: list[int] = []
    profiles: list[np.ndarray] = []
    counts: list[int] = []

    for slot in all_slots:
        if int(slot) in observed_profiles:
            profiles.append(observed_profiles[int(slot)])
            counts.append(observed_counts[int(slot)])
            continue

        missing_slots.append(int(slot))
        circular = np.minimum(
            np.abs(observed_slots - slot),
            288 - np.abs(observed_slots - slot),
        )
        nearest_slot = int(observed_slots[int(np.argmin(circular))])
        profiles.append(observed_profiles[nearest_slot].copy())
        counts.append(0)

    profile_matrix = np.vstack(profiles)
    profile_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    profile_matrix_scaled = profile_scaler.fit_transform(profile_matrix)

    return {
        "slots": all_slots,
        "observed_slots": observed_slots,
        "missing_slots": missing_slots,
        "profiles": profile_matrix,
        "profiles_scaled": profile_matrix_scaled,
        "slot_counts": np.asarray(counts, dtype=int),
        "input_scaler": clustering_input_scaler,
        "profile_scaler": profile_scaler,
    }


def build_segment_cost_matrix(
    x: np.ndarray,
    *,
    min_segment_slots: int,
) -> np.ndarray:
    """SSE cost for every contiguous [start, end) segment."""
    n, _d = x.shape
    m = int(min_segment_slots)
    cost = np.full((n + 1, n + 1), np.inf, dtype=np.float64)

    cum_sum = np.vstack(
        [np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(x, axis=0)]
    )
    cum_sq = np.concatenate(
        [[0.0], np.cumsum(np.sum(x * x, axis=1), dtype=np.float64)]
    )

    for start in range(n):
        ends = np.arange(start + m, n + 1, dtype=int)
        if ends.size == 0:
            continue
        sums = cum_sum[ends] - cum_sum[start]
        lengths = (ends - start).astype(np.float64)
        sse = (
            cum_sq[ends]
            - cum_sq[start]
            - np.sum(sums * sums, axis=1) / lengths
        )
        cost[start, ends] = np.maximum(sse, 0.0)

    return cost


def optimal_contiguous_partition(
    cost: np.ndarray,
    *,
    k: int,
    min_segment_slots: int,
) -> tuple[np.ndarray, list[int], float]:
    """Minimum-SSE partition of 00:00..23:55 into exactly K segments."""
    n = cost.shape[0] - 1
    m = int(min_segment_slots)

    if k < 1 or k * m > n:
        raise ValueError(
            f"infeasible K={k} for n_slots={n}, min_segment_slots={m}"
        )

    dp = np.full((k + 1, n + 1), np.inf, dtype=np.float64)
    prev = np.full((k + 1, n + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for seg in range(1, k + 1):
        j_min = seg * m
        j_max = n - (k - seg) * m

        for end in range(j_min, j_max + 1):
            start_min = (seg - 1) * m
            start_max = end - m
            starts = np.arange(start_min, start_max + 1, dtype=int)
            values = dp[seg - 1, starts] + cost[starts, end]
            best_pos = int(np.argmin(values))
            best_start = int(starts[best_pos])
            best_value = float(values[best_pos])

            if np.isfinite(best_value):
                dp[seg, end] = best_value
                prev[seg, end] = best_start

    if not np.isfinite(dp[k, n]):
        raise RuntimeError(f"failed to build contiguous partition for K={k}")

    boundaries = [n]
    end = n
    for seg in range(k, 0, -1):
        start = int(prev[seg, end])
        if start < 0:
            raise RuntimeError(f"invalid DP backtrack for K={k}, seg={seg}")
        boundaries.append(start)
        end = start
    boundaries = sorted(boundaries)

    labels = np.empty(n, dtype=np.int16)
    for context_id, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[a:b] = context_id

    return labels, boundaries, float(dp[k, n])


def fit_context_segmentation(
    slot_info: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Discover K contiguous temporal contexts from TRAIN profiles only."""
    x = np.asarray(slot_info["profiles_scaled"], dtype=np.float64)
    n_slots = len(x)

    if n_slots != 288:
        raise RuntimeError(f"expected 288 daily slots, got {n_slots}")

    m = int(cfg.min_segment_slots)
    if m < 1:
        raise RuntimeError("min_segment_slots must be >= 1")

    max_feasible_k = n_slots // m

    if cfg.k > 0:
        candidate_ks = [int(cfg.k)]
    else:
        lower = max(int(cfg.k_min), 2)
        upper = min(int(cfg.k_max), max_feasible_k, n_slots - 1)
        if lower > upper:
            raise RuntimeError(
                f"invalid K range: {lower}..{upper} "
                f"for min_segment_slots={m}"
            )
        candidate_ks = list(range(lower, upper + 1))

    cost = build_segment_cost_matrix(
        x,
        min_segment_slots=m,
    )

    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[float, int, np.ndarray, list[int], float]] = []

    for k in candidate_ks:
        if k < 2 or k > max_feasible_k:
            continue

        labels, boundaries, within_sse = optimal_contiguous_partition(
            cost,
            k=int(k),
            min_segment_slots=m,
        )

        sizes = np.bincount(labels, minlength=int(k))
        sil = float(silhouette_score(x, labels, metric="euclidean"))

        row = {
            "k": int(k),
            "silhouette": sil,
            "within_sse": float(within_sse),
            "min_segment_slots": int(sizes.min()),
            "max_segment_slots": int(sizes.max()),
            "boundaries": [int(v) for v in boundaries],
            "eligible": bool(int(sizes.min()) >= m),
        }
        candidates.append(row)
        accepted.append(
            (sil, int(k), labels.copy(), list(boundaries), float(within_sse))
        )

    if not accepted:
        raise RuntimeError("no feasible contiguous K segmentation")

    # Explicit --k fixes K. Otherwise choose the best TRAIN-only silhouette.
    accepted.sort(key=lambda item: (-item[0], item[1]))
    best_sil, selected_k, labels, boundaries, within_sse = accepted[0]

    segments: list[dict[str, Any]] = []
    for context_id, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segments.append(
            {
                "context_id": int(context_id),
                "start_slot": int(a),
                "end_slot": int(b - 1),
                "start": slot_label(int(a)),
                "end": slot_label(int(b - 1)),
                "n_slots": int(b - a),
            }
        )

    return {
        "selected_k": int(selected_k),
        "selected_silhouette": float(best_sil),
        "selected_within_sse": float(within_sse),
        "slot_to_context": labels.astype(np.int16),
        "segments": segments,
        "boundaries": [int(v) for v in boundaries],
        "missing_slots": list(slot_info.get("missing_slots") or []),
        "candidate_scores": sorted(candidates, key=lambda r: r["k"]),
    }


def context_ids_from_slots(
    slots: np.ndarray,
    slot_to_context: np.ndarray,
) -> np.ndarray:
    return slot_to_context[np.asarray(slots, dtype=int)].astype(np.int16)


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


def fit_context_model(
    x_context_raw: np.ndarray,
    *,
    context_id: int,
    cfg: Config,
    device: torch.device,
) -> ContextArtifacts:
    n = len(x_context_raw)
    if n < 20:
        raise RuntimeError(
            f"context {context_id}: too few train points ({n})"
        )

    # Keep the exact HourAE training recipe so only context partition changes.
    stage1_scaler = fit_scaler(x_context_raw)
    x1 = stage1_scaler.transform(x_context_raw).astype(np.float32)

    stage1_model, stage1_hist = train_ae(
        x1,
        n_features=x1.shape[1],
        cfg=cfg,
        epochs=cfg.stage1_epochs,
        device=device,
        seed=cfg.seed + context_id,
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
            f"context {context_id}: stage1 retained only "
            f"{int(clean_mask.sum())}/{n}"
        )

    clean_raw = x_context_raw[clean_mask]
    final_scaler = fit_scaler(clean_raw)
    x2 = final_scaler.transform(clean_raw).astype(np.float32)

    final_model, final_hist = train_ae(
        x2,
        n_features=x2.shape[1],
        cfg=cfg,
        epochs=cfg.final_epochs,
        device=device,
        seed=cfg.seed + 1000 + context_id,
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

    return ContextArtifacts(
        context_id=context_id,
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
    art: ContextArtifacts,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> PointAutoEncoder:
    model = PointAutoEncoder(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    )
    model.load_state_dict(art.final_state)
    model.to(device)
    model.eval()
    return model


def score_with_context_models(
    values_raw: np.ndarray,
    context_ids: np.ndarray,
    artifacts: dict[int, ContextArtifacts],
    *,
    n_features: int,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mse = np.full(len(values_raw), np.nan, dtype=np.float64)
    z_scores = np.full(len(values_raw), np.nan, dtype=np.float64)

    for context_id, art in artifacts.items():
        idx = np.flatnonzero(context_ids == context_id)
        if idx.size == 0:
            continue

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


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < len(m) and m[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def mask_to_labels(
    times: pd.Series,
    mask: np.ndarray,
    scores: np.ndarray,
    prefix: str,
) -> list[dict[str, Any]]:
    t = pd.to_datetime(pd.Series(times), utc=True)
    items: list[dict[str, Any]] = []

    for i, (a, b) in enumerate(segments(mask), start=1):
        ss = scores[a : b + 1]
        finite = ss[np.isfinite(ss)]
        items.append(
            {
                "id": f"{prefix}_{i:06d}",
                "kind": "range",
                "start": t.iloc[a].isoformat(),
                "end": t.iloc[b].isoformat(),
                "score": float(np.max(finite)) if finite.size else None,
                "source": prefix,
            }
        )

    return items


def metric_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    event = bundle["event"]
    point = bundle["point"]
    op = bundle["operational"]
    vus = bundle["vus"]
    return {
        "primary_metric": "vus_pr",
        "vus_pr": bundle["primary"]["vus_pr"],
        "pr_auc": bundle["pr_auc"],
        "r_auc_pr": vus.get("r_auc_pr"),
        "vus_window": bundle.get("vus_window"),
        "event_f1": event.get("f1"),
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "n_gt_events": event.get("n_gt_events"),
        "n_pred_events": event.get("n_pred_events"),
        "n_hit_gt": event.get("n_hit_gt"),
        "precision": point.get("precision"),
        "recall": point.get("recall"),
        "f1": point.get("f1"),
        "tp": point.get("tp"),
        "fp": point.get("fp"),
        "fn": point.get("fn"),
        "false_alarms_per_day": op.get("false_alarms_per_day"),
        "n_false_alarm_events": op.get("n_false_alarm_events"),
        "observation_days": op.get("observation_days"),
        "mean_detection_delay_min": op.get("mean_detection_delay_min"),
        "median_detection_delay_min": op.get("median_detection_delay_min"),
        "n_detected_events": op.get("n_detected"),
        "n_missed_events": op.get("n_missed"),
        "score_distribution": bundle.get("score_distribution"),
        "event_score_diagnostics": bundle.get("event_score_diagnostics"),
    }


def context_summary(
    artifacts: dict[int, ContextArtifacts],
    slot_to_context: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for context_id in sorted(artifacts):
        art = artifacts[context_id]
        slots = np.flatnonzero(slot_to_context == context_id)
        rows.append(
            {
                "context_id": int(context_id),
                "n_slots": int(len(slots)),
                "slots": [slot_label(int(s)) for s in slots],
                "n_train": art.n_train,
                "n_removed_stage1": art.n_removed,
                "removed_fraction": art.n_removed / art.n_train,
                "n_final_train": art.n_train - art.n_removed,
                "stage1_threshold_mse": art.stage1_threshold_mse,
                "final_train_mse_mean": art.final_score_mean,
                "final_train_mse_std": art.final_score_std,
                "final_reference_contamination_pct": cfg.final_contamination_pct,
                "final_reference_quantile_mse": art.final_reference_quantile_mse,
                "final_reference_quantile_z": art.final_reference_quantile_z,
                "decision_threshold_sigma": cfg.threshold_sigma,
                "decision_threshold_mse": (
                    art.final_score_mean
                    + cfg.threshold_sigma * art.final_score_std
                ),
                "stage1_final_train_loss": art.stage1_final_loss,
                "final_final_train_loss": art.final_final_loss,
            }
        )

    return rows


def segmentation_payload(
    segment_info: dict[str, Any],
) -> dict[str, Any]:
    slot_to_context = segment_info["slot_to_context"]
    return {
        "method": "train_only_robust_profile_contiguous_dp",
        "selected_k": int(segment_info["selected_k"]),
        "selected_silhouette": float(
            segment_info["selected_silhouette"]
        ),
        "candidate_scores": segment_info["candidate_scores"],
        "selected_within_sse": float(segment_info["selected_within_sse"]),
        "segments": segment_info["segments"],
        "missing_train_slots": [
            slot_label(int(x)) for x in segment_info["missing_slots"]
        ],
        "slot_to_context": {
            slot_label(slot): int(slot_to_context[slot])
            for slot in range(288)
        },
    }


def save_checkpoint(
    plmn: str,
    artifacts: dict[int, ContextArtifacts],
    cols: list[str],
    cfg: Config,
    segment_info: dict[str, Any],
) -> str:
    path = MODEL_DIR / plmn / "paperib" / "context_ae.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "feature_columns": cols,
            "config": asdict(cfg),
            "segmentation": segmentation_payload(segment_info),
            "contexts": {
                str(context_id): {
                    "final_state": art.final_state,
                    "stage1_state": art.stage1_state,
                    "stage1_scaler": art.stage1_scaler,
                    "final_scaler": art.final_scaler,
                    "stage1_threshold_mse": art.stage1_threshold_mse,
                    "final_score_mean": art.final_score_mean,
                    "final_score_std": art.final_score_std,
                    "final_reference_quantile_mse": (
                        art.final_reference_quantile_mse
                    ),
                    "final_reference_quantile_z": (
                        art.final_reference_quantile_z
                    ),
                    "n_train": art.n_train,
                    "n_removed": art.n_removed,
                }
                for context_id, art in artifacts.items()
            },
        },
        path,
    )
    return str(path)


def write_prediction(
    *,
    plmn: str,
    split_name: str,
    times: pd.Series,
    local_times: pd.Series,
    slots: np.ndarray,
    context_ids: np.ndarray,
    scores: np.ndarray,
    raw_mse: np.ndarray,
    threshold: float,
    bundle: dict[str, Any],
    cols: list[str],
    cfg: Config,
    artifacts: dict[int, ContextArtifacts],
    segment_info: dict[str, Any],
    split_meta: dict[str, str],
) -> str:
    pred = np.isfinite(scores) & (scores > threshold)
    labels = mask_to_labels(
        times,
        pred,
        scores,
        f"context_ae_{split_name}",
    )

    payload = {
        "plmn": plmn,
        "source": f"context_ae_{split_name}",
        "model": "context_ae",
        "run_name": "k_context_iterative",
        "eval_split": split_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_polarity": "higher_is_anomaly",
        "threshold": float(threshold),
        "threshold_method": (
            "per_context_clean_train_mse_mean_plus_sigma"
        ),
        "timezone": cfg.timezone,
        "n_context_models": int(segment_info["selected_k"]),
        "feature_columns": cols,
        "feature_mode": "all_raw",
        "n_scores": int(np.isfinite(scores).sum()),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "split": split_meta,
        "labels": labels,
        "metrics": {
            **metric_payload(bundle),
            "test_held_out": split_name != "test",
            "train_labels_used": False,
            "validation_labels_used_for_training": False,
            "validation_labels_used_for_context_selection": False,
        },
        "segmentation": segmentation_payload(segment_info),
        "training": {
            "method": (
                "train_only_contiguous_context_segmentation_plus_"
                "iterative_decontamination_autoencoder"
            ),
            "config": asdict(cfg),
            "context_models": context_summary(
                artifacts,
                segment_info["slot_to_context"],
                cfg,
            ),
            "note": (
                "K and contiguous time boundaries are determined from TRAIN profiles "
                "only. Direction-aware scoring is not included in this run so "
                "the temporal-segmentation effect can be isolated."
            ),
        },
        "score_series": {
            "times": [pd.Timestamp(x).isoformat() for x in times],
            "local_times": [
                pd.Timestamp(x).isoformat() for x in local_times
            ],
            "slots": [slot_label(int(x)) for x in slots],
            "context_ids": [int(x) for x in context_ids],
            "scores": [
                float(x) if np.isfinite(x) else None
                for x in scores
            ],
            "raw_reconstruction_mse": [
                float(x) if np.isfinite(x) else None
                for x in raw_mse
            ],
            "threshold": float(threshold),
        },
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{plmn}_context_ae_{split_name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def print_context_map(
    segment_info: dict[str, Any],
    train_context_ids: np.ndarray,
) -> None:
    k = int(segment_info["selected_k"])
    print(
        f"selected K={k} "
        f"silhouette={segment_info['selected_silhouette']:.4f} "
        f"within_SSE={segment_info['selected_within_sse']:.4f}",
        flush=True,
    )

    print("K candidates (TRAIN-only contiguous segmentation):", flush=True)
    for row in segment_info["candidate_scores"]:
        mark = "*" if row["k"] == k else " "
        print(
            f" {mark} K={row['k']:02d} "
            f"sil={row['silhouette']:.4f} "
            f"SSE={row['within_sse']:.2f} "
            f"min_slots={row['min_segment_slots']}",
            flush=True,
        )

    print("discovered contiguous contexts:", flush=True)
    for seg in segment_info["segments"]:
        c = int(seg["context_id"])
        n_train = int(np.sum(train_context_ids == c))
        print(
            f"  context={c} {seg['start']} ~ {seg['end']} "
            f"slots={seg['n_slots']} train_points={n_train}",
            flush=True,
        )



def _metric_delta(
    context_metrics: dict[str, Any],
    hour_metrics: dict[str, Any],
    key: str,
) -> float | None:
    a = context_metrics.get(key)
    b = hour_metrics.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def print_hour24_comparison(
    *,
    selected_k: int,
    context_metrics: dict[str, Any],
    hour_metrics: dict[str, Any],
) -> None:
    """Compact side-by-side validation comparison."""
    print("", flush=True)
    print("=" * 86, flush=True)
    print(
        f"VALID COMPARISON: adaptive K={selected_k} contexts vs fixed 24 hourly AEs",
        flush=True,
    )
    print("=" * 86, flush=True)
    print(
        f"{'metric':<20} {'adaptive-K':>16} {'hour-24':>16} {'delta(K-24)':>16}",
        flush=True,
    )

    rows = [
        ("VUS-PR", "vus_pr"),
        ("PR-AUC", "pr_auc"),
        ("Event-F1", "event_f1"),
        ("Event-Recall", "event_recall"),
        ("Point-F1", "f1"),
        ("FA/day", "false_alarms_per_day"),
    ]

    for label, key in rows:
        a = context_metrics.get(key)
        b = hour_metrics.get(key)
        d = _metric_delta(context_metrics, hour_metrics, key)

        def fmt(v: Any) -> str:
            return "None" if v is None else f"{float(v):.6f}"

        print(
            f"{label:<20} {fmt(a):>16} {fmt(b):>16} {fmt(d):>16}",
            flush=True,
        )

    print("=" * 86, flush=True)


def write_hour24_comparison(
    *,
    plmn: str,
    selected_k: int,
    selected_silhouette: float,
    context_result: dict[str, Any],
    hour_result: dict[str, Any],
    eval_test: bool,
) -> str:
    """Save a small comparison file without duplicating full score series."""
    context_valid = context_result["valid"]
    hour_valid = hour_result["valid"]

    keys = [
        "vus_pr",
        "pr_auc",
        "r_auc_pr",
        "event_f1",
        "event_precision",
        "event_recall",
        "n_gt_events",
        "n_pred_events",
        "n_hit_gt",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "false_alarms_per_day",
        "n_false_alarm_events",
        "n_detected_events",
        "n_missed_events",
    ]

    valid_delta = {
        key: _metric_delta(context_valid, hour_valid, key)
        for key in keys
        if isinstance(context_valid.get(key), (int, float))
        and isinstance(hour_valid.get(key), (int, float))
    }

    payload: dict[str, Any] = {
        "plmn": plmn,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "silhouette_selected_contiguous_context_vs_fixed_24_hour",
        "selected_k": int(selected_k),
        "selected_silhouette": float(selected_silhouette),
        "same_hyperparameters": True,
        "train_labels_used": False,
        "valid": {
            "adaptive_context": context_valid,
            "fixed_hour24": hour_valid,
            "delta_adaptive_minus_hour24": valid_delta,
        },
        "paths": {
            "adaptive_context": context_result.get("valid_path"),
            "fixed_hour24": hour_result.get("valid_path"),
        },
    }

    if eval_test and "test" in context_result and "test" in hour_result:
        context_test = context_result["test"]
        hour_test = hour_result["test"]
        payload["test"] = {
            "adaptive_context": context_test,
            "fixed_hour24": hour_test,
            "delta_adaptive_minus_hour24": {
                key: _metric_delta(context_test, hour_test, key)
                for key in keys
                if isinstance(context_test.get(key), (int, float))
                and isinstance(hour_test.get(key), (int, float))
            },
        }
        payload["paths"]["adaptive_context_test"] = context_result.get("test_path")
        payload["paths"]["fixed_hour24_test"] = hour_result.get("test_path")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{plmn}_context_vs_hour24.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def run_hour24_baseline(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool,
) -> dict[str, Any]:
    """Train the existing HourAE baseline with exactly the shared AE settings."""
    hourae = _load_hourae()
    hour_cfg = hourae.Config(
        timezone=cfg.timezone,
        stage1_remove_pct=cfg.stage1_remove_pct,
        final_contamination_pct=cfg.final_contamination_pct,
        threshold_sigma=cfg.threshold_sigma,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        stage1_epochs=cfg.stage1_epochs,
        final_epochs=cfg.final_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        sample_minutes=cfg.sample_minutes,
    )
    return hourae.run(
        plmn=plmn,
        cfg=hour_cfg,
        eval_test=eval_test,
    )


def run(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool,
    compare_hour24: bool = True,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = device_for_run()
    prep = prepare_unlabeled_plmn(plmn, cfg.timezone)

    print(
        f"device={device} plmn={plmn} features={len(prep['cols'])} "
        f"timezone={cfg.timezone}",
        flush=True,
    )
    print(
        "building robust 5-minute slot profiles and contiguous contexts from TRAIN only",
        flush=True,
    )

    slot_info = build_train_slot_profiles(
        prep["train"],
        prep["train_slots"],
        use_mad=cfg.profile_use_mad,
    )
    segment_info = fit_context_segmentation(slot_info, cfg)

    train_context_ids = context_ids_from_slots(
        prep["train_slots"],
        segment_info["slot_to_context"],
    )
    valid_context_ids = context_ids_from_slots(
        prep["valid_slots"],
        segment_info["slot_to_context"],
    )
    test_context_ids = context_ids_from_slots(
        prep["test_slots"],
        segment_info["slot_to_context"],
    )

    print_context_map(segment_info, train_context_ids)

    print(
        f"stage1 remove={cfg.stage1_remove_pct}% "
        f"final reference contamination={cfg.final_contamination_pct}% "
        f"threshold=mean+{cfg.threshold_sigma}sigma",
        flush=True,
    )
    print(
        "training K context AutoEncoders with NO anomaly labels loaded",
        flush=True,
    )

    t0 = time.time()
    artifacts: dict[int, ContextArtifacts] = {}

    for context_id in range(int(segment_info["selected_k"])):
        idx = np.flatnonzero(train_context_ids == context_id)
        art = fit_context_model(
            prep["train"][idx],
            context_id=context_id,
            cfg=cfg,
            device=device,
        )
        artifacts[context_id] = art

        print(
            f"  context={context_id} train={art.n_train} "
            f"removed={art.n_removed} final={art.n_train - art.n_removed} "
            f"stage1_loss={art.stage1_final_loss:.6f} "
            f"final_loss={art.final_final_loss:.6f}",
            flush=True,
        )

    train_scores, _train_raw = score_with_context_models(
        prep["train"],
        train_context_ids,
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    valid_scores, valid_raw = score_with_context_models(
        prep["valid"],
        valid_context_ids,
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    test_scores, test_raw = score_with_context_models(
        prep["test"],
        test_context_ids,
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )

    threshold = float(cfg.threshold_sigma)
    model_path = save_checkpoint(
        plmn,
        artifacts,
        prep["cols"],
        cfg,
        segment_info,
    )

    print(
        "training/scoring complete; loading expert labels for evaluation only",
        flush=True,
    )
    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(
        prep["df"],
        label_doc.get("labels"),
    ).astype(bool)
    y_valid = y_full[prep["train_end"] : prep["valid_end"]]
    y_test = y_full[prep["valid_end"] :]

    valid_vus_window = resolve_vus_window(y_valid, None)
    valid_bundle = evaluate_valid(
        y_valid,
        valid_scores,
        threshold,
        vus_window=valid_vus_window,
        sample_minutes=cfg.sample_minutes,
    )

    valid_times = prep["times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)
    valid_local_times = prep["local_times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)

    valid_path = write_prediction(
        plmn=plmn,
        split_name="valid",
        times=valid_times,
        local_times=valid_local_times,
        slots=prep["valid_slots"],
        context_ids=valid_context_ids,
        scores=valid_scores,
        raw_mse=valid_raw,
        threshold=threshold,
        bundle=valid_bundle,
        cols=prep["cols"],
        cfg=cfg,
        artifacts=artifacts,
        segment_info=segment_info,
        split_meta=prep["split"],
    )

    result: dict[str, Any] = {
        "plmn": plmn,
        "selected_k": int(segment_info["selected_k"]),
        "selected_silhouette": float(
            segment_info["selected_silhouette"]
        ),
        "model_path": model_path,
        "valid_path": valid_path,
        "valid": metric_payload(valid_bundle),
        "elapsed_sec": time.time() - t0,
        "train_labels_used": False,
    }

    print(
        f"VALID VUS-PR={valid_bundle['primary']['vus_pr']} "
        f"PR-AUC={valid_bundle['pr_auc']} "
        f"Event-F1={valid_bundle['event']['f1']} "
        f"Event-Recall={valid_bundle['event']['recall']} "
        f"FA/day={valid_bundle['operational']['false_alarms_per_day']} "
        f"VUS-window={valid_vus_window}",
        flush=True,
    )

    print("GT event max-score percentiles:", flush=True)
    for row in valid_bundle.get("event_score_diagnostics") or []:
        pct = row.get("max_score_percentile")
        pct_text = "None" if pct is None else f"{100.0 * pct:.1f}%"
        print(
            f"  event {row['event_id']}: {pct_text} "
            f"detected={row['detected']} "
            f"n_pred_points={row['n_pred_points']}",
            flush=True,
        )

    finite_train = train_scores[np.isfinite(train_scores)]
    if finite_train.size:
        print(
            f"train calibrated score: mean={np.mean(finite_train):.3f} "
            f"std={np.std(finite_train):.3f} "
            f"q99={np.quantile(finite_train, 0.99):.3f} "
            f"max={np.max(finite_train):.3f}",
            flush=True,
        )

    if eval_test:
        test_bundle = evaluate_valid(
            y_test,
            test_scores,
            threshold,
            vus_window=valid_vus_window,
            sample_minutes=cfg.sample_minutes,
        )
        test_times = prep["times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)
        test_local_times = prep["local_times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)

        test_path = write_prediction(
            plmn=plmn,
            split_name="test",
            times=test_times,
            local_times=test_local_times,
            slots=prep["test_slots"],
            context_ids=test_context_ids,
            scores=test_scores,
            raw_mse=test_raw,
            threshold=threshold,
            bundle=test_bundle,
            cols=prep["cols"],
            cfg=cfg,
            artifacts=artifacts,
            segment_info=segment_info,
            split_meta=prep["split"],
        )
        result["test"] = metric_payload(test_bundle)
        result["test_path"] = test_path

        print(
            f"TEST VUS-PR={test_bundle['primary']['vus_pr']} "
            f"PR-AUC={test_bundle['pr_auc']} "
            f"Event-F1={test_bundle['event']['f1']} "
            f"Event-Recall={test_bundle['event']['recall']} "
            f"FA/day={test_bundle['operational']['false_alarms_per_day']} "
            f"VUS-window={valid_vus_window} (fixed from validation)",
            flush=True,
        )

    if compare_hour24:
        print("", flush=True)
        print(
            "training fixed 24-hour HourAE baseline with identical AE hyperparameters",
            flush=True,
        )
        hour24_result = run_hour24_baseline(
            plmn=plmn,
            cfg=cfg,
            eval_test=eval_test,
        )
        result["hour24_baseline"] = hour24_result

        print_hour24_comparison(
            selected_k=int(segment_info["selected_k"]),
            context_metrics=result["valid"],
            hour_metrics=hour24_result["valid"],
        )

        comparison_path = write_hour24_comparison(
            plmn=plmn,
            selected_k=int(segment_info["selected_k"]),
            selected_silhouette=float(segment_info["selected_silhouette"]),
            context_result=result,
            hour_result=hour24_result,
            eval_test=eval_test,
        )
        result["comparison_path"] = comparison_path
        print(f"comparison saved: {comparison_path}", flush=True)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--timezone", default="Asia/Seoul")

    p.add_argument(
        "--k",
        type=int,
        default=0,
        help="fixed number of contiguous segments; 0 means train-only automatic K",
    )
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=24)
    p.add_argument("--min_segment_slots", type=int, default=3)
    p.add_argument(
        "--median_only_profile",
        action="store_true",
        help="segment slots by median only instead of median+MAD",
    )

    p.add_argument("--stage1_remove_pct", type=float, default=5.0)
    p.add_argument("--final_contamination_pct", type=float, default=0.01)
    p.add_argument("--threshold_sigma", type=float, default=3.0)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--stage1_epochs", type=int, default=50)
    p.add_argument("--final_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2)

    p.add_argument(
        "--skip_hour24_compare",
        action="store_true",
        help="skip retraining/comparison against the fixed 24-hour HourAE baseline",
    )
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="also evaluate held-out test after validation choices are frozen",
    )
    args = p.parse_args()

    if args.k < 0:
        p.error("--k must be 0 (auto) or >= 2")
    if args.k == 1:
        p.error("--k=1 is the global-AE baseline, not context segmentation")
    if args.k_min < 2:
        p.error("--k_min must be >= 2")
    if args.k_max < args.k_min:
        p.error("--k_max must be >= k_min")
    if args.min_segment_slots < 1:
        p.error("--min_segment_slots must be >= 1")

    if not (0.0 < args.stage1_remove_pct < 50.0):
        p.error("--stage1_remove_pct must be in (0,50)")
    if not (0.0 < args.final_contamination_pct < 50.0):
        p.error("--final_contamination_pct must be in (0,50)")
    if args.threshold_sigma <= 0:
        p.error("--threshold_sigma must be > 0")
    if args.hidden_dim < 2:
        p.error("--hidden_dim must be >= 2")
    if args.latent_dim < 1 or args.latent_dim >= args.hidden_dim:
        p.error("--latent_dim must be >=1 and < hidden_dim")
    if args.stage1_epochs < 1 or args.final_epochs < 1:
        p.error("epochs must be >= 1")
    if args.batch_size < 1:
        p.error("--batch_size must be >= 1")
    if args.lr <= 0:
        p.error("--lr must be > 0")

    cfg = Config(
        timezone=args.timezone,
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        min_segment_slots=args.min_segment_slots,
        profile_use_mad=not args.median_only_profile,
        stage1_remove_pct=args.stage1_remove_pct,
        final_contamination_pct=args.final_contamination_pct,
        threshold_sigma=args.threshold_sigma,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        stage1_epochs=args.stage1_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    run(
        plmn=args.plmn,
        cfg=cfg,
        eval_test=args.eval_test,
        compare_hour24=not args.skip_hour24_compare,
    )


if __name__ == "__main__":
    main()
