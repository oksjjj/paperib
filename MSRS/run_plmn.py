#!/usr/bin/env python3
"""MSRS: Multi-Scale Robust Sparse-aware TSAD for one PLMN.

Research prototype for paperib.

Design goals
------------
1. No anomaly labels are loaded or used during training.
2. Training may contain contamination.
3. Short and long temporal contexts are modeled separately.
4. Sparse-variable anomalies are preserved by top-k feature residual scoring.
5. Validation/test labels are loaded only after training and scoring finish.

Default pipeline
----------------
train raw counters
  -> train-only MinMax scaling
  -> GRU next-step predictor at windows 8 and 32
  -> Huber warm-up
  -> MAD-based sample down-weighting for contaminated windows
  -> feature-wise robust residual normalization
  -> top-k feature score at each scale
  -> max across scales
  -> threshold from train score percentile

Validation is always evaluated.
Test is evaluated only with --eval_test.
The test VUS tolerance is fixed to the validation-derived value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
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
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LABELING = REPO / "labeling"
MODEL_ROOT = ROOT / "model"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"

if str(LABELING) not in sys.path:
    sys.path.insert(0, str(LABELING))

from tool import (  # noqa: E402
    human_anomaly_mask,
    is_rate_metric,
    load_labels,
    load_plmn,
    metric_columns,
)


def _load_paper_metrics():
    path = REPO / "RiAD" / "metrics.py"
    spec = importlib.util.spec_from_file_location("paperib_metrics_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paper_metrics = _load_paper_metrics()
evaluate_valid = paper_metrics.evaluate_riad_valid
resolve_vus_window = paper_metrics.resolve_vus_window


@dataclass
class Config:
    scales: tuple[int, ...] = (8, 32)
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    warmup_epochs: int = 5
    robust_epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    huber_beta: float = 0.05
    robust_tau: float = 3.0
    topk_ratio: float = 0.10
    anomaly_ratio: float = 1.0
    seed: int = 2
    sample_minutes: float = 5.0


class NextStepDataset(Dataset):
    """Sequence -> next-point dataset. No labels are stored here."""

    def __init__(self, block: np.ndarray, window: int):
        x = np.asarray(block, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected (T,F), got {x.shape}")
        if len(x) <= window:
            raise ValueError(f"need T > window, got T={len(x)}, window={window}")
        self.block = torch.from_numpy(x)
        self.window = int(window)

    def __len__(self) -> int:
        return len(self.block) - self.window

    def __getitem__(self, idx: int):
        j = idx + self.window
        return self.block[idx:j], self.block[j]


class GRUPredictor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, train_ratio: float = 0.6, valid_ratio: float = 0.2):
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    return train_end, min(valid_end, n)


def prepare_unlabeled_plmn(plmn: str) -> dict[str, Any]:
    """Load and scale data without touching any anomaly label file."""
    df = load_plmn(plmn)
    cols = [c for c in metric_columns(df) if not is_rate_metric(c)]
    if not cols:
        raise RuntimeError(f"{plmn}: no raw feature columns")

    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    times = pd.to_datetime(df["time"], utc=True)

    train_end, valid_end = split_indices(len(df))
    scaler = MinMaxScaler()
    train = scaler.fit_transform(values[:train_end]).astype(np.float32)
    valid = scaler.transform(values[train_end:valid_end]).astype(np.float32)
    test = scaler.transform(values[valid_end:]).astype(np.float32)

    return {
        "df": df,
        "cols": cols,
        "times": times,
        "scaler": scaler,
        "train": train,
        "valid": valid,
        "test": test,
        "train_end": train_end,
        "valid_end": valid_end,
        "split": {
            "train_time_end": str(times.iloc[train_end - 1]),
            "valid_time_start": str(times.iloc[train_end]),
            "valid_time_end": str(times.iloc[valid_end - 1]),
            "test_time_start": str(times.iloc[valid_end]),
            "test_time_end": str(times.iloc[-1]),
        },
    }


def sample_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=float(beta),
    )
    return loss.mean(dim=1)


@torch.no_grad()
def estimate_loss_reference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> tuple[float, float]:
    model.eval()
    vals: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.float().to(device)
        yb = yb.float().to(device)
        loss = sample_huber_loss(model(xb), yb, beta)
        vals.append(loss.detach().cpu().numpy())

    arr = np.concatenate(vals).astype(np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_scale = max(1.4826 * mad, 1e-8)
    return med, robust_scale


def train_one_scale(
    train_block: np.ndarray,
    *,
    window: int,
    cfg: Config,
    device: torch.device,
) -> tuple[GRUPredictor, dict[str, Any]]:
    ds = NextStepDataset(train_block, window)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = GRUPredictor(
        n_features=train_block.shape[1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, cfg.warmup_epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            opt.zero_grad()
            sample_loss = sample_huber_loss(model(xb), yb, cfg.huber_beta)
            loss = sample_loss.mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "stage": "warmup",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
            }
        )

    loss_med, loss_scale = estimate_loss_reference(
        model, loader, device, cfg.huber_beta
    )

    for epoch in range(1, cfg.robust_epochs + 1):
        model.train()
        losses = []
        weights_seen: list[float] = []
        for xb, yb in loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            opt.zero_grad()

            sample_loss = sample_huber_loss(model(xb), yb, cfg.huber_beta)

            # Label-free contamination handling:
            # samples far above the warm-up loss median are softly down-weighted.
            z = torch.relu(
                (sample_loss.detach() - loss_med) / max(loss_scale, 1e-8)
            )
            weights = 1.0 / (1.0 + (z / max(cfg.robust_tau, 1e-6)) ** 2)
            loss = torch.sum(weights * sample_loss) / torch.clamp(
                weights.sum(), min=1e-8
            )

            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            weights_seen.append(float(weights.mean().detach().cpu()))

        history.append(
            {
                "stage": "robust",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "mean_weight": float(np.mean(weights_seen)),
            }
        )

    meta = {
        "window": int(window),
        "n_train_samples": len(ds),
        "warmup_loss_median": loss_med,
        "warmup_loss_robust_scale": loss_scale,
        "history": history,
    }
    return model, meta


@torch.no_grad()
def residual_matrix(
    model: nn.Module,
    block: np.ndarray,
    *,
    window: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return absolute next-step residuals aligned to block time indices."""
    t, f = block.shape
    out = np.full((t, f), np.nan, dtype=np.float64)
    if t <= window:
        return out

    ds = NextStepDataset(block, window)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    cursor = window
    model.eval()
    for xb, yb in loader:
        xb = xb.float().to(device)
        yb = yb.float().to(device)
        pred = model(xb)
        resid = torch.abs(pred - yb).detach().cpu().numpy().astype(np.float64)
        n = len(resid)
        out[cursor : cursor + n] = resid
        cursor += n
    return out


def feature_residual_reference(
    train_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(train_residual, axis=0)
    mad = np.nanmedian(np.abs(train_residual - med[None, :]), axis=0)
    scale = 1.4826 * mad
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1e-8)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.float64), scale.astype(np.float64)


def sparse_score(
    residual: np.ndarray,
    *,
    feature_median: np.ndarray,
    feature_scale: np.ndarray,
    topk_ratio: float,
) -> tuple[np.ndarray, int]:
    """Robust feature residuals -> log1p -> mean of largest k features."""
    z = (residual - feature_median[None, :]) / feature_scale[None, :]
    z = np.maximum(z, 0.0)
    z = np.log1p(z)

    n_features = residual.shape[1]
    k = max(1, min(n_features, int(math.ceil(n_features * topk_ratio))))

    scores = np.full(len(residual), np.nan, dtype=np.float64)
    valid = np.isfinite(z).all(axis=1)
    if valid.any():
        zv = z[valid]
        part = np.partition(zv, n_features - k, axis=1)[:, -k:]
        scores[valid] = np.mean(part, axis=1)
    return scores, k


def calibrate_scale_score(
    train_score: np.ndarray,
    other_scores: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    """Put each temporal scale on a comparable robust train-score scale."""
    finite = np.asarray(train_score, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite train scale scores")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = max(1.4826 * mad, 1e-8)

    def transform(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64).copy()
        ok = np.isfinite(x)
        x[ok] = np.log1p(np.maximum((x[ok] - med) / scale, 0.0))
        return x

    return (
        transform(train_score),
        [transform(x) for x in other_scores],
        {"median": med, "robust_scale": scale},
    )


def combine_scale_scores(
    scores_by_scale: dict[int, np.ndarray],
    *,
    max_window: int,
) -> np.ndarray:
    """Max aggregation preserves an anomaly visible at either temporal scale."""
    ordered = sorted(scores_by_scale)
    stack = np.vstack([scores_by_scale[s] for s in ordered])
    n = stack.shape[1]
    out = np.full(n, np.nan, dtype=np.float64)

    valid = np.arange(n) >= int(max_window)
    valid &= np.isfinite(stack).all(axis=0)
    if valid.any():
        out[valid] = np.max(stack[:, valid], axis=0)
    return out


def train_percentile_threshold(scores: np.ndarray, anomaly_ratio: float) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("no finite train scores")
    q = np.clip(100.0 - float(anomaly_ratio), 0.0, 100.0)
    return float(np.percentile(finite, q))


def _mask_to_labels(
    times: pd.Series,
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    t = pd.to_datetime(pd.Series(times), utc=True)
    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n = min(len(t), len(mask), len(scores))

    items: list[dict[str, Any]] = []
    i = 0
    idx = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        finite = scores[i:j][np.isfinite(scores[i:j])]
        idx += 1
        items.append(
            {
                "id": f"{prefix}_{idx:06d}",
                "kind": "range",
                "start": t.iloc[i].isoformat(),
                "end": t.iloc[j - 1].isoformat(),
                "score": float(np.max(finite)) if finite.size else None,
                "source": prefix,
            }
        )
        i = j
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


def write_prediction(
    *,
    plmn: str,
    split_name: str,
    times: pd.Series,
    scores: np.ndarray,
    scale_scores: dict[int, np.ndarray],
    threshold: float,
    y_true: np.ndarray,
    bundle: dict[str, Any],
    cols: list[str],
    cfg: Config,
    k_features: int,
    split_meta: dict[str, str],
    train_meta: dict[str, Any],
) -> str:
    pred = np.isfinite(scores) & (scores > threshold)
    labels = _mask_to_labels(
        times,
        pred,
        scores,
        prefix=f"msrs_{split_name}",
    )

    payload = {
        "plmn": plmn,
        "source": f"msrs_{split_name}",
        "model": "msrs",
        "run_name": "paperib",
        "eval_split": split_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_polarity": "higher_is_anomaly",
        "threshold": float(threshold),
        "threshold_method": "train_percentile",
        "feature_columns": cols,
        "feature_mode": "core",
        "scales": list(cfg.scales),
        "topk_ratio": cfg.topk_ratio,
        "topk_features": int(k_features),
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
        },
        "training": {
            "method": "unlabeled_contaminated",
            "config": asdict(cfg),
            "scale_meta": train_meta,
            "note": (
                "No anomaly labels are loaded until model training, residual "
                "calibration, train thresholding, and split scoring are complete."
            ),
        },
        "score_series": {
            "times": [pd.Timestamp(t).isoformat() for t in times],
            "scores": [float(x) if np.isfinite(x) else None for x in scores],
            "threshold": float(threshold),
        },
        "scale_score_series": {
            str(scale): [
                float(x) if np.isfinite(x) else None
                for x in scale_scores[scale]
            ]
            for scale in sorted(scale_scores)
        },
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{plmn}_msrs_{split_name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def run(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool = False,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = device_for_run()

    # IMPORTANT: this function does not read anomaly labels.
    prep = prepare_unlabeled_plmn(plmn)
    train = prep["train"]
    valid = prep["valid"]
    test = prep["test"]
    max_window = max(cfg.scales)

    print(
        f"device={device} plmn={plmn} features={len(prep['cols'])} "
        f"scales={cfg.scales} topk_ratio={cfg.topk_ratio}",
        flush=True,
    )
    print(
        "training starts with NO anomaly labels loaded",
        flush=True,
    )

    models: dict[int, GRUPredictor] = {}
    refs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    train_meta: dict[str, Any] = {}

    score_train_by_scale: dict[int, np.ndarray] = {}
    score_valid_by_scale: dict[int, np.ndarray] = {}
    score_test_by_scale: dict[int, np.ndarray] = {}

    k_features = 1
    t0 = time.time()

    for window in cfg.scales:
        print(f"[scale={window}] training", flush=True)
        model, meta = train_one_scale(
            train,
            window=window,
            cfg=cfg,
            device=device,
        )
        models[window] = model
        train_meta[str(window)] = meta

        train_resid = residual_matrix(
            model,
            train,
            window=window,
            batch_size=cfg.batch_size,
            device=device,
        )
        valid_resid = residual_matrix(
            model,
            valid,
            window=window,
            batch_size=cfg.batch_size,
            device=device,
        )
        test_resid = residual_matrix(
            model,
            test,
            window=window,
            batch_size=cfg.batch_size,
            device=device,
        )

        feat_med, feat_scale = feature_residual_reference(train_resid)
        refs[window] = (feat_med, feat_scale)

        train_score, k_features = sparse_score(
            train_resid,
            feature_median=feat_med,
            feature_scale=feat_scale,
            topk_ratio=cfg.topk_ratio,
        )
        valid_score, _ = sparse_score(
            valid_resid,
            feature_median=feat_med,
            feature_scale=feat_scale,
            topk_ratio=cfg.topk_ratio,
        )
        test_score, _ = sparse_score(
            test_resid,
            feature_median=feat_med,
            feature_scale=feat_scale,
            topk_ratio=cfg.topk_ratio,
        )

        train_score, calibrated, scale_score_ref = calibrate_scale_score(
            train_score,
            [valid_score, test_score],
        )
        valid_score, test_score = calibrated
        train_meta[str(window)]["score_reference"] = scale_score_ref

        score_train_by_scale[window] = train_score
        score_valid_by_scale[window] = valid_score
        score_test_by_scale[window] = test_score

    train_scores = combine_scale_scores(
        score_train_by_scale,
        max_window=max_window,
    )
    valid_scores = combine_scale_scores(
        score_valid_by_scale,
        max_window=max_window,
    )
    test_scores = combine_scale_scores(
        score_test_by_scale,
        max_window=max_window,
    )

    threshold = train_percentile_threshold(
        train_scores,
        cfg.anomaly_ratio,
    )

    # Strict separation: labels are loaded only after ALL model training,
    # train residual calibration, train threshold selection, and split scoring.
    print(
        "training/scoring complete; loading expert labels for evaluation only",
        flush=True,
    )
    doc = load_labels(plmn)
    y_full = human_anomaly_mask(prep["df"], doc.get("labels")).astype(bool)
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

    times_valid = prep["times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)
    valid_path = write_prediction(
        plmn=plmn,
        split_name="valid",
        times=times_valid,
        scores=valid_scores,
        scale_scores=score_valid_by_scale,
        threshold=threshold,
        y_true=y_valid,
        bundle=valid_bundle,
        cols=prep["cols"],
        cfg=cfg,
        k_features=k_features,
        split_meta=prep["split"],
        train_meta=train_meta,
    )

    checkpoint_dir = MODEL_ROOT / plmn / "paperib"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "models": {
            str(window): models[window].state_dict()
            for window in cfg.scales
        },
        "residual_reference": {
            str(window): {
                "median": refs[window][0],
                "scale": refs[window][1],
            }
            for window in cfg.scales
        },
        "feature_columns": prep["cols"],
        "threshold": threshold,
        "validation_vus_window": valid_vus_window,
        "config": asdict(cfg),
    }
    torch.save(checkpoint, checkpoint_dir / "msrs.pt")

    result: dict[str, Any] = {
        "plmn": plmn,
        "threshold": threshold,
        "valid": metric_payload(valid_bundle),
        "valid_path": valid_path,
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
            f"detected={row['detected']}",
            flush=True,
        )

    if eval_test:
        # Test uses the validation-derived VUS tolerance. It is NOT selected
        # from test labels.
        test_bundle = evaluate_valid(
            y_test,
            test_scores,
            threshold,
            vus_window=valid_vus_window,
            sample_minutes=cfg.sample_minutes,
        )
        times_test = prep["times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)
        test_path = write_prediction(
            plmn=plmn,
            split_name="test",
            times=times_test,
            scores=test_scores,
            scale_scores=score_test_by_scale,
            threshold=threshold,
            y_true=y_test,
            bundle=test_bundle,
            cols=prep["cols"],
            cfg=cfg,
            k_features=k_features,
            split_meta=prep["split"],
            train_meta=train_meta,
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

    return result


def parse_scales(text: str) -> tuple[int, ...]:
    vals = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if not vals or min(vals) < 2:
        raise argparse.ArgumentTypeError("scales must be comma-separated integers >= 2")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--scales", type=parse_scales, default=(8, 32))
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--robust_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--huber_beta", type=float, default=0.05)
    p.add_argument("--robust_tau", type=float, default=3.0)
    p.add_argument("--topk_ratio", type=float, default=0.10)
    p.add_argument("--anomaly_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="also evaluate held-out test; use only after validation choices are frozen",
    )
    args = p.parse_args()

    if not (0.0 < args.topk_ratio <= 1.0):
        p.error("--topk_ratio must be in (0,1]")
    if not (0.0 < args.anomaly_ratio < 100.0):
        p.error("--anomaly_ratio must be in (0,100)")

    cfg = Config(
        scales=args.scales,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        warmup_epochs=args.warmup_epochs,
        robust_epochs=args.robust_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        huber_beta=args.huber_beta,
        robust_tau=args.robust_tau,
        topk_ratio=args.topk_ratio,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )
    run(
        plmn=args.plmn,
        cfg=cfg,
        eval_test=args.eval_test,
    )


if __name__ == "__main__":
    main()
