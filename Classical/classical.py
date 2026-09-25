"""Classical IB baselines: per-feature IQR fences + EWMA residual spikes.

Binary fuse = OR across features and detectors. UI score = −(# fires) so
lower = more anomalous (same polarity as OmniAnomaly).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _iqr_bounds(
    train: np.ndarray,
    *,
    k: float = 1.5,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) per column from train IQR fences."""
    x = np.asarray(train, dtype=np.float64)
    if mask is not None:
        x = x[np.asarray(mask, dtype=bool)]
    if len(x) == 0:
        x = np.asarray(train, dtype=np.float64)
    q1 = np.nanpercentile(x, 25, axis=0)
    q3 = np.nanpercentile(x, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-8)
    return q1 - k * iqr, q3 + k * iqr


def _iqr_fire(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return (v < low) | (v > high)


def _ewma_residual_sigma(
    train: np.ndarray,
    *,
    alpha: float = 0.3,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit EWMA level on train; return (last_level, residual_std per dim)."""
    x = np.asarray(train, dtype=np.float64)
    n, d = x.shape
    level = x[0].copy()
    resid = np.zeros_like(x)
    for i in range(n):
        resid[i] = x[i] - level
        level = alpha * x[i] + (1.0 - alpha) * level
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        use = resid[m] if m.any() else resid
    else:
        use = resid
    sigma = np.nanstd(use, axis=0)
    sigma = np.maximum(sigma, 1e-8)
    return level, sigma


def _ewma_fire(
    values: np.ndarray,
    *,
    init_level: np.ndarray,
    sigma: np.ndarray,
    alpha: float = 0.3,
    z: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fire mask n×d, last level)."""
    x = np.asarray(values, dtype=np.float64)
    n, d = x.shape
    level = np.asarray(init_level, dtype=np.float64).reshape(d).copy()
    fire = np.zeros((n, d), dtype=bool)
    thr = z * np.asarray(sigma, dtype=np.float64).reshape(d)
    for i in range(n):
        resid = np.abs(x[i] - level)
        fire[i] = resid > thr
        level = alpha * x[i] + (1.0 - alpha) * level
    return fire, level


def score_classical(
    train: np.ndarray,
    eval_x: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    iqr_k: float = 1.5,
    ewma_alpha: float = 0.3,
    ewma_z: float = 3.0,
    use_iqr: bool = True,
    use_ewma: bool = True,
) -> dict[str, Any]:
    """Fit on train, score eval. ``score = -n_fires`` (lower = worse)."""
    train = np.asarray(train, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)
    n, d = eval_x.shape
    fires = np.zeros((n, d), dtype=np.int8)

    detail: dict[str, Any] = {
        "use_iqr": use_iqr,
        "use_ewma": use_ewma,
        "iqr_k": iqr_k,
        "ewma_alpha": ewma_alpha,
        "ewma_z": ewma_z,
    }

    if use_iqr:
        low, high = _iqr_bounds(train, k=iqr_k, mask=train_mask)
        iqr_f = _iqr_fire(eval_x, low, high)
        fires = np.maximum(fires, iqr_f.astype(np.int8))
        detail["iqr_low"] = low.tolist()
        detail["iqr_high"] = high.tolist()
        detail["iqr_pred_points"] = int(iqr_f.any(axis=1).sum())

    if use_ewma:
        level0, sigma = _ewma_residual_sigma(
            train, alpha=ewma_alpha, mask=train_mask
        )
        ewma_f, _ = _ewma_fire(
            eval_x,
            init_level=level0,
            sigma=sigma,
            alpha=ewma_alpha,
            z=ewma_z,
        )
        fires = np.maximum(fires, ewma_f.astype(np.int8))
        detail["ewma_sigma"] = sigma.tolist()
        detail["ewma_pred_points"] = int(ewma_f.any(axis=1).sum())

    n_fires = fires.sum(axis=1).astype(np.float64)
    score = -n_fires  # 0 = quiet, −d = all features fire
    pred = n_fires >= 1.0
    return {
        "score": score,
        "pred": pred.astype(bool),
        "n_fires": n_fires,
        "fires": fires,
        "detail": detail,
    }


def train_scores_for_threshold(
    train: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Self-score train (for percentile thr). Uses expanding fit lightly via full fit."""
    # Score train with stats from the same train (optimistic); thr still label-free.
    out = score_classical(train, train, train_mask=train_mask, **kwargs)
    return out["score"]
