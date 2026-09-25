"""Shared prep, metrics, and UI export for baseline models."""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OMNI = _REPO / "OmniAnomaly"
_LABELING = _REPO / "labeling"
for p in (_OMNI, _LABELING):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_plmn import (  # noqa: E402
    KNOWN_RUN_NAMES,
    experiment_name_for_run,
    features_for_run,
    prepare_arrays,
)
from tool import EXCLUDED_TRAIN_METRICS  # noqa: E402

PRED_DIR = _REPO / "data" / "ib_data" / "predictions"
MODEL_ROOT = _REPO / "LSTM_AD" / "model"


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_pred = np.asarray(y_pred).astype(bool).reshape(-1)
    m = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:m], y_pred[:m]
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _segment_bounds(y: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(y).astype(bool).reshape(-1)
    padded = np.concatenate([[False], y, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    segs = _segment_bounds(y_true)
    y_pred = np.asarray(y_pred).astype(bool).reshape(-1)
    if not segs:
        return {
            "n_gt_events": 0,
            "n_hit_events": 0,
            "event_recall": 0.0,
            "n_pred_events": len(_segment_bounds(y_pred)),
        }
    hit = sum(1 for a, b in segs if y_pred[a:b].any())
    return {
        "n_gt_events": len(segs),
        "n_hit_events": hit,
        "event_recall": float(hit / len(segs)),
        "n_pred_events": len(_segment_bounds(y_pred)),
    }


def load_prepared(
    plmn: str,
    *,
    run_name: str = "paperib",
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> dict[str, Any]:
    if run_name not in KNOWN_RUN_NAMES:
        raise SystemExit(
            f"unknown --run_name {run_name!r}; use {sorted(KNOWN_RUN_NAMES)}"
        )
    # features_for_run needs a df; prepare_arrays resolves columns internally.
    from tool import load_plmn

    df = load_plmn(plmn)
    selected = features_for_run(run_name, df)
    prepared = prepare_arrays(
        plmn,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        feature_columns=selected,
        run_name=run_name,
    )
    cols = list(prepared.get("cols") or [])
    bad = [c for c in cols if c in EXCLUDED_TRAIN_METRICS]
    if bad:
        raise SystemExit(f"excluded metrics leaked into features: {bad}")
    prepared["run_name"] = run_name
    prepared["experiment"] = experiment_name_for_run(run_name)
    return prepared


def mask_to_labels(
    times: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    id_prefix: str = "bl",
    source: str = "baseline",
) -> list[dict[str, Any]]:
    times = pd.to_datetime(pd.Series(times), utc=True)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(len(times), len(mask))
    items: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        seg_score = None
        if scores is not None:
            sl = np.asarray(scores[i:j], dtype=np.float64)
            finite = sl[np.isfinite(sl)]
            if len(finite):
                seg_score = float(np.min(finite))
        items.append(
            {
                "id": f"{id_prefix}_{uuid.uuid4().hex[:8]}",
                "kind": "point" if j == i + 1 else "range",
                "start": times.iloc[i].isoformat(),
                "end": times.iloc[j - 1].isoformat(),
                "score": seg_score,
                "source": source,
            }
        )
        i = j
    return items


def export_score_predictions(
    plmn: str,
    *,
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    source: str,
    model: str,
    threshold_method: str,
    score_note: str,
    pred_mask: np.ndarray | None = None,
) -> Path:
    """Export UI JSON. Convention: **lower score = more anomalous** (``score < thr``)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    times = np.asarray(times)
    n = min(len(scores), len(times))
    scores = scores[:n]
    times = times[:n]
    if pred_mask is None:
        ok = np.isfinite(scores)
        pred = np.zeros(n, dtype=bool)
        pred[ok] = scores[ok] < float(threshold)
    else:
        pred = np.asarray(pred_mask, dtype=bool).reshape(-1)[:n]

    labels = mask_to_labels(times, pred, scores, id_prefix=model[:2], source=source)
    payload = {
        "plmn": plmn,
        "source": source,
        "model": model,
        "run_name": meta.get("run_name"),
        "eval_split": "valid",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_method": threshold_method,
        "n_scores": int(np.isfinite(scores).sum()),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "feature_columns": meta.get("cols") or meta.get("feature_columns"),
        "feature_mode": meta.get("feature_mode"),
        "excluded_train_metrics": sorted(EXCLUDED_TRAIN_METRICS),
        "score_series": {
            "times": [
                pd.Timestamp(t).tz_convert("UTC").isoformat()
                if pd.Timestamp(t).tzinfo
                else pd.Timestamp(t, tz="UTC").isoformat()
                for t in times
            ],
            "scores": [
                float(v) if np.isfinite(v) else None for v in scores
            ],
            "threshold": float(threshold),
            "note": score_note,
        },
        "split": {
            "train_time_end": meta.get("train_time_end"),
            "valid_time_start": meta.get("valid_time_start"),
            "valid_time_end": meta.get("valid_time_end"),
            "test_time_start": meta.get("test_time_start"),
        },
        "metrics": metrics,
        "labels": labels,
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{plmn}_{source}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def train_percentile_threshold(
    train_scores: np.ndarray,
    *,
    anomaly_ratio: float = 1.0,
) -> float:
    """Lower-is-worse scores: thr = low percentile of train scores.

    ``anomaly_ratio`` percent of train points are expected below thr
    (e.g. 1 → percentile 1).
    """
    s = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0
    q = float(np.clip(anomaly_ratio, 0.01, 50.0))
    return float(np.percentile(s, q))
