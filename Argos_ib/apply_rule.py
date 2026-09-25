"""Apply an Argos ``inference`` rule file to a DataFrame; score metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def run_inference(
    rule_path: Path | str,
    eval_df: pd.DataFrame,
    *,
    chunk_size: int = 1000,
) -> np.ndarray:
    """Return int labels length ``len(eval_df)`` (1 = anomaly)."""
    rule = Path(rule_path).read_text(encoding="utf-8")
    local_env: dict[str, Any] = {}
    exec(rule, local_env)  # noqa: S102 — intentional rule sandbox like Argos
    inference_fn = local_env["inference"]
    n = len(eval_df)
    out = np.zeros(n, dtype=int)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = eval_df.iloc[start:end][["value", "index"]].to_numpy(dtype=float)
        pred = np.asarray(inference_fn(chunk)).reshape(-1)
        if len(pred) != end - start:
            raise RuntimeError(
                f"rule returned length {len(pred)}, expected {end - start}"
            )
        out[start:end] = (pred > 0).astype(int)
    return out


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


def segment_bounds(y: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(y).astype(bool).reshape(-1)
    padded = np.concatenate([[False], y, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Segment-level: GT segment hit if any predicted point inside overlaps."""
    segs = segment_bounds(y_true)
    y_pred = np.asarray(y_pred).astype(bool).reshape(-1)
    if not segs:
        return {
            "n_gt_events": 0,
            "n_hit_events": 0,
            "event_recall": 0.0,
            "n_pred_events": len(segment_bounds(y_pred)),
        }
    hit = 0
    for a, b in segs:
        if y_pred[a:b].any():
            hit += 1
    n_pred = len(segment_bounds(y_pred))
    return {
        "n_gt_events": len(segs),
        "n_hit_events": hit,
        "event_recall": float(hit / len(segs)),
        "n_pred_events": n_pred,
    }
