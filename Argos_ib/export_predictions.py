"""Export UI-compatible prediction JSON for Argos / IB heuristic rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_LABELING = _REPO / "labeling"
if str(_LABELING) not in sys.path:
    sys.path.insert(0, str(_LABELING))

from tool import (  # noqa: E402
    chrono_split_bounds,
    load_plmn,
    load_predictions,
    mask_to_pred_labels,
    parse_time,
)


def _valid_mask_and_bounds(plmn: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    df = load_plmn(plmn)
    try:
        preds_oa = load_predictions(plmn)
    except Exception:
        preds_oa = None
    bounds = chrono_split_bounds(df, preds_oa)
    times = pd.to_datetime(df["time"], utc=True)
    valid_mask = (
        (times >= bounds["valid_start"]) & (times <= bounds["valid_end"])
    ).to_numpy()
    return df, valid_mask, bounds


def _keep_valid_labels(labels: list[dict], bounds: dict) -> list[dict]:
    v0, v1 = bounds["valid_start"], bounds["valid_end"]
    kept = []
    for item in labels:
        s = parse_time(item["start"])
        e = parse_time(item["end"])
        if e < v0 or s > v1:
            continue
        kept.append(item)
    return kept


def build_prediction_doc(
    plmn: str,
    *,
    metric: str,
    pred_valid: np.ndarray,
    metrics: dict[str, Any],
    rule_path: str,
    source: str = "argos",
    score_valid: np.ndarray | None = None,
    ensemble_metrics: list[str] | None = None,
) -> dict[str, Any]:
    df, valid_mask, bounds = _valid_mask_and_bounds(plmn)
    if int(valid_mask.sum()) != len(pred_valid):
        raise ValueError(
            f"pred length {len(pred_valid)} != valid points {int(valid_mask.sum())}"
        )
    full_pred = np.zeros(len(df), dtype=bool)
    full_pred[valid_mask] = np.asarray(pred_valid).astype(bool)

    scores_full = None
    score_series: dict[str, Any]
    if score_valid is not None:
        score_valid = np.asarray(score_valid, dtype=np.float64).reshape(-1)
        if len(score_valid) != len(pred_valid):
            raise ValueError("score_valid length must match pred_valid")
        scores_full = np.full(len(df), np.nan, dtype=np.float64)
        scores_full[valid_mask] = score_valid
        times_iso = [
            pd.Timestamp(t).tz_convert("UTC").isoformat()
            if pd.Timestamp(t).tzinfo
            else pd.Timestamp(t, tz="UTC").isoformat()
            for t in df.loc[valid_mask, "time"].to_numpy()
        ]
        score_series = {
            "times": times_iso,
            "scores": [float(x) for x in score_valid],
            "threshold": 1.0,
            "note": (
                "ensemble anomaly score = # of metric axes firing "
                "(higher = more axes agree); binary thr defaults to 1 (OR)"
            ),
        }
    else:
        score_series = {
            "times": [],
            "scores": [],
            "note": "Argos emits binary rules, not continuous scores",
        }

    labels = mask_to_pred_labels(
        full_pred,
        df["time"].to_numpy(),
        scores=scores_full,
        id_prefix="ag",
        source=source,
    )
    kept = _keep_valid_labels(labels, bounds)
    doc: dict[str, Any] = {
        "plmn": plmn,
        "model": "argos",
        "source": source,
        "metric": metric,
        "eval_split": "valid",
        "threshold_method": "argos_rule",
        "rule_path": rule_path,
        "metrics": metrics,
        "split": {k: str(v) for k, v in bounds.items()},
        "labels": kept,
        "score_series": score_series,
    }
    if ensemble_metrics:
        doc["ensemble_metrics"] = list(ensemble_metrics)
        doc["feature_mode"] = "argos_ensemble"
    return doc


def write_prediction_json(doc: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
