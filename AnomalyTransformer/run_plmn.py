#!/usr/bin/env python3
"""Train / score Anomaly Transformer on a paperib PLMN (3 feature views).

Official model: https://github.com/thuml/Anomaly-Transformer
(ICLR 2022 — association discrepancy). Same chronological split and
``--run_name`` views as OmniAnomaly (`paperib` / `comb` / `comb_share`).

Example:
    cd AnomalyTransformer
    ../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10
    ../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10 --run_name comb
    ../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10 --run_name comb_share

Outputs (non-default ``win_size`` appends ``_w{N}``; default 100 keeps legacy paths):
    ../data/ib_data/predictions/{PLMN}_anomalytransformer.json
    ../data/ib_data/predictions/{PLMN}_anomalytransformer_comb.json
    ../data/ib_data/predictions/{PLMN}_anomalytransformer_comb_w200.json
    ../data/ib_data/predictions/{PLMN}_anomalytransformer_comb_share.json

Score polarity for the labeling UI (same as OmniAnomaly overlays):
    stored_score = -energy   (higher = more normal)
    anomaly when stored_score < threshold  (== energy > energy_threshold)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERIB_ROOT = os.path.dirname(ROOT)
OMNI_ROOT = os.path.join(PAPERIB_ROOT, "OmniAnomaly")
LABELING_ROOT = os.path.join(PAPERIB_ROOT, "labeling")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if LABELING_ROOT not in sys.path:
    sys.path.insert(0, LABELING_ROOT)

os.chdir(ROOT)

from plmn_engine import PlmnAnomalyTransformer  # noqa: E402
from tool import (  # noqa: E402
    ensure_score_column,
    load_labels,
    load_plmn,
    load_predictions,
)


PRED_DIR = os.path.join(PAPERIB_ROOT, "data", "ib_data", "predictions")
KNOWN_RUN_NAMES = frozenset({"paperib", "comb", "comb_share"})
DEFAULT_WIN_SIZE = 100


def experiment_name_for_run(
    run_name: str, win_size: int = DEFAULT_WIN_SIZE
) -> str:
    """Dir / artifact slug; non-default window gets ``_w{N}``."""
    base = run_name or "paperib"
    wl = int(win_size)
    if wl == DEFAULT_WIN_SIZE:
        return base
    return f"{base}_w{wl}"


def pred_source_for_run(
    run_name: str, win_size: int = DEFAULT_WIN_SIZE
) -> str:
    exp = experiment_name_for_run(run_name, win_size)
    if exp == "paperib":
        return "anomalytransformer"
    if exp.startswith("paperib_"):
        return "anomalytransformer" + exp[len("paperib") :]
    return f"anomalytransformer_{exp}"


def _load_omni_run_plmn():
    """Reuse OmniAnomaly feature prep / splits without duplicating COMB logic."""
    path = os.path.join(OMNI_ROOT, "run_plmn.py")
    cwd = os.getcwd()
    # OmniAnomaly/run_plmn.py chdirs into OmniAnomaly on import.
    if OMNI_ROOT not in sys.path:
        sys.path.insert(0, OMNI_ROOT)
    spec = importlib.util.spec_from_file_location("omni_run_plmn", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.chdir(cwd)
    return mod


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _top_metrics_from_dim_mse(
    dim_mse_slice: np.ndarray,
    feature_columns: list[str],
    *,
    baseline: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank features by reconstruction MSE (optionally vs train median)."""
    arr = np.asarray(dim_mse_slice, dtype=np.float64)
    if arr.ndim == 1:
        mean_mse = arr
    else:
        mean_mse = np.nanmean(arr, axis=0)
    if len(mean_mse) != len(feature_columns):
        return []
    if baseline is None:
        contrib = mean_mse
    else:
        contrib = mean_mse.reshape(-1) - np.asarray(baseline, dtype=np.float64).reshape(
            -1
        )
    order = np.argsort(-contrib)
    out = []
    for rank, di in enumerate(order[:top_k], start=1):
        out.append(
            {
                "rank": rank,
                "metric": str(feature_columns[int(di)]),
                "contribution": float(contrib[int(di)]),
                "log_prob": float(-mean_mse[int(di)]),  # UI field reuse
            }
        )
    return out


def _preds_to_label_items(
    times: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    window_length: int,
    score_dim: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    baseline: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Boolean preds → UI segments. score[i] aligns with times[i] (dense)."""
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = min(len(pred), len(scores), len(times))
    pred = pred[:n]
    scores = scores[:n]
    times = times[:n]
    items: list[dict[str, Any]] = []
    i = 0
    while i < n:
        if not pred[i]:
            i += 1
            continue
        j = i + 1
        while j < n and pred[j]:
            j += 1
        seg_scores = scores[i:j]
        finite = seg_scores[np.isfinite(seg_scores)]
        score_min = float(np.min(finite)) if len(finite) else None
        kind = "point" if j == i + 1 else "range"
        item: dict[str, Any] = {
            "id": f"at_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "start": pd.Timestamp(times[i]).tz_convert("UTC").isoformat(),
            "end": pd.Timestamp(times[j - 1]).tz_convert("UTC").isoformat(),
            "score": score_min,
            "source": "anomalytransformer",
        }
        if (
            score_dim is not None
            and feature_columns is not None
            and score_dim.shape[0] >= j
        ):
            item["top_metrics"] = _top_metrics_from_dim_mse(
                score_dim[i:j],
                feature_columns,
                baseline=baseline,
                top_k=top_k,
            )
        items.append(item)
        i = j
    return items


def _at_ui_scores_from_energy(
    energy: np.ndarray,
    energy_threshold: float,
) -> tuple[np.ndarray, float]:
    """Map AT energy → UI scores (higher = more normal; anomaly iff score < 0).

    ``score = log10(thr) - log10(energy)`` so the energy threshold lands at 0 and
    typical values are O(1) instead of ~1e-4 (which rounded to ``-0.00``).
    """
    eps = 1e-15
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    thr = max(float(energy_threshold), eps)
    scores = np.full(e.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(e)
    scores[ok] = np.log10(thr) - np.log10(np.maximum(e[ok], eps))
    return scores, 0.0


def export_ui_predictions(
    plmn: str,
    *,
    times: np.ndarray,
    energy: np.ndarray,
    energy_threshold: float,
    window_length: int,
    meta: dict,
    metrics: dict | None = None,
    eval_split: str = "valid",
    dim_mse: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    baseline: np.ndarray | None = None,
    top_k: int = 5,
    source: str = "anomalytransformer",
    anormly_ratio: float = 1.0,
) -> str:
    """Export labeling UI JSON. UI score uses log10(thr/energy); thr → 0."""
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    scores, threshold = _at_ui_scores_from_energy(energy, energy_threshold)
    pred = energy > float(energy_threshold)
    finite = np.isfinite(energy)
    pred = pred & finite

    labels = _preds_to_label_items(
        times,
        pred,
        scores,
        window_length=window_length,
        score_dim=dim_mse,
        feature_columns=feature_columns,
        baseline=baseline,
        top_k=top_k,
    )
    score_times = times[: len(scores)]
    payload = {
        "plmn": plmn,
        "source": source,
        "model": "anomalytransformer",
        "run_name": meta.get("run_name"),
        "eval_split": eval_split,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_method": "percentile",
        "anormly_ratio": float(anormly_ratio),
        "energy_threshold": float(energy_threshold),
        "score_polarity": "log10_thr_over_energy",
        "score_scale": "log10_thr_over_energy",
        "window_length": int(window_length),
        "n_scores": int(np.sum(finite)),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "feature_columns": meta.get("feature_columns"),
        "feature_mode": meta.get("feature_mode"),
        "comb_scaling": meta.get("comb_scaling"),
        "comb_raw_counters": meta.get("comb_raw_counters"),
        "score_series": {
            "times": [
                pd.Timestamp(t).tz_convert("UTC").isoformat() for t in score_times
            ],
            "scores": [
                float(v) if np.isfinite(v) else None for v in scores[: len(score_times)]
            ],
            "threshold": float(threshold),
            "note": (
                "score = log10(energy_threshold) - log10(energy); "
                "higher = more normal; anomaly when score < 0 "
                "(== energy > energy_threshold)"
            ),
        },
        "attribution": {
            "method": "per_dim_reconstruction_mse",
            "top_k": int(top_k),
            "note": (
                "contribution = segment mean MSE − train median MSE "
                "(larger = worse reconstruction on that metric)"
            ),
        },
        "split": {
            "train_time_end": meta.get("train_time_end"),
            "valid_time_start": meta.get("valid_time_start"),
            "valid_time_end": meta.get("valid_time_end"),
            "test_time_start": meta.get("test_time_start"),
        },
        "metrics": metrics or {},
        "labels": labels,
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"{plmn}_{source}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(
        f"wrote UI predictions ({eval_split} only) → {out_path} "
        f"({len(labels)} segments)",
        flush=True,
    )
    try:
        preds = load_predictions(plmn, source=source)
        df = load_plmn(plmn)
        ensure_score_column(df, preds, source=source, plmn=plmn)
        print(
            f"warmed score cache → labeling/cache/{plmn}_score_{source}.pkl",
            flush=True,
        )
    except Exception as exc:
        print(f"score cache warmup skipped: {exc}", flush=True)
    return out_path


def run(plmn: str, args: argparse.Namespace) -> None:
    if args.run_name not in KNOWN_RUN_NAMES:
        raise SystemExit(
            f"unknown --run_name {args.run_name!r}; "
            f"use one of: {', '.join(sorted(KNOWN_RUN_NAMES))}"
        )
    omni = _load_omni_run_plmn()
    os.chdir(ROOT)

    df_probe = load_plmn(plmn)
    selected = omni.features_for_run(args.run_name, df_probe)
    if selected is not None:
        print(
            f"run={args.run_name} features ({len(selected)}): {', '.join(selected)}",
            flush=True,
        )
    else:
        print(f"run={args.run_name} features: all raw metrics", flush=True)

    # Omni prepare_arrays writes under data/ib_data/pickle/...
    prepared = omni.prepare_arrays(
        plmn,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        drop_labeled_from_train=not args.keep_labeled_in_train,
        feature_columns=selected,
        run_name=args.run_name,
    )
    os.chdir(ROOT)

    train = np.asarray(prepared["train"], dtype=np.float32)
    valid = np.asarray(prepared["valid"], dtype=np.float32)
    times_valid = prepared["times_valid"]
    y_valid = np.asarray(prepared["y_valid"], dtype=bool)
    meta = dict(prepared["meta"])
    x_dim = int(meta["x_dim"])
    wl = int(args.win_size)
    exp = experiment_name_for_run(args.run_name, wl)

    model_dir = os.path.join(ROOT, "model", plmn, exp)
    result_dir = os.path.join(ROOT, "result", plmn, exp)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    ckpt_name = f"{plmn}_checkpoint.pth"
    ckpt_path = os.path.join(model_dir, ckpt_name)

    if args.device:
        device = torch.device(args.device)
    else:
        device = None

    engine = PlmnAnomalyTransformer(
        win_size=wl,
        input_c=x_dim,
        lr=args.lr,
        k=args.k,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        anormly_ratio=args.anormly_ratio,
        train_step=args.train_step,
        device=device,
        d_model=args.d_model,
        e_layers=args.e_layers,
    )
    print(
        f"device={engine.device}  x_dim={x_dim}  win={wl}  "
        f"epochs={args.num_epochs}  anomaly_ratio={args.anormly_ratio}%",
        flush=True,
    )

    metrics: dict[str, Any] = {
        "eval_split": "valid",
        "test_held_out": True,
        "model": "anomalytransformer",
    }

    if args.num_epochs > 0:
        train_metrics = engine.train(
            train, valid, model_dir=model_dir, checkpoint_name=ckpt_name
        )
        metrics.update(train_metrics or {})
        print(f"saved model → {ckpt_path}", flush=True)
    else:
        if not os.path.isfile(ckpt_path):
            raise SystemExit(f"no checkpoint at {ckpt_path}; train first")
        engine.load(ckpt_path)
        print(f"restored from {ckpt_path}", flush=True)

    print("scoring train (threshold) + valid (UI) ...", flush=True)
    train_energy, train_dim = engine.energy_series(
        train, step=args.score_step, with_dim_mse=True
    )
    valid_energy, valid_dim = engine.energy_series(
        valid, step=args.score_step, with_dim_mse=True
    )
    energy_thr = engine.threshold_from_train(train_energy)
    print(f"energy threshold (train p{100 - args.anormly_ratio}): {energy_thr:.6g}", flush=True)

    pred_valid = (valid_energy > energy_thr) & np.isfinite(valid_energy)
    metrics["valid-n-pred"] = int(pred_valid.sum())
    metrics["valid-n-human"] = int(y_valid.sum())
    metrics["energy_threshold"] = float(energy_thr)
    metrics["anormly_ratio"] = float(args.anormly_ratio)

    if y_valid.any() and pred_valid.any():
        from sklearn.metrics import precision_recall_fscore_support

        # Point-level vs human labels on valid (no point-adjust — honest).
        yt = y_valid.astype(int)
        yp = pred_valid.astype(int)
        # Align lengths if energy shorter (shouldn't be).
        m = min(len(yt), len(yp))
        precision, recall, f1, _ = precision_recall_fscore_support(
            yt[:m], yp[:m], average="binary", zero_division=0
        )
        metrics["valid-precision"] = float(precision)
        metrics["valid-recall"] = float(recall)
        metrics["valid-f1"] = float(f1)
        print(
            f"valid: human={int(y_valid.sum())}  pred={int(pred_valid.sum())}  "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}",
            flush=True,
        )
    else:
        print(
            f"valid: human={int(y_valid.sum())}  pred={int(pred_valid.sum())}",
            flush=True,
        )

    baseline = None
    if train_dim is not None:
        baseline = np.nanmedian(train_dim, axis=0)

    with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    with open(os.path.join(result_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            _jsonable(
                {
                    "plmn": plmn,
                    "run_name": args.run_name,
                    "win_size": wl,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "lr": args.lr,
                    "k": args.k,
                    "anormly_ratio": args.anormly_ratio,
                    "d_model": args.d_model,
                    "e_layers": args.e_layers,
                    "x_dim": x_dim,
                    "official": "https://github.com/thuml/Anomaly-Transformer",
                }
            ),
            f,
            indent=2,
        )

    export_ui_predictions(
        plmn,
        times=times_valid,
        energy=valid_energy,
        energy_threshold=float(energy_thr),
        window_length=wl,
        meta=meta,
        metrics=metrics,
        eval_split="valid",
        dim_mse=valid_dim,
        feature_columns=list(meta.get("feature_columns") or []),
        baseline=baseline,
        top_k=int(args.attr_top_k),
        source=pred_source_for_run(args.run_name, wl),
        anormly_ratio=float(args.anormly_ratio),
    )
    print("done. (test split held out)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--run_name", default="paperib", choices=sorted(KNOWN_RUN_NAMES))
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--win_size", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument(
        "--anormly_ratio",
        type=float,
        default=1.0,
        help="Train-energy percentile tail %% used as threshold (official name)",
    )
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--device", default=None)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument(
        "--train_step",
        type=int,
        default=1,
        help="Sliding-window step for training (1 = densest)",
    )
    p.add_argument(
        "--score_step",
        type=int,
        default=1,
        help="Sliding-window step when scoring (1 = densest, slower)",
    )
    p.add_argument("--attr_top_k", type=int, default=5)
    p.add_argument(
        "--keep_labeled_in_train",
        action="store_true",
        help="Do not drop human-labeled anomalies from the train split",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Silence unused import warning for load_labels (available for future metrics).
    _ = load_labels
    run(args.plmn, args)


if __name__ == "__main__":
    main()
