#!/usr/bin/env python3
"""LSTM-AD: next-step multivariate forecasting residual.

See docs/논문-스토리라인.md · LSTM_AD/PAPERIB.md.

Examples
--------
::

    cd LSTM_AD
    ../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10
    ../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10 --run_name comb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from common import (  # noqa: E402
    MODEL_ROOT,
    event_metrics,
    export_score_predictions,
    point_metrics,
    train_percentile_threshold,
)
from lstm_ad import predict_mse, scores_from_mse, train_lstm_ad  # noqa: E402


def _prep(plmn: str, run_name: str, *, drop_labeled: bool) -> dict:
    sys.path.insert(0, str(_REPO / "OmniAnomaly"))
    sys.path.insert(0, str(_REPO / "labeling"))
    from run_plmn import experiment_name_for_run, features_for_run, prepare_arrays
    from tool import EXCLUDED_TRAIN_METRICS, load_plmn

    df = load_plmn(plmn)
    selected = features_for_run(run_name, df)
    prepared = prepare_arrays(
        plmn,
        feature_columns=selected,
        run_name=run_name,
        drop_labeled_from_train=drop_labeled,
    )
    cols = list(prepared.get("cols") or [])
    if any(c in EXCLUDED_TRAIN_METRICS for c in cols):
        raise SystemExit("excluded metric in features")
    prepared["run_name"] = run_name
    prepared["experiment"] = experiment_name_for_run(run_name)
    return prepared


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", required=True)
    p.add_argument("--run_name", default="paperib", choices=["paperib", "comb", "comb_share"])
    p.add_argument("--anomaly_ratio", type=float, default=1.0)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    prepared = _prep(args.plmn, args.run_name, drop_labeled=False)
    train = prepared["train"]
    valid = prepared["valid"]
    y = prepared["y_valid"].astype(bool)
    times = prepared["times_valid"]
    meta = dict(prepared["meta"])
    meta["cols"] = prepared["cols"]
    meta["run_name"] = args.run_name
    meta["experiment"] = prepared.get("experiment") or args.run_name
    train_end = int(prepared["train_end"])
    human = prepared["human"]
    train_ok = ~human[:train_end]

    print(
        f"LSTM-AD train n={len(train)} valid={len(valid)} "
        f"window={args.window} epochs={args.epochs}",
        flush=True,
    )
    bundle = train_lstm_ad(
        train,
        window=args.window,
        hidden=args.hidden,
        layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train_mse, _ = predict_mse(bundle, train)
    valid_mse, _ = predict_mse(bundle, valid)
    train_scores = scores_from_mse(train_mse)
    fit_mask = np.isfinite(train_scores) & np.asarray(train_ok, dtype=bool)
    fit_scores = train_scores[fit_mask]
    thr = train_percentile_threshold(fit_scores, anomaly_ratio=args.anomaly_ratio)
    score = scores_from_mse(valid_mse)
    ok = np.isfinite(score)
    pred = np.zeros(len(score), dtype=bool)
    pred[ok] = score[ok] < thr

    point = point_metrics(y, pred)
    event = event_metrics(y, pred)
    metrics = {
        "valid-precision": point["precision"],
        "valid-recall": point["recall"],
        "valid-f1": point["f1"],
        "valid-TP": point["tp"],
        "valid-FP": point["fp"],
        "valid-FN": point["fn"],
        "point": point,
        "event": event,
        "anomaly_ratio": args.anomaly_ratio,
        "threshold": thr,
        "window": args.window,
        "epochs": args.epochs,
        "train_loss": bundle["history"],
    }
    source = "lstm_ad" if args.run_name == "paperib" else f"lstm_ad_{args.run_name}"
    path = export_score_predictions(
        args.plmn,
        times=times,
        scores=score,
        threshold=thr,
        meta=meta,
        metrics=metrics,
        source=source,
        model="lstm_ad",
        threshold_method="percentile",
        score_note=(
            "LSTM-AD next-step residual; score = −MSE; "
            "lower = more anomalous; thr = train residual percentile"
        ),
        pred_mask=pred,
    )
    out_dir = MODEL_ROOT / args.plmn / meta["experiment"] / "lstm_ad"
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": bundle["model"].state_dict(),
            "window": bundle["window"],
            "n_features": bundle["n_features"],
            "hidden": args.hidden,
            "layers": args.layers,
        },
        out_dir / "model.pt",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "train_loss"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    metrics["prediction_json"] = str(path)
    print(json.dumps(metrics, indent=2, default=str))
    pt, ev = metrics["point"], metrics["event"]
    print(
        f"\nvalid point: P={pt['precision']:.3f} R={pt['recall']:.3f} "
        f"F1={pt['f1']:.3f} (TP={pt['tp']} FP={pt['fp']} FN={pt['fn']})"
    )
    print(
        f"valid event: hit={ev['n_hit_events']}/{ev['n_gt_events']} "
        f"recall={ev['event_recall']:.3f}"
    )


if __name__ == "__main__":
    main()
