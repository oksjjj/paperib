#!/usr/bin/env python3
"""LSTM-AD: next-step multivariate forecasting residual.

Default: score / metrics / UI export on **valid only** (HP tuning).
Pass ``--with_test`` to also score test (final paper table).
See docs/baseline-하이퍼파라미터.md · LSTM_AD/PAPERIB.md.

Examples
--------
::

    cd LSTM_AD
    ../.venv/bin/python run_plmn.py --plmn P0480
    ../.venv/bin/python run_plmn.py --plmn P0480 --anomaly_ratio 2.0
    ../.venv/bin/python run_plmn.py --plmn P0480 --with_test
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
    export_score_predictions,
    pack_split_metrics,
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


def _print_split(name: str, block: dict) -> None:
    pt, ev = block["point"], block["event"]
    print(
        f"{name} point: P={pt['precision']:.3f} R={pt['recall']:.3f} "
        f"F1={pt['f1']:.3f} (TP={pt['tp']} FP={pt['fp']} FN={pt['fn']})"
    )
    print(
        f"{name} event: hit={ev['n_hit_events']}/{ev['n_gt_events']} "
        f"recall={ev['event_recall']:.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", required=True)
    p.add_argument("--run_name", default="paperib", choices=["paperib", "comb", "comb_share"])
    p.add_argument("--anomaly_ratio", type=float, default=1.0)
    # Improved defaults: window=100 (OA/AT align); hidden=128 (vs 64).
    # epochs=10 aligned with OA/AT defaults (20 only as optional sweep).
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--with_test",
        action="store_true",
        help="Also score/export test (default: valid only for HP tuning)",
    )
    args = p.parse_args()

    prepared = _prep(args.plmn, args.run_name, drop_labeled=False)
    train = prepared["train"]
    valid = prepared["valid"]
    test = prepared["test"]
    y_valid = prepared["y_valid"].astype(bool)
    y_test = prepared["y_test"].astype(bool)
    times_valid = prepared["times_valid"]
    times_test = prepared["times_test"]
    meta = dict(prepared["meta"])
    meta["cols"] = prepared["cols"]
    meta["run_name"] = args.run_name
    meta["experiment"] = prepared.get("experiment") or args.run_name
    train_end = int(prepared["train_end"])
    human = prepared["human"]
    train_ok = ~human[:train_end]

    print(
        f"LSTM-AD train n={len(train)} valid={len(valid)} test={len(test)} "
        f"window={args.window} epochs={args.epochs} with_test={args.with_test}",
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
    train_scores = scores_from_mse(train_mse)
    fit_mask = np.isfinite(train_scores)
    if len(train_ok) == len(train_scores):
        fit_mask = fit_mask & np.asarray(train_ok, dtype=bool)
    fit_scores = train_scores[fit_mask]
    thr = train_percentile_threshold(fit_scores, anomaly_ratio=args.anomaly_ratio)

    w = int(bundle["window"])
    context = train[-w:] if len(train) >= w else train
    if args.with_test:
        eval_block = np.vstack([context, valid, test])
        mse_block, _ = predict_mse(bundle, eval_block)
        mse_eval = mse_block[len(context) :]
        score = scores_from_mse(mse_eval)
        n_v = len(valid)
        score_v, score_t = score[:n_v], score[n_v:]
        times = np.concatenate([times_valid, times_test])
        eval_split = "valid+test"
        note = (
            "LSTM-AD residual −MSE; thr = train percentile; "
            "evaluate valid+test (train window context for continuity)"
        )
        series_note = "series = valid+test"
    else:
        eval_block = np.vstack([context, valid])
        mse_block, _ = predict_mse(bundle, eval_block)
        mse_eval = mse_block[len(context) :]
        score = scores_from_mse(mse_eval)
        score_v = score
        times = times_valid
        eval_split = "valid"
        note = (
            "LSTM-AD residual −MSE; thr = train percentile; "
            "evaluate valid only (HP tuning; train window context)"
        )
        series_note = "series = valid only"

    ok = np.isfinite(score)
    pred = np.zeros(len(score), dtype=bool)
    pred[ok] = score[ok] < thr
    if args.with_test:
        n_v = len(valid)
        pred_v, pred_t = pred[:n_v], pred[n_v:]
    else:
        pred_v = pred

    metrics: dict = {
        "eval_splits": ["valid", "test"] if args.with_test else ["valid"],
        "eval_split": eval_split,
        "test_held_out": not args.with_test,
        "anomaly_ratio": args.anomaly_ratio,
        "threshold": thr,
        "window": args.window,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "train_loss": bundle["history"],
        "note": note,
    }
    metrics.update(pack_split_metrics(y_valid, pred_v, prefix="valid"))
    if args.with_test:
        metrics.update(pack_split_metrics(y_test, pred_t, prefix="test"))
    metrics["point"] = metrics["valid"]["point"]
    metrics["event"] = metrics["valid"]["event"]

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
            f"lower = more anomalous; thr = train residual percentile; {series_note}"
        ),
        pred_mask=pred,
        eval_split=eval_split,
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
    to_save = {k: v for k, v in metrics.items() if k != "train_loss"}
    (out_dir / "metrics.json").write_text(
        json.dumps(to_save, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics["prediction_json"] = str(path)
    print(json.dumps(to_save, indent=2, default=str))
    _print_split("valid", metrics["valid"])
    if args.with_test:
        _print_split("test", metrics["test"])


if __name__ == "__main__":
    main()
