#!/usr/bin/env python3
"""Classical IB baseline: per-feature IQR + EWMA (OR fuse).

See docs/논문-스토리라인.md · Classical/PAPERIB.md.

Examples
--------
::

    cd Classical
    ../.venv/bin/python run_plmn.py --plmn P0480
    ../.venv/bin/python run_plmn.py --plmn P0480 --run_name comb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from classical import score_classical, train_scores_for_threshold  # noqa: E402
from common import (  # noqa: E402
    MODEL_ROOT,
    event_metrics,
    export_score_predictions,
    point_metrics,
    train_percentile_threshold,
)


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
    p.add_argument("--iqr_k", type=float, default=1.5)
    p.add_argument("--ewma_alpha", type=float, default=0.3)
    p.add_argument("--ewma_z", type=float, default=3.0)
    p.add_argument("--no_iqr", action="store_true")
    p.add_argument("--no_ewma", action="store_true")
    args = p.parse_args()

    prepared = _prep(args.plmn, args.run_name, drop_labeled=True)
    train = prepared["train"]
    valid = prepared["valid"]
    y = prepared["y_valid"].astype(bool)
    times = prepared["times_valid"]
    meta = dict(prepared["meta"])
    meta["cols"] = prepared["cols"]
    meta["run_name"] = args.run_name
    meta["experiment"] = prepared.get("experiment") or args.run_name

    fit_kw = dict(
        iqr_k=args.iqr_k,
        ewma_alpha=args.ewma_alpha,
        ewma_z=args.ewma_z,
        use_iqr=not args.no_iqr,
        use_ewma=not args.no_ewma,
    )
    train_scores = train_scores_for_threshold(train, **fit_kw)
    thr = train_percentile_threshold(train_scores, anomaly_ratio=args.anomaly_ratio)
    scored = score_classical(train, valid, **fit_kw)
    score = scored["score"]
    pred = scored["pred"]
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
        "classical": scored["detail"],
        "note": "binary=OR(IQR,EWMA) across features; score=−#fires",
    }
    source = (
        "classical" if args.run_name == "paperib" else f"classical_{args.run_name}"
    )
    path = export_score_predictions(
        args.plmn,
        times=times,
        scores=score,
        threshold=thr,
        meta=meta,
        metrics=metrics,
        source=source,
        model="classical",
        threshold_method="percentile",
        score_note=(
            "classical IQR+EWMA; score = −(# feature detectors firing); "
            "lower = more anomalous; binary overlay uses OR (any fire)"
        ),
        pred_mask=pred,
    )
    out_dir = MODEL_ROOT / args.plmn / meta["experiment"] / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
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
