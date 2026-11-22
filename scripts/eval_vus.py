#!/usr/bin/env python3
"""저장된 prediction score JSON에서 VUS metric을 다시 계산하는 스크립트.

모델 재학습은 하지 않는다.
저장된 validation score series와 같은 시간축의 전문가 label mask를 만든 뒤
common.metrics.vus_pr()로 VUS를 계산하고, prediction JSON에 다시 기록한다.
마지막으로 모델/사업자별 결과를 summary CSV/JSON으로 저장한다.

사용 예:
  .venv/bin/python scripts/eval_vus.py --all
  .venv/bin/python scripts/eval_vus.py --plmn P0480
  .venv/bin/python scripts/eval_vus.py --all --models context_ae hour_ae roca
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import vus_pr

PRED_DIR = ROOT / "data" / "ib_data" / "predictions"
LABEL_DIR = ROOT / "data" / "ib_data" / "labels"
RESULT_DIR = ROOT / "results"

PAPER_PLMNS = [
    "P0480",
    "P0714",
    "P1888",
    "P0975",
    "P2309",
    "P1921",
    "P1575",
    "P0139",
    "P2291",
    "P0110",
]

MODEL_FILES = {
    "context_ae": "{plmn}_context_ae_valid.json",
    "hour_ae": "{plmn}_hour_ae_valid.json",
    "roca": "{plmn}_roca.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _score_series(doc: dict[str, Any]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ss = doc.get("score_series") or {}
    times = ss.get("times") or []
    scores = ss.get("scores") or []
    if not times or not scores:
        raise ValueError("score_series.times/scores not found")
    n = min(len(times), len(scores))
    t = pd.to_datetime(times[:n], utc=True)
    s = np.asarray(
        [np.nan if x is None else float(x) for x in scores[:n]],
        dtype=np.float64,
    )
    return t, s


def _label_mask(times: pd.DatetimeIndex, label_doc: dict[str, Any]) -> np.ndarray:
    y = np.zeros(len(times), dtype=bool)
    for item in label_doc.get("labels") or []:
        start = item.get("start")
        end = item.get("end") or start
        if not start:
            continue
        a = pd.Timestamp(start)
        b = pd.Timestamp(end)
        if a.tzinfo is None:
            a = a.tz_localize("UTC")
        else:
            a = a.tz_convert("UTC")
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        else:
            b = b.tz_convert("UTC")
        y |= (times >= a) & (times <= b)
    return y


def evaluate_file(path: Path, *, vus_window: int | None = None) -> dict[str, Any]:
    doc = _load_json(path)
    plmn = str(doc.get("plmn") or path.name.split("_")[0])
    label_path = LABEL_DIR / f"{plmn}_labels.json"
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    times, scores = _score_series(doc)
    labels = _load_json(label_path)
    y = _label_mask(times, labels)

    polarity = str(doc.get("score_polarity") or "higher_is_anomaly")
    score_for_vus = scores if polarity != "lower_is_anomaly" else -scores

    result = vus_pr(
        y,
        score_for_vus,
        sliding_window=vus_window,
    )

    paper = dict(doc.get("paper_metrics") or {})
    paper.update(
        {
            "primary_metric": "vus_pr",
            "vus_pr": result.get("vus_pr"),
            "vus_roc": result.get("vus_roc"),
            "r_auc_pr": result.get("r_auc_pr"),
            "auc_pr": result.get("auc_pr"),
            "vus_window": result.get("vus_window"),
            "eval_split": str(doc.get("eval_split") or "valid"),
        }
    )
    doc["paper_metrics"] = paper

    metrics = dict(doc.get("metrics") or {})
    metrics.update(
        {
            "primary_metric": "vus_pr",
            "vus_pr": result.get("vus_pr"),
            "vus_roc": result.get("vus_roc"),
            "r_auc_pr": result.get("r_auc_pr"),
            "auc_pr": result.get("auc_pr"),
            "vus_window": result.get("vus_window"),
        }
    )
    doc["metrics"] = metrics
    doc["vus_evaluated_at"] = datetime.now(timezone.utc).isoformat()

    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "plmn": plmn,
        "model": str(doc.get("model") or doc.get("source") or path.stem),
        "file": path.name,
        "n_scores": int(np.isfinite(scores).sum()),
        "n_gt_points": int(y.sum()),
        "vus_pr": result.get("vus_pr"),
        "vus_roc": result.get("vus_roc"),
        "r_auc_pr": result.get("r_auc_pr"),
        "auc_pr": result.get("auc_pr"),
        "vus_window": result.get("vus_window"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="evaluate all 10 paper PLMNs")
    scope.add_argument("--plmn", help="evaluate one PLMN")
    p.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_FILES),
        default=["context_ae", "hour_ae", "roca"],
        help="prediction models to evaluate",
    )
    p.add_argument(
        "--vus_window",
        type=int,
        default=0,
        help="fixed VUS tolerance window in points; 0 = median GT event length",
    )
    args = p.parse_args()

    plmns = PAPER_PLMNS if args.all else [str(args.plmn)]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for plmn in plmns:
        for model in args.models:
            path = PRED_DIR / MODEL_FILES[model].format(plmn=plmn)
            if not path.exists():
                msg = f"[SKIP] {path.name}: file not found"
                print(msg)
                failures.append(msg)
                continue
            try:
                row = evaluate_file(
                    path,
                    vus_window=(None if args.vus_window <= 0 else args.vus_window),
                )
                rows.append(row)
                print(
                    f"{plmn:>5} {model:<10} "
                    f"VUS-PR={row['vus_pr']} "
                    f"VUS-ROC={row['vus_roc']} "
                    f"R-AUC-PR={row['r_auc_pr']} "
                    f"AUC-PR={row['auc_pr']} "
                    f"window={row['vus_window']}"
                )
            except Exception as exc:
                msg = f"[ERROR] {path.name}: {exc}"
                print(msg, file=sys.stderr)
                failures.append(msg)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "vus_summary.csv"
    json_path = RESULT_DIR / "vus_summary.json"

    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"summary CSV:  {csv_path}")
        print(f"summary JSON: {json_path}")

    if failures:
        print(f"completed with {len(failures)} skipped/failed file(s)")
    if not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
