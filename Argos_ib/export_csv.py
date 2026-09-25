"""Export paperib PLMN series to Argos CSV layout (value, label, index).

Argos ([arXiv:2501.14170](https://arxiv.org/abs/2501.14170),
[microsoft/argos](https://github.com/microsoft/argos)) expects univariate CSVs.
We export train+valid only and set ``train_test_split`` so Argos's test fold
equals paperib **valid** (chronological 60/20 of the 80% prefix).
"""

from __future__ import annotations

import json
import os
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
    A_RATE_KEY,
    EXCLUDED_TRAIN_METRICS,
    M971_COL,
    S_RATE_KEY,
    chrono_split_bounds,
    human_anomaly_mask,
    load_labels,
    load_plmn,
    load_predictions,
)

DEFAULT_METRICS = tuple(
    m for m in (S_RATE_KEY, A_RATE_KEY, M971_COL) if m not in EXCLUDED_TRAIN_METRICS
)
DEFAULT_OUT = _REPO / "data" / "ib_data" / "argos"


def _split_masks(times: pd.Series, bounds: dict) -> dict[str, np.ndarray]:
    t = pd.to_datetime(times, utc=True)
    return {
        "train": ((t >= bounds["t_min"]) & (t <= bounds["train_end"])).to_numpy(),
        "valid": ((t >= bounds["valid_start"]) & (t <= bounds["valid_end"])).to_numpy(),
        "test": ((t >= bounds["test_start"]) & (t <= bounds["t_max"])).to_numpy(),
    }


def export_plmn_metric(
    plmn: str,
    metric: str,
    *,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Write ``{out}/{plmn}/{metric}.csv`` covering train+valid only.

    Returns paths and the ``train_test_split`` to pass to Argos so that
    Argos ``test_df`` == paperib valid.
    """
    out_root = Path(out_dir or DEFAULT_OUT)
    plmn_dir = out_root / plmn
    plmn_dir.mkdir(parents=True, exist_ok=True)

    df = load_plmn(plmn)
    if metric in EXCLUDED_TRAIN_METRICS:
        raise SystemExit(
            f"{plmn}: metric {metric!r} is in EXCLUDED_TRAIN_METRICS "
            f"({sorted(EXCLUDED_TRAIN_METRICS)}); excluded from all models"
        )
    if metric not in df.columns:
        raise SystemExit(f"{plmn}: metric {metric!r} not in dataframe")
    doc = load_labels(plmn)
    try:
        preds = load_predictions(plmn)
    except Exception:
        preds = None
    bounds = chrono_split_bounds(df, preds)
    if not bounds:
        raise SystemExit(f"{plmn}: cannot resolve chrono split")

    y = human_anomaly_mask(df, doc.get("labels") or []).astype(int)
    masks = _split_masks(df["time"], bounds)
    # Prefix used by Argos: train + valid only (hold out paperib test).
    keep = masks["train"] | masks["valid"]
    n_train = int(masks["train"].sum())
    n_valid = int(masks["valid"].sum())
    n_keep = int(keep.sum())
    if n_train < 10 or n_valid < 1:
        raise SystemExit(
            f"{plmn}: need train and valid points; got train={n_train} valid={n_valid}"
        )
    # Argos: full_train = first train_test_split of CSV; test = remainder.
    # Want full_train == paperib train → ratio = n_train / n_keep.
    train_test_split = float(n_train) / float(n_keep)

    values = df.loc[keep, metric].to_numpy(dtype=np.float64)
    labels = y[keep]
    # Stable row index 0..n-1 inside the exported CSV (Argos convention).
    index = np.arange(len(values), dtype=np.int64)
    csv_path = plmn_dir / f"{metric}.csv"
    pd.DataFrame({"value": values, "label": labels, "index": index}).to_csv(
        csv_path, index=False
    )

    meta = {
        "plmn": plmn,
        "metric": metric,
        "csv_path": str(csv_path),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_csv": n_keep,
        "train_anom": int(y[masks["train"]].sum()),
        "valid_anom": int(y[masks["valid"]].sum()),
        "train_test_split": train_test_split,
        "split": {k: str(v) for k, v in bounds.items()},
        "note": (
            "CSV = paperib train+valid. Pass train_test_split to Argos so "
            "Argos test fold == paperib valid. paperib test is held out."
        ),
    }
    meta_path = plmn_dir / f"{metric}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def export_plmn(
    plmn: str,
    metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
    *,
    out_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    return [export_plmn_metric(plmn, m, out_dir=out_dir) for m in metrics]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", required=True)
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Univariate metrics to export (default: S_RATE A_RATE M971)",
    )
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    for meta in export_plmn(args.plmn, args.metrics, out_dir=args.out_dir):
        print(
            f"{meta['metric']}: {meta['csv_path']}  "
            f"split={meta['train_test_split']:.4f}  "
            f"train_anom={meta['train_anom']} valid_anom={meta['valid_anom']}"
        )
