#!/usr/bin/env python3
"""24-hour contextual point anomaly detection with iterative decontamination.

For each local hour h in 0..23:
1. Fit a stage-1 Isolation Forest on unlabeled train points for hour h.
2. Remove stage-1 anomaly candidates.
3. Refit scaler and final Isolation Forest on retained points.
4. Calibrate final IF scores using retained train score mean/std.
5. Score validation/test points only with the model for their local hour.

No anomaly labels are read until all 24 models are trained and all train/valid/test
scores are generated. Labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LABELING = REPO / "labeling"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"
MODEL_DIR = ROOT / "model"

if str(LABELING) not in sys.path:
    sys.path.insert(0, str(LABELING))

from tool import human_anomaly_mask, is_rate_metric, load_labels, load_plmn, metric_columns  # noqa: E402


def _load_metrics():
    path = REPO / "RiAD" / "metrics.py"
    spec = importlib.util.spec_from_file_location("paperib_metrics_hourif", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paper_metrics = _load_metrics()
evaluate_valid = paper_metrics.evaluate_riad_valid
resolve_vus_window = paper_metrics.resolve_vus_window


@dataclass
class Config:
    timezone: str = "Asia/Seoul"
    stage1_remove_pct: float = 5.0
    stage2_contamination_pct: float = 0.01
    threshold_sigma: float = 3.0
    n_estimators: int = 300
    max_samples: str | int = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    seed: int = 2
    sample_minutes: float = 5.0


@dataclass
class HourModel:
    hour: int
    stage1_scaler: RobustScaler
    stage1_model: IsolationForest
    final_scaler: RobustScaler
    final_model: IsolationForest
    clean_mask_local: np.ndarray
    score_mean: float
    score_std: float
    native_offset: float
    n_train: int
    n_removed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_indices(n: int, train_ratio: float = 0.6, valid_ratio: float = 0.2) -> tuple[int, int]:
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    return train_end, min(valid_end, n)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in metric_columns(df) if not is_rate_metric(c)]


def prepare_unlabeled_plmn(plmn: str, tz_name: str) -> dict[str, Any]:
    df = load_plmn(plmn)
    cols = feature_columns(df)
    if not cols:
        raise RuntimeError(f"{plmn}: no raw feature columns")

    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    times = pd.to_datetime(df["time"], utc=True)
    local_times = times.dt.tz_convert(tz_name)
    hours = local_times.dt.hour.to_numpy(dtype=np.int16)

    train_end, valid_end = split_indices(len(df))

    return {
        "df": df,
        "cols": cols,
        "values": values,
        "times": times,
        "local_times": local_times,
        "hours": hours,
        "train_end": train_end,
        "valid_end": valid_end,
        "train": values[:train_end],
        "valid": values[train_end:valid_end],
        "test": values[valid_end:],
        "train_hours": hours[:train_end],
        "valid_hours": hours[train_end:valid_end],
        "test_hours": hours[valid_end:],
        "split": {
            "train_time_end": str(times.iloc[train_end - 1]),
            "valid_time_start": str(times.iloc[train_end]),
            "valid_time_end": str(times.iloc[valid_end - 1]),
            "test_time_start": str(times.iloc[valid_end]),
            "test_time_end": str(times.iloc[-1]),
        },
    }


def contamination_fraction(percent: float) -> float:
    frac = float(percent) / 100.0
    if not (0.0 < frac <= 0.5):
        raise ValueError(f"contamination must be in (0,50%], got {percent}%")
    return frac


def build_if(contamination_pct: float, cfg: Config, random_state: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(cfg.n_estimators),
        max_samples=cfg.max_samples,
        contamination=contamination_fraction(contamination_pct),
        max_features=float(cfg.max_features),
        bootstrap=bool(cfg.bootstrap),
        random_state=int(random_state),
        n_jobs=-1,
    )


def raw_anomaly_score(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x).astype(np.float64)


def fit_hour_model(x_hour: np.ndarray, hour: int, cfg: Config) -> HourModel:
    n = len(x_hour)
    if n < 20:
        raise RuntimeError(f"hour {hour:02d}: too few train points ({n})")

    stage1_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    x1 = stage1_scaler.fit_transform(x_hour)

    stage1_model = build_if(
        cfg.stage1_remove_pct,
        cfg,
        cfg.seed + hour,
    )
    stage1_pred = stage1_model.fit_predict(x1)
    clean_mask = stage1_pred == 1

    if int(clean_mask.sum()) < max(20, int(0.5 * n)):
        raise RuntimeError(
            f"hour {hour:02d}: stage1 retained only {int(clean_mask.sum())}/{n}"
        )

    clean_x = x_hour[clean_mask]

    final_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    clean_x2 = final_scaler.fit_transform(clean_x)

    final_model = build_if(
        cfg.stage2_contamination_pct,
        cfg,
        cfg.seed + 1000 + hour,
    )
    final_model.fit(clean_x2)

    clean_scores = raw_anomaly_score(final_model, clean_x2)
    score_mean = float(np.mean(clean_scores))
    score_std = float(np.std(clean_scores))
    if not np.isfinite(score_std) or score_std < 1e-8:
        score_std = 1e-8

    return HourModel(
        hour=hour,
        stage1_scaler=stage1_scaler,
        stage1_model=stage1_model,
        final_scaler=final_scaler,
        final_model=final_model,
        clean_mask_local=clean_mask,
        score_mean=score_mean,
        score_std=score_std,
        native_offset=float(final_model.offset_),
        n_train=n,
        n_removed=int((~clean_mask).sum()),
    )


def score_with_hour_models(
    values: np.ndarray,
    hours: np.ndarray,
    models: dict[int, HourModel],
) -> tuple[np.ndarray, np.ndarray]:
    calibrated = np.full(len(values), np.nan, dtype=np.float64)
    raw = np.full(len(values), np.nan, dtype=np.float64)

    for hour in range(24):
        idx = np.flatnonzero(hours == hour)
        if idx.size == 0:
            continue
        hm = models[hour]
        x = hm.final_scaler.transform(values[idx])
        s = raw_anomaly_score(hm.final_model, x)
        raw[idx] = s
        calibrated[idx] = (s - hm.score_mean) / hm.score_std

    return calibrated, raw


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < len(m) and m[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def mask_to_labels(times: pd.Series, mask: np.ndarray, scores: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    t = pd.to_datetime(pd.Series(times), utc=True)
    items: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(segments(mask), start=1):
        ss = scores[a : b + 1]
        finite = ss[np.isfinite(ss)]
        items.append(
            {
                "id": f"{prefix}_{i:06d}",
                "kind": "range",
                "start": t.iloc[a].isoformat(),
                "end": t.iloc[b].isoformat(),
                "score": float(np.max(finite)) if finite.size else None,
                "source": prefix,
            }
        )
    return items


def metric_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    event = bundle["event"]
    point = bundle["point"]
    op = bundle["operational"]
    vus = bundle["vus"]
    return {
        "primary_metric": "vus_pr",
        "vus_pr": bundle["primary"]["vus_pr"],
        "pr_auc": bundle["pr_auc"],
        "r_auc_pr": vus.get("r_auc_pr"),
        "vus_window": bundle.get("vus_window"),
        "event_f1": event.get("f1"),
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "n_gt_events": event.get("n_gt_events"),
        "n_pred_events": event.get("n_pred_events"),
        "n_hit_gt": event.get("n_hit_gt"),
        "precision": point.get("precision"),
        "recall": point.get("recall"),
        "f1": point.get("f1"),
        "tp": point.get("tp"),
        "fp": point.get("fp"),
        "fn": point.get("fn"),
        "false_alarms_per_day": op.get("false_alarms_per_day"),
        "n_false_alarm_events": op.get("n_false_alarm_events"),
        "observation_days": op.get("observation_days"),
        "mean_detection_delay_min": op.get("mean_detection_delay_min"),
        "median_detection_delay_min": op.get("median_detection_delay_min"),
        "n_detected_events": op.get("n_detected"),
        "n_missed_events": op.get("n_missed"),
        "score_distribution": bundle.get("score_distribution"),
        "event_score_diagnostics": bundle.get("event_score_diagnostics"),
    }


def hour_summary(models: dict[int, HourModel], cfg: Config) -> list[dict[str, Any]]:
    rows = []
    for hour in range(24):
        hm = models[hour]
        rows.append(
            {
                "hour": hour,
                "n_train": hm.n_train,
                "n_removed_stage1": hm.n_removed,
                "removed_fraction": hm.n_removed / hm.n_train,
                "n_final_train": hm.n_train - hm.n_removed,
                "final_score_mean": hm.score_mean,
                "final_score_std": hm.score_std,
                "final_if_native_offset": hm.native_offset,
                "decision_threshold_z": cfg.threshold_sigma,
                "raw_threshold_equivalent": hm.score_mean + cfg.threshold_sigma * hm.score_std,
            }
        )
    return rows


def write_prediction(
    plmn: str,
    split_name: str,
    times: pd.Series,
    local_times: pd.Series,
    hours: np.ndarray,
    scores: np.ndarray,
    raw_scores: np.ndarray,
    threshold: float,
    bundle: dict[str, Any],
    cols: list[str],
    cfg: Config,
    models: dict[int, HourModel],
    split_meta: dict[str, str],
) -> str:
    pred = np.isfinite(scores) & (scores > threshold)
    labels = mask_to_labels(times, pred, scores, f"hour_if_{split_name}")

    payload = {
        "plmn": plmn,
        "source": f"hour_if_{split_name}",
        "model": "hour_if",
        "run_name": "24h_iterative",
        "eval_split": split_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_polarity": "higher_is_anomaly",
        "threshold": float(threshold),
        "threshold_method": "per_hour_clean_train_mean_plus_sigma",
        "timezone": cfg.timezone,
        "n_hour_models": 24,
        "feature_columns": cols,
        "feature_mode": "all_raw",
        "n_scores": int(np.isfinite(scores).sum()),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "split": split_meta,
        "labels": labels,
        "metrics": {
            **metric_payload(bundle),
            "test_held_out": split_name != "test",
            "train_labels_used": False,
            "validation_labels_used_for_training": False,
        },
        "training": {
            "method": "24_hour_contextual_iterative_decontamination",
            "config": asdict(cfg),
            "hour_models": hour_summary(models, cfg),
            "note": "All models, decontamination masks, calibration, and split scores are fixed before human labels are loaded.",
        },
        "score_series": {
            "times": [pd.Timestamp(x).isoformat() for x in times],
            "local_times": [pd.Timestamp(x).isoformat() for x in local_times],
            "hours": [int(x) for x in hours],
            "scores": [float(x) if np.isfinite(x) else None for x in scores],
            "raw_if_scores": [float(x) if np.isfinite(x) else None for x in raw_scores],
            "threshold": float(threshold),
        },
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{plmn}_hour_if_{split_name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def save_models(plmn: str, models: dict[int, HourModel], cols: list[str], cfg: Config) -> str:
    path = MODEL_DIR / plmn / "paperib" / "hour_if.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {"models": models, "feature_columns": cols, "config": asdict(cfg)},
            f,
        )
    return str(path)


def run(plmn: str, cfg: Config, eval_test: bool) -> dict[str, Any]:
    set_seed(cfg.seed)
    prep = prepare_unlabeled_plmn(plmn, cfg.timezone)

    print(
        f"plmn={plmn} features={len(prep['cols'])} timezone={cfg.timezone}",
        flush=True,
    )
    print(
        f"stage1 remove={cfg.stage1_remove_pct}% "
        f"stage2 contamination={cfg.stage2_contamination_pct}% "
        f"threshold=mean+{cfg.threshold_sigma}sigma",
        flush=True,
    )
    print("training 24 hourly models with NO anomaly labels loaded", flush=True)

    t0 = time.time()
    models: dict[int, HourModel] = {}

    for hour in range(24):
        idx = np.flatnonzero(prep["train_hours"] == hour)
        hm = fit_hour_model(prep["train"][idx], hour, cfg)
        models[hour] = hm
        print(
            f"  hour={hour:02d} train={hm.n_train} "
            f"removed={hm.n_removed} final={hm.n_train - hm.n_removed}",
            flush=True,
        )

    train_scores, _train_raw = score_with_hour_models(
        prep["train"], prep["train_hours"], models
    )
    valid_scores, valid_raw = score_with_hour_models(
        prep["valid"], prep["valid_hours"], models
    )
    test_scores, test_raw = score_with_hour_models(
        prep["test"], prep["test_hours"], models
    )

    threshold = float(cfg.threshold_sigma)
    model_path = save_models(plmn, models, prep["cols"], cfg)

    print(
        "training/scoring complete; loading expert labels for evaluation only",
        flush=True,
    )
    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(prep["df"], label_doc.get("labels")).astype(bool)
    y_valid = y_full[prep["train_end"] : prep["valid_end"]]
    y_test = y_full[prep["valid_end"] :]

    valid_vus_window = resolve_vus_window(y_valid, None)
    valid_bundle = evaluate_valid(
        y_valid,
        valid_scores,
        threshold,
        vus_window=valid_vus_window,
        sample_minutes=cfg.sample_minutes,
    )

    valid_times = prep["times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)
    valid_local_times = prep["local_times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)

    valid_path = write_prediction(
        plmn,
        "valid",
        valid_times,
        valid_local_times,
        prep["valid_hours"],
        valid_scores,
        valid_raw,
        threshold,
        valid_bundle,
        prep["cols"],
        cfg,
        models,
        prep["split"],
    )

    result: dict[str, Any] = {
        "plmn": plmn,
        "model_path": model_path,
        "valid_path": valid_path,
        "valid": metric_payload(valid_bundle),
        "elapsed_sec": time.time() - t0,
        "train_labels_used": False,
    }

    print(
        f"VALID VUS-PR={valid_bundle['primary']['vus_pr']} "
        f"PR-AUC={valid_bundle['pr_auc']} "
        f"Event-F1={valid_bundle['event']['f1']} "
        f"Event-Recall={valid_bundle['event']['recall']} "
        f"FA/day={valid_bundle['operational']['false_alarms_per_day']} "
        f"VUS-window={valid_vus_window}",
        flush=True,
    )
    print("GT event max-score percentiles:", flush=True)
    for row in valid_bundle.get("event_score_diagnostics") or []:
        pct = row.get("max_score_percentile")
        pct_text = "None" if pct is None else f"{100.0 * pct:.1f}%"
        print(
            f"  event {row['event_id']}: {pct_text} "
            f"detected={row['detected']} n_pred_points={row['n_pred_points']}",
            flush=True,
        )

    finite_train = train_scores[np.isfinite(train_scores)]
    if finite_train.size:
        print(
            f"train calibrated score: mean={np.mean(finite_train):.3f} "
            f"std={np.std(finite_train):.3f} "
            f"q99={np.quantile(finite_train, 0.99):.3f} "
            f"max={np.max(finite_train):.3f}",
            flush=True,
        )

    if eval_test:
        test_bundle = evaluate_valid(
            y_test,
            test_scores,
            threshold,
            vus_window=valid_vus_window,
            sample_minutes=cfg.sample_minutes,
        )
        test_times = prep["times"].iloc[prep["valid_end"] :].reset_index(drop=True)
        test_local_times = prep["local_times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)

        test_path = write_prediction(
            plmn,
            "test",
            test_times,
            test_local_times,
            prep["test_hours"],
            test_scores,
            test_raw,
            threshold,
            test_bundle,
            prep["cols"],
            cfg,
            models,
            prep["split"],
        )
        result["test"] = metric_payload(test_bundle)
        result["test_path"] = test_path

        print(
            f"TEST VUS-PR={test_bundle['primary']['vus_pr']} "
            f"PR-AUC={test_bundle['pr_auc']} "
            f"Event-F1={test_bundle['event']['f1']} "
            f"Event-Recall={test_bundle['event']['recall']} "
            f"FA/day={test_bundle['operational']['false_alarms_per_day']} "
            f"VUS-window={valid_vus_window} (fixed from validation)",
            flush=True,
        )

    return result


def parse_max_samples(text: str) -> str | int:
    if text == "auto":
        return "auto"
    try:
        value = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--max_samples must be auto or an integer"
        ) from e
    if value < 1:
        raise argparse.ArgumentTypeError("--max_samples integer must be >= 1")
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--timezone", default="Asia/Seoul")
    p.add_argument("--stage1_remove_pct", type=float, default=5.0)
    p.add_argument("--stage2_contamination_pct", type=float, default=0.01)
    p.add_argument("--threshold_sigma", type=float, default=3.0)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_samples", type=parse_max_samples, default="auto")
    p.add_argument("--max_features", type=float, default=1.0)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="also evaluate held-out test after validation choices are frozen",
    )
    args = p.parse_args()

    if not (0.0 < args.stage1_remove_pct <= 50.0):
        p.error("--stage1_remove_pct must be in (0,50]")
    if not (0.0 < args.stage2_contamination_pct <= 50.0):
        p.error("--stage2_contamination_pct must be in (0,50]")
    if args.threshold_sigma <= 0:
        p.error("--threshold_sigma must be > 0")
    if args.n_estimators < 10:
        p.error("--n_estimators must be >= 10")
    if not (0.0 < args.max_features <= 1.0):
        p.error("--max_features must be in (0,1]")

    cfg = Config(
        timezone=args.timezone,
        stage1_remove_pct=args.stage1_remove_pct,
        stage2_contamination_pct=args.stage2_contamination_pct,
        threshold_sigma=args.threshold_sigma,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    run(args.plmn, cfg, args.eval_test)


if __name__ == "__main__":
    main()
