#!/usr/bin/env python3
"""하루를 24개 고정 시간대로 나누는 HourAE 실행 코드.

각 local hour h ∈ {0,...,23}에 대해 다음 과정을 독립적으로 수행한다.

1. 해당 hour의 모든 unlabeled TRAIN point로 Stage-1 AutoEncoder를 학습한다.
2. TRAIN point별 reconstruction MSE를 계산한다.
3. reconstruction error 상위 q%(기본 5%)를 anomaly candidate로 제거한다.
4. 남은 point만 사용해 새로운 Final AutoEncoder를 처음부터 다시 학습한다.
5. retained TRAIN의 최종 reconstruction error 분포를 계산한다.
6. 각 point score를 z=(MSE-mean_h)/std_h 형태로 보정한다.
7. z > threshold_sigma이면 anomaly point로 판단한다.

final_contamination_pct(기본 0.01%)는 과거 구현과의 비교를 위해 남겨둔
참고용 upper-tail quantile이다. AutoEncoder 학습 weight를 바꾸는 contamination
파라미터가 아니며, 실제 기본 decision threshold는 clean TRAIN의 mean+sigma 규칙을 사용한다.

모델 학습, decontamination, score 생성에는 anomaly label을 사용하지 않는다.
전문가 label은 모든 score가 결정된 뒤 VALID/TEST 평가에만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "HourAE"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"
MODEL_DIR = ROOT / "model"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.labels import human_anomaly_mask, load_labels
from core.metrics import evaluate_valid, resolve_vus_window


from models.hourae.model import PointAutoEncoder
from models.hourae.training import (
    Config,
    HourArtifacts,
    set_seed,
    device_for_run,
    split_indices,
    prepare_unlabeled_plmn,
    fit_scaler,
    train_ae,
    reconstruction_mse,
    cpu_state_dict,
    quantile_for_contamination,
    fit_hour,
    build_final_model,
    score_with_hour_models,
)

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


def mask_to_labels(
    times: pd.Series,
    mask: np.ndarray,
    scores: np.ndarray,
    prefix: str,
) -> list[dict[str, Any]]:
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


def hour_summary(
    artifacts: dict[int, HourArtifacts],
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for hour in range(24):
        art = artifacts[hour]
        rows.append(
            {
                "hour": hour,
                "n_train": art.n_train,
                "n_removed_stage1": art.n_removed,
                "removed_fraction": art.n_removed / art.n_train,
                "n_final_train": art.n_train - art.n_removed,
                "stage1_threshold_mse": art.stage1_threshold_mse,
                "final_train_mse_mean": art.final_score_mean,
                "final_train_mse_std": art.final_score_std,
                "final_reference_contamination_pct": cfg.final_contamination_pct,
                "final_reference_quantile_mse": art.final_reference_quantile_mse,
                "final_reference_quantile_z": art.final_reference_quantile_z,
                "decision_threshold_sigma": cfg.threshold_sigma,
                "decision_threshold_mse": (
                    art.final_score_mean
                    + cfg.threshold_sigma * art.final_score_std
                ),
                "stage1_final_train_loss": art.stage1_final_loss,
                "final_final_train_loss": art.final_final_loss,
            }
        )

    return rows


def save_checkpoint(
    plmn: str,
    artifacts: dict[int, HourArtifacts],
    cols: list[str],
    cfg: Config,
) -> str:
    path = MODEL_DIR / plmn / "paperib" / "hour_ae.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "feature_columns": cols,
            "config": asdict(cfg),
            "hours": {
                str(hour): {
                    "final_state": art.final_state,
                    "stage1_state": art.stage1_state,
                    "stage1_scaler": art.stage1_scaler,
                    "final_scaler": art.final_scaler,
                    "stage1_threshold_mse": art.stage1_threshold_mse,
                    "final_score_mean": art.final_score_mean,
                    "final_score_std": art.final_score_std,
                    "final_reference_quantile_mse": art.final_reference_quantile_mse,
                    "final_reference_quantile_z": art.final_reference_quantile_z,
                    "n_train": art.n_train,
                    "n_removed": art.n_removed,
                }
                for hour, art in artifacts.items()
            },
        },
        path,
    )
    return str(path)


def write_prediction(
    *,
    plmn: str,
    split_name: str,
    times: pd.Series,
    local_times: pd.Series,
    hours: np.ndarray,
    scores: np.ndarray,
    raw_mse: np.ndarray,
    threshold: float,
    bundle: dict[str, Any],
    cols: list[str],
    cfg: Config,
    artifacts: dict[int, HourArtifacts],
    split_meta: dict[str, str],
) -> str:
    pred = np.isfinite(scores) & (scores > threshold)
    labels = mask_to_labels(
        times,
        pred,
        scores,
        f"hour_ae_{split_name}",
    )

    payload = {
        "plmn": plmn,
        "source": f"hour_ae_{split_name}",
        "model": "hour_ae",
        "run_name": "24h_iterative",
        "eval_split": split_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_polarity": "higher_is_anomaly",
        "threshold": float(threshold),
        "threshold_method": "per_hour_clean_train_mse_mean_plus_sigma",
        "timezone": cfg.timezone,
        "n_hour_models": 24,
        "feature_columns": cols,
        "feature_mode": "core",
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
            "method": "24_hour_autoencoder_iterative_decontamination",
            "config": asdict(cfg),
            "hour_models": hour_summary(artifacts, cfg),
            "note": (
                "Stage-1 removes the top reconstruction-error tail. A fresh "
                "AutoEncoder is then trained on retained train points. "
                "All model fitting and split scoring finish before labels load."
            ),
        },
        "score_series": {
            "times": [pd.Timestamp(x).isoformat() for x in times],
            "local_times": [
                pd.Timestamp(x).isoformat() for x in local_times
            ],
            "hours": [int(x) for x in hours],
            "scores": [
                float(x) if np.isfinite(x) else None
                for x in scores
            ],
            "raw_reconstruction_mse": [
                float(x) if np.isfinite(x) else None
                for x in raw_mse
            ],
            "threshold": float(threshold),
        },
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{plmn}_hour_ae_{split_name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def run(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = device_for_run()
    prep = prepare_unlabeled_plmn(plmn, cfg.timezone)

    print(
        f"device={device} plmn={plmn} features={len(prep['cols'])} "
        f"timezone={cfg.timezone}",
        flush=True,
    )
    print(
        f"stage1 remove={cfg.stage1_remove_pct}% "
        f"final reference contamination={cfg.final_contamination_pct}% "
        f"threshold=mean+{cfg.threshold_sigma}sigma",
        flush=True,
    )
    print(
        "training 24 hourly AutoEncoders with NO anomaly labels loaded",
        flush=True,
    )

    t0 = time.time()
    artifacts: dict[int, HourArtifacts] = {}

    for hour in range(24):
        idx = np.flatnonzero(prep["train_hours"] == hour)
        art = fit_hour(
            prep["train"][idx],
            hour=hour,
            cfg=cfg,
            device=device,
        )
        artifacts[hour] = art

        print(
            f"  hour={hour:02d} train={art.n_train} "
            f"removed={art.n_removed} final={art.n_train - art.n_removed} "
            f"stage1_loss={art.stage1_final_loss:.6f} "
            f"final_loss={art.final_final_loss:.6f}",
            flush=True,
        )

    train_scores, _train_raw = score_with_hour_models(
        prep["train"],
        prep["train_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    valid_scores, valid_raw = score_with_hour_models(
        prep["valid"],
        prep["valid_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    test_scores, test_raw = score_with_hour_models(
        prep["test"],
        prep["test_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )

    threshold = float(cfg.threshold_sigma)
    model_path = save_checkpoint(
        plmn,
        artifacts,
        prep["cols"],
        cfg,
    )

    print(
        "training/scoring complete; loading expert labels for evaluation only",
        flush=True,
    )
    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(
        prep["df"],
        label_doc.get("labels"),
    ).astype(bool)
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
        plmn=plmn,
        split_name="valid",
        times=valid_times,
        local_times=valid_local_times,
        hours=prep["valid_hours"],
        scores=valid_scores,
        raw_mse=valid_raw,
        threshold=threshold,
        bundle=valid_bundle,
        cols=prep["cols"],
        cfg=cfg,
        artifacts=artifacts,
        split_meta=prep["split"],
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
            f"detected={row['detected']} "
            f"n_pred_points={row['n_pred_points']}",
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
        test_times = prep["times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)
        test_local_times = prep["local_times"].iloc[
            prep["valid_end"] :
        ].reset_index(drop=True)

        test_path = write_prediction(
            plmn=plmn,
            split_name="test",
            times=test_times,
            local_times=test_local_times,
            hours=prep["test_hours"],
            scores=test_scores,
            raw_mse=test_raw,
            threshold=threshold,
            bundle=test_bundle,
            cols=prep["cols"],
            cfg=cfg,
            artifacts=artifacts,
            split_meta=prep["split"],
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--timezone", default="Asia/Seoul")
    p.add_argument("--stage1_remove_pct", type=float, default=5.0)
    p.add_argument("--final_contamination_pct", type=float, default=0.01)
    p.add_argument("--threshold_sigma", type=float, default=3.0)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--stage1_epochs", type=int, default=50)
    p.add_argument("--final_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="also evaluate held-out test after validation choices are frozen",
    )
    args = p.parse_args()

    if not (0.0 < args.stage1_remove_pct < 50.0):
        p.error("--stage1_remove_pct must be in (0,50)")
    if not (0.0 < args.final_contamination_pct < 50.0):
        p.error("--final_contamination_pct must be in (0,50)")
    if args.threshold_sigma <= 0:
        p.error("--threshold_sigma must be > 0")
    if args.hidden_dim < 2:
        p.error("--hidden_dim must be >= 2")
    if args.latent_dim < 1 or args.latent_dim >= args.hidden_dim:
        p.error("--latent_dim must be >=1 and < hidden_dim")
    if args.stage1_epochs < 1 or args.final_epochs < 1:
        p.error("epochs must be >= 1")
    if args.batch_size < 1:
        p.error("--batch_size must be >= 1")
    if args.lr <= 0:
        p.error("--lr must be > 0")

    cfg = Config(
        timezone=args.timezone,
        stage1_remove_pct=args.stage1_remove_pct,
        final_contamination_pct=args.final_contamination_pct,
        threshold_sigma=args.threshold_sigma,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        stage1_epochs=args.stage1_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    run(
        plmn=args.plmn,
        cfg=cfg,
        eval_test=args.eval_test,
    )


if __name__ == "__main__":
    main()
