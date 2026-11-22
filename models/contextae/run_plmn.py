#!/usr/bin/env python3
"""TRAIN 데이터에서 시간 context를 자동으로 찾는 ContextAE 실행 코드.

HourAE와 비교했을 때 AE 구조와 decontamination 방식은 그대로 두고,
시간대를 나누는 방식만 데이터 기반 adaptive segmentation으로 바꾼 실험이다.

전체 흐름:
1. 데이터를 시간순으로 TRAIN / VALID / TEST로 분할한다.
2. TRAIN만 사용해 하루의 각 5분 slot을 robust temporal profile로 표현한다.
3. 하루 288개 slot을 K개의 연속 시간 context로 나눈다.
4. 각 K에 대해 Dynamic Programming으로 구간 내부 SSE가 최소인 연속 분할을 찾는다.
5. --k를 고정하지 않았다면 TRAIN-only silhouette score가 가장 높은 K를 선택한다.
6. 각 context마다 iterative decontamination AutoEncoder를 학습한다.
7. 각 point는 자신의 5분 slot이 속한 context의 AutoEncoder로 score를 계산한다.
8. 모든 모델 학습과 score 생성이 끝난 뒤에만 전문가 label을 읽어 평가한다.
9. 같은 하이퍼파라미터의 고정 24시간 HourAE baseline도 함께 학습해 VALID 성능을 비교한다.

Anomaly score는 11개 core feature의 대칭적인 reconstruction MSE를 사용한다.
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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "ContextAE"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"
MODEL_DIR = ROOT / "model"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.labels import human_anomaly_mask, load_labels
from core.metrics import evaluate_valid, resolve_vus_window


def _load_hourae():
    """고정 24시간 분할 baseline인 새 HourAE runner 모듈을 불러온다."""
    from models.hourae import run_plmn as mod
    return mod


from models.contextae.model import PointAutoEncoder
from models.contextae.context import (
    robust_mad,
    build_train_slot_profiles,
    build_segment_cost_matrix,
    optimal_contiguous_partition,
    fit_context_segmentation,
    context_ids_from_slots,
    slot_label,
)
from models.contextae.training import (
    Config,
    ContextArtifacts,
    device_for_run,
    split_indices,
    feature_columns,
    local_slot_ids,
    prepare_unlabeled_plmn,
    fit_scaler,
    train_ae,
    reconstruction_mse,
    quantile_for_contamination,
    fit_context_model,
    build_final_model,
    score_with_context_models,
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


def context_summary(
    artifacts: dict[int, ContextArtifacts],
    slot_to_context: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for context_id in sorted(artifacts):
        art = artifacts[context_id]
        slots = np.flatnonzero(slot_to_context == context_id)
        rows.append(
            {
                "context_id": int(context_id),
                "n_slots": int(len(slots)),
                "slots": [slot_label(int(s)) for s in slots],
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


def segmentation_payload(
    segment_info: dict[str, Any],
) -> dict[str, Any]:
    slot_to_context = segment_info["slot_to_context"]
    return {
        "method": "train_only_robust_profile_contiguous_dp",
        "selected_k": int(segment_info["selected_k"]),
        "selected_silhouette": float(
            segment_info["selected_silhouette"]
        ),
        "candidate_scores": segment_info["candidate_scores"],
        "selected_within_sse": float(segment_info["selected_within_sse"]),
        "segments": segment_info["segments"],
        "missing_train_slots": [
            slot_label(int(x)) for x in segment_info["missing_slots"]
        ],
        "slot_to_context": {
            slot_label(slot): int(slot_to_context[slot])
            for slot in range(288)
        },
    }


def save_checkpoint(
    plmn: str,
    artifacts: dict[int, ContextArtifacts],
    cols: list[str],
    cfg: Config,
    segment_info: dict[str, Any],
) -> str:
    path = MODEL_DIR / plmn / "paperib" / "context_ae.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "feature_columns": cols,
            "config": asdict(cfg),
            "segmentation": segmentation_payload(segment_info),
            "contexts": {
                str(context_id): {
                    "final_state": art.final_state,
                    "stage1_state": art.stage1_state,
                    "stage1_scaler": art.stage1_scaler,
                    "final_scaler": art.final_scaler,
                    "stage1_threshold_mse": art.stage1_threshold_mse,
                    "final_score_mean": art.final_score_mean,
                    "final_score_std": art.final_score_std,
                    "final_reference_quantile_mse": (
                        art.final_reference_quantile_mse
                    ),
                    "final_reference_quantile_z": (
                        art.final_reference_quantile_z
                    ),
                    "n_train": art.n_train,
                    "n_removed": art.n_removed,
                }
                for context_id, art in artifacts.items()
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
    slots: np.ndarray,
    context_ids: np.ndarray,
    scores: np.ndarray,
    raw_mse: np.ndarray,
    threshold: float,
    bundle: dict[str, Any],
    cols: list[str],
    cfg: Config,
    artifacts: dict[int, ContextArtifacts],
    segment_info: dict[str, Any],
    split_meta: dict[str, str],
) -> str:
    pred = np.isfinite(scores) & (scores > threshold)
    labels = mask_to_labels(
        times,
        pred,
        scores,
        f"context_ae_{split_name}",
    )

    payload = {
        "plmn": plmn,
        "source": f"context_ae_{split_name}",
        "model": "context_ae",
        "run_name": "k_context_iterative",
        "eval_split": split_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "score_polarity": "higher_is_anomaly",
        "threshold": float(threshold),
        "threshold_method": (
            "per_context_clean_train_mse_mean_plus_sigma"
        ),
        "timezone": cfg.timezone,
        "n_context_models": int(segment_info["selected_k"]),
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
            "validation_labels_used_for_context_selection": False,
        },
        "segmentation": segmentation_payload(segment_info),
        "training": {
            "method": (
                "train_only_contiguous_context_segmentation_plus_"
                "iterative_decontamination_autoencoder"
            ),
            "config": asdict(cfg),
            "context_models": context_summary(
                artifacts,
                segment_info["slot_to_context"],
                cfg,
            ),
            "note": (
                "K and contiguous time boundaries are determined from TRAIN profiles "
                "only. Anomaly scoring uses symmetric mean squared reconstruction error."
            ),
        },
        "score_series": {
            "times": [pd.Timestamp(x).isoformat() for x in times],
            "local_times": [
                pd.Timestamp(x).isoformat() for x in local_times
            ],
            "slots": [slot_label(int(x)) for x in slots],
            "context_ids": [int(x) for x in context_ids],
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
    path = PRED_DIR / f"{plmn}_context_ae_{split_name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def print_context_map(
    segment_info: dict[str, Any],
    train_context_ids: np.ndarray,
) -> None:
    k = int(segment_info["selected_k"])
    print(
        f"selected K={k} "
        f"silhouette={segment_info['selected_silhouette']:.4f} "
        f"within_SSE={segment_info['selected_within_sse']:.4f}",
        flush=True,
    )

    print("K candidates (TRAIN-only contiguous segmentation):", flush=True)
    for row in segment_info["candidate_scores"]:
        mark = "*" if row["k"] == k else " "
        print(
            f" {mark} K={row['k']:02d} "
            f"sil={row['silhouette']:.4f} "
            f"SSE={row['within_sse']:.2f} "
            f"min_slots={row['min_segment_slots']}",
            flush=True,
        )

    print("discovered contiguous contexts:", flush=True)
    for seg in segment_info["segments"]:
        c = int(seg["context_id"])
        n_train = int(np.sum(train_context_ids == c))
        print(
            f"  context={c} {seg['start']} ~ {seg['end']} "
            f"slots={seg['n_slots']} train_points={n_train}",
            flush=True,
        )



def _metric_delta(
    context_metrics: dict[str, Any],
    hour_metrics: dict[str, Any],
    key: str,
) -> float | None:
    a = context_metrics.get(key)
    b = hour_metrics.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def print_hour24_comparison(
    *,
    selected_k: int,
    context_metrics: dict[str, Any],
    hour_metrics: dict[str, Any],
) -> None:
    """ContextAE와 HourAE의 validation metric을 한눈에 비교할 수 있는 간단한 표를 출력한다."""
    print("", flush=True)
    print("=" * 86, flush=True)
    print(
        f"VALID COMPARISON: adaptive K={selected_k} contexts vs fixed 24 hourly AEs",
        flush=True,
    )
    print("=" * 86, flush=True)
    print(
        f"{'metric':<20} {'adaptive-K':>16} {'hour-24':>16} {'delta(K-24)':>16}",
        flush=True,
    )

    rows = [
        ("VUS-PR", "vus_pr"),
        ("PR-AUC", "pr_auc"),
        ("Event-F1", "event_f1"),
        ("Event-Recall", "event_recall"),
        ("Point-F1", "f1"),
        ("FA/day", "false_alarms_per_day"),
    ]

    for label, key in rows:
        a = context_metrics.get(key)
        b = hour_metrics.get(key)
        d = _metric_delta(context_metrics, hour_metrics, key)

        def fmt(v: Any) -> str:
            return "None" if v is None else f"{float(v):.6f}"

        print(
            f"{label:<20} {fmt(a):>16} {fmt(b):>16} {fmt(d):>16}",
            flush=True,
        )

    print("=" * 86, flush=True)


def run_hour24_baseline(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool,
) -> dict[str, Any]:
    """공정한 비교를 위해 ContextAE와 동일한 AE/decontamination 설정으로 기존 HourAE baseline을 다시 학습한다."""
    hourae = _load_hourae()
    hour_cfg = hourae.Config(
        timezone=cfg.timezone,
        stage1_remove_pct=cfg.stage1_remove_pct,
        final_contamination_pct=cfg.final_contamination_pct,
        threshold_sigma=cfg.threshold_sigma,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        stage1_epochs=cfg.stage1_epochs,
        final_epochs=cfg.final_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        sample_minutes=cfg.sample_minutes,
    )
    return hourae.run(
        plmn=plmn,
        cfg=hour_cfg,
        eval_test=eval_test,
    )


def run(
    *,
    plmn: str,
    cfg: Config,
    eval_test: bool,
    compare_hour24: bool = True,
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
        "building robust 5-minute slot profiles and contiguous contexts from TRAIN only",
        flush=True,
    )

    slot_info = build_train_slot_profiles(
        prep["train"],
        prep["train_slots"],
        use_mad=cfg.profile_use_mad,
    )
    segment_info = fit_context_segmentation(slot_info, cfg)

    train_context_ids = context_ids_from_slots(
        prep["train_slots"],
        segment_info["slot_to_context"],
    )
    valid_context_ids = context_ids_from_slots(
        prep["valid_slots"],
        segment_info["slot_to_context"],
    )
    test_context_ids = context_ids_from_slots(
        prep["test_slots"],
        segment_info["slot_to_context"],
    )

    print_context_map(segment_info, train_context_ids)

    print(
        f"stage1 remove={cfg.stage1_remove_pct}% "
        f"final reference contamination={cfg.final_contamination_pct}% "
        f"threshold=mean+{cfg.threshold_sigma}sigma",
        flush=True,
    )
    print(
        "training K context AutoEncoders with NO anomaly labels loaded",
        flush=True,
    )

    t0 = time.time()
    artifacts: dict[int, ContextArtifacts] = {}

    for context_id in range(int(segment_info["selected_k"])):
        idx = np.flatnonzero(train_context_ids == context_id)
        art = fit_context_model(
            prep["train"][idx],
            context_id=context_id,
            cfg=cfg,
            device=device,
        )
        artifacts[context_id] = art

        print(
            f"  context={context_id} train={art.n_train} "
            f"removed={art.n_removed} final={art.n_train - art.n_removed} "
            f"stage1_loss={art.stage1_final_loss:.6f} "
            f"final_loss={art.final_final_loss:.6f}",
            flush=True,
        )

    train_scores, _train_raw = score_with_context_models(
        prep["train"],
        train_context_ids,
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    valid_scores, valid_raw = score_with_context_models(
        prep["valid"],
        valid_context_ids,
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    test_scores, test_raw = score_with_context_models(
        prep["test"],
        test_context_ids,
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
        segment_info,
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
        slots=prep["valid_slots"],
        context_ids=valid_context_ids,
        scores=valid_scores,
        raw_mse=valid_raw,
        threshold=threshold,
        bundle=valid_bundle,
        cols=prep["cols"],
        cfg=cfg,
        artifacts=artifacts,
        segment_info=segment_info,
        split_meta=prep["split"],
    )

    result: dict[str, Any] = {
        "plmn": plmn,
        "selected_k": int(segment_info["selected_k"]),
        "selected_silhouette": float(
            segment_info["selected_silhouette"]
        ),
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
            slots=prep["test_slots"],
            context_ids=test_context_ids,
            scores=test_scores,
            raw_mse=test_raw,
            threshold=threshold,
            bundle=test_bundle,
            cols=prep["cols"],
            cfg=cfg,
            artifacts=artifacts,
            segment_info=segment_info,
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

    if compare_hour24:
        print("", flush=True)
        print(
            "training fixed 24-hour HourAE baseline with identical AE hyperparameters",
            flush=True,
        )
        hour24_result = run_hour24_baseline(
            plmn=plmn,
            cfg=cfg,
            eval_test=eval_test,
        )
        result["hour24_baseline"] = hour24_result

        print_hour24_comparison(
            selected_k=int(segment_info["selected_k"]),
            context_metrics=result["valid"],
            hour_metrics=hour24_result["valid"],
        )


    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--timezone", default="Asia/Seoul")

    p.add_argument(
        "--k",
        type=int,
        default=0,
        help="fixed number of contiguous segments; 0 means train-only automatic K",
    )
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=24)
    p.add_argument("--min_segment_slots", type=int, default=3)
    p.add_argument(
        "--median_only_profile",
        action="store_true",
        help="segment slots by median only instead of median+MAD",
    )

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
        "--skip_hour24_compare",
        action="store_true",
        help="skip retraining/comparison against the fixed 24-hour HourAE baseline",
    )
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="also evaluate held-out test after validation choices are frozen",
    )
    args = p.parse_args()

    if args.k < 0:
        p.error("--k must be 0 (auto) or >= 2")
    if args.k == 1:
        p.error("--k=1 is the global-AE baseline, not context segmentation")
    if args.k_min < 2:
        p.error("--k_min must be >= 2")
    if args.k_max < args.k_min:
        p.error("--k_max must be >= k_min")
    if args.min_segment_slots < 1:
        p.error("--min_segment_slots must be >= 1")

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
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        min_segment_slots=args.min_segment_slots,
        profile_use_mad=not args.median_only_profile,
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
        compare_hour24=not args.skip_hour24_compare,
    )


if __name__ == "__main__":
    main()
