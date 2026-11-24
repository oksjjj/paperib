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

모델 학습, decontamination, score 생성에는 anomaly label을 사용하지 않는다.
전문가 label은 모든 score가 결정된 뒤 VALID/TEST 평가에만 사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "hourae" / "model"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.data import prepare_unlabeled_base
from core.labels import human_anomaly_mask, load_labels
from core.metrics import evaluate_valid, resolve_vus_window
from core.predictions import mask_to_labels, metric_payload
from models.pointae import (
    DecontaminatedAEArtifacts,
    device_for_run,
    fit_decontaminated_group,
    score_grouped_models,
    set_seed,
)


from models.hourae.training import Config

def hour_summary(
    artifacts: dict[int, DecontaminatedAEArtifacts],
    cfg: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # 하루를 고정 24시간대로 나눈 후 같은 hour에 속한 TRAIN 행만 모아 독립 AE를 학습한다.
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
    artifacts: dict[int, DecontaminatedAEArtifacts],
    cols: list[str],
    cfg: Config,
) -> str:
    # 학습 결과를 PLMN별 .pt 한 파일에 묶어 저장한다.
    # 경로 예: MODEL_DIR/{plmn}/paperib/hour_ae.pt
    path = MODEL_DIR / plmn / "paperib" / "hour_ae.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    # torch.save에 넣는 dict 구조:
    #   feature_columns — 학습에 쓴 metric 컬럼 이름 목록 (추론 시 동일 순서 필요)
    #   config          — Config를 dict로 변환한 하이퍼파라미터
    #   hours           — 시간대(0~23)별 모델·스케일러·점수 통계
    #     final_state / stage1_state — AE 가중치 (추론은 주로 final_state)
    #     stage1_scaler / final_scaler — 각 단계 RobustScaler
    #     stage1_threshold_mse — Stage-1에서 상위 % 자를 때 쓴 MSE 경계
    #     final_score_mean / final_score_std — clean TRAIN MSE의 평균·표준편차 (z-score용)
    #     n_train / n_removed — 원래 TRAIN 수, Stage-1에서 제거한 수
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
    artifacts: dict[int, DecontaminatedAEArtifacts],
    split_meta: dict[str, str],
) -> str:
    # NaN 등 점수가 없는 위치는 제외하고 TRAIN 기준 임계치를 넘는 시점만 이상으로 예측한다.
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
    # 실행 장치 결정 후 라벨 없이 원본 데이터와 시간순 split을 준비한다.
    device = device_for_run()
    prep = prepare_unlabeled_base(plmn, cfg.timezone)
    hours = prep["local_times"].dt.hour.to_numpy(dtype=np.int16)
    a, b = prep["train_end"], prep["valid_end"]
    prep["train_hours"] = hours[:a]
    prep["valid_hours"] = hours[a:b]
    prep["test_hours"] = hours[b:]

    print(
        f"device={device} plmn={plmn} features={len(prep['cols'])} "
        f"timezone={cfg.timezone}",
        flush=True,
    )
    print(
        f"stage1 remove={cfg.stage1_remove_pct}% "        f"threshold=mean+{cfg.threshold_sigma}sigma",
        flush=True,
    )
    print(
        "training 24 hourly AutoEncoders with NO anomaly labels loaded",
        flush=True,
    )

    t0 = time.time()
    artifacts: dict[int, DecontaminatedAEArtifacts] = {}

    for hour in range(24):
        idx = np.flatnonzero(prep["train_hours"] == hour)
        # HourAE에서는 hour 번호를 group_seed로 사용한다.
        # 공통 fit_decontaminated_group() 안에서
        #   Stage-1 seed = cfg.seed + hour
        #   Final seed   = cfg.seed + 1000 + hour
        # 로 계산된다. 따라서 hour별 AE끼리, 그리고 같은 hour의 Stage-1/Final끼리
        # 서로 다른 초기 weight와 shuffle 순서에서 시작한다.
        art = fit_decontaminated_group(
            prep["train"][idx],
            cfg=cfg,
            device=device,
            group_seed=hour,
            group_name=f"hour {hour:02d}",
        )
        artifacts[hour] = art

        print(
            f"  hour={hour:02d} train={art.n_train} "
            f"removed={art.n_removed} final={art.n_train - art.n_removed} "
            f"stage1_loss={art.stage1_final_loss:.6f} "
            f"final_loss={art.final_final_loss:.6f}",
            flush=True,
        )

    # 모델 학습이 끝난 후 각 split의 모든 시점에 해당 그룹의 최종 모델을 적용한다.
    # TRAIN의 제거된 시점도 여기서 점수를 계산한다. 추론이므로 파라미터를 갱신하지 않는다.
    train_scores, _train_raw = score_grouped_models(
        prep["train"],
        prep["train_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    valid_scores, valid_raw = score_grouped_models(
        prep["valid"],
        prep["valid_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )
    test_scores, test_raw = score_grouped_models(
        prep["test"],
        prep["test_hours"],
        artifacts,
        n_features=len(prep["cols"]),
        cfg=cfg,
        device=device,
    )

    # 그룹별 clean TRAIN 평균/표준편차로 이미 z-score를 만들었으므로 공통 z 임계치를 사용한다.
    threshold = float(cfg.threshold_sigma)
    # 후속 임계치 실험에서 모델을 다시 학습할 필요가 없도록 scaler와 가중치 등을 저장한다.
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
    # 중요: 학습과 점수 산출이 완료된 뒤에야 전문가 GT를 읽는다. 아래에서는 성능 평가에만 사용한다.
    label_doc = load_labels(plmn)
    y_full = human_anomaly_mask(
        prep["df"],
        label_doc.get("labels"),
    ).astype(bool)
    # 점수와 GT가 동일한 시간순 split 및 인덱스를 사용하도록 원본 마스크를 자른다.
    y_valid = y_full[prep["train_end"] : prep["valid_end"]]
    y_test = y_full[prep["valid_end"] :]

    # VUS 평가용 범위 설정과 운영 threshold 기반 point/event 성능을 공통 평가 함수에서 계산한다.
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

    # UI에서 동일 시점의 점수, threshold, 예측 이벤트와 평가 결과를 확인할 수 있도록 저장한다.
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

    # TEST 평가를 요청한 경우에만 최종 TEST 결과를 계산하고 저장한다.
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

    # argparse로 받은 실험 설정을 Config에 전달한다. 실제 모델 학습은 run()에서 시작한다.
    cfg = Config(
        timezone=args.timezone,
        stage1_remove_pct=args.stage1_remove_pct,
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
