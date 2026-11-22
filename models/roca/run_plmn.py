#!/usr/bin/env python3
"""한 사업자의 contaminated train data로 RoCA를 학습하고 validation 결과를 저장한다.

원본 ruiking04/RoCA의 network와 loss를 사용하되 Merlion 의존성은 제거하고,
paperib의 공통 데이터/평가 형식에 연결한 실행 wrapper다.

전체 흐름:
    사업자 데이터 로드
    → train window 생성
    → weak/strong augmentation 생성
    → RoCA backbone 학습
    → hypersphere center 기준 anomaly score 계산
    → TRAIN score quantile로 label-free threshold 결정
    → VALID score를 전문가 label과 비교
    → UI용 prediction JSON 저장

실행 예:
  ../.venv/bin/python run_plmn.py --plmn P0480 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.roca.config import RoCAConfig
from models.roca.dataset import RoCAWindowDataset, block_to_nct
from models.roca.export_predictions import write_roca_prediction
from core.metrics import evaluate_valid
from models.roca.official.loss import center_c, roca_loss
from models.roca.official.network import base_Model

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "RoCA"
MODEL_ROOT = ROOT / "model"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.data import prepare_plmn


def _device() -> torch.device:
    """사용 가능한 가속기 중 MPS → CUDA → CPU 순서로 선택한다."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_quantile_threshold(train_scores: np.ndarray, quantile: float = 0.99) -> float:
    """TRAIN anomaly score만 이용해 label-free quantile threshold를 계산한다.

    예를 들어 quantile=0.99이면 train score의 99 percentile을 threshold로 사용한다.
    validation label은 threshold 계산에 사용하지 않는다.
    """
    finite = train_scores[np.isfinite(train_scores)]
    if finite.size == 0:
        raise RuntimeError("no finite train scores for threshold")
    q = max(0.0, min(1.0, float(quantile)))
    return float(np.quantile(finite, q))


@torch.no_grad()
def score_block(
    model: base_Model,
    center: torch.Tensor,
    block: np.ndarray,
    *,
    cfg: RoCAConfig,
    device: torch.device,
    time_step: int | None = None,
) -> np.ndarray:
    """연속 block을 window로 나누고 각 window의 loss_n을 마지막 point에 배치한다.

    RoCA는 window 하나당 하나의 score를 만든다.
    따라서 window가 원본 시계열에서 끝나는 위치에 score를 기록하여
    최종적으로 원본 길이와 같은 point-wise score 배열을 만든다.

    loss_n은 hypersphere center와의 거리 기반 값이며 클수록 anomaly에 가깝다.
    window가 아직 완성되지 않는 앞부분은 NaN으로 남는다.
    """
    model.eval()
    step = int(time_step if time_step is not None else cfg.time_step)
    t = len(block)
    scores = np.full(t, np.nan, dtype=np.float64)
    if t < cfg.window_size:
        return scores
    # (T, F) block을 Conv1d 입력인 (N, F, window) 형태로 변환한다.
    # ends에는 각 window가 원본 block의 어느 point에서 끝나는지가 저장된다.
    wins, ends = block_to_nct(block, window=cfg.window_size, time_step=step)
    if wins.shape[0] == 0:
        return scores
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(wins)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    outs: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.float().to(device)

        # backbone은 원본 window representation과 복원 representation을 반환한다.
        feat, feat_dec = model(xb)

        # 평가에서는 warmup 이후 상태의 loss 정의를 사용한다.
        # 반환값 중 loss_n이 실제 anomaly score다.
        _, _, loss_n, _ = roca_loss(
            feat,
            feat_dec,
            center,
            epoch=cfg.warmup + 1,
            config=cfg,
            device=device,
        )
        outs.append(loss_n.detach().cpu().numpy())
    err = np.concatenate(outs, axis=0)
    for e, s in zip(ends, err):
        if 0 <= int(e) < t:
            scores[int(e)] = float(s)
    return scores


def train_and_eval(
    *,
    plmn: str,
    epochs: int | None = None,
    seed: int = 2,
    anomaly_ratio: float | None = None,
    threshold_quantile: float = 0.99,
    window: int | None = None,
    time_step: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    nu: float | None = None,
    export_ui: bool = True,
    save_checkpoint: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    # 실험 재현성을 위해 PyTorch와 NumPy seed를 고정한다.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # prepare_plmn()은 11개 core feature 선택, 시간순 60/20/20 split,
    # TRAIN-only MinMaxScaler fit을 수행한다.
    # 이 공통 함수는 y_valid/y_test도 함께 준비하지만,
    # 아래 학습 loop에서는 label을 전혀 사용하지 않고 평가 시점에만 y_valid를 읽는다.
    prep = prepare_plmn(plmn, window=window or 32)
    n_feat = len(prep["cols"])

    # config의 input_channels를 현재 실제 feature 수(현재 11개)에 맞춘다.
    cfg = RoCAConfig(input_channels=n_feat)
    if epochs is not None:
        cfg.num_epoch = int(epochs)
    if anomaly_ratio is not None:
        cfg.anomaly_ratio = float(anomaly_ratio)
    if window is not None:
        cfg.window_size = int(window)
    if time_step is not None:
        cfg.time_step = int(time_step)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if lr is not None:
        cfg.lr = float(lr)
    if nu is not None:
        cfg.nu = float(nu)

    device = _device()
    if not quiet:
        print(
            f"device={device} plmn={plmn} window={cfg.window_size} "
            f"step={cfg.time_step} epochs={cfg.num_epoch} ch={n_feat} "
            f"ar={cfg.anomaly_ratio} nu={cfg.nu} q={threshold_quantile}",
            flush=True,
        )

    # TRAIN 시계열을 겹치는 window로 변환한다.
    # 기본값 window=32, time_step=16이므로 학습 window는 16 point씩 이동한다.
    train_nct, _ = block_to_nct(
        prep["train"],
        window=cfg.window_size,
        time_step=cfg.time_step,
    )

    # RoCA 학습은 실제 anomaly label을 사용하지 않는다.
    # Dataset interface를 맞추기 위한 dummy label이므로 모두 0으로 둔다.
    train_y = np.zeros(len(train_nct), dtype=np.int64)
    train_ds = RoCAWindowDataset(train_nct, train_y, cfg)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.drop_last,
        num_workers=0,
    )
    if not quiet:
        print(f"train_windows={len(train_ds)} valid_len={len(prep['valid'])}", flush=True)

    # 원본 RoCA backbone: Conv1d + LSTM encoder/decoder + projection head.
    model = base_Model(cfg, device).to(device)

    # optimizer 설정은 RoCAConfig 값을 그대로 사용한다.
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight,
    )
    # epoch loss가 개선되지 않으면 learning rate를 낮춘다.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")

    # one-class 학습의 기준이 되는 hypersphere center를 먼저 계산한다.
    # center는 원본/증강 representation들의 평균으로 구한다.
    center = torch.zeros(cfg.project_channels, device=device)
    center = center_c(
        train_dl,
        model,
        device,
        center,
        cfg,
        eps=cfg.center_eps,
    )

    history: list[dict[str, float]] = []
    t0 = time.time()
    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        losses: list[float] = []
        for data, _target, aug1, aug2 in train_dl:
            data = data.float().to(device)
            aug1 = aug1.float().to(device)
            aug2 = aug2.float().to(device)

            # 한 batch 안에서 원본/weak/strong view를 합쳐 동일 backbone에 통과시킨다.
            all_data = torch.cat((data, aug1, aug2), dim=0)

            opt.zero_grad()
            feature1, feature_dec1 = model(all_data)

            # loss_oc은 center 기반 one-class objective,
            # loss_sigam은 representation collapse를 막는 variance regularizer다.
            loss_oc, loss_sigam, _loss_n, _loss_a = roca_loss(
                feature1, feature_dec1, center, epoch, cfg, device
            )
            # 최종 학습 loss = one-class loss + 가중된 variance regularizer.
            loss = loss_oc + cfg.omega * loss_sigam
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # 설정된 초기 epoch 동안 representation이 바뀐 것을 반영해 center를 다시 계산한다.
        if epoch < cfg.change_center_epoch or epoch == 1:
            center = center_c(
                train_dl,
                model,
                device,
                center,
                cfg,
                eps=cfg.center_eps,
            )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        scheduler.step(mean_loss)
        history.append({"epoch": epoch, "loss": mean_loss})
        if not quiet:
            print(f"epoch {epoch}/{cfg.num_epoch} loss={mean_loss:.4f}", flush=True)

    # 학습 때는 stride(time_step)=16을 사용하지만,
    # 모델 간 VUS 비교에서는 가능한 모든 point에 score를 붙이는 것이 필요하다.
    # 따라서 평가 score는 stride=1로 다시 계산해 dense point series를 만든다.
    train_scores = score_block(
        model, center, prep["train"], cfg=cfg, device=device, time_step=1
    )
    valid_scores = score_block(
        model, center, prep["valid"], cfg=cfg, device=device, time_step=1
    )
    # detection threshold 역시 label 없이 TRAIN score에서만 결정한다.
    thr = train_quantile_threshold(train_scores, threshold_quantile)

    # 여기서부터 validation label을 실제 평가에 사용한다.
    y_valid = prep["y_valid"]
    eval_bundle = evaluate_valid(
        y_valid, valid_scores, thr, vus_window=None, sample_minutes=5.0
    )

    out: dict[str, Any] = {
        "plmn": plmn,
        "threshold": thr,
        "threshold_quantile": float(threshold_quantile),
        "eval_split": "valid",
        "metrics": {
            "event_f1": eval_bundle["event"].get("f1"),
            "false_alarms_per_day": eval_bundle["operational"].get(
                "false_alarms_per_day"
            ),
            **{k: eval_bundle["point"].get(k) for k in ("precision", "recall", "f1")},
        },
        "config": {
            "window_size": cfg.window_size,
            "time_step": cfg.time_step,
            "epochs": cfg.num_epoch,
            "anomaly_ratio": cfg.anomaly_ratio,
            "threshold_quantile": float(threshold_quantile),
            "nu": cfg.nu,
            "lr": cfg.lr,
            "seed": seed,
        },
        "history": history,
        "elapsed_sec": time.time() - t0,
    }

    # 재현과 후속 분석을 위해 학습된 weight, center, config를 저장한다.
    run_dir = MODEL_ROOT / plmn / "paperib"
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        ckpt = {
            "model": model.state_dict(),
            "center": center.detach().cpu(),
            "config": out["config"],
            "n_features": n_feat,
            "cols": prep["cols"],
        }
        torch.save(ckpt, run_dir / "roca.pt")
        (run_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    # UI 및 공통 평가 파이프라인에서 사용할 validation prediction JSON을 만든다.
    times_valid = prep["times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ]
    if export_ui:
        path = write_roca_prediction(
            plmn=plmn,
            times_valid=times_valid,
            scores_valid=valid_scores,
            y_valid=y_valid,
            threshold=thr,
            feature_columns=prep["cols"],
            window=cfg.window_size,
            split=prep["split"],
            metrics_extra={
                "model": "roca",
                "threshold_quantile": float(threshold_quantile),
                "score_distribution": eval_bundle.get("score_distribution"),
                "event_score_diagnostics": eval_bundle.get("event_score_diagnostics"),
                "upstream": "ruiking04/RoCA",
                "train_method": cfg.train_method,
                "nu": cfg.nu,
                "anomaly_ratio": cfg.anomaly_ratio,
                "seed": seed,
                "epochs": cfg.num_epoch,
                "lr": cfg.lr,
            },
        )
        out["pred_path"] = path
        if not quiet:
            print(f"exported {path}", flush=True)
            m = out["metrics"]
            print(
                f"valid Event-F1={m['event_f1']} "
                f"P={m['precision']} R={m['recall']} F1={m['f1']} "
                f"FA/day={m['false_alarms_per_day']}",
                flush=True,
            )
            sd = eval_bundle.get("score_distribution") or {}
            print(
                f"valid score>thr fraction={sd.get('fraction_above_threshold')} "
                f"q95={sd.get('q95')} max={sd.get('max')}",
                flush=True,
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--anomaly_ratio", type=float, default=1.0)
    p.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.99,
        help="TRAIN score quantile 기반 detection threshold (기본값: 0.99)",
    )
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--time_step", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--no_export", action="store_true")
    args = p.parse_args()
    train_and_eval(
        plmn=args.plmn,
        epochs=args.epochs,
        seed=args.seed,
        anomaly_ratio=args.anomaly_ratio,
        threshold_quantile=args.threshold_quantile,
        window=args.window,
        time_step=args.time_step,
        batch_size=args.batch_size,
        lr=args.lr,
        nu=args.nu,
        export_ui=not args.no_export,
    )


if __name__ == "__main__":
    # RoCA 디렉터리 안에서 python run_plmn.py 형태로 실행해도
    # config.py, dataset.py 같은 로컬 모듈을 찾을 수 있도록 경로를 보정한다.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
